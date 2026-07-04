"""Execute build tasks via LLM (NVIDIA / Anthropic / OpenAI)."""

from __future__ import annotations

import json
import re
from typing import Callable

from autocrew.analyzer.llm_client import LLMError, call_with_json_retry
from autocrew.analyzer.project_model import ProjectContext
from autocrew.config import settings
from autocrew.crew.task_context import inject_task_context
from autocrew.crew.crew_logger import CrewLogger
from autocrew.metrics import begin_session, end_session
from autocrew.metrics.instrumentation import instrument_llm_call
from autocrew.squad.squad_model import AgentConfig
from autocrew.tasks.task_model import TaskConfig
from autocrew.context.path_filter import is_scannable_path, iter_source_files
from autocrew.crew.build_rules import find_pattern_examples
from autocrew.tools.file_tools import read_file, write_file


BUILD_TASK_PROMPT = """You are {agent_name}, role: {agent_role}.
Goal: {goal}
Backstory: {backstory}

Project: {project_name}
Task ID: {task_id}
Task title: {task_title}

Instructions:
{task_body}

Output format: {output_format}
Primary output path: {output_path}

You may write files only under these path prefixes: {write_scopes}
Use paths relative to the project root with forward slashes.

If the task modifies existing code, produce complete updated file contents (not diffs).
Match the project's existing stack, patterns, and conventions from the context files.

Return JSON only:
{{
  "files": [
    {{"path": "relative/path.ext", "content": "full file content as a string"}}
  ],
  "summary": "one sentence describing what you implemented"
}}

Rules:
- Include at least one file when output_format is file, markdown, or code.
- For report tasks, write markdown to the primary output path or output/reports/.
- Do not wrap the JSON in markdown fences.
- Escape newlines inside content strings properly for valid JSON.
"""


DOC_TASK_PROMPT = """You are {agent_name}, role: {agent_role}.
Goal: {goal}
Backstory: {backstory}

Project: {project_name}
Task: {task_title}

Instructions:
{task_body}

Write ONE markdown document for: {output_path}
Keep it concise but complete (under 6000 words).

Return JSON only:
{{
  "content": "full markdown document as one JSON string",
  "summary": "one sentence"
}}

Rules:
- Valid JSON only, no markdown fences around the JSON.
- Escape newlines in the content string.
"""


def _is_doc_task(task: TaskConfig) -> bool:
    return bool(
        task.output_path
        and task.output_path.endswith(".md")
        and task.task_id in ("arch_design", "po_product_spec", "review_code", "track_progress")
    )


def _list_project_tree(project_root: str, max_entries: int = 40) -> str:
    lines = iter_source_files(project_root, max_entries=max_entries)
    if not lines:
        return "(no source files yet)"
    if len(lines) >= max_entries:
        lines.append("...")
    return "\n".join(lines)


def _parse_files_payload(data: dict, task: TaskConfig) -> list[dict[str, str]]:
    files = data.get("files")
    if isinstance(files, list) and files:
        result: list[dict[str, str]] = []
        for item in files:
            if isinstance(item, dict) and item.get("path") and item.get("content") is not None:
                result.append({"path": str(item["path"]), "content": str(item["content"])})
        if result:
            return result

    content = data.get("content") or data.get("markdown") or data.get("report")
    if isinstance(content, str) and content.strip():
        path = task.output_path or f"output/build/{task.task_id}.md"
        return [{"path": path, "content": content}]

    summary = data.get("summary")
    if isinstance(summary, str) and summary.strip() and task.output_path:
        return [{"path": task.output_path, "content": summary}]

    raise LLMError("LLM response missing 'files' array with path and content")


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def execute_task_with_llm(
    task: TaskConfig,
    agent: AgentConfig,
    context: ProjectContext,
    project_root: str,
    logger: CrewLogger,
    llm_call: Callable[[str], str],
    *,
    model_name: str = "",
) -> str:
    """Run one task through the LLM and write returned files to disk."""
    task_body = inject_task_context(task, project_root)
    tree = _list_project_tree(project_root)

    if _is_doc_task(task):
        prompt = DOC_TASK_PROMPT.format(
            agent_name=agent.name,
            agent_role=agent.role.value,
            goal=agent.goal,
            backstory=agent.backstory,
            project_name=context.project_name,
            task_title=task.title,
            task_body=task_body[:8000],
            output_path=task.output_path,
        )
    else:
        prompt = BUILD_TASK_PROMPT.format(
            agent_name=agent.name,
            agent_role=agent.role.value,
            goal=agent.goal,
            backstory=agent.backstory,
            project_name=context.project_name,
            task_id=task.task_id,
            task_title=task.title,
            task_body=task_body[:14000],
            output_format=task.output_format,
            output_path=task.output_path or "(choose an appropriate path under allowed scopes)",
            write_scopes=", ".join(agent.can_write_to),
        )
        prompt += f"\n\nExisting project files (sample):\n{tree}\n"
        patterns = find_pattern_examples(project_root, task, max_files=2)
        if patterns:
            prompt += "\nPattern examples (follow these conventions):\n"
            for pattern_path in patterns:
                try:
                    content = read_file(
                        pattern_path,
                        project_root,
                        agent.can_read,
                        enforce_scope=settings.enforce_scope,
                    )
                    prompt += f"\n--- Pattern: {pattern_path} ---\n{content[:4000]}\n"
                except (OSError, LLMError):
                    prompt += f"\n--- Pattern: {pattern_path} (unreadable) ---\n"

    model_label = model_name.split("/")[-1] if model_name else "LLM"
    logger.log(f"Calling {model_label} for task '{task.title}'")

    measured_call = instrument_llm_call(
        llm_call,
        phase="build",
        agent_name=agent.name,
        agent_role=agent.role.value,
        model_name=model_name or "LLM",
        task_id=task.task_id,
    )

    try:
        data = call_with_json_retry(measured_call, prompt, max_retries=1)
    except LLMError:
        try:
            raw = measured_call(prompt)
            data = _extract_json_object(raw)
        except (json.JSONDecodeError, LLMError):
            # Fallback: treat raw text as document content for doc tasks
            if _is_doc_task(task) and task.output_path:
                raw_stripped = raw.strip() if raw else ""
                if raw_stripped:
                    data = {"content": raw_stripped, "summary": f"Generated: {task.title}"}
                else:
                    raise LLMError(f"LLM returned empty response for task '{task.title}'")
            else:
                raise LLMError(f"LLM response could not be parsed as JSON for task '{task.title}'")

    if not isinstance(data, dict):
        raise LLMError("LLM build response must be a JSON object")

    files = _parse_files_payload(data, task)
    written: list[str] = []
    for item in files:
        path = item["path"].replace("\\", "/").lstrip("./")
        write_file(
            path,
            item["content"],
            project_root,
            agent.can_write_to,
            enforce_scope=settings.enforce_scope,
        )
        written.append(path)
        logger.log(f"Wrote {path} ({len(item['content'])} bytes)")

    summary = str(data.get("summary", f"Completed: {task.title}"))
    if written:
        summary += f" → {', '.join(written)}"
    return summary