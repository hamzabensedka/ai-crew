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
from autocrew.squad.squad_model import AgentConfig
from autocrew.tasks.task_model import TaskConfig
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


def _list_project_tree(project_root: str, max_entries: int = 40) -> str:
    from pathlib import Path

    root = Path(project_root)
    if not root.is_dir():
        return "(empty project root)"
    skip = {".git", "node_modules", "__pycache__", ".nx", "dist", "build", ".venv"}
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            lines.append(rel)
            if len(lines) >= max_entries:
                lines.append("...")
                break
    return "\n".join(lines) if lines else "(no files yet)"


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

    model_label = model_name.split("/")[-1] if model_name else "LLM"
    logger.log(f"Calling {model_label} for task '{task.title}'")

    try:
        data = call_with_json_retry(llm_call, prompt, max_retries=1)
    except LLMError:
        raw = llm_call(prompt)
        data = _extract_json_object(raw)

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
