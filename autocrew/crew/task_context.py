"""Shared task prompt context for crew build and LLM execution."""

from __future__ import annotations

from autocrew.context.path_filter import is_scannable_path
from autocrew.tasks.task_model import TaskConfig
from autocrew.tools.file_tools import read_file


def inject_task_context(task: TaskConfig, project_root: str) -> str:
    parts = [task.description]
    for ctx_file in task.context_files:
        if not is_scannable_path(ctx_file, project_root):
            continue
        try:
            content = read_file(ctx_file, project_root, ["*"], enforce_scope=False)
            parts.append(f"\n--- Context: {ctx_file} ---\n{content}")
        except (OSError, FileNotFoundError):
            continue
    parts.append(f"\nExpected output: {task.expected_output}")
    return "\n".join(parts)
