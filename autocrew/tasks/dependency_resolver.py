"""Topological sort and cycle detection for task dependencies."""

from __future__ import annotations

from autocrew.tasks.task_model import TaskConfig


class DependencyError(Exception):
    pass


def resolve_dependencies(tasks: list[TaskConfig]) -> list[TaskConfig]:
    """Return tasks in dependency order. Raises DependencyError on cycles or missing deps."""
    task_map = {t.task_id: t for t in tasks}
    visited: set[str] = set()
    temp: set[str] = set()
    ordered: list[TaskConfig] = []

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in temp:
            raise DependencyError(f"Cycle detected involving task: {task_id}")
        if task_id not in task_map:
            raise DependencyError(f"Unknown dependency: {task_id}")
        temp.add(task_id)
        for dep in task_map[task_id].depends_on:
            visit(dep)
        temp.remove(task_id)
        visited.add(task_id)
        ordered.append(task_map[task_id])

    for task in tasks:
        visit(task.task_id)

    return ordered
