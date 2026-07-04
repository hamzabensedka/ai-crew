"""Build-phase rules: skip redundant debate, pattern examples, one feature at a time."""

from __future__ import annotations

from pathlib import Path

from autocrew.context.path_filter import filter_scannable_paths, is_scannable_path
from autocrew.tasks.task_model import TaskConfig

# Established module patterns — debate skipped when these exist.
ESTABLISHED_PATTERNS = (
    "**/businesses.controller.ts",
    "**/*.controller.ts",
    "**/*.service.ts",
    "**/routes/*.ts",
)


def _glob_matches(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def pattern_established(project_root: str) -> bool:
    """True when the codebase already has established controller/service patterns."""
    root = Path(project_root)
    if not root.is_dir():
        return False
    for pattern in ESTABLISHED_PATTERNS:
        matches = _glob_matches(root, pattern)
        scannable = [p for p in matches if is_scannable_path(p.relative_to(root).as_posix(), project_root)]
        if scannable:
            return True
    return False


def should_skip_architecture_debate(project_root: str, *, feature_task: TaskConfig | None = None) -> bool:
    """Skip re-debate when an established pattern module exists."""
    if not pattern_established(project_root):
        return False
    if feature_task is None:
        return True
    title_lower = feature_task.title.lower()
    skip_keywords = ("crud", "controller", "endpoint", "api route", "module")
    return any(k in title_lower for k in skip_keywords)


def find_pattern_examples(
    project_root: str,
    task: TaskConfig,
    *,
    max_files: int = 2,
) -> list[str]:
    """Return 1-2 existing similar source files as pattern references."""
    root = Path(project_root)
    if not root.is_dir():
        return []

    candidates: list[str] = []
    keywords = _task_keywords(task)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not is_scannable_path(rel, project_root):
            continue
        if path.suffix.lower() not in {".ts", ".tsx", ".py", ".js", ".jsx"}:
            continue
        name_lower = path.name.lower()
        if any(k in name_lower or k in rel.lower() for k in keywords):
            candidates.append(rel)

    if not candidates:
        for pattern in ("**/*.controller.ts", "**/*.service.ts", "**/routes/*.ts"):
            for path in _glob_matches(root, pattern):
                rel = path.relative_to(root).as_posix()
                if is_scannable_path(rel, project_root):
                    candidates.append(rel)

    return filter_scannable_paths(candidates[:max_files], project_root)


def _task_keywords(task: TaskConfig) -> tuple[str, ...]:
    text = f"{task.title} {task.description} {task.task_id}".lower()
    keys: list[str] = []
    for token in ("controller", "service", "route", "api", "module", "component", "screen"):
        if token in text:
            keys.append(token)
    if not keys:
        keys.append("controller")
    return tuple(keys)
