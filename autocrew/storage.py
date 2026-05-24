"""Persistence helpers for context, squad, and task snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from autocrew.analyzer.project_model import ProjectContext
from autocrew.squad.squad_model import Squad
from autocrew.tasks.task_model import TaskConfig


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def save_context(context: ProjectContext, contexts_dir: str) -> str:
    path = Path(contexts_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(context.project_name)}_context.json"
    filepath = path / filename
    filepath.write_text(json.dumps(context.to_dict(), indent=2), encoding="utf-8")
    return str(filepath)


def load_context(path: str) -> ProjectContext:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectContext.from_dict(data)


def save_squad(squad: Squad, squads_dir: str) -> str:
    path = Path(squads_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(squad.project_name)}_squad.json"
    filepath = path / filename
    filepath.write_text(json.dumps(squad.to_dict(), indent=2), encoding="utf-8")
    return str(filepath)


def load_squad(path: str) -> Squad:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Squad.from_dict(data)


def save_tasks(tasks: list[TaskConfig], output_dir: str, project_name: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(project_name)}_tasks.json"
    filepath = path / filename
    filepath.write_text(
        json.dumps([t.to_dict() for t in tasks], indent=2),
        encoding="utf-8",
    )
    return str(filepath)


def load_tasks(path: str) -> list[TaskConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TaskConfig.from_dict(item) for item in data]


def find_latest_context(contexts_dir: str) -> str | None:
    path = Path(contexts_dir)
    if not path.exists():
        return None
    files = sorted(path.glob("*_context.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def find_latest_squad(squads_dir: str) -> str | None:
    path = Path(squads_dir)
    if not path.exists():
        return None
    files = sorted(path.glob("*_squad.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def find_latest_tasks(output_dir: str, project_name: str | None = None) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    pattern = f"{_slug(project_name)}_tasks.json" if project_name else "*_tasks.json"
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def find_latest_report(reports_dir: str) -> str | None:
    path = Path(reports_dir)
    if not path.exists():
        return None
    files = sorted(path.glob("progress_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def find_latest_debate(output_dir: str, project_name: str) -> str | None:
    slug = _slug(project_name)
    path = Path(output_dir) / "debate" / slug / "debate_result.json"
    return str(path) if path.is_file() else None
