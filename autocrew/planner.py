"""Render planning documents from templates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from autocrew.analyzer.project_model import ProjectContext
from autocrew.squad.squad_model import Squad
from autocrew.tasks.task_model import TaskConfig

TEMPLATES_DIR = Path(__file__).parent / "docs" / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(default=False),
    )


def render_product_doc(context: ProjectContext) -> str:
    env = _env()
    template = env.get_template("product.md.j2")
    return template.render(
        project_name=context.project_name,
        description=context.description,
        domain=context.domain.value,
        frontend=context.tech_stack.frontend,
        backend=context.tech_stack.backend,
        devops=context.tech_stack.devops,
        special_requirements=context.special_requirements,
        features=context.features,
        missing_parts=context.missing_parts,
    )


def render_architecture_doc(context: ProjectContext) -> str:
    env = _env()
    template = env.get_template("architecture.md.j2")
    return template.render(
        project_name=context.project_name,
        description=context.description,
        frontend=context.tech_stack.frontend,
        backend=context.tech_stack.backend,
        devops=context.tech_stack.devops,
    )


def render_tasks_doc(squad: Squad, tasks: list[TaskConfig]) -> str:
    env = _env()
    template = env.get_template("tasks.md.j2")
    return template.render(
        project_name=squad.project_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        execution_order=squad.execution_order,
        parallel_groups=squad.parallel_groups,
        tasks=tasks,
    )


def write_plan_docs(
    context: ProjectContext,
    squad: Squad,
    tasks: list[TaskConfig],
    project_root: str,
) -> dict[str, str]:
    root = Path(project_root)
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "product": str(docs_dir / "product.md"),
        "architecture": str(docs_dir / "architecture.md"),
        "tasks": str(docs_dir / "tasks.md"),
    }

    (docs_dir / "product.md").write_text(render_product_doc(context), encoding="utf-8")
    (docs_dir / "architecture.md").write_text(render_architecture_doc(context), encoding="utf-8")
    (docs_dir / "tasks.md").write_text(render_tasks_doc(squad, tasks), encoding="utf-8")

    return paths
