"""Generates TaskConfig list from Squad + ProjectContext."""

from __future__ import annotations

import json
from typing import Callable

from autocrew.analyzer.project_model import ProjectContext
from autocrew.analyzer.llm_client import LLMClient, call_with_json_retry
from autocrew.squad.squad_model import AgentRole, Squad
from autocrew.tasks.dependency_resolver import resolve_dependencies
from autocrew.tasks.task_model import TaskConfig

TASK_PROMPT = """You are a project manager creating a task plan for a software project.

Project: {project_name}
Description: {description}
Features: {features}
Tech stack: {tech_stack}

Squad agents (roles): {agent_roles}

Generate a JSON array of tasks. Each task:
{{
  "task_id": "unique_snake_case_id",
  "title": "Short title",
  "description": "Full instruction for the agent",
  "assigned_agent_role": "one of the agent roles listed above",
  "depends_on": ["task_ids this depends on"],
  "output_format": "file|report|code|markdown",
  "output_path": "relative/path/or/null",
  "expected_output": "what success looks like",
  "context_files": ["docs/product.md"]
}}

Rules:
- Product Owner's first task creates docs/product.md
- Architect's first task creates docs/architecture.md and folder scaffold
- Group dev tasks by feature area, not one task per feature
- Include depends_on correctly so PO -> Architect -> Devs -> Reviewer flow works

Return only valid JSON array. No explanation.
"""


def _standard_tasks(squad: Squad, context: ProjectContext) -> list[TaskConfig]:
    """Inject reviewer and tracker tasks if not already present."""
    existing_roles = {t.assigned_agent_role for t in []}
    tasks: list[TaskConfig] = []

    role_values = {a.role.value for a in squad.agents}

    if AgentRole.PRODUCT_OWNER.value in role_values:
        tasks.append(
            TaskConfig(
                task_id="po_product_spec",
                title="Write Product Specification",
                description=(
                    f"Create a complete product specification for {context.project_name}. "
                    f"Cover all features: {[f.name for f in context.features]}"
                ),
                assigned_agent_role=AgentRole.PRODUCT_OWNER.value,
                depends_on=[],
                output_format="file",
                output_path="docs/product.md",
                expected_output="Complete product.md with features, acceptance criteria, and priorities",
            )
        )

    if AgentRole.ARCHITECT.value in role_values:
        tasks.append(
            TaskConfig(
                task_id="arch_design",
                title="Design Architecture",
                description=(
                    f"Design system architecture for {context.project_name} using "
                    f"{context.tech_stack.frontend + context.tech_stack.backend}. "
                    "Create folder scaffold."
                ),
                assigned_agent_role=AgentRole.ARCHITECT.value,
                depends_on=["po_product_spec"],
                output_format="file",
                output_path="docs/architecture.md",
                expected_output="Complete architecture.md with folder structure and service boundaries",
                context_files=["docs/product.md"],
            )
        )

    if AgentRole.CODE_REVIEWER.value in role_values:
        tasks.append(
            TaskConfig(
                task_id="review_code",
                title="Code Review",
                description="Review all code produced for bugs, security issues, and maintainability.",
                assigned_agent_role=AgentRole.CODE_REVIEWER.value,
                depends_on=[],
                output_format="report",
                output_path="output/reports/code_review.md",
                expected_output="Detailed code review report with findings and recommendations",
                context_files=["docs/product.md", "docs/architecture.md"],
            )
        )

    if AgentRole.PROGRESS_TRACKER.value in role_values:
        tasks.append(
            TaskConfig(
                task_id="track_progress",
                title="Progress Report",
                description="Compare codebase against product spec and report completion status.",
                assigned_agent_role=AgentRole.PROGRESS_TRACKER.value,
                depends_on=["review_code"],
                output_format="report",
                output_path="output/reports/progress_report.md",
                expected_output="Progress report with completion percentage and next priorities",
                context_files=["docs/product.md"],
            )
        )

    _ = existing_roles
    return tasks


def _parse_task_list(raw: list[dict]) -> list[TaskConfig]:
    return [TaskConfig.from_dict(item) for item in raw]


def _merge_tasks(llm_tasks: list[TaskConfig], standard: list[TaskConfig]) -> list[TaskConfig]:
    """Merge LLM tasks with standard tasks, avoiding duplicate task_ids."""
    seen = {t.task_id for t in standard}
    merged = list(standard)
    for task in llm_tasks:
        if task.task_id not in seen:
            merged.append(task)
            seen.add(task.task_id)
    return merged


def build_tasks(
    squad: Squad,
    context: ProjectContext,
    llm: LLMClient | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> list[TaskConfig]:
    features_str = json.dumps(
        [{"name": f.name, "description": f.description, "priority": f.priority} for f in context.features]
    )
    tech_str = json.dumps(context.tech_stack.__dict__)
    agent_roles = [a.role.value for a in squad.agents]

    prompt = TASK_PROMPT.format(
        project_name=context.project_name,
        description=context.description,
        features=features_str,
        tech_stack=tech_str,
        agent_roles=", ".join(agent_roles),
    )

    llm_tasks: list[TaskConfig] = []
    if llm_call is not None or llm is not None:
        if llm_call is not None:
            raw = call_with_json_retry(llm_call, prompt)
        else:
            raw = call_with_json_retry(llm.complete, prompt)  # type: ignore[union-attr]
        if isinstance(raw, dict) and "tasks" in raw:
            raw = raw["tasks"]
        llm_tasks = _parse_task_list(raw)

    standard = _standard_tasks(squad, context)
    tasks = _merge_tasks(llm_tasks, standard)

    dev_roles = {
        AgentRole.BACKEND_DEV.value,
        AgentRole.FRONTEND_DEV.value,
        AgentRole.FULLSTACK_DEV.value,
        AgentRole.DEVOPS.value,
        AgentRole.DATA_ENGINEER.value,
        AgentRole.AI_ENGINEER.value,
    }
    for task in tasks:
        if task.assigned_agent_role in dev_roles and "arch_design" not in task.depends_on:
            if task.task_id not in ("po_product_spec", "arch_design"):
                if "arch_design" not in task.depends_on:
                    task.depends_on = list(set(task.depends_on + ["arch_design"]))

    if any(t.task_id == "review_code" for t in tasks):
        dev_task_ids = [t.task_id for t in tasks if t.assigned_agent_role in dev_roles]
        review = next(t for t in tasks if t.task_id == "review_code")
        review.depends_on = list(set(review.depends_on + dev_task_ids))

    return resolve_dependencies(tasks)
