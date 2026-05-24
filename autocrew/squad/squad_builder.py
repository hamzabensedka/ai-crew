"""Builds a tailored Squad from ProjectContext."""

from __future__ import annotations

from datetime import datetime, timezone

from autocrew.analyzer.project_model import ProjectContext, ProjectDomain
from autocrew.squad.role_templates import ROLE_TEMPLATES
from autocrew.squad.squad_model import AgentConfig, AgentRole, Squad


def _format_stack(items: list[str]) -> str:
    return ", ".join(items) if items else "general purpose stack"


def _determine_roles(context: ProjectContext) -> list[AgentRole]:
    roles: list[AgentRole] = [
        AgentRole.PRODUCT_OWNER,
        AgentRole.ARCHITECT,
    ]

    has_frontend = bool(context.tech_stack.frontend)
    has_backend = bool(context.tech_stack.backend)
    feature_count = len(context.features)

    if has_frontend and has_backend and feature_count <= 4:
        roles.append(AgentRole.FULLSTACK_DEV)
    else:
        if has_backend:
            roles.append(AgentRole.BACKEND_DEV)
        if has_frontend:
            roles.append(AgentRole.FRONTEND_DEV)

    if context.tech_stack.devops or any(
        kw in " ".join(context.special_requirements).lower()
        for kw in ("docker", "ci", "deploy")
    ):
        roles.append(AgentRole.DEVOPS)

    if context.domain == ProjectDomain.DATA_PIPELINE or any(
        kw in f.name.lower() + f.description.lower()
        for f in context.features
        for kw in ("etl", "migration", "pipeline", "database")
    ):
        roles.append(AgentRole.DATA_ENGINEER)

    if context.domain == ProjectDomain.AI_TOOL or any(
        kw in f.name.lower() + f.description.lower()
        for f in context.features
        for kw in ("llm", "embedding", "ai", "fine-tun")
    ):
        roles.append(AgentRole.AI_ENGINEER)

    if feature_count > 5:
        roles.append(AgentRole.TESTER)

    roles.extend([AgentRole.CODE_REVIEWER, AgentRole.PROGRESS_TRACKER])
    return roles


def _monorepo_write_scopes(role: AgentRole) -> list[str]:
    """Extra write paths for Nx/pnpm monorepos (apps/, packages/)."""
    scopes: dict[AgentRole, list[str]] = {
        AgentRole.PRODUCT_OWNER: ["docs"],
        AgentRole.ARCHITECT: ["apps", "packages", "docs"],
        AgentRole.BACKEND_DEV: ["apps/api", "packages/shared", "packages/database"],
        AgentRole.FRONTEND_DEV: ["apps/mobile", "packages/ui", "packages/shared"],
        AgentRole.FULLSTACK_DEV: ["apps", "packages"],
        AgentRole.DEVOPS: ["apps", ".github", "docker-compose.yml"],
        AgentRole.DATA_ENGINEER: ["apps/api", "packages/database", "apps/api/prisma"],
        AgentRole.AI_ENGINEER: ["apps/api", "apps/mobile", "packages/shared"],
        AgentRole.TESTER: ["apps", "packages"],
        AgentRole.CODE_REVIEWER: ["output/reports", "docs"],
        AgentRole.PROGRESS_TRACKER: ["output/reports", "docs"],
    }
    return scopes.get(role, [])


def _build_agent(role: AgentRole, context: ProjectContext) -> AgentConfig:
    template = ROLE_TEMPLATES[role]
    tech_stack_str = _format_stack(
        context.tech_stack.frontend
        + context.tech_stack.backend
        + context.tech_stack.devops
        + context.tech_stack.other
    )
    fmt = {
        "project_name": context.project_name,
        "domain": context.domain.value.replace("_", " "),
        "tech_stack": tech_stack_str,
        "frontend_stack": _format_stack(context.tech_stack.frontend),
        "backend_stack": _format_stack(context.tech_stack.backend),
        "devops_stack": _format_stack(context.tech_stack.devops),
    }
    can_write_to = list(template["can_write_to"])
    if any(req.lower() == "monorepo" for req in context.special_requirements):
        for scope in _monorepo_write_scopes(role):
            if scope not in can_write_to:
                can_write_to.append(scope)

    return AgentConfig(
        role=role,
        name=template["name"],
        goal=template["goal_template"].format(**fmt),
        backstory=template["backstory_template"].format(**fmt),
        tools=list(template["tools"]),
        can_write_to=can_write_to,
        can_read=list(template["can_read"]),
        allow_delegation=template.get("allow_delegation", False),
    )


def _determine_execution_plan(
    roles: list[AgentRole],
) -> tuple[list[str], list[list[str]]]:
    dev_roles = {
        AgentRole.BACKEND_DEV,
        AgentRole.FRONTEND_DEV,
        AgentRole.FULLSTACK_DEV,
        AgentRole.DEVOPS,
        AgentRole.DATA_ENGINEER,
        AgentRole.AI_ENGINEER,
    }

    order: list[str] = []
    parallel_groups: list[list[str]] = []

    for role in [AgentRole.PRODUCT_OWNER, AgentRole.ARCHITECT]:
        if role in roles:
            order.append(role.value)

    active_devs = [r for r in roles if r in dev_roles]
    if active_devs:
        if len(active_devs) >= 2:
            parallel_groups.append([r.value for r in active_devs])
        else:
            order.extend(r.value for r in active_devs)

    for role in [AgentRole.TESTER, AgentRole.CODE_REVIEWER, AgentRole.PROGRESS_TRACKER]:
        if role in roles:
            order.append(role.value)

    if AgentRole.PRODUCT_OWNER in roles:
        order.append(AgentRole.PRODUCT_OWNER.value)

    return order, parallel_groups


def build_squad(context: ProjectContext) -> Squad:
    roles_needed = _determine_roles(context)
    agents = [_build_agent(role, context) for role in roles_needed]
    order, parallel_groups = _determine_execution_plan(roles_needed)
    return Squad(
        project_name=context.project_name,
        agents=agents,
        execution_order=order,
        parallel_groups=parallel_groups,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
