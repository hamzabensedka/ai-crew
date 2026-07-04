"""Debate-phase agent tiers — core debaters, consultants, post-debate review."""

from __future__ import annotations

from autocrew.analyzer.model_registry import CONSULTANT_ROLES, CORE_DEBATER_ROLES
from autocrew.squad.squad_model import AgentConfig, AgentRole, Squad


def build_core_debate_tiers(squad: Squad) -> list[list[AgentConfig]]:
    """Full multi-round debate participants: PO → Architect → DevOps."""
    by_role = {agent.role: agent for agent in squad.agents}
    tiers: list[list[AgentConfig]] = []
    for role in (AgentRole.PRODUCT_OWNER, AgentRole.ARCHITECT, AgentRole.DEVOPS):
        if role in by_role and role in CORE_DEBATER_ROLES:
            tiers.append([by_role[role]])
    return tiers


def get_consultant_agents(squad: Squad) -> list[AgentConfig]:
    """One-shot consultants — constraints injected before round 1."""
    order = (
        AgentRole.BACKEND_DEV,
        AgentRole.FRONTEND_DEV,
        AgentRole.FULLSTACK_DEV,
        AgentRole.DATA_ENGINEER,
        AgentRole.AI_ENGINEER,
        AgentRole.TESTER,
    )
    by_role = {agent.role: agent for agent in squad.agents}
    return [by_role[role] for role in order if role in by_role and role in CONSULTANT_ROLES]


def get_code_reviewer(squad: Squad) -> AgentConfig | None:
    if AgentRole.CODE_REVIEWER in {a.role for a in squad.agents}:
        return next(a for a in squad.agents if a.role == AgentRole.CODE_REVIEWER)
    return None


def build_debate_tiers(
    squad: Squad,
    *,
    randomize_dev_order: bool | None = None,
    seed: int | None = None,
) -> list[list[AgentConfig]]:
    """Legacy tier builder — delegates to core debater tiers only."""
    _ = randomize_dev_order, seed
    return build_core_debate_tiers(squad)


def flatten_tiers(tiers: list[list[AgentConfig]]) -> list[AgentConfig]:
    agents: list[AgentConfig] = []
    for tier in tiers:
        agents.extend(tier)
    return agents


def is_parallel_tier(tier: list[AgentConfig]) -> bool:
    return len(tier) > 1


def get_dev_tier_order(tiers: list[list[AgentConfig]]) -> list[AgentRole]:
    for tier in tiers:
        if is_parallel_tier(tier):
            return [agent.role for agent in tier]
    return []
