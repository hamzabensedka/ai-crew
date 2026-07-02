"""Debate-phase agent tiers for parallel execution."""

from __future__ import annotations

import random

from autocrew.config import settings
from autocrew.squad.squad_model import AgentConfig, AgentRole, Squad

DEV_ROLES = (
    AgentRole.BACKEND_DEV,
    AgentRole.FRONTEND_DEV,
    AgentRole.FULLSTACK_DEV,
    AgentRole.DEVOPS,
    AgentRole.DATA_ENGINEER,
    AgentRole.AI_ENGINEER,
)

SEQUENTIAL_TAIL = (
    AgentRole.TESTER,
    AgentRole.CODE_REVIEWER,
    AgentRole.PROGRESS_TRACKER,
)


def build_debate_tiers(
    squad: Squad,
    *,
    randomize_dev_order: bool | None = None,
    seed: int | None = None,
) -> list[list[AgentConfig]]:
    """
    Debate tiers: PO → Architect → [dev roles in parallel] → Tester → Reviewer → Tracker.

    Mirrors build-phase dependency grouping for the dev-adjacent seats; Tester, Reviewer,
    and Tracker remain single-agent tiers so Step 3 full-context rules still apply.

    When ``randomize_dev_order`` is True (or the global config flag is set), the dev
    roles within the parallel tier are shuffled per debate round. This supports the
    Step 7 order-effect experiment. PO and Architect are always fixed at the front
    (they are deliberately ordered); the tail (Tester/Reviewer/Tracker) is also fixed.
    """
    by_role = {agent.role: agent for agent in squad.agents}
    tiers: list[list[AgentConfig]] = []

    for role in (AgentRole.PRODUCT_OWNER, AgentRole.ARCHITECT):
        if role in by_role:
            tiers.append([by_role[role]])

    dev_agents = [by_role[role] for role in DEV_ROLES if role in by_role]

    should_randomize = (
        randomize_dev_order
        if randomize_dev_order is not None
        else settings.debate_randomize_dev_order
    )
    if should_randomize and len(dev_agents) > 1:
        rng = random.Random(seed)
        rng.shuffle(dev_agents)

    if dev_agents:
        tiers.append(dev_agents)

    for role in SEQUENTIAL_TAIL:
        if role in by_role:
            tiers.append([by_role[role]])

    return tiers


def flatten_tiers(tiers: list[list[AgentConfig]]) -> list[AgentConfig]:
    agents: list[AgentConfig] = []
    for tier in tiers:
        agents.extend(tier)
    return agents


def is_parallel_tier(tier: list[AgentConfig]) -> bool:
    return len(tier) > 1


def get_dev_tier_order(tiers: list[list[AgentConfig]]) -> list[AgentRole]:
    """Extract the role order of the dev-adjacent tier (for logging/comparison)."""
    for tier in tiers:
        if is_parallel_tier(tier):
            return [agent.role for agent in tier]
    return []