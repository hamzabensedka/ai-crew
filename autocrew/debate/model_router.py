"""Route debate/build agents to different LLM models.

Supports routing modes:
- RoleModelRouter: free-tier chain per role tier (reasoning / coder / reviewer)
- DualModelRouter: 2 groups (planning vs implementation)
- PerAgentModelRouter: a unique model per agent role (10 models max)
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from autocrew.analyzer.llm_client import LLMClient
from autocrew.analyzer.litellm_chain import LiteLLMFallbackClient, create_chain_client_for_tier
from autocrew.analyzer.model_registry import (
    model_tier_for_role,
    nim_model_for_tier,
)
from autocrew.squad.squad_model import AgentConfig, AgentRole
from autocrew.tasks.task_model import TaskConfig


class ModelRouter(Protocol):
    def for_agent(self, agent: AgentConfig) -> tuple[LLMClient, str]: ...

    def for_build_task(self, agent: AgentConfig, task: TaskConfig) -> tuple[LLMClient, str]: ...

# Heavy doc/code generation tasks — use implementation model even for planning roles
# Code-heavy doc tasks only; PO product spec stays on reasoning model (Kimi)
BUILD_IMPLEMENTATION_TASKS = frozenset({"arch_design"})

# Planning / review agents — typically stronger reasoning models (e.g. Kimi)
PLANNING_ROLES = {
    AgentRole.PRODUCT_OWNER,
    AgentRole.ARCHITECT,
    AgentRole.CODE_REVIEWER,
    AgentRole.PROGRESS_TRACKER,
    AgentRole.TESTER,
}

# Implementation agents — typically code-focused models (e.g. DeepSeek)
IMPLEMENTATION_ROLES = {
    AgentRole.BACKEND_DEV,
    AgentRole.FRONTEND_DEV,
    AgentRole.FULLSTACK_DEV,
    AgentRole.DEVOPS,
    AgentRole.DATA_ENGINEER,
    AgentRole.AI_ENGINEER,
}


class DualModelRouter:
    """Send planning agents to model A and implementation agents to model B."""

    def __init__(
        self,
        planning_llm: LLMClient,
        implementation_llm: LLMClient,
        planning_model: str,
        implementation_model: str,
    ) -> None:
        self.planning_llm = planning_llm
        self.implementation_llm = implementation_llm
        self.planning_model = planning_model
        self.implementation_model = implementation_model

    def for_agent(self, agent: AgentConfig) -> tuple[LLMClient, str]:
        if agent.role in PLANNING_ROLES:
            return self.planning_llm, self.planning_model
        if agent.role in IMPLEMENTATION_ROLES:
            return self.implementation_llm, self.implementation_model
        return self.planning_llm, self.planning_model

    def for_build_task(self, agent: AgentConfig, task: TaskConfig) -> tuple[LLMClient, str]:
        """Debate stays dual-model; large file generation uses the implementation model."""
        if task.task_id in BUILD_IMPLEMENTATION_TASKS:
            return self.implementation_llm, self.implementation_model
        if task.output_path and task.output_path.endswith(".md") and task.task_id == "arch_design":
            return self.implementation_llm, self.implementation_model
        return self.for_agent(agent)

    def summary(self) -> str:
        return (
            f"Planning agents -> {self.planning_model}\n"
            f"Implementation agents -> {self.implementation_model}"
        )


class PerAgentModelRouter:
    """Route each agent role to a specific LLM model.

    A per-agent model mapping overrides the dual-model split. Any role not in
    the mapping falls back to the default (planning) model.
    """

    def __init__(
        self,
        role_model_map: dict[str, tuple[LLMClient, str]],
        *,
        default_llm: LLMClient,
        default_model: str,
    ) -> None:
        self.role_model_map = role_model_map
        self.default_llm = default_llm
        self.default_model = default_model

    @property
    def planning_model(self) -> str:
        """Model used by planning agents (product_owner, or default)."""
        if "product_owner" in self.role_model_map:
            return self.role_model_map["product_owner"][1]
        return self.default_model

    @property
    def implementation_model(self) -> str:
        """Model used by implementation agents (backend_developer, or default)."""
        if "backend_developer" in self.role_model_map:
            return self.role_model_map["backend_developer"][1]
        if "fullstack_developer" in self.role_model_map:
            return self.role_model_map["fullstack_developer"][1]
        return self.default_model

    def for_agent(self, agent: AgentConfig) -> tuple[LLMClient, str]:
        key = agent.role.value
        if key in self.role_model_map:
            return self.role_model_map[key]
        return self.default_llm, self.default_model

    def for_build_task(self, agent: AgentConfig, task: TaskConfig) -> tuple[LLMClient, str]:
        """Per-agent routing for build tasks (same as debate — each agent keeps its model)."""
        return self.for_agent(agent)

    def summary(self) -> str:
        lines = ["Per-agent model routing:"]
        for role in AgentRole:
            key = role.value
            if key in self.role_model_map:
                _, model = self.role_model_map[key]
                lines.append(f"  {role.value} -> {model}")
            else:
                lines.append(f"  {role.value} -> {self.default_model} (default)")
        return "\n".join(lines)

    def models_used(self) -> dict[str, str]:
        """Return a dict of role -> model_name for logging."""
        result: dict[str, str] = {}
        for role in AgentRole:
            key = role.value
            if key in self.role_model_map:
                result[key] = self.role_model_map[key][1]
            else:
                result[key] = self.default_model
        return result


class RoleModelRouter:
    """Route each agent role to a model tier with NIM→Groq→Cerebras→OpenRouter fallback."""

    def __init__(self) -> None:
        self._clients: dict[str, LiteLLMFallbackClient] = {}
        self._models: dict[str, str] = {}

    def _client_for_tier(self, tier: str) -> LiteLLMFallbackClient:
        if tier not in self._clients:
            self._clients[tier] = create_chain_client_for_tier(tier)
            self._models[tier] = nim_model_for_tier(tier)
        return self._clients[tier]

    def for_agent(self, agent: AgentConfig) -> tuple[LLMClient, str]:
        tier = model_tier_for_role(agent.role)
        if tier is None:
            raise ValueError(f"No LLM model for role {agent.role.value}")
        client = self._client_for_tier(tier)
        return client, self._models[tier]

    def for_build_task(self, agent: AgentConfig, task: TaskConfig) -> tuple[LLMClient, str]:
        if task.task_id in BUILD_IMPLEMENTATION_TASKS:
            client = self._client_for_tier("coder")
            return client, self._models.get("coder", nim_model_for_tier("coder"))
        return self.for_agent(agent)

    @property
    def planning_model(self) -> str:
        return self._models.get("reasoning", nim_model_for_tier("reasoning"))

    @property
    def implementation_model(self) -> str:
        return self._models.get("coder", nim_model_for_tier("coder"))

    def summary(self) -> str:
        lines = ["Role-based model routing (NIM primary, Groq/Cerebras/OpenRouter fallback):"]
        for tier in ("reasoning", "coder", "reviewer"):
            if tier in self._models:
                lines.append(f"  {tier} -> {self._models[tier]}")
        return "\n".join(lines)

    def models_used(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for role in AgentRole:
            tier = model_tier_for_role(role)
            if tier is None:
                result[role.value] = "deterministic"
            else:
                result[role.value] = self._models.get(tier, nim_model_for_tier(tier))
        return result


def parse_per_agent_models(config_str: str) -> dict[str, str]:
    """Parse a JSON string mapping role names to model names.

    Returns empty dict if config_str is empty or invalid.
    """
    if not config_str or not config_str.strip():
        return {}
    try:
        data = json.loads(config_str)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if v}
    except (json.JSONDecodeError, TypeError):
        return {}
