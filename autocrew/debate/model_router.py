"""Route debate agents to different LLM models."""

from __future__ import annotations

from autocrew.analyzer.llm_client import LLMClient
from autocrew.squad.squad_model import AgentConfig, AgentRole
from autocrew.tasks.task_model import TaskConfig

# Heavy doc/code generation tasks — use implementation model even for planning roles
BUILD_IMPLEMENTATION_TASKS = frozenset({"arch_design", "po_product_spec"})

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
