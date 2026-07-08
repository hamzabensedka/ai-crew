"""Model assignments per agent role with free-tier provider equivalents."""

from __future__ import annotations

from autocrew.squad.squad_model import AgentRole

# NIM (primary) model IDs — OpenAI-compatible on integrate.api.nvidia.com
# Verified on integrate.api.nvidia.com (kimi-k2.6 / deepseek-v3.1 often 404 on free tier)
NIM_REASONING_MODEL = "meta/llama-3.3-70b-instruct"
NIM_CODER_MODEL = "deepseek-ai/deepseek-v4-pro"
NIM_REVIEWER_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Groq free-tier equivalents (no nested vendor/ prefix in model id)
GROQ_REASONING_MODEL = "moonshotai/kimi-k2-instruct"
GROQ_CODER_MODEL = "qwen-qwen3-32b"

# Cerebras free-tier equivalents
CEREBRAS_REASONING_MODEL = "llama-3.3-70b"
CEREBRAS_CODER_MODEL = "qwen-3-32b"

# OpenRouter (paid last resort)
OR_REASONING_MODEL = "moonshotai/kimi-k2"
OR_CODER_MODEL = "deepseek/deepseek-chat"
OR_REVIEWER_MODEL = "moonshotai/kimi-k2"

REASONING_ROLES = frozenset({
    AgentRole.PRODUCT_OWNER,
    AgentRole.ARCHITECT,
})

CODER_ROLES = frozenset({
    AgentRole.DEVOPS,
    AgentRole.BACKEND_DEV,
    AgentRole.FRONTEND_DEV,
    AgentRole.FULLSTACK_DEV,
    AgentRole.DATA_ENGINEER,
    AgentRole.AI_ENGINEER,
    AgentRole.TESTER,
})

REVIEWER_ROLES = frozenset({AgentRole.CODE_REVIEWER})

CORE_DEBATER_ROLES = frozenset({
    AgentRole.PRODUCT_OWNER,
    AgentRole.ARCHITECT,
    AgentRole.DEVOPS,
})

CONSULTANT_ROLES = frozenset({
    AgentRole.BACKEND_DEV,
    AgentRole.FRONTEND_DEV,
    AgentRole.FULLSTACK_DEV,
    AgentRole.DATA_ENGINEER,
    AgentRole.AI_ENGINEER,
    AgentRole.TESTER,
})

ModelTier = str  # "reasoning" | "coder" | "reviewer"


def model_tier_for_role(role: AgentRole) -> ModelTier | None:
    if role == AgentRole.PROGRESS_TRACKER:
        return None
    if role in REASONING_ROLES:
        return "reasoning"
    if role in REVIEWER_ROLES:
        return "reviewer"
    if role in CODER_ROLES:
        return "coder"
    return "coder"


def nim_model_for_tier(tier: ModelTier) -> str:
    if tier == "reasoning":
        return NIM_REASONING_MODEL
    if tier == "reviewer":
        return NIM_REVIEWER_MODEL
    return NIM_CODER_MODEL


def groq_model_for_tier(tier: ModelTier) -> str:
    if tier == "reasoning" or tier == "reviewer":
        return GROQ_REASONING_MODEL
    return GROQ_CODER_MODEL


def cerebras_model_for_tier(tier: ModelTier) -> str:
    if tier == "reasoning" or tier == "reviewer":
        return CEREBRAS_REASONING_MODEL
    return CEREBRAS_CODER_MODEL


def openrouter_model_for_tier(tier: ModelTier) -> str:
    if tier == "reasoning":
        return OR_REASONING_MODEL
    if tier == "reviewer":
        return OR_REVIEWER_MODEL
    return OR_CODER_MODEL
