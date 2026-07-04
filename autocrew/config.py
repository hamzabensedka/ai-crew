from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    zenmux_api_key: str = ""
    openrouter_api_key: str = ""
    llm_provider: str = "auto"  # auto | anthropic | openai | nvidia | zenmux | openrouter
    llm_free_tier_chain: bool = True  # NIM → Groq → Cerebras → OpenRouter per role
    default_llm: str = "deepseek-ai/deepseek-v4-pro"
    fallback_llm: str = "moonshotai/kimi-k2.6"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    zenmux_base_url: str = "https://zenmux.ai/api/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    nvidia_enable_thinking: bool = False
    nvidia_max_tokens: int = 16384
    nvidia_temperature: float = 1.0
    nvidia_top_p: float = 0.95
    nvidia_reasoning_budget: int = 4096

    # Step 5: parallel debate tiers for independent seats
    debate_parallel_tiers: bool = True

    # Step 4: early-exit when debate stops raising new concerns/questions
    debate_early_exit: bool = True
    debate_min_rounds: int = 1

    # Step 3: structured critique schema + selective full-text context
    debate_structured_critiques: bool = True
    debate_context_max_chars: int = 12000

    # Step 2: cost/latency/round-count instrumentation
    metrics_enabled: bool = True
    metrics_dir: str = "./output/metrics"

    # Step 1: deterministic Progress Tracker in debate (no LLM for Avery)
    debate_deterministic_tracker: bool = True

    # Step 7: randomize order of dev-adjacent roles within their tier
    debate_randomize_dev_order: bool = False

    # Dual-model debate: planning agents vs implementation agents
    debate_dual_model: bool = True
    debate_planning_model: str = ""  # empty = use FALLBACK_LLM (Kimi)
    debate_implementation_model: str = ""  # empty = use DEFAULT_LLM (DeepSeek)

    # Per-agent model routing: JSON mapping role -> model name
    # e.g. {"product_owner":"claude-3-5-sonnet-20241022","backend_developer":"deepseek-ai/deepseek-v4-pro"}
    debate_per_agent_models: str = ""

    # Provider safety limits (paid OpenRouter fallback)
    openrouter_fallback_ratio_limit: float = 0.20
    openrouter_daily_spend_cap_eur: float = 10.0

    # Early exit: stop after N consecutive rounds with no meaningful change
    debate_stable_rounds_required: int = 2

    # Build: one feature task at a time; skip debate when pattern exists
    build_one_feature_at_a_time: bool = True
    build_skip_debate_if_pattern_exists: bool = True

    output_dir: str = "./output"
    squads_dir: str = "./output/squads"
    reports_dir: str = "./output/reports"
    logs_dir: str = "./output/logs"
    contexts_dir: str = "./output/contexts"

    llm_max_retries: int = 6
    llm_retry_backoff_seconds: float = 10.0
    llm_request_timeout_seconds: int = 600
    max_retries_per_task: int = 5
    task_timeout_seconds: int = 300
    continue_on_task_failure: bool = True
    parallel_execution: bool = True
    parallel_git: bool = True
    git_push: bool = False

    enforce_scope: bool = True
    require_confirmation: bool = True

    def has_api_keys(self) -> bool:
        return bool(
            self.anthropic_api_key.strip()
            or self.openai_api_key.strip()
            or self.nvidia_api_key.strip()
            or self.groq_api_key.strip()
            or self.cerebras_api_key.strip()
            or self.zenmux_api_key.strip()
            or self.openrouter_api_key.strip()
        )

    def sync_provider_env(self) -> None:
        """Push API keys into os.environ for LiteLLM provider chain."""
        import os

        mapping = {
            "NVIDIA_API_KEY": self.nvidia_api_key,
            "GROQ_API_KEY": self.groq_api_key,
            "CEREBRAS_API_KEY": self.cerebras_api_key,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
        }
        for env_key, value in mapping.items():
            if value.strip() and not os.environ.get(env_key, "").strip():
                os.environ[env_key] = value.strip()

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.squads_dir,
            self.reports_dir,
            self.logs_dir,
            self.contexts_dir,
            self.metrics_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()