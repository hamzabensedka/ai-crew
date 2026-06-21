from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    nvidia_api_key: str = ""
    zenmux_api_key: str = ""
    llm_provider: str = "auto"  # auto | anthropic | openai | nvidia | zenmux
    default_llm: str = "deepseek-ai/deepseek-v4-pro"
    fallback_llm: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    zenmux_base_url: str = "https://zenmux.ai/api/v1"
    nvidia_enable_thinking: bool = False
    nvidia_max_tokens: int = 16384
    nvidia_temperature: float = 1.0
    nvidia_top_p: float = 0.95
    nvidia_reasoning_budget: int = 16384

    # Dual-model debate: planning agents vs implementation agents
    debate_dual_model: bool = True
    debate_planning_model: str = ""  # empty = use FALLBACK_LLM (Nemotron)
    debate_implementation_model: str = ""  # empty = use DEFAULT_LLM (DeepSeek)

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
            or self.zenmux_api_key.strip()
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.squads_dir,
            self.reports_dir,
            self.logs_dir,
            self.contexts_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
