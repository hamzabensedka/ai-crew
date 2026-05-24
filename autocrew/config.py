from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    nvidia_api_key: str = ""
    llm_provider: str = "auto"  # auto | anthropic | openai | nvidia
    default_llm: str = "deepseek-ai/deepseek-v4-flash"
    fallback_llm: str = "moonshotai/kimi-k2.6"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_enable_thinking: bool = False
    nvidia_max_tokens: int = 16384

    # Dual-model debate: planning agents vs implementation agents
    debate_dual_model: bool = True
    debate_planning_model: str = ""  # empty = use FALLBACK_LLM (Kimi)
    debate_implementation_model: str = ""  # empty = use DEFAULT_LLM (DeepSeek)

    output_dir: str = "./output"
    squads_dir: str = "./output/squads"
    reports_dir: str = "./output/reports"
    logs_dir: str = "./output/logs"
    contexts_dir: str = "./output/contexts"

    max_retries_per_task: int = 2
    task_timeout_seconds: int = 300
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
