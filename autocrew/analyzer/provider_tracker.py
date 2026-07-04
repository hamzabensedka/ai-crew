"""Track LLM provider usage, rate-limit waits, and paid fallback spend."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from autocrew.config import settings
from autocrew.progress_log import progress_log

PAID_PROVIDER = "OpenRouter"

# Rough EUR estimate per 1M tokens (input+output blended) for OpenRouter fallbacks.
_OPENROUTER_EUR_PER_M_TOKEN = 2.0


class ProviderBudgetError(RuntimeError):
    """Raised when paid fallback limits are exceeded."""


@dataclass
class ProviderSessionStats:
    total_calls: int = 0
    calls_by_provider: dict[str, int] = field(default_factory=dict)
    rate_limit_wait_ms_by_provider: dict[str, float] = field(default_factory=dict)
    paid_fallback_calls: int = 0
    paid_spend_eur: float = 0.0

    def record_call(self, provider: str, *, is_paid: bool, cost_eur: float = 0.0) -> None:
        self.total_calls += 1
        self.calls_by_provider[provider] = self.calls_by_provider.get(provider, 0) + 1
        if is_paid:
            self.paid_fallback_calls += 1
            self.paid_spend_eur += cost_eur

    def record_rate_limit_wait(self, provider: str, wait_ms: float) -> None:
        current = self.rate_limit_wait_ms_by_provider.get(provider, 0.0)
        self.rate_limit_wait_ms_by_provider[provider] = current + wait_ms

    @property
    def paid_fallback_ratio(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.paid_fallback_calls / self.total_calls

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "calls_by_provider": dict(self.calls_by_provider),
            "rate_limit_wait_ms_by_provider": dict(self.rate_limit_wait_ms_by_provider),
            "paid_fallback_calls": self.paid_fallback_calls,
            "paid_fallback_ratio": round(self.paid_fallback_ratio, 4),
            "paid_spend_eur": round(self.paid_spend_eur, 4),
        }


_active: ProviderSessionStats | None = None
_daily_spend_path: Path | None = None


def _daily_spend_file() -> Path:
    global _daily_spend_path
    if _daily_spend_path is None:
        _daily_spend_path = Path(settings.metrics_dir) / "openrouter_daily_spend.json"
    return _daily_spend_path


def _load_daily_spend() -> dict[str, float]:
    path = _daily_spend_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {}


def _save_daily_spend(data: dict[str, float]) -> None:
    path = _daily_spend_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _today_key() -> str:
    return date.today().isoformat()


def get_daily_openrouter_spend_eur() -> float:
    return _load_daily_spend().get(_today_key(), 0.0)


def add_daily_openrouter_spend(cost_eur: float) -> float:
    data = _load_daily_spend()
    key = _today_key()
    data[key] = round(data.get(key, 0.0) + cost_eur, 6)
    _save_daily_spend(data)
    return data[key]


def begin_provider_session() -> ProviderSessionStats:
    global _active
    _active = ProviderSessionStats()
    return _active


def get_provider_tracker() -> ProviderSessionStats:
    global _active
    if _active is None:
        _active = ProviderSessionStats()
    return _active


def end_provider_session() -> ProviderSessionStats | None:
    global _active
    stats = _active
    _active = None
    return stats


def estimate_openrouter_cost_eur(input_tokens: int, output_tokens: int) -> float:
    total = input_tokens + output_tokens
    return (total / 1_000_000) * _OPENROUTER_EUR_PER_M_TOKEN


def check_provider_limits_before_call() -> None:
    """Hard stop if daily OpenRouter spend exceeds cap."""
    daily = get_daily_openrouter_spend_eur()
    cap = settings.openrouter_daily_spend_cap_eur
    if daily >= cap:
        raise ProviderBudgetError(
            f"OpenRouter daily spend {daily:.2f} EUR reached cap ({cap:.2f} EUR). "
            "Manual confirmation required to continue."
        )


def check_paid_fallback_ratio_after_call() -> None:
    """Stop session if paid fallback exceeds configured ratio."""
    stats = get_provider_tracker()
    threshold = settings.openrouter_fallback_ratio_limit
    if stats.total_calls < 5:
        return
    ratio = stats.paid_fallback_ratio
    if ratio > threshold:
        raise ProviderBudgetError(
            f"OpenRouter served {ratio:.0%} of calls (limit {threshold:.0%}). "
            "Free tiers may be insufficient — rethink model split before spending more."
        )


def log_provider_summary() -> None:
    stats = get_provider_tracker()
    if stats.total_calls == 0:
        return
    lines = [
        f"Provider summary: {stats.total_calls} calls",
        f"  by provider: {stats.calls_by_provider}",
        f"  paid fallback: {stats.paid_fallback_calls} ({stats.paid_fallback_ratio:.1%})",
        f"  paid spend (session): {stats.paid_spend_eur:.4f} EUR",
        f"  paid spend (today): {get_daily_openrouter_spend_eur():.4f} EUR",
    ]
    for provider, wait_ms in stats.rate_limit_wait_ms_by_provider.items():
        if wait_ms > 0:
            lines.append(f"  rate-limit wait {provider}: {wait_ms / 1000:.1f}s")
    progress_log("\n".join(lines))
