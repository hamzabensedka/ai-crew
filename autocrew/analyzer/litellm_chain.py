"""LiteLLM multi-provider fallback chain: NIM → Groq → Cerebras → OpenRouter."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from autocrew.analyzer.llm_client import LLMError, extract_message_text, parse_retry_after_seconds
from autocrew.analyzer.model_registry import ModelTier, cerebras_model_for_tier, groq_model_for_tier, nim_model_for_tier, openrouter_model_for_tier
from autocrew.analyzer.provider_tracker import (
    PAID_PROVIDER,
    add_daily_openrouter_spend,
    check_paid_fallback_ratio_after_call,
    check_provider_limits_before_call,
    estimate_openrouter_cost_eur,
    get_provider_tracker,
)
from autocrew.config import settings
from autocrew.progress_log import progress_log

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Do NOT use LiteLLM's `fallbacks=` param — it reuses the primary api_base for all hops.
_FALLBACK_MARKERS = (
    "429",
    "503",
    "502",
    "504",
    "524",
    "410",
    "404",
    "410",
    "gone",
    "end of life",
    "not found for account",
    "rate limit",
    "timeout",
    "timed out",
    "gone",
    "end of life",
    "not found",
    "not available",
)


@dataclass(frozen=True)
class ProviderHop:
    provider: str
    model: str
    api_base: str | None
    api_key_env: str
    is_paid: bool = False


def _should_fallback(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _FALLBACK_MARKERS)


def _build_hops(tier: ModelTier) -> list[ProviderHop]:
    hops: list[ProviderHop] = [
        ProviderHop("NIM", nim_model_for_tier(tier), settings.nvidia_base_url, "NVIDIA_API_KEY"),
        ProviderHop("Groq", groq_model_for_tier(tier), GROQ_BASE_URL, "GROQ_API_KEY"),
        ProviderHop("Cerebras", cerebras_model_for_tier(tier), CEREBRAS_BASE_URL, "CEREBRAS_API_KEY"),
        ProviderHop(
            "OpenRouter",
            openrouter_model_for_tier(tier),
            settings.openrouter_base_url,
            "OPENROUTER_API_KEY",
            is_paid=True,
        ),
    ]
    return [h for h in hops if os.environ.get(h.api_key_env, "").strip()]


def _litellm_model_id(hop: ProviderHop) -> str:
    if hop.provider == "Groq":
        return f"groq/{hop.model}"
    if hop.provider == "Cerebras":
        return f"cerebras/{hop.model}"
    if hop.provider == "OpenRouter":
        return f"openrouter/{hop.model}"
    return f"openai/{hop.model}"


def _extract_content(response: Any) -> str:
    content = extract_message_text(response.choices[0].message)
    if not content:
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        detail = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise LLMError(f"Empty response from provider{detail}")
    return content


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


class LiteLLMFallbackClient:
    """Try NIM → Groq → Cerebras → OpenRouter — one provider per attempt, correct api_base each time."""

    def __init__(self, tier: ModelTier, *, label: str | None = None) -> None:
        self.tier = tier
        self.label = label or tier
        self._last_usage: dict[str, int] | None = None
        self._last_provider: str | None = None
        self._last_model: str | None = None

    @property
    def provider_used(self) -> str | None:
        return self._last_provider

    @property
    def model_used(self) -> str | None:
        return self._last_model

    def complete(self, prompt: str) -> str:
        import litellm

        litellm.drop_params = True
        litellm.suppress_debug_info = True
        hops = _build_hops(self.tier)
        if not hops:
            raise LLMError(
                "No provider API keys configured (need NVIDIA_API_KEY at minimum)"
            )

        last_error: Exception | None = None
        tracker = get_provider_tracker()

        for index, hop in enumerate(hops):
            api_key = os.environ.get(hop.api_key_env, "").strip()
            if not api_key:
                continue

            if hop.is_paid:
                check_provider_limits_before_call()

            params: dict[str, Any] = {
                "model": _litellm_model_id(hop),
                "messages": [{"role": "user", "content": prompt}],
                "timeout": settings.llm_request_timeout_seconds,
                "api_key": api_key,
            }
            if hop.api_base:
                params["api_base"] = hop.api_base

            try:
                progress_log(
                    f"LLM [{self.label}] -> {hop.provider}/{hop.model.split('/')[-1]} "
                    f"({hop.api_base or 'default'})",
                )
                start = time.perf_counter()
                response = litellm.completion(**params)
                elapsed_ms = (time.perf_counter() - start) * 1000

                content = _extract_content(response)
                in_tok, out_tok = _extract_usage(response)
                self._last_usage = {"input_tokens": in_tok, "output_tokens": out_tok}
                self._last_provider = hop.provider
                self._last_model = hop.model

                cost_eur = 0.0
                if hop.is_paid:
                    cost_eur = estimate_openrouter_cost_eur(in_tok, out_tok)
                    add_daily_openrouter_spend(cost_eur)

                tracker.record_call(hop.provider, is_paid=hop.is_paid, cost_eur=cost_eur)
                progress_log(
                    f"LLM [{self.label}] served by {hop.provider} "
                    f"({hop.model.split('/')[-1]}, {elapsed_ms:.0f}ms)"
                )
                check_paid_fallback_ratio_after_call()
                return content

            except Exception as exc:
                last_error = exc
                wait_s = parse_retry_after_seconds(exc)
                if wait_s > 0:
                    tracker.record_rate_limit_wait(hop.provider, wait_s * 1000)
                    time.sleep(wait_s)

                if index < len(hops) - 1:
                    next_hop = hops[index + 1]
                    progress_log(
                        f"LLM [{self.label}] {hop.provider} failed — "
                        f"trying {next_hop.provider}: {str(exc)[:120]}"
                    )
                    continue
                break

        if last_error is not None:
            raise LLMError(f"All providers failed for {self.label}: {last_error}") from last_error
        raise LLMError(f"All providers failed for {self.label}")


def create_chain_client_for_tier(tier: ModelTier) -> LiteLLMFallbackClient:
    return LiteLLMFallbackClient(tier, label=tier)
