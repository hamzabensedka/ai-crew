"""LiteLLM multi-provider fallback chain: NIM → Groq → Cerebras → OpenRouter."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from autocrew.analyzer.llm_client import LLMError, NVIDIA_DEFAULT_BASE_URL, OPENROUTER_DEFAULT_BASE_URL
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

_FALLBACK_MARKERS = ("429", "503", "502", "504", "524", "rate limit", "timeout", "timed out")


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


def _parse_retry_after_seconds(exc: Exception) -> float:
    msg = str(exc)
    match = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", msg, re.I)
    if match:
        return float(match.group(1))
    if "429" in msg.lower() or "rate limit" in msg.lower():
        return float(settings.llm_retry_backoff_seconds)
    return 0.0


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
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None) or ""
    if not content.strip():
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            content = reasoning
    if not content or not content.strip():
        raise LLMError("Empty response from provider")
    return content


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


class LiteLLMFallbackClient:
    """Try NIM → Groq → Cerebras → OpenRouter using LiteLLM completion API."""

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
        hops = _build_hops(self.tier)
        if not hops:
            raise LLMError(
                "No provider API keys configured (need NVIDIA_API_KEY at minimum)"
            )

        fallback_model_ids = [_litellm_model_id(h) for h in hops[1:]]
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

            if index == 0 and fallback_model_ids:
                params["fallbacks"] = fallback_model_ids

            try:
                progress_log(
                    f"LLM [{self.label}] → {hop.provider}/{hop.model.split('/')[-1]}",
                    verbose_only=True,
                )
                start = time.perf_counter()
                response = litellm.completion(**params)
                elapsed_ms = (time.perf_counter() - start) * 1000

                content = _extract_content(response)
                in_tok, out_tok = _extract_usage(response)
                self._last_usage = {"input_tokens": in_tok, "output_tokens": out_tok}

                served_provider = hop.provider
                served_model = hop.model
                hidden = getattr(response, "_hidden_params", {}) or {}
                if hidden.get("fallback_used"):
                    served_provider = str(hidden.get("fallback_provider", served_provider))
                    served_model = str(hidden.get("fallback_model", served_model))

                self._last_provider = served_provider
                self._last_model = served_model

                cost_eur = 0.0
                if hop.is_paid or served_provider == PAID_PROVIDER:
                    cost_eur = estimate_openrouter_cost_eur(in_tok, out_tok)
                    add_daily_openrouter_spend(cost_eur)

                tracker.record_call(served_provider, is_paid=(hop.is_paid or served_provider == PAID_PROVIDER), cost_eur=cost_eur)
                progress_log(
                    f"LLM [{self.label}] served by {served_provider} "
                    f"({served_model.split('/')[-1]}, {elapsed_ms:.0f}ms, "
                    f"paid session total {tracker.paid_spend_eur:.4f} EUR)"
                )
                check_paid_fallback_ratio_after_call()
                return content

            except Exception as exc:
                last_error = exc
                wait_s = _parse_retry_after_seconds(exc)
                if wait_s > 0:
                    tracker.record_rate_limit_wait(hop.provider, wait_s * 1000)
                    progress_log(
                        f"LLM [{self.label}] {hop.provider} rate-limited — "
                        f"waited {wait_s:.0f}s before fallback"
                    )
                    time.sleep(wait_s)

                if _should_fallback(exc) and index < len(hops) - 1:
                    next_hop = hops[index + 1]
                    progress_log(
                        f"LLM [{self.label}] {hop.provider} failed ({exc}) — "
                        f"falling back to {next_hop.provider}"
                    )
                    continue
                if index < len(hops) - 1:
                    progress_log(
                        f"LLM [{self.label}] {hop.provider} error — trying next provider"
                    )
                    continue
                break

        if last_error is not None:
            raise LLMError(f"All providers failed for {self.label}: {last_error}") from last_error
        raise LLMError(f"All providers failed for {self.label}")


def create_chain_client_for_tier(tier: ModelTier) -> LiteLLMFallbackClient:
    return LiteLLMFallbackClient(tier, label=tier)
