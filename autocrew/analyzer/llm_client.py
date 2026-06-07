"""LLM client abstraction — Anthropic, OpenAI, and NVIDIA NIM."""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Protocol

NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_RETRYABLE_MARKERS = (
    "504",
    "503",
    "502",
    "429",
    "timeout",
    "timed out",
    "connection",
    "overloaded",
    "rate limit",
    "empty response",
)


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class LLMError(Exception):
    pass


def is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _backoff_delay(attempt: int, base_seconds: float, cap_seconds: float = 120.0) -> float:
    return min(base_seconds * (2 ** attempt), cap_seconds)


def extract_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if not text:
        raise LLMError("Invalid JSON from LLM: empty response")
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON from LLM: {exc}") from exc


def call_with_json_retry(
    llm_call: Callable[[str], str],
    prompt: str,
    *,
    max_retries: int = 3,
) -> dict:
    """Call LLM and parse JSON, retrying with a stricter prompt on failure."""
    last_error: Exception | None = None
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            response = llm_call(current_prompt)
            return extract_json(response)
        except (LLMError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_retries:
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid JSON. "
                    "Return ONLY a valid JSON object. No markdown, no explanation."
                )
                if is_retryable_error(exc):
                    time.sleep(_backoff_delay(attempt, 5.0))
    raise LLMError(f"Failed to parse JSON after {max_retries + 1} attempts: {last_error}")


class ResilientLLMClient:
    """Retry transient API failures with backoff, then optional fallback model."""

    def __init__(
        self,
        primary: LLMClient,
        fallback: LLMClient | None = None,
        *,
        max_retries: int = 6,
        backoff_seconds: float = 10.0,
        label: str = "LLM",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.label = label

    def complete(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.primary.complete(prompt)
                if not result or not result.strip():
                    raise LLMError(f"{self.label} returned empty response")
                return result
            except LLMError as exc:
                last_error = exc
                if attempt < self.max_retries and is_retryable_error(exc):
                    delay = _backoff_delay(attempt, self.backoff_seconds)
                    time.sleep(delay)
                    continue
                break

        if self.fallback is not None:
            for attempt in range(min(3, self.max_retries) + 1):
                try:
                    result = self.fallback.complete(prompt)
                    if not result or not result.strip():
                        raise LLMError(f"{self.label} fallback returned empty response")
                    return result
                except LLMError as exc:
                    last_error = exc
                    if attempt < min(3, self.max_retries) and is_retryable_error(exc):
                        time.sleep(_backoff_delay(attempt, self.backoff_seconds))
                        continue
                    break

        if last_error is not None:
            raise last_error
        raise LLMError(f"{self.label} call failed")


class AnthropicClient:
    def __init__(self, api_key: str, model: str, *, timeout_seconds: int = 600) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMError("Anthropic API key not configured")
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_seconds)
            message = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text
            if not text or not text.strip():
                raise LLMError("Anthropic returned empty response")
            return text
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc


class OpenAICompatibleClient:
    """OpenAI SDK client — works with OpenAI and any compatible API (e.g. NVIDIA NIM)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        max_tokens: int = 16384,
        temperature: float = 0.2,
        enable_thinking: bool = False,
        reasoning_effort: str = "high",
        provider_label: str = "OpenAI",
        timeout_seconds: int = 600,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.provider_label = provider_label
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMError(f"{self.provider_label} API key not configured")
        try:
            from openai import OpenAI

            client_kwargs: dict = {
                "api_key": self.api_key,
                "timeout": self.timeout_seconds,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client = OpenAI(**client_kwargs)

            request_kwargs: dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            if self.enable_thinking:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": self.reasoning_effort,
                    }
                }

            response = client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            content = message.content or ""
            if not content and hasattr(message, "reasoning_content") and message.reasoning_content:
                content = message.reasoning_content
            if not content or not content.strip():
                raise LLMError(f"{self.provider_label} returned empty response")
            return content
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.provider_label} API error: {exc}") from exc


class OpenAIClient(OpenAICompatibleClient):
    def __init__(self, api_key: str, model: str, *, timeout_seconds: int = 600) -> None:
        super().__init__(
            api_key,
            model,
            provider_label="OpenAI",
            temperature=0.2,
            timeout_seconds=timeout_seconds,
        )


class NvidiaClient(OpenAICompatibleClient):
    """NVIDIA integrate.api.nvidia.com — OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = NVIDIA_DEFAULT_BASE_URL,
        max_tokens: int = 16384,
        temperature: float = 0.2,
        enable_thinking: bool = False,
        reasoning_effort: str = "high",
        timeout_seconds: int = 600,
    ) -> None:
        super().__init__(
            api_key,
            model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            provider_label="NVIDIA",
            timeout_seconds=timeout_seconds,
        )


class FallbackLLMClient:
    """Try primary LLM, fall back to secondary on failure."""

    def __init__(self, primary: LLMClient, fallback: LLMClient | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    def complete(self, prompt: str) -> str:
        try:
            return self.primary.complete(prompt)
        except LLMError:
            if self.fallback is None:
                raise
            return self.fallback.complete(prompt)


def _wrap_resilient(
    client: LLMClient,
    *,
    fallback: LLMClient | None = None,
    label: str,
    max_retries: int,
    backoff_seconds: float,
) -> LLMClient:
    return ResilientLLMClient(
        client,
        fallback=fallback,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        label=label,
    )


def _is_nvidia_model(model: str) -> bool:
    return "/" in model or model.startswith(("deepseek", "moonshotai", "meta", "nvidia"))


def create_llm_client(
    anthropic_key: str = "",
    openai_key: str = "",
    nvidia_key: str = "",
    default_model: str = "claude-3-5-sonnet-20241022",
    fallback_model: str = "gpt-4o",
    llm_provider: str = "auto",
    nvidia_base_url: str = NVIDIA_DEFAULT_BASE_URL,
    nvidia_enable_thinking: bool = False,
    nvidia_max_tokens: int = 16384,
    llm_max_retries: int = 6,
    llm_retry_backoff_seconds: float = 10.0,
    llm_request_timeout_seconds: int = 600,
) -> LLMClient:
    provider = llm_provider.strip().lower()
    timeout = llm_request_timeout_seconds

    def _anthropic(model: str) -> AnthropicClient | None:
        if not anthropic_key.strip():
            return None
        return AnthropicClient(anthropic_key, model, timeout_seconds=timeout)

    def _openai(model: str) -> OpenAIClient | None:
        if not openai_key.strip():
            return None
        return OpenAIClient(openai_key, model, timeout_seconds=timeout)

    def _nvidia(model: str) -> NvidiaClient | None:
        if not nvidia_key.strip():
            return None
        return NvidiaClient(
            nvidia_key,
            model,
            base_url=nvidia_base_url,
            max_tokens=nvidia_max_tokens,
            enable_thinking=nvidia_enable_thinking,
            timeout_seconds=timeout,
        )

    def _resilient_pair(primary: LLMClient | None, fallback: LLMClient | None, label: str) -> LLMClient | None:
        if primary is None:
            return None
        return _wrap_resilient(
            primary,
            fallback=fallback,
            label=label,
            max_retries=llm_max_retries,
            backoff_seconds=llm_retry_backoff_seconds,
        )

    if provider == "nvidia":
        primary = _nvidia(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set")
        fallback = _nvidia(fallback_model) or _openai(fallback_model) or _anthropic(fallback_model)
        return _resilient_pair(primary, fallback, default_model) or primary

    if provider == "openai":
        primary = _openai(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        fallback = _anthropic(fallback_model) or _nvidia(fallback_model)
        return _resilient_pair(primary, fallback, default_model) or primary

    if provider == "anthropic":
        primary = _anthropic(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        fallback = _openai(fallback_model) or _nvidia(fallback_model)
        return _resilient_pair(primary, fallback, default_model) or primary

    if nvidia_key.strip() and (
        _is_nvidia_model(default_model)
        or (not anthropic_key.strip() and not openai_key.strip())
    ):
        primary = _nvidia(default_model)
        if primary:
            fallback = _nvidia(fallback_model) or _openai(fallback_model) or _anthropic(fallback_model)
            return _resilient_pair(primary, fallback, default_model) or primary

    if default_model.startswith("claude"):
        primary = _anthropic(default_model)
        fallback = _openai(fallback_model) or _nvidia(fallback_model)
    else:
        primary = _openai(default_model) or _nvidia(default_model)
        fallback = _anthropic(fallback_model) or _nvidia(fallback_model)

    if primary is None:
        raise LLMError("No LLM API key configured (Anthropic, OpenAI, or NVIDIA)")
    return _resilient_pair(primary, fallback, default_model) or primary


def create_model_client(
    model: str,
    *,
    anthropic_key: str = "",
    openai_key: str = "",
    nvidia_key: str = "",
    llm_provider: str = "auto",
    nvidia_base_url: str = NVIDIA_DEFAULT_BASE_URL,
    nvidia_enable_thinking: bool = False,
    nvidia_max_tokens: int = 16384,
    llm_max_retries: int = 6,
    llm_retry_backoff_seconds: float = 10.0,
    llm_request_timeout_seconds: int = 600,
    fallback: LLMClient | None = None,
) -> LLMClient:
    """Create a client for a specific model (used for dual-model debate)."""
    provider = llm_provider.strip().lower()
    timeout = llm_request_timeout_seconds

    if provider == "nvidia" or (provider == "auto" and nvidia_key.strip()):
        if not nvidia_key.strip():
            raise LLMError("NVIDIA API key required for NVIDIA models")
        client: LLMClient = NvidiaClient(
            nvidia_key,
            model,
            base_url=nvidia_base_url,
            max_tokens=nvidia_max_tokens,
            enable_thinking=nvidia_enable_thinking,
            timeout_seconds=timeout,
        )
    elif model.startswith("claude"):
        if not anthropic_key.strip():
            raise LLMError("Anthropic API key required for Claude models")
        client = AnthropicClient(anthropic_key, model, timeout_seconds=timeout)
    else:
        if not openai_key.strip():
            raise LLMError("OpenAI API key required for OpenAI-compatible models")
        client = OpenAIClient(openai_key, model, timeout_seconds=timeout)

    return _wrap_resilient(
        client,
        fallback=fallback,
        label=model,
        max_retries=llm_max_retries,
        backoff_seconds=llm_retry_backoff_seconds,
    )
