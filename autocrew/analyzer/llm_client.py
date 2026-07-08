"""LLM client abstraction — Anthropic, OpenAI, NVIDIA NIM, and ZenMux."""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Protocol

from autocrew.progress_log import progress_log

NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
ZENMUX_DEFAULT_BASE_URL = "https://zenmux.ai/api/v1"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
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


def is_gateway_timeout_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "504" in msg or "524" in msg or "gateway timeout" in msg


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate-limited" in msg


def parse_retry_after_seconds(exc: Exception) -> float:
    """Read Retry-After / retry_after_seconds from provider error payloads."""
    msg = str(exc)
    for pattern in (
        r"retry_after_seconds(?:_raw)?['\"]?\s*:\s*(\d+(?:\.\d+)?)",
        r"retry[- _]after['\"]?\s*:\s*['\"]?\s*(\d+(?:\.\d+)?)",
        r"'Retry-After'\s*:\s*['\"]?\s*(\d+)",
    ):
        match = re.search(pattern, msg, re.I)
        if match:
            return float(match.group(1))
    if is_rate_limit_error(exc):
        return 30.0
    return 0.0


def _retry_delay(attempt: int, base_seconds: float, exc: Exception) -> float:
    return max(_backoff_delay(attempt, base_seconds), parse_retry_after_seconds(exc))


def _backoff_delay(attempt: int, base_seconds: float, cap_seconds: float = 120.0) -> float:
    return min(base_seconds * (2 ** attempt), cap_seconds)


def _text_from_content_part(part: object) -> str:
    if isinstance(part, dict):
        if part.get("type") == "text":
            return str(part.get("text") or "")
        return str(part.get("text") or part.get("content") or "")
    text = getattr(part, "text", None)
    if text:
        return str(text)
    content = getattr(part, "content", None)
    return str(content) if content else ""


def _text_from_reasoning_details(details: object) -> str:
    if not details:
        return ""
    parts: list[str] = []
    for item in details:
        if isinstance(item, dict):
            if item.get("type") in {"reasoning.text", "reasoning.summary"} and item.get("text"):
                parts.append(str(item["text"]))
            elif item.get("text"):
                parts.append(str(item["text"]))
        else:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
    return "".join(parts).strip()


def extract_message_text(message: object) -> str:
    """Normalize chat message content across OpenAI-compatible providers.

    OpenRouter thinking models (e.g. tencent/hy3:free) often return output in
    ``reasoning`` or ``reasoning_details`` while ``content`` is empty.
    """
    raw = getattr(message, "content", None)
    content = ""
    if isinstance(raw, list):
        content = "".join(_text_from_content_part(part) for part in raw)
    elif raw is not None:
        content = str(raw)

    if content.strip():
        return content.strip()

    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if value and str(value).strip():
            return str(value).strip()

    reasoning_details = getattr(message, "reasoning_details", None)
    details_text = _text_from_reasoning_details(reasoning_details)
    if details_text:
        return details_text

    return ""


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
                if self.fallback is not None and (
                    is_gateway_timeout_error(exc)
                    or (is_rate_limit_error(exc) and attempt >= 1)
                ):
                    reason = "rate limited" if is_rate_limit_error(exc) else "gateway timeout"
                    progress_log(f"{reason.title()} on {self.label} — switching to fallback model")
                    break
                if attempt < self.max_retries and is_retryable_error(exc):
                    delay = _retry_delay(attempt, self.backoff_seconds, exc)
                    progress_log(
                        f"LLM retry {attempt + 1}/{self.max_retries} for {self.label} "
                        f"in {delay:.0f}s — {exc}",
                    )
                    time.sleep(delay)
                    continue
                break

        if self.fallback is not None:
            progress_log(f"Trying fallback model for {self.label}")
            for attempt in range(min(3, self.max_retries) + 1):
                try:
                    result = self.fallback.complete(prompt)
                    if not result or not result.strip():
                        raise LLMError(f"{self.label} fallback returned empty response")
                    return result
                except LLMError as exc:
                    last_error = exc
                    if attempt < min(3, self.max_retries) and is_retryable_error(exc):
                        delay = _retry_delay(attempt, self.backoff_seconds, exc)
                        progress_log(
                            f"Fallback retry {attempt + 1} for {self.label} in {delay:.0f}s — {exc}"
                        )
                        time.sleep(delay)
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
        self._last_usage: dict[str, int] | None = None

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
            if getattr(response, "usage", None) is not None:
                self._last_usage = {
                    "input_tokens": int(response.usage.prompt_tokens or 0),
                    "output_tokens": int(response.usage.completion_tokens or 0),
                }
            else:
                self._last_usage = None
            message = response.choices[0].message
            content = extract_message_text(message)
            if not content:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                raise LLMError(
                    f"{self.provider_label} returned empty response"
                    + (f" (finish_reason={finish_reason})" if finish_reason else "")
                )
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
        temperature: float = 1.0,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        reasoning_effort: str = "high",
        reasoning_budget: int = 4096,
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
        self.top_p = top_p
        self.reasoning_budget = reasoning_budget

    def _build_request_kwargs(self, prompt: str) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        model_lower = self.model.lower()
        if "nemotron" in model_lower:
            kwargs["stream"] = True
            kwargs["max_tokens"] = min(self.max_tokens, 8192)
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": self.reasoning_budget,
            }
        elif "deepseek" in model_lower:
            kwargs["stream"] = False
            kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        elif self.enable_thinking:
            kwargs["stream"] = False
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": self.reasoning_effort,
                }
            }
        else:
            kwargs["stream"] = False
        return kwargs

    def _complete_stream(self, client, request_kwargs: dict) -> str:
        stream = client.chat.completions.create(**request_kwargs)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        chunk_count = 0
        start = time.perf_counter()
        last_ping = start
        progress_log(f"Streaming {self.model} (thinking enabled)...")
        for chunk in stream:
            chunk_count += 1
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)
            now = time.perf_counter()
            if now - last_ping >= 15:
                content_len = sum(len(part) for part in content_parts)
                reasoning_len = sum(len(part) for part in reasoning_parts)
                progress_log(
                    f"  … still streaming ({now - start:.0f}s, "
                    f"{chunk_count} chunks, {content_len:,} content chars, "
                    f"{reasoning_len:,} reasoning chars)"
                )
                last_ping = now
        elapsed = time.perf_counter() - start
        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        if content:
            progress_log(
                f"Stream complete ({elapsed:.1f}s, {chunk_count} chunks, {len(content):,} chars)"
            )
            return content
        if reasoning:
            progress_log(
                f"Stream complete — using reasoning ({elapsed:.1f}s, {len(reasoning):,} chars)"
            )
            return reasoning
        raise LLMError("NVIDIA returned empty response")

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMError("NVIDIA API key not configured")
        try:
            from openai import OpenAI

            client_kwargs: dict = {
                "api_key": self.api_key,
                "timeout": self.timeout_seconds,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client = OpenAI(**client_kwargs)

            request_kwargs = self._build_request_kwargs(prompt)
            if request_kwargs.get("stream"):
                return self._complete_stream(client, request_kwargs)

            progress_log(f"Calling {self.model} (non-stream)...", verbose_only=True)
            start = time.perf_counter()
            response = client.chat.completions.create(**request_kwargs)
            elapsed = time.perf_counter() - start
            message = response.choices[0].message
            content = extract_message_text(message)
            if not content:
                raise LLMError("NVIDIA returned empty response")
            progress_log(
                f"Response from {self.model} ({elapsed:.1f}s, {len(content):,} chars)",
                verbose_only=True,
            )
            return content
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"NVIDIA API error: {exc}") from exc


class ZenMuxClient(OpenAICompatibleClient):
    """ZenMux — OpenAI-compatible gateway (https://zenmux.ai/api/v1)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = ZENMUX_DEFAULT_BASE_URL,
        max_tokens: int = 16384,
        temperature: float = 0.2,
        timeout_seconds: int = 600,
    ) -> None:
        super().__init__(
            api_key,
            model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_label="ZenMux",
            timeout_seconds=timeout_seconds,
        )


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter — unified gateway to hundreds of models (https://openrouter.ai/api/v1).

    Supports Claude, GPT, DeepSeek, Kimi, Llama, Mistral, and more with a single API key.
    Model names use the provider/model format, e.g.:
      - "anthropic/claude-3.5-sonnet"
      - "openai/gpt-4o"
      - "deepseek/deepseek-chat"
      - "moonshotai/kimi-k2"
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = OPENROUTER_DEFAULT_BASE_URL,
        max_tokens: int = 16384,
        temperature: float = 0.2,
        timeout_seconds: int = 600,
    ) -> None:
        super().__init__(
            api_key,
            model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_label="OpenRouter",
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


def _is_openrouter_routed_model(model: str) -> bool:
    """Models that must use OpenRouter (not available on NIM)."""
    normalized = model.strip().lower()
    if ":free" in normalized:
        return True
    return normalized.startswith(("tencent/", "anthropic/", "google/", "openai/", "meta-llama/"))


def _make_nvidia_client(
    nvidia_key: str,
    model: str,
    *,
    nvidia_base_url: str,
    nvidia_max_tokens: int,
    nvidia_enable_thinking: bool,
    nvidia_temperature: float,
    nvidia_top_p: float,
    nvidia_reasoning_budget: int,
    timeout_seconds: int,
) -> NvidiaClient:
    return NvidiaClient(
        nvidia_key,
        model,
        base_url=nvidia_base_url,
        max_tokens=nvidia_max_tokens,
        temperature=nvidia_temperature,
        top_p=nvidia_top_p,
        enable_thinking=nvidia_enable_thinking,
        reasoning_budget=nvidia_reasoning_budget,
        timeout_seconds=timeout_seconds,
    )


def create_llm_client(
    anthropic_key: str = "",
    openai_key: str = "",
    nvidia_key: str = "",
    zenmux_key: str = "",
    openrouter_key: str = "",
    openrouter_base_url: str = OPENROUTER_DEFAULT_BASE_URL,
    default_model: str = "claude-3-5-sonnet-20241022",
    fallback_model: str = "gpt-4o",
    llm_provider: str = "auto",
    nvidia_base_url: str = NVIDIA_DEFAULT_BASE_URL,
    zenmux_base_url: str = ZENMUX_DEFAULT_BASE_URL,
    nvidia_enable_thinking: bool = False,
    nvidia_max_tokens: int = 16384,
    nvidia_temperature: float = 1.0,
    nvidia_top_p: float = 0.95,
    nvidia_reasoning_budget: int = 4096,
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
        return _make_nvidia_client(
            nvidia_key,
            model,
            nvidia_base_url=nvidia_base_url,
            nvidia_max_tokens=nvidia_max_tokens,
            nvidia_enable_thinking=nvidia_enable_thinking,
            nvidia_temperature=nvidia_temperature,
            nvidia_top_p=nvidia_top_p,
            nvidia_reasoning_budget=nvidia_reasoning_budget,
            timeout_seconds=timeout,
        )

    def _openrouter(model: str) -> OpenRouterClient | None:
        if not openrouter_key.strip():
            return None
        return OpenRouterClient(
            openrouter_key,
            model,
            base_url=openrouter_base_url,
            max_tokens=nvidia_max_tokens,
            timeout_seconds=timeout,
        )

    def _zenmux(model: str) -> ZenMuxClient | None:
        if not zenmux_key.strip():
            return None
        return ZenMuxClient(
            zenmux_key,
            model,
            base_url=zenmux_base_url,
            max_tokens=nvidia_max_tokens,
            timeout_seconds=timeout,
        )

    def _resilient_pair(
        primary: LLMClient | None,
        fallback: LLMClient | None,
        label: str,
        *,
        fallback_model: str = "",
    ) -> LLMClient | None:
        if primary is None:
            return None
        if fallback_model.strip().lower() == default_model.strip().lower():
            fallback = None
        return _wrap_resilient(
            primary,
            fallback=fallback,
            label=label,
            max_retries=llm_max_retries,
            backoff_seconds=llm_retry_backoff_seconds,
        )

    if provider == "zenmux":
        primary = _zenmux(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=zenmux but ZENMUX_API_KEY is not set")
        fallback = (
            _zenmux(fallback_model)
            or _nvidia(fallback_model)
            or _openai(fallback_model)
            or _anthropic(fallback_model)
        )
        return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary

    if provider == "nvidia":
        primary = _nvidia(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set")
        fallback = _nvidia(fallback_model) or _openai(fallback_model) or _anthropic(fallback_model)
        return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary

    if provider == "openai":
        primary = _openai(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        fallback = _anthropic(fallback_model) or _nvidia(fallback_model)
        return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary

    if provider == "anthropic":
        primary = _anthropic(default_model)
        if primary is None:
            raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        fallback = _openai(fallback_model) or _nvidia(fallback_model)
        return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary

    if provider == "openrouter" or (
        openrouter_key.strip() and _is_openrouter_routed_model(default_model)
    ):
        primary = _openrouter(default_model)
        if primary is None:
            raise LLMError("OpenRouter API key required for OpenRouter-routed models")
        fallback = (
            _openrouter(fallback_model)
            or _nvidia(fallback_model)
            or _openai(fallback_model)
            or _anthropic(fallback_model)
        )
        return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary

    if nvidia_key.strip() and (
        _is_nvidia_model(default_model)
        and not _is_openrouter_routed_model(default_model)
        or (not anthropic_key.strip() and not openai_key.strip() and not openrouter_key.strip())
    ):
        primary = _nvidia(default_model)
        if primary:
            fallback = _nvidia(fallback_model) or _openrouter(fallback_model) or _openai(fallback_model) or _anthropic(fallback_model)
            return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary

    if default_model.startswith("claude"):
        primary = _anthropic(default_model)
        fallback = _openai(fallback_model) or _nvidia(fallback_model)
    else:
        primary = _openai(default_model) or _nvidia(default_model)
        fallback = _anthropic(fallback_model) or _nvidia(fallback_model)

    if primary is None:
        raise LLMError("No LLM API key configured (Anthropic, OpenAI, NVIDIA, or ZenMux)")
    return _resilient_pair(primary, fallback, default_model, fallback_model=fallback_model) or primary


def create_model_client(
    model: str,
    *,
    anthropic_key: str = "",
    openai_key: str = "",
    nvidia_key: str = "",
    zenmux_key: str = "",
    openrouter_key: str = "",
    llm_provider: str = "auto",
    nvidia_base_url: str = NVIDIA_DEFAULT_BASE_URL,
    zenmux_base_url: str = ZENMUX_DEFAULT_BASE_URL,
    openrouter_base_url: str = OPENROUTER_DEFAULT_BASE_URL,
    nvidia_enable_thinking: bool = False,
    nvidia_max_tokens: int = 16384,
    nvidia_temperature: float = 1.0,
    nvidia_top_p: float = 0.95,
    nvidia_reasoning_budget: int = 4096,
    llm_max_retries: int = 6,
    llm_retry_backoff_seconds: float = 10.0,
    llm_request_timeout_seconds: int = 600,
    fallback: LLMClient | None = None,
) -> LLMClient:
    """Create a client for a specific model (used for dual-model debate)."""
    provider = llm_provider.strip().lower()
    timeout = llm_request_timeout_seconds

    if provider == "zenmux":
        if not zenmux_key.strip():
            raise LLMError("ZenMux API key required when LLM_PROVIDER=zenmux")
        client: LLMClient = ZenMuxClient(
            zenmux_key,
            model,
            base_url=zenmux_base_url,
            max_tokens=nvidia_max_tokens,
            timeout_seconds=timeout,
        )
    elif provider == "openrouter" or (openrouter_key.strip() and _is_openrouter_routed_model(model)):
        if not openrouter_key.strip():
            raise LLMError("OpenRouter API key required for OpenRouter-routed models")
        client: LLMClient = OpenRouterClient(
            openrouter_key,
            model,
            base_url=openrouter_base_url,
            max_tokens=nvidia_max_tokens,
            timeout_seconds=timeout,
        )
    elif provider == "nvidia" or (provider == "auto" and nvidia_key.strip()):
        if not nvidia_key.strip():
            raise LLMError("NVIDIA API key required for NVIDIA models")
        client: LLMClient = _make_nvidia_client(
            nvidia_key,
            model,
            nvidia_base_url=nvidia_base_url,
            nvidia_max_tokens=nvidia_max_tokens,
            nvidia_enable_thinking=nvidia_enable_thinking,
            nvidia_temperature=nvidia_temperature,
            nvidia_top_p=nvidia_top_p,
            nvidia_reasoning_budget=nvidia_reasoning_budget,
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
