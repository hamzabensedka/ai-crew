"""Wrap LLM calls to capture latency and token usage."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from autocrew.metrics.collector import record_agent_call
from autocrew.metrics.tokens import estimate_tokens


def _read_api_usage(llm_call: Callable[[str], str]) -> tuple[int | None, int | None]:
    """Read prompt/completion tokens from client if the provider returned usage."""
    client = getattr(llm_call, "__self__", None)
    usage = getattr(client, "_last_usage", None)
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    if input_tokens is None or output_tokens is None:
        return None, None
    return int(input_tokens), int(output_tokens)


def instrument_llm_call(
    llm_call: Callable[[str], str],
    *,
    phase: str,
    agent_name: str,
    agent_role: str,
    model_name: str,
    round_number: int | None = None,
    task_id: str | None = None,
) -> Callable[[str], str]:
    """Return a wrapped callable that records metrics after each completion."""

    def wrapped(prompt: str) -> str:
        wall_start = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        result = llm_call(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        wall_end = datetime.now(timezone.utc).isoformat()

        api_input, api_output = _read_api_usage(llm_call)
        if api_input is not None and api_output is not None:
            input_tokens = api_input
            output_tokens = api_output
            tokens_estimated = False
        else:
            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(result)
            tokens_estimated = True

        record_agent_call(
            phase=phase,
            round_number=round_number,
            agent_name=agent_name,
            agent_role=agent_role,
            model_used=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_estimated=tokens_estimated,
            latency_ms=round(elapsed_ms, 2),
            wall_clock_start=wall_start,
            wall_clock_end=wall_end,
            task_id=task_id,
        )
        return result

    return wrapped


def record_non_llm_agent_call(
    *,
    phase: str,
    agent_name: str,
    agent_role: str,
    model_used: str,
    round_number: int | None = None,
    task_id: str | None = None,
    latency_ms: float = 0.0,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    record_agent_call(
        phase=phase,
        round_number=round_number,
        agent_name=agent_name,
        agent_role=agent_role,
        model_used=model_used,
        input_tokens=0,
        output_tokens=0,
        tokens_estimated=False,
        latency_ms=latency_ms,
        wall_clock_start=now,
        wall_clock_end=now,
        task_id=task_id,
    )
