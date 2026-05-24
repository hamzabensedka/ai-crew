#!/usr/bin/env python3
"""Verify NVIDIA NIM (integrate.api.nvidia.com) connectivity and model access."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as: python scripts/check_nvidia_api.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocrew.config import Settings


def _log(message: str = "") -> None:
    print(message, flush=True)


def _mask_key(key: str) -> str:
    key = key.strip()
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def check_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    enable_thinking: bool,
    timeout: float,
) -> tuple[bool, str, float]:
    start = time.perf_counter()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        request_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if enable_thinking:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "high",
                }
            }

        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = message.content or ""
        if not content and hasattr(message, "reasoning_content") and message.reasoning_content:
            content = message.reasoning_content

        elapsed = time.perf_counter() - start
        preview = content.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return True, preview or "(empty response)", elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, str(exc), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the NVIDIA LLM provider API is reachable and responding."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="Model to test (repeatable). Defaults to DEFAULT_LLM and FALLBACK_LLM from .env.",
    )
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="Only test DEFAULT_LLM, not FALLBACK_LLM.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Prompt sent for the completion test.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60).",
    )
    args = parser.parse_args()

    settings = Settings()

    _log("NVIDIA LLM API health check")
    _log("-" * 40)
    _log(f"Base URL:  {settings.nvidia_base_url}")
    _log(f"Provider:  {settings.llm_provider}")

    if not settings.nvidia_api_key.strip():
        _log("\nFAIL: NVIDIA_API_KEY is not set in .env")
        return 1

    _log(f"API key:   {_mask_key(settings.nvidia_api_key)}")

    models = args.models or [settings.default_llm]
    if not args.skip_fallback and settings.fallback_llm.strip():
        if settings.fallback_llm not in models:
            models.append(settings.fallback_llm)

    max_tokens = min(settings.nvidia_max_tokens, 64)
    all_ok = True

    for model in models:
        _log(f"\nTesting model: {model}")
        ok, detail, elapsed = check_model(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=model,
            prompt=args.prompt,
            max_tokens=max_tokens,
            enable_thinking=settings.nvidia_enable_thinking,
            timeout=args.timeout,
        )
        if ok:
            _log(f"  OK ({elapsed:.2f}s)")
            _log(f"  Response: {detail}")
        else:
            all_ok = False
            _log(f"  FAIL ({elapsed:.2f}s)")
            _log(f"  Error: {detail}")

    _log("\n" + "-" * 40)
    if all_ok:
        _log("Result: NVIDIA API is working.")
        return 0

    _log("Result: NVIDIA API check failed for one or more models.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
