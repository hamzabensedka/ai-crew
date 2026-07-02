"""Token counting — API usage when available, char-based estimate otherwise."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English prose/code)."""
    if not text:
        return 0
    return max(1, len(text) // 4)
