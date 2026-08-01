"""Replace real-looking secrets in example env files before commit or push."""

from __future__ import annotations

import re
from pathlib import Path

ENV_EXAMPLE_NAMES = frozenset({".env.example", "env.example"})

SECRET_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk_live_[a-zA-Z0-9]{10,}"), "sk_live_REPLACE_ME"),
    (re.compile(r"sk_test_[a-zA-Z0-9]{10,}"), "sk_test_your_key_here"),
    (re.compile(r"whsec_[a-zA-Z0-9]{10,}"), "whsec_your_webhook_secret_here"),
    (re.compile(r"pk_live_[a-zA-Z0-9]{10,}"), "pk_live_REPLACE_ME"),
    (re.compile(r"pk_test_[a-zA-Z0-9]{10,}"), "pk_test_your_key_here"),
    (re.compile(r"nvapi-[a-zA-Z0-9_-]{20,}"), "nvapi-REPLACE_ME"),
)

PLACEHOLDER_HINTS = (
    "your_",
    "replace",
    "example",
    "changeme",
    "placeholder",
    "xxx",
    "here",
)


def _is_env_example(path: Path) -> bool:
    return path.name in ENV_EXAMPLE_NAMES or path.name.endswith(".env.example")


def _looks_like_real_secret(match: str) -> bool:
    lower = match.lower()
    return not any(hint in lower for hint in PLACEHOLDER_HINTS)


def sanitize_text(content: str) -> tuple[str, int]:
    """Return sanitized content and number of replacements."""
    replacements = 0
    updated = content
    for pattern, replacement in SECRET_REPLACEMENTS:
        def _sub(m: re.Match[str]) -> str:
            nonlocal replacements
            if _looks_like_real_secret(m.group(0)):
                replacements += 1
                return replacement
            return m.group(0)

        updated = pattern.sub(_sub, updated)
    return updated, replacements


def sanitize_file(path: Path) -> int:
    """Sanitize one env example file in place. Returns replacement count."""
    if not path.is_file() or not _is_env_example(path):
        return 0
    original = path.read_text(encoding="utf-8", errors="ignore")
    updated, count = sanitize_text(original)
    if count > 0 and updated != original:
        path.write_text(updated, encoding="utf-8")
    return count


def sanitize_project(project_root: str) -> list[str]:
    """Sanitize all .env.example / env.example files under project_root."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        return []

    messages: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "node_modules" in path.parts:
            continue
        if not _is_env_example(path):
            continue
        count = sanitize_file(path)
        if count > 0:
            rel = path.relative_to(root).as_posix()
            messages.append(f"Sanitized {count} secret(s) in {rel}")
    return messages
