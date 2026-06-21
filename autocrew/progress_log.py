"""Timestamped progress logs for long LLM calls (debate, autopilot, build)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

_active: "ProgressLogger | None" = None


class ProgressLogger:
    def __init__(self, log_path: str | None = None, *, verbose: bool = False) -> None:
        self.console = Console()
        self.log_path = log_path
        self.verbose = verbose
        self._entries: list[str] = []

    def log(self, message: str, *, verbose_only: bool = False) -> None:
        if verbose_only and not self.verbose:
            return
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._entries.append(entry)
        if verbose_only:
            self.console.print(f"[dim]{entry}[/dim]")
        else:
            self.console.print(f"[dim][{timestamp}][/dim] {message}")
        self._flush()

    def _flush(self) -> None:
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self._entries), encoding="utf-8")


def set_progress_logger(logger: ProgressLogger | None) -> None:
    global _active
    _active = logger


def get_progress_logger() -> ProgressLogger | None:
    return _active


def progress_log(message: str, *, verbose_only: bool = False) -> None:
    logger = _active
    if logger is not None:
        logger.log(message, verbose_only=verbose_only)
