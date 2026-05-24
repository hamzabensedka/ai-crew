"""Real-time and persistent logging for crew execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table


class CrewLogger:
    def __init__(self, log_path: str | None = None) -> None:
        self.console = Console()
        self.log_path = log_path
        self._entries: list[str] = []
        self._agent_status: dict[str, str] = {}

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._entries.append(entry)
        self.console.print(entry)

    def set_agent_status(self, agent: str, status: str, task: str = "") -> None:
        self._agent_status[agent] = f"{status}: {task}" if task else status

    def build_table(self) -> Table:
        table = Table(title="AutoCrew Execution")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="green")
        for agent, status in self._agent_status.items():
            table.add_row(agent, status)
        return table

    def flush(self) -> None:
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.log_path).write_text("\n".join(self._entries), encoding="utf-8")


class LiveCrewDisplay:
    def __init__(self, logger: CrewLogger) -> None:
        self.logger = logger
        self._live: Live | None = None

    def __enter__(self) -> "LiveCrewDisplay":
        self._live = Live(self.logger.build_table(), refresh_per_second=4, console=self.logger.console)
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)

    def refresh(self) -> None:
        if self._live:
            self._live.update(self.logger.build_table())
