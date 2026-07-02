"""Dataclasses for agent-call and phase-summary metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentCallRecord:
    session_id: str
    phase: str
    round_number: int | None
    agent_name: str
    agent_role: str
    model_used: str
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool
    latency_ms: float
    wall_clock_start: str
    wall_clock_end: str
    task_id: str | None = None
    project_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseSummaryRecord:
    session_id: str
    project_name: str
    phase: str
    wall_clock_start: str
    wall_clock_end: str
    total_rounds: int | None
    total_agent_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    debate_rounds: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extra"] = dict(self.extra)
        return data
