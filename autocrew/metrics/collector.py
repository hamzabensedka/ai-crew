"""In-memory session collector with persistence hooks."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from autocrew.config import settings
from autocrew.metrics.models import AgentCallRecord, PhaseSummaryRecord
from autocrew.metrics.store import MetricsStore

_active: SessionMetricsCollector | None = None


class SessionMetricsCollector:
    def __init__(
        self,
        *,
        project_name: str,
        session_id: str | None = None,
        metrics_dir: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.session_id = session_id or uuid.uuid4().hex
        self.metrics_dir = metrics_dir or settings.metrics_dir
        self.store = MetricsStore(self.metrics_dir)
        self.agent_calls: list[AgentCallRecord] = []
        self.phase_summaries: list[PhaseSummaryRecord] = []
        self._phase_starts: dict[str, str] = {}

    def start_phase(self, phase: str) -> None:
        self._phase_starts[phase] = datetime.now(timezone.utc).isoformat()

    def record_agent_call(
        self,
        *,
        phase: str,
        agent_name: str,
        agent_role: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        wall_clock_start: str,
        wall_clock_end: str,
        round_number: int | None = None,
        task_id: str | None = None,
        tokens_estimated: bool = True,
        **kwargs: Any,
    ) -> AgentCallRecord:
        record = AgentCallRecord(
            session_id=self.session_id,
            project_name=self.project_name,
            phase=phase,
            round_number=round_number,
            agent_name=agent_name,
            agent_role=agent_role,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_estimated=tokens_estimated,
            latency_ms=latency_ms,
            wall_clock_start=wall_clock_start,
            wall_clock_end=wall_clock_end,
            task_id=task_id,
            provider_used=kwargs.get("provider_used", ""),
            is_paid_fallback=bool(kwargs.get("is_paid_fallback", False)),
            rate_limit_wait_ms=float(kwargs.get("rate_limit_wait_ms", 0.0)),
        )
        self.agent_calls.append(record)
        if settings.metrics_enabled:
            self.store.insert_agent_call(record)
        return record

    def end_phase(
        self,
        phase: str,
        *,
        debate_rounds: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PhaseSummaryRecord:
        phase_calls = [c for c in self.agent_calls if c.phase == phase]
        wall_clock_end = datetime.now(timezone.utc).isoformat()
        wall_clock_start = self._phase_starts.get(phase, wall_clock_end)

        summary = PhaseSummaryRecord(
            session_id=self.session_id,
            project_name=self.project_name,
            phase=phase,
            wall_clock_start=wall_clock_start,
            wall_clock_end=wall_clock_end,
            total_rounds=debate_rounds if phase == "debate" else None,
            total_agent_calls=len(phase_calls),
            total_input_tokens=sum(c.input_tokens for c in phase_calls),
            total_output_tokens=sum(c.output_tokens for c in phase_calls),
            total_latency_ms=sum(c.latency_ms for c in phase_calls),
            debate_rounds=debate_rounds,
            extra=extra or {},
        )
        self.phase_summaries.append(summary)
        if settings.metrics_enabled:
            self.store.insert_phase_summary(summary)
        return summary


def set_metrics_collector(collector: SessionMetricsCollector | None) -> None:
    global _active
    _active = collector


def get_metrics_collector() -> SessionMetricsCollector | None:
    return _active


def begin_session(project_name: str, *, phase: str) -> SessionMetricsCollector | None:
    if not settings.metrics_enabled:
        return None
    collector = SessionMetricsCollector(project_name=project_name)
    collector.start_phase(phase)
    set_metrics_collector(collector)
    return collector


def end_session(*, phase: str, debate_rounds: int | None = None, extra: dict | None = None) -> None:
    collector = get_metrics_collector()
    if collector is None:
        return
    collector.end_phase(phase, debate_rounds=debate_rounds, extra=extra)
    set_metrics_collector(None)


def record_agent_call(**kwargs: Any) -> None:
    collector = get_metrics_collector()
    if collector is None or not settings.metrics_enabled:
        return
    collector.record_agent_call(**kwargs)
