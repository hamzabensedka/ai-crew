"""Backfill metrics from saved debate_result.json files (retroactive round/token estimates)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from autocrew.debate.debate_model import DebateResult
from autocrew.metrics.collector import SessionMetricsCollector
from autocrew.metrics.tokens import estimate_tokens


def backfill_debate_from_results(
    debate_results: list[Path],
    *,
    metrics_dir: str,
    tokens_estimated: bool = True,
) -> list[str]:
    """Import historical debate sessions; tokens are estimated from critique text."""
    session_ids: list[str] = []
    for path in debate_results:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result = DebateResult.from_dict(data)
        collector = SessionMetricsCollector(
            project_name=result.project_name,
            session_id=f"backfill_{path.parent.name}_{uuid.uuid4().hex[:8]}",
            metrics_dir=metrics_dir,
        )
        collector.start_phase("debate")
        wall_start = result.timestamp or datetime.now(timezone.utc).isoformat()

        for round_data in result.rounds:
            plan_chars = len(round_data.revised_plan_excerpt or "")
            for critique in round_data.critiques:
                output_text = "\n".join(
                    critique.blockers + critique.concerns + critique.suggestions
                )
                model = critique.model_used or "unknown"
                if model == "deterministic":
                    input_tokens = 0
                    output_tokens = 0
                    estimated = False
                else:
                    input_tokens = estimate_tokens("x" * min(plan_chars, 8000)) + 1500
                    output_tokens = estimate_tokens(output_text)
                    estimated = tokens_estimated

                collector.record_agent_call(
                    phase="debate",
                    round_number=round_data.round_number,
                    agent_name=critique.agent_name,
                    agent_role=critique.agent_role,
                    model_used=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_estimated=estimated,
                    latency_ms=0.0,
                    wall_clock_start=wall_start,
                    wall_clock_end=wall_start,
                )

        collector.end_phase("debate", debate_rounds=len(result.rounds))
        session_ids.append(collector.session_id)

    return session_ids
