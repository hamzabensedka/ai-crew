"""Persist metrics to SQLite and JSONL."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autocrew.metrics.models import AgentCallRecord, PhaseSummaryRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    project_name TEXT NOT NULL,
    phase TEXT NOT NULL,
    round_number INTEGER,
    agent_name TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    model_used TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    tokens_estimated INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    wall_clock_start TEXT NOT NULL,
    wall_clock_end TEXT NOT NULL,
    task_id TEXT
);

CREATE TABLE IF NOT EXISTS phase_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    project_name TEXT NOT NULL,
    phase TEXT NOT NULL,
    wall_clock_start TEXT NOT NULL,
    wall_clock_end TEXT NOT NULL,
    total_rounds INTEGER,
    total_agent_calls INTEGER NOT NULL,
    total_input_tokens INTEGER NOT NULL,
    total_output_tokens INTEGER NOT NULL,
    total_latency_ms REAL NOT NULL,
    debate_rounds INTEGER,
    extra_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_calls_session ON agent_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_calls_phase ON agent_calls(phase);
CREATE INDEX IF NOT EXISTS idx_phase_summaries_phase ON phase_summaries(phase);
"""


def default_db_path(metrics_dir: str) -> Path:
    return Path(metrics_dir) / "session_metrics.db"


def default_jsonl_path(metrics_dir: str) -> Path:
    return Path(metrics_dir) / "agent_calls.jsonl"


class MetricsStore:
    def __init__(self, metrics_dir: str) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = default_db_path(str(self.metrics_dir))
        self.jsonl_path = default_jsonl_path(str(self.metrics_dir))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def insert_agent_call(self, record: AgentCallRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_calls (
                    session_id, project_name, phase, round_number, agent_name, agent_role,
                    model_used, input_tokens, output_tokens, tokens_estimated, latency_ms,
                    wall_clock_start, wall_clock_end, task_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.project_name,
                    record.phase,
                    record.round_number,
                    record.agent_name,
                    record.agent_role,
                    record.model_used,
                    record.input_tokens,
                    record.output_tokens,
                    int(record.tokens_estimated),
                    record.latency_ms,
                    record.wall_clock_start,
                    record.wall_clock_end,
                    record.task_id,
                ),
            )

        line = {"type": "agent_call", **record.to_dict()}
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=True) + "\n")

    def insert_phase_summary(self, record: PhaseSummaryRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO phase_summaries (
                    session_id, project_name, phase, wall_clock_start, wall_clock_end,
                    total_rounds, total_agent_calls, total_input_tokens, total_output_tokens,
                    total_latency_ms, debate_rounds, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.project_name,
                    record.phase,
                    record.wall_clock_start,
                    record.wall_clock_end,
                    record.total_rounds,
                    record.total_agent_calls,
                    record.total_input_tokens,
                    record.total_output_tokens,
                    record.total_latency_ms,
                    record.debate_rounds,
                    json.dumps(record.extra),
                ),
            )

        line = {"type": "phase_summary", **record.to_dict()}
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=True) + "\n")

    def fetch_agent_calls(self, phase: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if phase:
                rows = conn.execute(
                    "SELECT * FROM agent_calls WHERE phase = ? ORDER BY id",
                    (phase,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM agent_calls ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def fetch_phase_summaries(self, phase: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if phase:
                rows = conn.execute(
                    "SELECT * FROM phase_summaries WHERE phase = ? ORDER BY id",
                    (phase,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM phase_summaries ORDER BY id").fetchall()
        return [dict(row) for row in rows]
