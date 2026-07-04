"""Tests for session metrics instrumentation."""

from __future__ import annotations

import json

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.crew.llm_task_executor import execute_task_with_llm
from autocrew.crew.crew_logger import CrewLogger
from autocrew.debate.debate_runner import run_debate
from autocrew.metrics import begin_session, end_session, query_metrics
from autocrew.metrics.collector import SessionMetricsCollector
from autocrew.metrics.instrumentation import instrument_llm_call
from autocrew.metrics.store import MetricsStore
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole
from autocrew.tasks.task_model import TaskConfig


def _context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="MetricsTest",
        domain=ProjectDomain.API,
        description="Test",
        tech_stack=TechStack(backend=["FastAPI"]),
        features=[
            FeatureItem(name="Auth", description="JWT", status="not_started", priority="high"),
            FeatureItem(name="Health", description="Health", status="done", priority="low"),
        ],
        missing_parts=["authentication"],
        codebase_path=".",
    )


class TestMetricsStore:
    def test_persists_agent_call_and_summary(self, isolated_output_dirs):
        collector = SessionMetricsCollector(
            project_name="StoreTest",
            metrics_dir=str(isolated_output_dirs / "metrics"),
        )
        collector.start_phase("debate")
        collector.record_agent_call(
            phase="debate",
            round_number=1,
            agent_name="Alex",
            agent_role="product_owner",
            model_used="test-model",
            input_tokens=100,
            output_tokens=50,
            tokens_estimated=True,
            latency_ms=1200.5,
            wall_clock_start="2026-01-01T00:00:00+00:00",
            wall_clock_end="2026-01-01T00:00:02+00:00",
        )
        collector.end_phase("debate", debate_rounds=1)

        store = MetricsStore(str(isolated_output_dirs / "metrics"))
        calls = store.fetch_agent_calls("debate")
        summaries = store.fetch_phase_summaries("debate")
        assert len(calls) == 1
        assert calls[0]["input_tokens"] == 100
        assert len(summaries) == 1
        assert summaries[0]["debate_rounds"] == 1

        jsonl = (isolated_output_dirs / "metrics" / "agent_calls.jsonl").read_text(encoding="utf-8")
        assert "agent_call" in jsonl
        assert "phase_summary" in jsonl


class TestInstrumentation:
    def test_instrument_llm_call_records_tokens_and_latency(self, isolated_output_dirs, monkeypatch):
        monkeypatch.setattr(settings, "metrics_dir", str(isolated_output_dirs / "metrics"))
        begin_session("InstrumentTest", phase="debate")

        def fake_llm(prompt: str) -> str:
            return '{"approved": true, "concerns": [], "suggestions": [], "blockers": []}'

        wrapped = instrument_llm_call(
            fake_llm,
            phase="debate",
            agent_name="Alex",
            agent_role="product_owner",
            model_name="test/model",
            round_number=1,
        )
        wrapped("hello " * 100)
        end_session(phase="debate", debate_rounds=1)

        metrics = query_metrics(str(isolated_output_dirs / "metrics"))
        assert metrics["sessions"]["debate"] == 1
        assert metrics["debate"]["tokens_median"] > 0
        assert metrics["debate"]["latency_ms_median"] >= 0


class TestDebateMetricsIntegration:
    def test_debate_records_per_agent_calls(self, tmp_path, isolated_output_dirs, monkeypatch):
        monkeypatch.setattr(settings, "metrics_enabled", True)
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)

        context = _context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nPlan", encoding="utf-8")

        def fake_llm(prompt: str) -> str:
            return json.dumps({
                "approved": True,
                "concerns": [],
                "suggestions": [],
                "blockers": [],
            })

        run_debate(
            context,
            squad,
            str(project_root),
            str(isolated_output_dirs),
            max_rounds=1,
            llm_call=fake_llm,
        )

        metrics = query_metrics(str(isolated_output_dirs / "metrics"))
        store = MetricsStore(str(isolated_output_dirs / "metrics"))
        calls = store.fetch_agent_calls("debate")

        assert metrics["sessions"]["debate"] == 1
        assert metrics["debate"]["rounds_median"] == 1.0
        assert len(calls) >= 3  # consultants + core debaters + plan review

        tracker_calls = [c for c in calls if c["agent_role"] == AgentRole.PROGRESS_TRACKER.value]
        assert len(tracker_calls) == 0  # tracker no longer participates in debate rounds


class TestBuildMetricsIntegration:
    def test_build_llm_task_records_metrics(self, tmp_path, isolated_output_dirs, monkeypatch):
        monkeypatch.setattr(settings, "metrics_enabled", True)
        begin_session("BuildMetrics", phase="build")

        context = _context()
        squad = build_squad(context)
        agent = squad.agents[0]
        project_root = tmp_path / "proj"
        project_root.mkdir()
        (project_root / "docs").mkdir()

        task = TaskConfig(
            task_id="po_product_spec",
            title="Write spec",
            description="Write product spec",
            assigned_agent_role=agent.role.value,
            depends_on=[],
            output_format="markdown",
            output_path="docs/product.md",
            expected_output="product.md",
        )

        payload = json.dumps({
            "content": "# Product\nHello",
            "summary": "wrote spec",
        })

        execute_task_with_llm(
            task,
            agent,
            context,
            str(project_root),
            CrewLogger(),
            lambda _: payload,
            model_name="test/model",
        )
        end_session(phase="build")

        metrics = query_metrics(str(isolated_output_dirs / "metrics"))
        assert metrics["sessions"]["build"] == 1
        assert metrics["build"]["tokens_median"] > 0
