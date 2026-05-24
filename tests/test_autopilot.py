"""Tests for autopilot loop."""

from unittest.mock import MagicMock

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.autopilot import _is_mission_complete, is_build_complete, run_autopilot, run_project_tests
from autocrew.debate.debate_model import DebateResult, DebateRound
from autocrew.security_audit import SecurityReport
from autocrew.squad.squad_builder import build_squad


class TestAutopilot:
    def test_mission_complete_requires_all_gates(self):
        ok, _ = _is_mission_complete(
            consensus=False,
            build_ok=True,
            security_ok=True,
            tests_passed=True,
            require_tests=True,
        )
        assert not ok

        ok, reason = _is_mission_complete(
            consensus=True,
            build_ok=True,
            security_ok=True,
            tests_passed=True,
            require_tests=True,
        )
        assert ok
        assert "crew approved" in reason

    def test_build_complete_flags_missing_features(self):
        context = ProjectContext(
            project_type=ProjectType.EXISTING_CODE,
            project_name="X",
            domain=ProjectDomain.MOBILE_APP,
            description="d",
            tech_stack=TechStack(),
            features=[
                FeatureItem(name="Payments", description="p", status="not_started", priority="high"),
            ],
        )
        ok, msg = is_build_complete(context, 100.0, 100.0)
        assert not ok
        assert "Payments" in msg

    def test_run_project_tests_skips_without_script(self, tmp_path):
        (tmp_path / "package.json").write_text('{"scripts": {}}', encoding="utf-8")
        passed, msg = run_project_tests(str(tmp_path))
        assert passed
        assert "skipped" in msg

    def test_autopilot_stops_when_fully_done(self, tmp_path, isolated_output_dirs, monkeypatch):
        context = ProjectContext(
            project_type=ProjectType.EXISTING_CODE,
            project_name="AutoTest",
            domain=ProjectDomain.MOBILE_APP,
            description="Test autopilot",
            tech_stack=TechStack(frontend=["Expo"], backend=["NestJS"]),
            features=[FeatureItem(name="Auth", description="Login", status="done", priority="high")],
            codebase_path=str(tmp_path),
            missing_parts=[],
        )
        squad = build_squad(context)
        project_root = tmp_path / "proj"
        project_root.mkdir()

        fake_debate = DebateResult(
            project_name="AutoTest",
            timestamp="2026-01-01T00:00:00Z",
            rounds=[
                DebateRound(
                    round_number=1,
                    critiques=[],
                    revised_plan_excerpt="",
                    all_approved=True,
                    total_blockers=0,
                )
            ],
            consensus_reached=True,
            final_plan_path="plan.md",
            debate_dir=str(isolated_output_dirs / "debate"),
            action_items=[],
        )

        monkeypatch.setattr("autocrew.autopilot.run_debate", lambda *a, **k: fake_debate)
        monkeypatch.setattr("autocrew.autopilot.build_tasks_from_debate", lambda *a, **k: [])
        monkeypatch.setattr("autocrew.autopilot.run_crew", lambda *a, **k: "ok")
        monkeypatch.setattr("autocrew.autopilot.run_project_tests", lambda *a, **k: (True, "passed"))
        monkeypatch.setattr(
            "autocrew.autopilot.run_security_audit",
            lambda *a, **k: SecurityReport(passed=True, summary="ok"),
        )
        monkeypatch.setattr("autocrew.autopilot.run_llm_security_review", lambda *a, **k: SecurityReport(passed=True))

        result = run_autopilot(
            context,
            squad,
            str(project_root),
            str(isolated_output_dirs),
            max_cycles=10,
            min_completion=100,
            llm_security=False,
        )
        assert result.consensus_reached
        assert result.build_complete
        assert result.security_passed
        assert len(result.cycles) == 1

    def test_autopilot_continues_on_security_fail(self, tmp_path, isolated_output_dirs, monkeypatch):
        context = ProjectContext(
            project_type=ProjectType.EXISTING_CODE,
            project_name="AutoSec",
            domain=ProjectDomain.API,
            description="Test",
            tech_stack=TechStack(backend=["NestJS"]),
            features=[FeatureItem(name="API", description="REST", status="done", priority="high")],
            missing_parts=[],
        )
        squad = build_squad(context)
        project_root = tmp_path / "proj"
        project_root.mkdir()

        debate_ok = DebateResult(
            project_name="AutoSec",
            timestamp="2026-01-01T00:00:00Z",
            rounds=[
                DebateRound(
                    round_number=1,
                    critiques=[],
                    revised_plan_excerpt="",
                    all_approved=True,
                    total_blockers=0,
                )
            ],
            consensus_reached=True,
            final_plan_path="plan.md",
            debate_dir=str(isolated_output_dirs / "debate"),
            action_items=[],
        )

        calls = {"n": 0}

        def fake_security(*_a, **_k):
            calls["n"] += 1
            if calls["n"] >= 2:
                return SecurityReport(passed=True, summary="fixed")
            return SecurityReport(
                passed=False,
                summary="webhook issue",
            )

        monkeypatch.setattr("autocrew.autopilot.run_debate", lambda *a, **k: debate_ok)
        monkeypatch.setattr("autocrew.autopilot.build_tasks_from_debate", lambda *a, **k: [])
        monkeypatch.setattr("autocrew.autopilot.run_crew", lambda *a, **k: "ok")
        monkeypatch.setattr("autocrew.autopilot.run_project_tests", lambda *a, **k: (True, "ok"))
        monkeypatch.setattr("autocrew.autopilot.run_security_audit", fake_security)
        monkeypatch.setattr("autocrew.autopilot.run_llm_security_review", lambda r, *a, **k: r)

        result = run_autopilot(
            context,
            squad,
            str(project_root),
            str(isolated_output_dirs),
            max_cycles=5,
            llm_security=False,
        )
        assert result.security_passed
        assert len(result.cycles) == 2
