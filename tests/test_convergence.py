"""Tests for debate convergence early-exit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.debate.convergence import (
    collect_round_items,
    diff_rounds,
    should_early_exit,
)
from autocrew.debate.critique_types import StructuredConcern, StructuredOpenQuestion
from autocrew.debate.debate_model import AgentCritique
from autocrew.debate.debate_runner import run_debate
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole


def _planity_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="Debate Test",
        domain=ProjectDomain.MOBILE_APP,
        description="Salon booking app",
        tech_stack=TechStack(
            frontend=["Expo", "React Native"],
            backend=["NestJS", "Prisma"],
            devops=["Docker"],
        ),
        features=[
            FeatureItem(name="User Auth", description="JWT auth", status="done", priority="high"),
            FeatureItem(name="Payment Integration", description="Stripe", status="not_started", priority="high"),
            FeatureItem(name="Admin Dashboard", description="Next.js admin", status="not_started", priority="medium"),
        ],
        missing_parts=["Stripe checkout", "Admin dashboard"],
        codebase_path=".",
    )


def _critique(
    role: AgentRole,
    name: str,
    *,
    concerns: list[StructuredConcern] | None = None,
    questions: list[StructuredOpenQuestion] | None = None,
) -> AgentCritique:
    critique = AgentCritique(
        agent_role=role.value,
        agent_name=name,
        round_number=1,
        approved=False,
        structured_concerns=concerns or [],
        structured_open_questions=questions or [],
    )
    critique.concerns = [c.text for c in critique.structured_concerns]
    return critique


class TestConvergenceDiff:
    def test_detects_net_new_concern_by_id(self):
        prev = [
            _critique(
                AgentRole.BACKEND_DEV,
                "Sam",
                concerns=[StructuredConcern("c1", "high", "Need REST API", [])],
            )
        ]
        cur = [
            _critique(
                AgentRole.BACKEND_DEV,
                "Sam",
                concerns=[
                    StructuredConcern("c1", "high", "Need REST API", []),
                    StructuredConcern("c2", "medium", "Add pagination", []),
                ],
            )
        ]
        diff = diff_rounds(prev, cur, previous_round=1, current_round=2)
        assert len(diff.net_new_concerns) == 1
        assert diff.net_new_concerns[0].item_id == "c2"
        assert diff.has_net_new

    def test_same_ids_and_text_are_not_net_new(self):
        concern = StructuredConcern("c1", "high", "Need REST API", [])
        prev = [_critique(AgentRole.BACKEND_DEV, "Sam", concerns=[concern])]
        cur = [_critique(AgentRole.BACKEND_DEV, "Sam", concerns=[concern])]
        diff = diff_rounds(prev, cur, previous_round=1, current_round=2)
        assert not diff.net_new_concerns
        assert not diff.net_new_open_questions
        assert not diff.has_net_new

    def test_text_match_without_same_id_counts_as_known(self):
        prev = [
            _critique(
                AgentRole.BACKEND_DEV,
                "Sam",
                concerns=[StructuredConcern("c1", "high", "Need REST API", [])],
            )
        ]
        cur = [
            _critique(
                AgentRole.BACKEND_DEV,
                "Sam",
                concerns=[StructuredConcern("c9", "high", "Need REST API", [])],
            )
        ]
        diff = diff_rounds(prev, cur, previous_round=1, current_round=2)
        assert not diff.has_net_new

    def test_excludes_progress_tracker_from_round_items(self):
        critiques = [
            _critique(
                AgentRole.BACKEND_DEV,
                "Sam",
                concerns=[StructuredConcern("c1", "high", "Backend gap", [])],
            ),
            _critique(
                AgentRole.PROGRESS_TRACKER,
                "Avery",
                concerns=[StructuredConcern("c_tracker", "high", "Echoed concern", [])],
            ),
        ]
        concerns, questions = collect_round_items(critiques)
        assert len(concerns) == 1
        assert concerns[0].item_id == "c1"

    def test_should_early_exit_respects_min_rounds(self):
        diff = diff_rounds([], [], previous_round=1, current_round=2)
        assert not should_early_exit(diff, round_number=1, min_rounds=1)
        assert should_early_exit(diff, round_number=2, min_rounds=1)
        assert not should_early_exit(diff, round_number=2, min_rounds=3)


class TestDebateEarlyExitIntegration:
    def test_stops_when_round_raises_no_new_items(
        self, tmp_path, isolated_output_dirs, monkeypatch
    ):
        monkeypatch.setattr(settings, "debate_early_exit", True)
        monkeypatch.setattr(settings, "debate_min_rounds", 1)
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)
        monkeypatch.setattr(settings, "debate_structured_critiques", True)

        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nPlan", encoding="utf-8")

        call_counts = {"n": 0}

        def fake_llm(prompt: str) -> str:
            call_counts["n"] += 1
            if call_counts["n"] <= len(squad.agents):
                return json.dumps({
                    "approved": False,
                    "concerns": [
                        {
                            "id": "c1",
                            "severity": "high",
                            "text": "Payment spec missing",
                            "targets": [],
                        }
                    ],
                    "decisions": [],
                    "open_questions": [],
                    "blockers": [],
                })
            return json.dumps({
                "approved": False,
                "concerns": [
                    {
                        "id": "c1",
                        "severity": "high",
                        "text": "Payment spec missing",
                        "targets": [],
                    }
                ],
                "decisions": [],
                "open_questions": [],
                "blockers": [],
            })

        result = run_debate(
            context,
            squad,
            str(project_root),
            str(isolated_output_dirs),
            max_rounds=5,
            llm_call=fake_llm,
        )

        assert result.converged_early
        assert result.early_exit_round == 2
        assert len(result.rounds) == 2
        assert Path(result.early_exit_log_path).is_file()
        log_lines = Path(result.early_exit_log_path).read_text(encoding="utf-8").strip().splitlines()
        event = json.loads(log_lines[-1])
        assert event["event"] == "debate_early_exit"
        assert event["round_number"] == 2
        assert event["task_id"]

    def test_min_rounds_prevents_exit_until_threshold(
        self, tmp_path, isolated_output_dirs, monkeypatch
    ):
        monkeypatch.setattr(settings, "debate_early_exit", True)
        monkeypatch.setattr(settings, "debate_min_rounds", 3)
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)

        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nPlan", encoding="utf-8")

        stable = json.dumps({
            "approved": False,
            "concerns": [{"id": "c1", "severity": "high", "text": "Same issue", "targets": []}],
            "decisions": [],
            "open_questions": [],
            "blockers": [{"id": "b1", "severity": "high", "text": "Blocker", "targets": []}],
        })

        result = run_debate(
            context,
            squad,
            str(project_root),
            str(isolated_output_dirs),
            max_rounds=5,
            llm_call=lambda _: stable,
        )

        assert len(result.rounds) >= 3
        if result.converged_early:
            assert result.early_exit_round is not None
            assert result.early_exit_round >= 3
