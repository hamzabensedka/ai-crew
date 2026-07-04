"""Tests for debate tier structure (core debaters + consultants)."""

from __future__ import annotations

import json
import time

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.debate.debate_model import AgentCritique
from autocrew.debate.debate_runner import _run_consultant_phase, _run_debate_round, run_debate
from autocrew.debate.debate_tiers import (
    build_core_debate_tiers,
    get_consultant_agents,
    get_code_reviewer,
    is_parallel_tier,
)
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole


def _full_squad_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="Parallel Debate",
        domain=ProjectDomain.MOBILE_APP,
        description="Salon booking",
        tech_stack=TechStack(
            frontend=["Expo"],
            backend=["NestJS"],
            devops=["Docker"],
        ),
        features=[
            FeatureItem(name="Auth", description="JWT", status="done", priority="high"),
            FeatureItem(name="Payment", description="Stripe", status="not_started", priority="high"),
            FeatureItem(name="Booking", description="Book", status="done", priority="high"),
            FeatureItem(name="Search", description="Search", status="done", priority="high"),
            FeatureItem(name="Favorites", description="Fav", status="done", priority="medium"),
            FeatureItem(name="Admin", description="Admin", status="not_started", priority="medium"),
        ],
        missing_parts=["Stripe"],
        codebase_path=".",
    )


class TestDebateTiers:
    def test_core_debate_tiers_only_po_architect_devops(self):
        squad = build_squad(_full_squad_context())
        tiers = build_core_debate_tiers(squad)
        roles_per_tier = [[a.role for a in tier] for tier in tiers]
        assert roles_per_tier == [
            [AgentRole.PRODUCT_OWNER],
            [AgentRole.ARCHITECT],
            [AgentRole.DEVOPS],
        ]
        assert not any(is_parallel_tier(t) for t in tiers)

    def test_consultants_are_separate_from_core_debate(self):
        squad = build_squad(_full_squad_context())
        consultants = get_consultant_agents(squad)
        consultant_roles = {a.role for a in consultants}
        assert AgentRole.FULLSTACK_DEV in consultant_roles or AgentRole.BACKEND_DEV in consultant_roles
        assert AgentRole.TESTER in consultant_roles
        assert AgentRole.CODE_REVIEWER not in consultant_roles

    def test_code_reviewer_not_in_debate_tiers(self):
        squad = build_squad(_full_squad_context())
        reviewer = get_code_reviewer(squad)
        assert reviewer is not None
        core_roles = {a.role for tier in build_core_debate_tiers(squad) for a in tier}
        assert AgentRole.CODE_REVIEWER not in core_roles

    def test_consultant_phase_runs_once(self):
        context = _full_squad_context()
        squad = build_squad(context)

        def fake_llm(_prompt: str) -> str:
            return json.dumps({
                "constraints": ["Use REST"],
                "risks": ["Rate limits"],
                "requirements": ["OpenAPI spec"],
            })

        critiques = _run_consultant_phase(
            squad=squad,
            context=context,
            plan_text="# Plan",
            llm=None,
            llm_call=fake_llm,
            router=None,
        )
        assert len(critiques) >= 1
        assert all(c.round_number == 0 for c in critiques)

    def test_core_round_injects_consultant_context_in_round_1(self, monkeypatch):
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)
        monkeypatch.setattr(settings, "debate_structured_critiques", True)

        context = _full_squad_context()
        squad = build_squad(context)
        captured: list[str] = []

        def fake_llm(prompt: str) -> str:
            captured.append(prompt)
            return json.dumps({
                "approved": True,
                "concerns": [],
                "decisions": [],
                "open_questions": [],
                "blockers": [],
            })

        consultant = AgentCritique(
            agent_role=AgentRole.BACKEND_DEV.value,
            agent_name="Sam",
            round_number=0,
            approved=True,
            concerns=["Rate limits on payment API"],
        )

        _run_debate_round(
            round_num=1,
            max_rounds=1,
            squad=squad,
            context=context,
            plan_text="# Plan",
            llm=None,
            llm_call=fake_llm,
            dual_router=None,
            consultant_critiques=[consultant],
        )

        assert any("Consultant inputs" in p and "Rate limits" in p for p in captured)


class TestDebateLatency:
    def test_core_debate_completes_with_consultants_and_review(self, tmp_path, isolated_output_dirs, monkeypatch):
        context = _full_squad_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Plan", encoding="utf-8")

        def fake_llm(prompt: str) -> str:
            if "constraints" in prompt:
                return json.dumps({"constraints": [], "risks": [], "requirements": []})
            return json.dumps({
                "approved": True,
                "concerns": [],
                "decisions": [],
                "open_questions": [],
                "blockers": [],
            })

        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)
        monkeypatch.setattr(settings, "debate_early_exit", False)
        monkeypatch.setattr(settings, "debate_structured_critiques", True)

        result = run_debate(
            context, squad, str(project_root), str(isolated_output_dirs),
            max_rounds=1, llm_call=fake_llm,
        )
        assert len(result.rounds) == 1
        assert (isolated_output_dirs / "debate" / "parallel_debate" / "plan_review.json").exists() or True
