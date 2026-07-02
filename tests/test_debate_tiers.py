"""Tests for parallel debate tiers."""

from __future__ import annotations

import json
import time

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.debate.critique_types import StructuredConcern
from autocrew.debate.debate_model import AgentCritique
from autocrew.debate.debate_runner import (
    _context_critiques_for_agent,
    _run_debate_round,
    run_debate,
)
from autocrew.debate.debate_tiers import build_debate_tiers, is_parallel_tier
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentConfig, AgentRole


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
    def test_builds_expected_tier_structure(self):
        squad = build_squad(_full_squad_context())
        tiers = build_debate_tiers(squad)
        roles_per_tier = [[a.role for a in tier] for tier in tiers]
        assert roles_per_tier[0] == [AgentRole.PRODUCT_OWNER]
        assert roles_per_tier[1] == [AgentRole.ARCHITECT]
        assert len(roles_per_tier[2]) >= 2
        assert is_parallel_tier(tiers[2])
        assert roles_per_tier[-1] == [AgentRole.PROGRESS_TRACKER]
        assert AgentRole.TESTER in roles_per_tier[-3]
        assert AgentRole.CODE_REVIEWER in roles_per_tier[-2]

    def test_parallel_tier_uses_only_previous_tier_context(self):
        architect = AgentCritique(
            agent_role=AgentRole.ARCHITECT.value,
            agent_name="Jordan",
            round_number=1,
            approved=False,
            structured_concerns=[
                StructuredConcern("c_arch", "high", "Define OpenAPI first", []),
            ],
        )
        architect.concerns = ["Define OpenAPI first"]
        backend = AgentConfig(
            role=AgentRole.BACKEND_DEV,
            name="Sam",
            goal="g",
            backstory="b",
            tools=[],
            can_write_to=[],
            can_read=[],
        )
        frontend_critique = AgentCritique(
            agent_role=AgentRole.FRONTEND_DEV.value,
            agent_name="Riley",
            round_number=1,
            approved=False,
            structured_concerns=[
                StructuredConcern("c_fe", "high", "GraphQL for mobile", []),
            ],
        )
        frontend_critique.concerns = ["GraphQL for mobile"]

        tier = [
            backend,
            AgentConfig(
                role=AgentRole.FRONTEND_DEV,
                name="Riley",
                goal="g",
                backstory="b",
                tools=[],
                can_write_to=[],
                can_read=[],
            ),
        ]
        ctx, _ = _context_critiques_for_agent(
            backend,
            tier=tier,
            round_critiques=[architect, frontend_critique],
            previous_tier_critiques=[architect],
            parallel_tiers_enabled=True,
        )
        assert len(ctx) == 1
        assert ctx[0].agent_role == AgentRole.ARCHITECT.value

    def test_tester_gets_full_round_context_in_parallel_mode(self):
        po = AgentCritique(
            agent_role=AgentRole.PRODUCT_OWNER.value,
            agent_name="Alex",
            round_number=1,
            approved=True,
        )
        backend = AgentCritique(
            agent_role=AgentRole.BACKEND_DEV.value,
            agent_name="Sam",
            round_number=1,
            approved=False,
            structured_concerns=[StructuredConcern("c_be", "high", "REST API", [])],
        )
        backend.concerns = ["REST API"]
        tester = AgentConfig(
            role=AgentRole.TESTER,
            name="Jamie",
            goal="g",
            backstory="b",
            tools=[],
            can_write_to=[],
            can_read=[],
        )
        ctx, _ = _context_critiques_for_agent(
            tester,
            tier=[tester],
            round_critiques=[po, backend],
            previous_tier_critiques=[backend],
            parallel_tiers_enabled=True,
        )
        assert len(ctx) == 2


class TestParallelDebateLatency:
    def test_parallel_tiers_faster_than_sequential(self, tmp_path, isolated_output_dirs, monkeypatch):
        delay = 0.03
        context = _full_squad_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Plan", encoding="utf-8")

        def slow_llm(_prompt: str) -> str:
            time.sleep(delay)
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

        monkeypatch.setattr(settings, "debate_parallel_tiers", False)
        start_seq = time.perf_counter()
        run_debate(
            context, squad, str(project_root), str(isolated_output_dirs),
            max_rounds=1, llm_call=slow_llm,
        )
        sequential_ms = (time.perf_counter() - start_seq) * 1000

        monkeypatch.setattr(settings, "debate_parallel_tiers", True)
        start_par = time.perf_counter()
        run_debate(
            context, squad, str(project_root), str(isolated_output_dirs),
            max_rounds=1, llm_call=slow_llm,
        )
        parallel_ms = (time.perf_counter() - start_par) * 1000

        tiers = build_debate_tiers(squad)
        parallel_devs = sum(len(t) for t in tiers if is_parallel_tier(t))
        assert parallel_devs >= 2
        assert parallel_ms < sequential_ms * 0.85, (
            f"parallel {parallel_ms:.0f}ms vs sequential {sequential_ms:.0f}ms"
        )

    def test_parallel_round_preserves_distinct_dev_concerns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "debate_parallel_tiers", True)
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)
        monkeypatch.setattr(settings, "debate_structured_critiques", True)

        context = _full_squad_context()
        squad = build_squad(context)

        captured_prompts: list[str] = []

        def fake_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            if "Backend" in prompt or "backend_developer" in prompt:
                return json.dumps({
                    "approved": False,
                    "concerns": [
                        {
                            "id": "c_be_rest",
                            "severity": "high",
                            "text": "Backend requires REST endpoints with /v1 prefix",
                            "targets": [],
                        }
                    ],
                    "decisions": [],
                    "open_questions": [],
                    "blockers": [],
                })
            if "Frontend" in prompt or "frontend_developer" in prompt:
                return json.dumps({
                    "approved": False,
                    "concerns": [
                        {
                            "id": "c_fe_graphql",
                            "severity": "high",
                            "text": "Mobile client should use GraphQL for bandwidth",
                            "targets": [],
                        }
                    ],
                    "decisions": [],
                    "open_questions": [],
                    "blockers": [],
                })
            return json.dumps({
                "approved": True,
                "concerns": [],
                "decisions": [],
                "open_questions": [],
                "blockers": [],
            })

        critiques = _run_debate_round(
            round_num=1,
            max_rounds=1,
            squad=squad,
            context=context,
            plan_text="# Plan",
            llm=None,
            llm_call=fake_llm,
            dual_router=None,
        )

        roles = {c.agent_role: c for c in critiques}
        assert "c_be_rest" in str(roles[AgentRole.BACKEND_DEV.value].structured_concerns)
        assert "c_fe_graphql" in str(roles[AgentRole.FRONTEND_DEV.value].structured_concerns)

        tester_prompt = next(
            p for p in captured_prompts if "tester" in p.lower() or "Jamie" in p
        )
        assert "c_be_rest" in tester_prompt
        assert "c_fe_graphql" in tester_prompt
