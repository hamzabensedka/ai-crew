"""Tests for structured critique schema and selective context pass-through."""

from __future__ import annotations

import json

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.debate.critique_context import (
    build_critique_context,
    context_contains_concern_ids,
    context_contains_texts,
)
from autocrew.debate.critique_schema import parse_critique_response
from autocrew.debate.critique_types import StructuredConcern
from autocrew.debate.debate_model import AgentCritique
from autocrew.debate.debate_runner import _format_other_critiques
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


def _structured_critique(
    role: AgentRole,
    name: str,
    *,
    concern_id: str,
    concern_text: str,
    round_number: int = 1,
) -> AgentCritique:
    critique = AgentCritique(
        agent_role=role.value,
        agent_name=name,
        round_number=round_number,
        approved=False,
        structured_concerns=[
            StructuredConcern(
                id=concern_id,
                severity="high",
                text=concern_text,
                targets=["architect"],
            )
        ],
    )
    critique.concerns = [concern_text]
    return critique


@pytest.fixture
def disagreeing_prior() -> list[AgentCritique]:
    return [
        _structured_critique(
            AgentRole.PRODUCT_OWNER,
            "Alex",
            concern_id="c_po_scope",
            concern_text="MVP scope should exclude provider portal until Q2",
        ),
        _structured_critique(
            AgentRole.ARCHITECT,
            "Jordan",
            concern_id="c_arch_api",
            concern_text="API must use REST with explicit OpenAPI 3.0 versioned paths",
        ),
        _structured_critique(
            AgentRole.BACKEND_DEV,
            "Sam",
            concern_id="c_be_rest",
            concern_text="Backend requires REST endpoints with /v1 prefix for all resources",
        ),
        _structured_critique(
            AgentRole.FRONTEND_DEV,
            "Riley",
            concern_id="c_fe_graphql",
            concern_text="Mobile client should use GraphQL to reduce over-fetching on slow networks",
        ),
    ]


class TestCritiqueSchema:
    def test_parse_structured_llm_response(self):
        context = _planity_context()
        context.features.extend([
            FeatureItem(name="Booking", description="Book", status="done", priority="high"),
            FeatureItem(name="Search", description="Search", status="done", priority="high"),
            FeatureItem(name="Favorites", description="Fav", status="done", priority="medium"),
        ])
        squad = build_squad(context)
        agent = next(
            a for a in squad.agents
            if a.role in (AgentRole.BACKEND_DEV, AgentRole.FULLSTACK_DEV)
        )
        data = {
            "approved": False,
            "concerns": [
                {
                    "id": "c1",
                    "severity": "high",
                    "text": "Missing payment webhook design",
                    "targets": ["architect"],
                }
            ],
            "decisions": [{"id": "d1", "text": "Add Stripe webhook handler spec"}],
            "open_questions": [{"id": "q1", "text": "Offline payments?", "for": ["product_owner"]}],
            "blockers": [
                {"id": "b1", "severity": "high", "text": "No Prisma schema", "targets": []}
            ],
        }
        critique = parse_critique_response(data, agent, 1, model_used="test")
        assert critique.structured_concerns[0].id == "c1"
        assert critique.structured_blockers[0].text == "No Prisma schema"
        assert critique.concerns == ["Missing payment webhook design"]
        assert critique.blockers == ["No Prisma schema"]


class TestCritiqueContext:
    def test_immediate_predecessor_is_full_json(self, disagreeing_prior, monkeypatch):
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        # DevOps is index 4 — predecessor Riley (GraphQL) must be full
        context = build_critique_context(
            disagreeing_prior,
            AgentRole.DEVOPS.value,
            receiver_index=4,
        )
        payload = json.loads(context)
        assert payload[-1]["agent"] == "Riley"
        assert "_summary" not in payload[-1]
        assert context_contains_texts(context, ["GraphQL to reduce over-fetching"])

    def test_two_plus_turns_back_are_summarized_for_regular_agents(
        self, disagreeing_prior, monkeypatch
    ):
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        context = build_critique_context(
            disagreeing_prior,
            AgentRole.DEVOPS.value,
            receiver_index=4,
        )
        payload = json.loads(context)
        # Alex is 4 turns back — summarized
        assert payload[0]["_summary"] is True
        assert context_contains_concern_ids(context, ["c_po_scope"])
        # Sam is 2 turns back — summarized
        assert payload[2]["_summary"] is True

    def test_tester_receives_full_prior_critiques(self, disagreeing_prior, monkeypatch):
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        context = build_critique_context(
            disagreeing_prior,
            AgentRole.TESTER.value,
            receiver_index=8,
        )
        payload = json.loads(context)
        assert all("_summary" not in section for section in payload)
        assert context_contains_texts(
            context,
            [
                "REST endpoints with /v1 prefix",
                "GraphQL to reduce over-fetching",
            ],
        )
        assert context_contains_concern_ids(context, ["c_be_rest", "c_fe_graphql"])

    def test_reviewer_receives_full_prior_critiques(self, disagreeing_prior, monkeypatch):
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        context = build_critique_context(
            disagreeing_prior,
            AgentRole.CODE_REVIEWER.value,
            receiver_index=9,
        )
        assert context_contains_concern_ids(context, ["c_be_rest", "c_fe_graphql"])
        assert "REST endpoints" in context
        assert "GraphQL" in context

    def test_disagreement_not_flattened_for_tester_and_reviewer(
        self, disagreeing_prior, monkeypatch
    ):
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        for role in (AgentRole.TESTER.value, AgentRole.CODE_REVIEWER.value):
            context = build_critique_context(disagreeing_prior, role, receiver_index=9)
            assert "c_be_rest" in context
            assert "c_fe_graphql" in context
            assert "some concerns raised" not in context.lower()

    def test_context_respects_char_budget(self, disagreeing_prior, monkeypatch):
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        context = build_critique_context(
            disagreeing_prior,
            AgentRole.TESTER.value,
            receiver_index=8,
            max_chars=1200,
        )
        assert len(context) <= 1250
        assert context_contains_concern_ids(context, ["c_be_rest", "c_fe_graphql"])

    def test_legacy_mode_uses_prose_summary(self, disagreeing_prior, monkeypatch):
        monkeypatch.setattr(settings, "debate_structured_critiques", False)
        text = _format_other_critiques(
            disagreeing_prior,
            AgentRole.DEVOPS.value,
            AgentRole.DEVOPS.value,
            4,
        )
        assert "GraphQL" in text
        assert '"concerns"' not in text
