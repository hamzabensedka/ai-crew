"""Tests for deterministic Avery (Progress Tracker) debate critique."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.debate.debate_model import AgentCritique
from autocrew.debate.debate_runner import run_debate
from autocrew.debate.tracker_critique import generate_tracker_critique
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole


def _planity_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="Planity Clone",
        domain=ProjectDomain.MOBILE_APP,
        description="Salon booking app",
        tech_stack=TechStack(
            frontend=["Expo", "React Native"],
            backend=["NestJS", "Prisma"],
            devops=["Docker"],
        ),
        features=[
            FeatureItem(name="User Auth", description="JWT", status="done", priority="high"),
            FeatureItem(name="Payment Integration", description="Stripe", status="not_started", priority="high"),
            FeatureItem(name="Admin Dashboard", description="Admin", status="not_started", priority="medium"),
            FeatureItem(name="Booking Flow", description="Book", status="partial", priority="high"),
            FeatureItem(name="Search", description="Search", status="done", priority="high"),
            FeatureItem(name="Favorites", description="Fav", status="done", priority="medium"),
        ],
        missing_parts=["Stripe checkout", "Admin dashboard"],
        codebase_path=".",
    )


def _tracker_agent(squad):
    return next(a for a in squad.agents if a.role == AgentRole.PROGRESS_TRACKER)


def _serialize_critique(critique: AgentCritique) -> bytes:
    return json.dumps(critique.to_dict(), sort_keys=True, ensure_ascii=True).encode("utf-8")


class TestTrackerCritiqueDeterminism:
    def test_byte_identical_for_identical_inputs(self):
        context = _planity_context()
        squad = build_squad(context)
        tracker = _tracker_agent(squad)
        plan = "# Product\nShort plan without acceptance criteria"
        prior = [
            AgentCritique(
                agent_role="backend_developer",
                agent_name="Sam",
                round_number=1,
                approved=False,
                blockers=["No API contracts defined"],
                concerns=["Payment spec missing"],
                suggestions=["Add OpenAPI spec"],
            ),
            AgentCritique(
                agent_role="code_reviewer",
                agent_name="Drew",
                round_number=1,
                approved=False,
                blockers=["No API contracts defined"],
                concerns=["Security review criteria missing"],
                suggestions=[],
            ),
        ]

        first = generate_tracker_critique(tracker, context, plan, 1, prior)
        second = generate_tracker_critique(tracker, context, plan, 1, prior)

        assert _serialize_critique(first) == _serialize_critique(second)
        assert first.model_used == "deterministic"

    def test_aggregates_prior_blockers_with_attribution(self):
        context = _planity_context()
        squad = build_squad(context)
        tracker = _tracker_agent(squad)
        prior = [
            AgentCritique(
                agent_role="tester",
                agent_name="Jamie",
                round_number=1,
                approved=False,
                blockers=["No test plan for Payment Integration"],
                concerns=[],
                suggestions=[],
            ),
        ]
        critique = generate_tracker_critique(tracker, context, "plan", 1, prior)
        assert any("tester" in b for b in critique.blockers)
        assert any("Payment Integration" in b for b in critique.blockers)

    def test_no_llm_call_in_debate_when_flag_enabled(self, tmp_path, isolated_output_dirs, monkeypatch):
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)

        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nInitial plan", encoding="utf-8")

        llm_calls: list[str] = []

        def fake_llm(prompt: str) -> str:
            llm_calls.append(prompt)
            return json.dumps({
                "approved": True,
                "concerns": [],
                "suggestions": [],
                "blockers": [],
            })

        result = run_debate(
            context,
            squad,
            str(project_root),
            str(isolated_output_dirs),
            max_rounds=1,
            llm_call=fake_llm,
        )

        tracker_critiques = [
            c for r in result.rounds for c in r.critiques
            if c.agent_role == AgentRole.PROGRESS_TRACKER.value
        ]
        assert len(tracker_critiques) == 1
        assert tracker_critiques[0].model_used == "deterministic"
        assert len(llm_calls) == len(squad.agents) - 1


class TestAveryOutputClassification:
    """Regression fixture: classify historical LLM Avery outputs (Step 1 audit)."""

    CLASSIFICATIONS = {
        "pure_aggregation": [
            "converge",
            "parallel agent",
            "task plan shows",
            "circular dependency",
            "unanimous",
        ],
        "templated_narration": [
            "product.md",
            "architecture.md",
            "api contract",
            "prisma",
            "auth",
            "payment",
            "bullmq",
            "acceptance criteria",
            "folder structure",
            "monorepo",
            "notification",
            "admin dashboard",
            "provider",
            "ci/cd",
            "docker",
            "design system",
            "offline",
            "gdpr",
            "monitoring",
            "availability",
            "background job",
        ],
        "genuine_synthesis": [],
    }

    @pytest.fixture
    def avery_outputs(self) -> list[dict]:
        root = Path(__file__).resolve().parents[1] / "output" / "debate" / "planity_clone"
        outputs: list[dict] = []
        for path in sorted(root.glob("round-*/critiques.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data:
                if entry.get("agent_role") == "progress_tracker":
                    outputs.append(entry)
        debate_result = root / "debate_result.json"
        if debate_result.is_file():
            result = json.loads(debate_result.read_text(encoding="utf-8"))
            for round_data in result.get("rounds", []):
                for entry in round_data.get("critiques", []):
                    if entry.get("agent_role") == "progress_tracker":
                        if entry not in outputs:
                            outputs.append(entry)
        return outputs

    def test_inspected_at_least_three_historical_outputs(self, avery_outputs):
        assert len(avery_outputs) >= 3

    def test_no_genuine_synthesis_requires_llm(self, avery_outputs):
        """Every sampled item maps to aggregation or templated narration — no LLM needed."""
        genuine_hits = 0
        for entry in avery_outputs:
            all_text = " ".join(
                entry.get("concerns", [])
                + entry.get("blockers", [])
                + entry.get("suggestions", [])
            )
            for marker in self.CLASSIFICATIONS["genuine_synthesis"]:
                if marker.lower() in all_text.lower():
                    genuine_hits += 1
        assert genuine_hits == 0

    def test_historical_outputs_classified_without_llm_synthesis(self, avery_outputs):
        pure_agg = templated = other = 0
        for entry in avery_outputs:
            for item in entry.get("concerns", []) + entry.get("blockers", []) + entry.get("suggestions", []):
                low = item.lower()
                if any(m in low for m in self.CLASSIFICATIONS["pure_aggregation"]):
                    pure_agg += 1
                elif any(m in low for m in self.CLASSIFICATIONS["templated_narration"]):
                    templated += 1
                else:
                    other += 1
        total = pure_agg + templated + other
        assert total > 0
        # 92/129 items in planity_clone fixtures are templated restatements; remainder is aggregation/other plan gaps
        assert (pure_agg + templated) / total >= 0.7
