"""Tests for Step 7 â€” order-effect check and dev-role randomization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocrew.analyzer.project_model import (
    FeatureItem,
    ProjectContext,
    ProjectDomain,
    ProjectType,
    TechStack,
)
from autocrew.config import settings
from autocrew.debate.debate_tiers import build_debate_tiers, get_dev_tier_order
from autocrew.debate.model_diversity_eval import EvalTask
from autocrew.debate.order_effect_eval import (
    OrderComparison,
    OrderEffectReport,
    OrderRunResult,
    compare_orderings,
    make_conclusion,
    run_order_effect_eval,
    save_order_effect_report,
    format_order_effect_report,
)
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole


def _full_squad_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="Order Effect Test",
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


class TestDevOrderRandomization:
    def test_default_order_is_deterministic(self):
        squad = build_squad(_full_squad_context())
        tiers = build_debate_tiers(squad, randomize_dev_order=False)
        dev_order = get_dev_tier_order(tiers)
        # Should always be in DEV_ROLES order
        assert dev_order[0] == AgentRole.BACKEND_DEV
        assert dev_order[1] == AgentRole.FRONTEND_DEV

    def test_randomized_order_changes_with_seed(self):
        squad = build_squad(_full_squad_context())
        default_tiers = build_debate_tiers(squad, randomize_dev_order=False)
        randomized_tiers = build_debate_tiers(squad, randomize_dev_order=True, seed=42)
        default_order = get_dev_tier_order(default_tiers)
        randomized_order = get_dev_tier_order(randomized_tiers)
        assert default_order != randomized_order

    def test_same_seed_produces_same_order(self):
        squad = build_squad(_full_squad_context())
        tiers1 = build_debate_tiers(squad, randomize_dev_order=True, seed=99)
        tiers2 = build_debate_tiers(squad, randomize_dev_order=True, seed=99)
        assert get_dev_tier_order(tiers1) == get_dev_tier_order(tiers2)

    def test_different_seeds_produce_different_orders(self):
        squad = build_squad(_full_squad_context())
        # Try multiple seed pairs — with 3 dev roles, most seed pairs will differ
        found_difference = False
        for seed_a, seed_b in [(1, 2), (1, 100), (42, 99), (7, 13), (3, 77)]:
            tiers1 = build_debate_tiers(squad, randomize_dev_order=True, seed=seed_a)
            tiers2 = build_debate_tiers(squad, randomize_dev_order=True, seed=seed_b)
            if get_dev_tier_order(tiers1) != get_dev_tier_order(tiers2):
                found_difference = True
                break
        assert found_difference, "All tested seed pairs produced the same order"

    def test_po_and_architect_always_fixed(self):
        squad = build_squad(_full_squad_context())
        tiers = build_debate_tiers(squad, randomize_dev_order=True, seed=42)
        assert tiers[0][0].role == AgentRole.PRODUCT_OWNER
        assert tiers[1][0].role == AgentRole.ARCHITECT

    def test_tail_always_fixed(self):
        squad = build_squad(_full_squad_context())
        tiers = build_debate_tiers(squad, randomize_dev_order=True, seed=42)
        assert tiers[-1][0].role == AgentRole.PROGRESS_TRACKER
        assert tiers[-2][0].role == AgentRole.CODE_REVIEWER
        assert tiers[-3][0].role == AgentRole.TESTER

    def test_randomize_with_single_dev_role_no_change(self):
        """When only one dev role exists, randomization has no effect."""
        context = ProjectContext(
            project_type=ProjectType.EXISTING_CODE,
            project_name="Single Dev",
            domain=ProjectDomain.API,
            description="API only",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[
                FeatureItem(name="Auth", description="JWT", status="not_started", priority="high"),
            ],
            missing_parts=["auth"],
            codebase_path=".",
        )
        squad = build_squad(context)
        default_tiers = build_debate_tiers(squad, randomize_dev_order=False)
        randomized_tiers = build_debate_tiers(squad, randomize_dev_order=True, seed=42)
        assert get_dev_tier_order(default_tiers) == get_dev_tier_order(randomized_tiers)

    def test_config_flag_controls_randomization(self, monkeypatch):
        squad = build_squad(_full_squad_context())
        monkeypatch.setattr(settings, "debate_randomize_dev_order", True)
        tiers = build_debate_tiers(squad, seed=42)
        default_tiers = build_debate_tiers(squad, randomize_dev_order=False)
        assert get_dev_tier_order(tiers) != get_dev_tier_order(default_tiers)

    def test_explicit_param_overrides_config_flag(self, monkeypatch):
        squad = build_squad(_full_squad_context())
        monkeypatch.setattr(settings, "debate_randomize_dev_order", True)
        # Explicit False should override the config flag
        tiers = build_debate_tiers(squad, randomize_dev_order=False)
        default_tiers = build_debate_tiers(squad, randomize_dev_order=False)
        assert get_dev_tier_order(tiers) == get_dev_tier_order(default_tiers)


class TestCompareOrderings:
    def test_overlap_and_delta_calculation(self):
        default = OrderRunResult(
            task_id="t1", condition="default_order", dev_tier_order=["a", "b"],
            total_concerns=3, distinct_concern_texts=["A", "B", "C"],
            concerns_by_agent={}, debate_rounds=1,
        )
        randomized = OrderRunResult(
            task_id="t1", condition="randomized_order", dev_tier_order=["b", "a"],
            total_concerns=4, distinct_concern_texts=["B", "C", "D", "E"],
            concerns_by_agent={}, debate_rounds=1,
        )
        comp = compare_orderings(default, randomized, "t1")
        assert comp.concern_delta == 1
        assert comp.concern_overlap == 2  # "B" and "C"
        assert comp.concern_only_default == 1  # "A"
        assert comp.concern_only_randomized == 2  # "D" and "E"
        assert comp.order_changed is True

    def test_same_order_no_change(self):
        default = OrderRunResult(
            task_id="t1", condition="default_order", dev_tier_order=["a", "b"],
            total_concerns=2, distinct_concern_texts=["A", "B"],
            concerns_by_agent={}, debate_rounds=1,
        )
        randomized = OrderRunResult(
            task_id="t1", condition="randomized_order", dev_tier_order=["a", "b"],
            total_concerns=2, distinct_concern_texts=["A", "B"],
            concerns_by_agent={}, debate_rounds=1,
        )
        comp = compare_orderings(default, randomized, "t1")
        assert comp.order_changed is False
        assert comp.concern_delta == 0
        assert comp.concern_overlap == 2


class TestMakeConclusion:
    def test_order_not_meaningful_single_dev(self):
        default = OrderRunResult("t1", "default", ["a"], 2, ["A", "B"], {}, 1)
        randomized = OrderRunResult("t1", "randomized", ["a"], 2, ["A", "B"], {}, 1)
        comp = compare_orderings(default, randomized, "t1")
        conclusion = make_conclusion([comp])
        assert "not meaningful" in conclusion.lower()

    def test_order_does_not_matter(self):
        """When delta is small and overlap is high, order doesn't matter."""
        default = OrderRunResult("t1", "default", ["a", "b"], 10, [f"c{i}" for i in range(10)], {}, 1)
        randomized = OrderRunResult("t1", "randomized", ["b", "a"], 10, [f"c{i}" for i in range(10)], {}, 1)
        comp = compare_orderings(default, randomized, "t1")
        conclusion = make_conclusion([comp])
        assert "does not meaningfully matter" in conclusion.lower()

    def test_order_matters_when_large_delta(self):
        """When delta is large, order matters."""
        default = OrderRunResult("t1", "default", ["a", "b"], 2, ["A", "B"], {}, 1)
        randomized = OrderRunResult("t1", "randomized", ["b", "a"], 8, ["C", "D", "E", "F", "G", "H", "I", "J"], {}, 1)
        comp = compare_orderings(default, randomized, "t1")
        conclusion = make_conclusion([comp])
        assert "appears to matter" in conclusion.lower()

    def test_no_concerns_inconclusive(self):
        default = OrderRunResult("t1", "default", ["a", "b"], 0, [], {}, 1)
        randomized = OrderRunResult("t1", "randomized", ["b", "a"], 0, [], {}, 1)
        comp = compare_orderings(default, randomized, "t1")
        conclusion = make_conclusion([comp])
        assert "inconclusive" in conclusion.lower()


class TestRunOrderEffectEval:
    def test_eval_runs_both_orderings_and_produces_report(
        self, tmp_path, isolated_output_dirs, monkeypatch
    ):
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        monkeypatch.setattr(settings, "debate_early_exit", False)
        monkeypatch.setattr(settings, "debate_parallel_tiers", False)

        context = _full_squad_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nPlan", encoding="utf-8")

        def fake_llm(prompt: str) -> str:
            return json.dumps({
                "approved": True,
                "concerns": [
                    {"id": "c1", "severity": "high", "text": "Missing API design", "targets": []}
                ],
                "decisions": [],
                "open_questions": [],
                "blockers": [],
            })

        task = EvalTask(
            task_id="test_task",
            context=context,
            squad=squad,
            project_root=str(project_root),
            max_rounds=1,
        )

        report = run_order_effect_eval(
            [task],
            llm_call=fake_llm,
            metrics_dir=str(isolated_output_dirs / "metrics"),
            seed=42,
        )

        assert report.task_count == 1
        assert len(report.comparisons) == 1
        assert report.comparisons[0].default.condition == "default_order"
        assert report.comparisons[0].randomized.condition == "randomized_order"
        assert "Conclusion" in report.conclusion


class TestSaveAndFormatReport:
    def test_save_report_writes_json_and_md(self, tmp_path):
        default = OrderRunResult("t1", "default", ["a", "b"], 3, ["A", "B", "C"], {"Sam": 1}, 1)
        randomized = OrderRunResult("t1", "randomized", ["b", "a"], 3, ["A", "B", "C"], {"Sam": 1}, 1)
        comp = compare_orderings(default, randomized, "t1")
        report = OrderEffectReport(
            timestamp="2026-01-01T00:00:00+00:00",
            task_count=1,
            comparisons=[comp],
            conclusion="Test conclusion",
        )
        output = tmp_path / "report"
        md_path = save_order_effect_report(report, str(output))
        assert Path(md_path).is_file()
        assert Path(str(output) + ".json").is_file()
        md_content = Path(md_path).read_text(encoding="utf-8")
        assert "Order-Effect Experiment Report" in md_content
        assert "Test conclusion" in md_content

    def test_format_report_contains_all_sections(self):
        default = OrderRunResult("t1", "default", ["a", "b"], 3, ["A", "B", "C"], {}, 1)
        randomized = OrderRunResult("t1", "randomized", ["b", "a"], 3, ["A", "B", "C"], {}, 1)
        comp = compare_orderings(default, randomized, "t1")
        report = OrderEffectReport(
            timestamp="2026-01-01T00:00:00+00:00",
            task_count=1,
            comparisons=[comp],
            conclusion="Order does not matter.",
        )
        text = format_order_effect_report(report)
        assert "Per-task comparison" in text
        assert "Dev-tier orderings" in text
        assert "Order does not matter" in text