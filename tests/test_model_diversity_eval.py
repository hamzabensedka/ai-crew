"""Tests for Step 6 — model diversity experiment harness."""

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
from autocrew.debate.critique_types import StructuredConcern
from autocrew.debate.debate_model import AgentCritique, DebateResult, DebateRound
from autocrew.debate.model_diversity_eval import (
    EvalReport,
    EvalRunResult,
    EvalTask,
    aggregate_results,
    compare_conditions,
    extract_concerns_from_debate,
    format_eval_report,
    label_concerns,
    make_recommendation,
    run_model_diversity_eval,
    save_eval_report,
)
from autocrew.debate.debate_runner import run_debate
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole


def _planity_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="Diversity Eval Test",
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


def _make_debate_result(
    project_name: str = "Test",
    concerns_by_agent: dict[str, list[str]] | None = None,
) -> DebateResult:
    """Build a synthetic DebateResult with structured concerns for unit tests."""
    critiques: list[AgentCritique] = []
    concerns_map = concerns_by_agent or {
        "Sam": ["Missing payment webhook design", "No API rate limiting"],
        "Drew": ["Security review criteria missing"],
    }

    for agent_name, texts in concerns_map.items():
        structured = [
            StructuredConcern(id=f"c_{i}", severity="high", text=t, targets=[])
            for i, t in enumerate(texts, 1)
        ]
        critique = AgentCritique(
            agent_role="backend_developer",
            agent_name=agent_name,
            round_number=1,
            approved=False,
            structured_concerns=structured,
        )
        critique.concerns = texts
        critiques.append(critique)

    round_data = DebateRound(
        round_number=1,
        critiques=critiques,
        revised_plan_excerpt="plan excerpt",
        all_approved=False,
        total_blockers=0,
    )
    return DebateResult(
        project_name=project_name,
        timestamp="2026-01-01T00:00:00+00:00",
        rounds=[round_data],
        consensus_reached=True,
        final_plan_path="/tmp/plan.md",
        debate_dir="/tmp/debate",
    )


class TestExtractConcerns:
    def test_extracts_distinct_concerns(self):
        result = _make_debate_result(
            concerns_by_agent={
                "Sam": ["Missing payment webhook", "No rate limiting"],
                "Drew": ["Security review missing", "Missing payment webhook"],  # duplicate
            }
        )
        count, texts, by_agent = extract_concerns_from_debate(result)
        assert count == 3  # "Missing payment webhook" deduped
        assert "Missing payment webhook" in texts
        assert "No rate limiting" in texts
        assert "Security review missing" in texts
        assert by_agent["Sam"] == 2
        assert by_agent["Drew"] == 2  # count before dedupe

    def test_falls_back_to_legacy_concerns(self):
        critique = AgentCritique(
            agent_role="tester",
            agent_name="Jamie",
            round_number=1,
            approved=False,
            concerns=["No test plan", "Missing E2E tests"],
        )
        round_data = DebateRound(
            round_number=1,
            critiques=[critique],
            revised_plan_excerpt="",
            all_approved=False,
            total_blockers=0,
        )
        result = DebateResult(
            project_name="Test",
            timestamp="2026-01-01T00:00:00+00:00",
            rounds=[round_data],
            consensus_reached=True,
            final_plan_path="",
            debate_dir="",
        )
        count, texts, by_agent = extract_concerns_from_debate(result)
        assert count == 2
        assert "No test plan" in texts


class TestLabelConcerns:
    def test_no_known_issues_all_unverified(self):
        confirmed, false_pos, unverified = label_concerns(
            ["Missing API", "No tests"], known_real_issues=None
        )
        assert confirmed == 0
        assert false_pos == 0
        assert unverified == 2

    def test_matches_known_real_issues(self):
        confirmed, false_pos, unverified = label_concerns(
            ["Missing payment webhook design", "No API rate limiting", "Consider adding docs"],
            known_real_issues=["payment webhook", "rate limiting"],
        )
        assert confirmed == 2
        assert false_pos == 1  # "Consider adding docs" has generic marker
        assert unverified == 0

    def test_generic_markers_are_false_positives(self):
        confirmed, false_pos, unverified = label_concerns(
            ["Maybe add caching", "Could improve UX"],
            known_real_issues=["caching"],
        )
        assert confirmed == 1
        assert false_pos == 1  # "Could improve UX"
        assert unverified == 0


class TestCompareConditions:
    def test_delta_calculation(self):
        baseline = EvalRunResult(
            task_id="t1",
            condition="baseline",
            total_concerns=5,
            distinct_concern_texts=[],
            concerns_by_agent={},
            total_tokens=10000,
            total_latency_ms=5000.0,
            debate_rounds=2,
            confirmed_real=3,
            false_positives=1,
        )
        cross = EvalRunResult(
            task_id="t1",
            condition="cross",
            total_concerns=7,
            distinct_concern_texts=[],
            concerns_by_agent={},
            total_tokens=12000,
            total_latency_ms=6000.0,
            debate_rounds=2,
            confirmed_real=5,
            false_positives=1,
        )
        comp = compare_conditions(baseline, cross, "t1")
        assert comp.concern_delta == 2
        assert comp.token_delta == 2000
        assert comp.latency_delta_ms == 1000.0
        assert comp.confirmed_real_delta == 2
        assert comp.false_positive_delta == 0


class TestAggregateResults:
    def test_aggregates_across_tasks(self):
        results = [
            EvalRunResult(
                task_id="t1", condition="baseline", total_concerns=3,
                distinct_concern_texts=["A", "B", "C"], concerns_by_agent={"Sam": 3},
                total_tokens=5000, total_latency_ms=3000.0, debate_rounds=2,
            ),
            EvalRunResult(
                task_id="t2", condition="baseline", total_concerns=2,
                distinct_concern_texts=["B", "D"], concerns_by_agent={"Drew": 2},
                total_tokens=4000, total_latency_ms=2000.0, debate_rounds=1,
            ),
        ]
        agg = aggregate_results(results)
        assert agg.total_concerns == 4  # A, B, C, D — "B" deduped across tasks
        assert agg.total_tokens == 9000
        assert agg.total_latency_ms == 5000.0
        assert agg.debate_rounds == 3
        assert agg.concerns_by_agent == {"Sam": 3, "Drew": 2}

    def test_empty_list_returns_zeros(self):
        agg = aggregate_results([])
        assert agg.total_concerns == 0
        assert agg.total_tokens == 0


class TestMakeRecommendation:
    def test_adopt_when_more_confirmed_no_extra_false_pos(self):
        baseline = EvalRunResult("agg", "baseline", 5, [], {}, 10000, 5000, 2, confirmed_real=3, false_positives=1)
        cross = EvalRunResult("agg", "cross", 7, [], {}, 12000, 6000, 2, confirmed_real=5, false_positives=1)
        comp = compare_conditions(baseline, cross, "t1")
        rec = make_recommendation(baseline, cross, [comp])
        assert "Adopt" in rec

    def test_revert_when_no_improvement_and_higher_cost(self):
        baseline = EvalRunResult("agg", "baseline", 5, [], {}, 10000, 5000, 2, confirmed_real=3, false_positives=1)
        cross = EvalRunResult("agg", "cross", 5, [], {}, 15000, 6000, 2, confirmed_real=3, false_positives=1)
        comp = compare_conditions(baseline, cross, "t1")
        rec = make_recommendation(baseline, cross, [comp])
        assert "Revert" in rec

    def test_expand_cautiously_when_more_confirmed_but_also_false_pos(self):
        baseline = EvalRunResult("agg", "baseline", 5, [], {}, 10000, 5000, 2, confirmed_real=3, false_positives=1)
        cross = EvalRunResult("agg", "cross", 8, [], {}, 12000, 6000, 2, confirmed_real=5, false_positives=3)
        comp = compare_conditions(baseline, cross, "t1")
        rec = make_recommendation(baseline, cross, [comp])
        assert "cautiously" in rec.lower()


class TestRunModelDiversityEval:
    def test_eval_runs_both_conditions_and_produces_report(
        self, tmp_path, isolated_output_dirs, monkeypatch
    ):
        monkeypatch.setattr(settings, "debate_deterministic_tracker", True)
        monkeypatch.setattr(settings, "debate_structured_critiques", True)
        monkeypatch.setattr(settings, "debate_early_exit", False)
        monkeypatch.setattr(settings, "debate_parallel_tiers", False)

        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nPlan", encoding="utf-8")

        call_state = {"n": 0}

        def baseline_llm(prompt: str) -> str:
            call_state["n"] += 1
            return json.dumps({
                "approved": True,
                "concerns": [
                    {"id": "c1", "severity": "high", "text": "Baseline concern: missing API", "targets": []}
                ],
                "decisions": [],
                "open_questions": [],
                "blockers": [],
            })

        def cross_llm(prompt: str) -> str:
            call_state["n"] += 1
            return json.dumps({
                "approved": True,
                "concerns": [
                    {"id": "c1", "severity": "high", "text": "Cross concern: missing API", "targets": []},
                    {"id": "c2", "severity": "medium", "text": "Cross concern: no rate limiting", "targets": []},
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

        report = run_model_diversity_eval(
            [task],
            baseline_llm_call=baseline_llm,
            cross_model_router=None,
            cross_model_llm_call=cross_llm,
            metrics_dir=str(isolated_output_dirs / "metrics"),
        )

        assert report.task_count == 1
        assert len(report.comparisons) == 1
        assert report.baseline.condition == "baseline_all_kimi"
        assert report.cross_model.condition == "cross_model_drew_jordan"
        # Cross-model found more concerns
        assert report.comparisons[0].concern_delta > 0
        assert "Recommendation" in report.recommendation


class TestSaveAndFormatReport:
    def test_save_report_writes_json_and_md(self, tmp_path):
        baseline = EvalRunResult("agg", "baseline", 5, ["c1"], {"Sam": 1}, 10000, 5000, 2)
        cross = EvalRunResult("agg", "cross", 7, ["c1", "c2"], {"Drew": 2}, 12000, 6000, 2)
        comp = compare_conditions(baseline, cross, "t1")
        report = EvalReport(
            timestamp="2026-01-01T00:00:00+00:00",
            task_count=1,
            baseline=baseline,
            cross_model=cross,
            comparisons=[comp],
            recommendation="Test recommendation",
        )
        output = tmp_path / "report"
        md_path = save_eval_report(report, str(output))
        assert Path(md_path).is_file()
        assert Path(str(output) + ".json").is_file()
        md_content = Path(md_path).read_text(encoding="utf-8")
        assert "Model Diversity Experiment Report" in md_content
        assert "Test recommendation" in md_content

    def test_format_report_contains_all_sections(self):
        baseline = EvalRunResult("agg", "baseline", 5, [], {}, 10000, 5000, 2)
        cross = EvalRunResult("agg", "cross", 7, [], {}, 12000, 6000, 2)
        comp = compare_conditions(baseline, cross, "t1")
        report = EvalReport(
            timestamp="2026-01-01T00:00:00+00:00",
            task_count=1,
            baseline=baseline,
            cross_model=cross,
            comparisons=[comp],
            recommendation="Adopt the split.",
        )
        text = format_eval_report(report)
        assert "Baseline (all-Kimi)" in text
        assert "Cross-model" in text
        assert "Per-task comparison" in text
        assert "Adopt the split" in text