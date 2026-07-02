"""Step 6 — Model diversity experiment harness.

Runs debate twice per task (all-Kimi baseline vs cross-model Drew+Jordan) and
compares concern counts, false-positive estimates, and token/cost delta using
Step 3's structured schema and Step 2's metrics instrumentation.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from autocrew.analyzer.project_model import ProjectContext
from autocrew.debate.debate_model import AgentCritique, DebateResult
from autocrew.debate.model_router import DualModelRouter
from autocrew.squad.squad_model import Squad


@dataclass
class EvalTask:
    """A single task to evaluate (context + squad + project root)."""

    task_id: str
    context: ProjectContext
    squad: Squad
    project_root: str
    max_rounds: int = 2


@dataclass
class EvalRunResult:
    """Captured outcome of one debate run for one condition."""

    task_id: str
    condition: str
    total_concerns: int
    distinct_concern_texts: list[str]
    concerns_by_agent: dict[str, int]
    total_tokens: int
    total_latency_ms: float
    debate_rounds: int
    confirmed_real: int = 0
    false_positives: int = 0
    unverified: int = 0
    models_used: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionComparison:
    """Per-task comparison between baseline and cross-model conditions."""

    task_id: str
    baseline: EvalRunResult
    cross_model: EvalRunResult
    concern_delta: int
    token_delta: int
    latency_delta_ms: float
    confirmed_real_delta: int
    false_positive_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    """Aggregate comparison report across all tasks."""

    timestamp: str
    task_count: int
    baseline: EvalRunResult
    cross_model: EvalRunResult
    comparisons: list[ConditionComparison]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_count": self.task_count,
            "baseline": self.baseline.to_dict(),
            "cross_model": self.cross_model.to_dict(),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "recommendation": self.recommendation,
        }


def extract_concerns_from_debate(
    result: DebateResult,
) -> tuple[int, list[str], dict[str, int]]:
    """Extract distinct concern texts and per-agent counts from a debate result."""
    all_texts: list[str] = []
    by_agent: dict[str, int] = {}

    for round_data in result.rounds:
        for critique in round_data.critiques:
            agent_key = critique.agent_name or critique.agent_role
            count = 0
            for concern in critique.structured_concerns:
                if concern.text.strip():
                    all_texts.append(concern.text.strip())
                    count += 1
            # Fallback to legacy concerns list if structured is empty
            if not critique.structured_concerns:
                for text in critique.concerns:
                    if text.strip():
                        all_texts.append(text.strip())
                        count += 1
            if count:
                by_agent[agent_key] = by_agent.get(agent_key, 0) + count

    # Dedupe by normalized text
    seen: set[str] = set()
    distinct: list[str] = []
    for text in all_texts:
        key = text.lower().strip()
        if key not in seen:
            seen.add(key)
            distinct.append(text)

    return len(distinct), distinct, by_agent


def extract_tokens_from_debate(
    result: DebateResult,
    metrics_dir: str | None = None,
) -> tuple[int, float]:
    """Extract total tokens and latency from metrics store or debate result.

    When metrics_dir is provided, queries the SQLite store for the most recent
    debate session. Otherwise estimates from critique text length.
    """
    if metrics_dir:
        from autocrew.metrics.store import MetricsStore

        store = MetricsStore(metrics_dir)
        calls = store.fetch_agent_calls("debate")
        if calls:
            total_tokens = sum(
                (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
                for c in calls
            )
            total_latency = sum(c.get("latency_ms") or 0.0 for c in calls)
            return total_tokens, total_latency

    # Fallback: estimate from critique text
    from autocrew.metrics.tokens import estimate_tokens

    total_tokens = 0
    for round_data in result.rounds:
        for critique in round_data.critiques:
            text = " ".join(
                critique.concerns + critique.suggestions + critique.blockers
            )
            total_tokens += estimate_tokens(text) + 500  # prompt overhead estimate
    return total_tokens, 0.0


def label_concerns(
    distinct_concerns: list[str],
    known_real_issues: list[str] | None = None,
) -> tuple[int, int, int]:
    """Label concerns as confirmed-real, false-positive, or unverified.

    Uses simple keyword matching against known_real_issues. When no issue list
    is provided, all concerns are marked unverified (human review needed).
    """
    if not known_real_issues:
        return 0, 0, len(distinct_concerns)

    real = 0
    false_pos = 0
    unverified = 0
    real_lower = [r.lower().strip() for r in known_real_issues]

    for concern in distinct_concerns:
        concern_lower = concern.lower()
        matched = any(
            real_kw in concern_lower or concern_lower in real_kw
            for real_kw in real_lower
        )
        if matched:
            real += 1
        else:
            # Heuristic: concerns with severity markers or specific technical terms
            # are more likely real; generic concerns are potential false positives
            generic_markers = ["consider", "might", "could", "perhaps", "maybe"]
            if any(m in concern_lower for m in generic_markers):
                false_pos += 1
            else:
                unverified += 1

    return real, false_pos, unverified


def run_eval_condition(
    task: EvalTask,
    *,
    condition: str,
    dual_router: DualModelRouter | None,
    llm_call: Callable[[str], str] | None,
    metrics_dir: str | None = None,
    known_real_issues: list[str] | None = None,
) -> EvalRunResult:
    """Run debate for one task under one model condition."""
    from autocrew.debate.debate_runner import run_debate

    result = run_debate(
        task.context,
        task.squad,
        task.project_root,
        str(Path(metrics_dir or "./output") / "eval"),
        max_rounds=task.max_rounds,
        llm_call=llm_call,
        dual_router=dual_router,
    )

    total_concerns, distinct_texts, by_agent = extract_concerns_from_debate(result)
    total_tokens, total_latency = extract_tokens_from_debate(result, metrics_dir)
    confirmed, false_pos, unverified = label_concerns(distinct_texts, known_real_issues)

    models_used = result.models_used if result.models_used else {}

    return EvalRunResult(
        task_id=task.task_id,
        condition=condition,
        total_concerns=total_concerns,
        distinct_concern_texts=distinct_texts,
        concerns_by_agent=by_agent,
        total_tokens=total_tokens,
        total_latency_ms=total_latency,
        debate_rounds=len(result.rounds),
        confirmed_real=confirmed,
        false_positives=false_pos,
        unverified=unverified,
        models_used=models_used,
    )


def compare_conditions(
    baseline: EvalRunResult,
    cross_model: EvalRunResult,
    task_id: str,
) -> ConditionComparison:
    """Compare baseline vs cross-model results for one task."""
    return ConditionComparison(
        task_id=task_id,
        baseline=baseline,
        cross_model=cross_model,
        concern_delta=cross_model.total_concerns - baseline.total_concerns,
        token_delta=cross_model.total_tokens - baseline.total_tokens,
        latency_delta_ms=cross_model.total_latency_ms - baseline.total_latency_ms,
        confirmed_real_delta=cross_model.confirmed_real - baseline.confirmed_real,
        false_positive_delta=cross_model.false_positives - baseline.false_positives,
    )


def aggregate_results(
    results: list[EvalRunResult],
) -> EvalRunResult:
    """Aggregate a list of per-task results into a single summary."""
    if not results:
        return EvalRunResult(
            task_id="aggregate",
            condition="",
            total_concerns=0,
            distinct_concern_texts=[],
            concerns_by_agent={},
            total_tokens=0,
            total_latency_ms=0.0,
            debate_rounds=0,
        )

    all_texts: list[str] = []
    by_agent: dict[str, int] = {}
    total_tokens = 0
    total_latency = 0.0
    total_rounds = 0
    confirmed = 0
    false_pos = 0
    unverified = 0

    for r in results:
        all_texts.extend(r.distinct_concern_texts)
        for agent, count in r.concerns_by_agent.items():
            by_agent[agent] = by_agent.get(agent, 0) + count
        total_tokens += r.total_tokens
        total_latency += r.total_latency_ms
        total_rounds += r.debate_rounds
        confirmed += r.confirmed_real
        false_pos += r.false_positives
        unverified += r.unverified

    # Dedupe across tasks
    seen: set[str] = set()
    distinct: list[str] = []
    for text in all_texts:
        key = text.lower().strip()
        if key not in seen:
            seen.add(key)
            distinct.append(text)

    return EvalRunResult(
        task_id="aggregate",
        condition=results[0].condition,
        total_concerns=len(distinct),
        distinct_concern_texts=distinct,
        concerns_by_agent=by_agent,
        total_tokens=total_tokens,
        total_latency_ms=total_latency,
        debate_rounds=total_rounds,
        confirmed_real=confirmed,
        false_positives=false_pos,
        unverified=unverified,
    )


def make_recommendation(
    baseline: EvalRunResult,
    cross_model: EvalRunResult,
    comparisons: list[ConditionComparison],
) -> str:
    """Generate a written recommendation based on the comparison data."""
    avg_concern_delta = statistics.mean([c.concern_delta for c in comparisons]) if comparisons else 0
    avg_token_delta = statistics.mean([c.token_delta for c in comparisons]) if comparisons else 0
    avg_confirmed_delta = statistics.mean([c.confirmed_real_delta for c in comparisons]) if comparisons else 0
    avg_false_pos_delta = statistics.mean([c.false_positive_delta for c in comparisons]) if comparisons else 0

    lines = [
        f"## Model Diversity Recommendation",
        f"",
        f"Tasks evaluated: {len(comparisons)}",
        f"Average concern delta: {avg_concern_delta:+.1f} (positive = cross-model found more)",
        f"Average confirmed-real delta: {avg_confirmed_delta:+.1f}",
        f"Average false-positive delta: {avg_false_pos_delta:+.1f}",
        f"Average token delta: {avg_token_delta:+,.0f} (positive = cross-model cost more)",
        f"",
    ]

    if avg_confirmed_delta > 0 and avg_false_pos_delta <= 0:
        lines.append("**Recommendation: Adopt the model split permanently.**")
        lines.append(
            "Cross-model critique catches more real issues without increasing false positives. "
            "The token cost delta is justified by the quality improvement."
        )
    elif avg_confirmed_delta > 0 and avg_false_pos_delta > 0:
        lines.append("**Recommendation: Expand cautiously.**")
        lines.append(
            "Cross-model critique catches more real issues but also raises more false positives. "
            "Consider expanding to more seats only if the false-positive review burden is acceptable."
        )
    elif avg_confirmed_delta <= 0 and avg_token_delta > 0:
        lines.append("**Recommendation: Revert to single-model.**")
        lines.append(
            "Cross-model critique did not catch more real issues and cost more tokens. "
            "The model split is not worth it for this task set."
        )
    else:
        lines.append("**Recommendation: Inconclusive — needs more data.**")
        lines.append(
            "No clear quality or cost advantage. Run with more tasks or a different model pair."
        )

    return "\n".join(lines)


def run_model_diversity_eval(
    tasks: list[EvalTask],
    *,
    baseline_llm_call: Callable[[str], str],
    cross_model_router: DualModelRouter | None,
    cross_model_llm_call: Callable[[str], str] | None = None,
    metrics_dir: str | None = None,
    known_real_issues: dict[str, list[str]] | None = None,
) -> EvalReport:
    """Run the full model diversity experiment across all tasks.

    Args:
        tasks: List of tasks to evaluate.
        baseline_llm_call: LLM callable for the all-Kimi baseline.
        cross_model_router: DualModelRouter for the cross-model condition.
            If None, cross_model_llm_call is used directly.
        cross_model_llm_call: Fallback LLM callable when no router is available.
        metrics_dir: Directory for metrics persistence.
        known_real_issues: Map of task_id -> list of known real issue keywords
            for labeling concerns as confirmed-real vs false-positive.
    """
    baseline_results: list[EvalRunResult] = []
    cross_results: list[EvalRunResult] = []
    comparisons: list[ConditionComparison] = []

    for task in tasks:
        issues = (known_real_issues or {}).get(task.task_id)

        baseline = run_eval_condition(
            task,
            condition="baseline_all_kimi",
            dual_router=None,
            llm_call=baseline_llm_call,
            metrics_dir=metrics_dir,
            known_real_issues=issues,
        )
        baseline_results.append(baseline)

        cross = run_eval_condition(
            task,
            condition="cross_model_drew_jordan",
            dual_router=cross_model_router,
            llm_call=cross_model_llm_call,
            metrics_dir=metrics_dir,
            known_real_issues=issues,
        )
        cross_results.append(cross)

        comparisons.append(compare_conditions(baseline, cross, task.task_id))

    agg_baseline = aggregate_results(baseline_results)
    agg_cross = aggregate_results(cross_results)
    recommendation = make_recommendation(agg_baseline, agg_cross, comparisons)

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_count=len(tasks),
        baseline=agg_baseline,
        cross_model=agg_cross,
        comparisons=comparisons,
        recommendation=recommendation,
    )


def format_eval_report(report: EvalReport) -> str:
    """Render the eval report as markdown."""
    lines = [
        "# Model Diversity Experiment Report",
        f"**Generated:** {report.timestamp}",
        f"**Tasks evaluated:** {report.task_count}",
        "",
        "## Baseline (all-Kimi)",
        f"- Total distinct concerns: {report.baseline.total_concerns}",
        f"- Confirmed real: {report.baseline.confirmed_real}",
        f"- False positives: {report.baseline.false_positives}",
        f"- Unverified: {report.baseline.unverified}",
        f"- Total tokens: {report.baseline.total_tokens:,}",
        f"- Total latency: {report.baseline.total_latency_ms:,.0f} ms",
        "",
        "## Cross-model (Drew+Jordan on different model)",
        f"- Total distinct concerns: {report.cross_model.total_concerns}",
        f"- Confirmed real: {report.cross_model.confirmed_real}",
        f"- False positives: {report.cross_model.false_positives}",
        f"- Unverified: {report.cross_model.unverified}",
        f"- Total tokens: {report.cross_model.total_tokens:,}",
        f"- Total latency: {report.cross_model.total_latency_ms:,.0f} ms",
        "",
        "## Per-task comparison",
        "| Task | Concern Δ | Token Δ | Confirmed Δ | False-pos Δ |",
        "|------|-----------|---------|------------|-------------|",
    ]
    for c in report.comparisons:
        lines.append(
            f"| {c.task_id} | {c.concern_delta:+d} | {c.token_delta:+,d} | "
            f"{c.confirmed_real_delta:+d} | {c.false_positive_delta:+d} |"
        )
    lines.append("")
    lines.append(report.recommendation)
    return "\n".join(lines)


def save_eval_report(report: EvalReport, output_path: str) -> str:
    """Save report as JSON and markdown, return the markdown path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(format_eval_report(report), encoding="utf-8")
    return str(md_path)