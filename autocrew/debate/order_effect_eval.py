"""Step 7 — Order-effect check harness.

Runs debate on the same tasks with default dev-role order vs randomized order
within the dev-adjacent tier, and compares concern counts and content to
determine whether order matters for that subset.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from autocrew.debate.debate_model import DebateResult
from autocrew.debate.debate_tiers import get_dev_tier_order
from autocrew.debate.model_diversity_eval import EvalTask, extract_concerns_from_debate
from autocrew.squad.squad_model import AgentRole


@dataclass
class OrderRunResult:
    """Captured outcome of one debate run under one ordering condition."""

    task_id: str
    condition: str  # "default_order" or "randomized_order"
    dev_tier_order: list[str]
    total_concerns: int
    distinct_concern_texts: list[str]
    concerns_by_agent: dict[str, int]
    debate_rounds: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderComparison:
    """Per-task comparison between default and randomized ordering."""

    task_id: str
    default: OrderRunResult
    randomized: OrderRunResult
    concern_delta: int
    concern_overlap: int
    concern_only_default: int
    concern_only_randomized: int
    order_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderEffectReport:
    """Aggregate order-effect report across all tasks."""

    timestamp: str
    task_count: int
    comparisons: list[OrderComparison]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_count": self.task_count,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "conclusion": self.conclusion,
        }


def _run_order_condition(
    task: EvalTask,
    *,
    condition: str,
    randomize_dev_order: bool,
    llm_call: Callable[[str], str] | None,
    seed: int | None = None,
    metrics_dir: str | None = None,
) -> OrderRunResult:
    """Run debate for one task under one ordering condition."""
    from autocrew.config import settings
    from autocrew.debate.debate_runner import run_debate

    # Temporarily set the config flag for this run
    original_flag = settings.debate_randomize_dev_order
    settings.debate_randomize_dev_order = randomize_dev_order

    try:
        result = run_debate(
            task.context,
            task.squad,
            task.project_root,
            str(Path(metrics_dir or "./output") / "order_eval"),
            max_rounds=task.max_rounds,
            llm_call=llm_call,
        )
    finally:
        settings.debate_randomize_dev_order = original_flag

    from autocrew.debate.debate_tiers import build_debate_tiers

    tiers = build_debate_tiers(task.squad, randomize_dev_order=randomize_dev_order, seed=seed)
    dev_order = [r.value for r in get_dev_tier_order(tiers)]

    total_concerns, distinct_texts, by_agent = extract_concerns_from_debate(result)

    return OrderRunResult(
        task_id=task.task_id,
        condition=condition,
        dev_tier_order=dev_order,
        total_concerns=total_concerns,
        distinct_concern_texts=distinct_texts,
        concerns_by_agent=by_agent,
        debate_rounds=len(result.rounds),
    )


def compare_orderings(
    default: OrderRunResult,
    randomized: OrderRunResult,
    task_id: str,
) -> OrderComparison:
    """Compare default vs randomized ordering for one task."""
    default_set = {t.lower().strip() for t in default.distinct_concern_texts}
    randomized_set = {t.lower().strip() for t in randomized.distinct_concern_texts}

    overlap = len(default_set & randomized_set)
    only_default = len(default_set - randomized_set)
    only_randomized = len(randomized_set - default_set)

    return OrderComparison(
        task_id=task_id,
        default=default,
        randomized=randomized,
        concern_delta=randomized.total_concerns - default.total_concerns,
        concern_overlap=overlap,
        concern_only_default=only_default,
        concern_only_randomized=only_randomized,
        order_changed=default.dev_tier_order != randomized.dev_tier_order,
    )


def make_conclusion(comparisons: list[OrderComparison]) -> str:
    """Generate a written conclusion about whether order matters."""
    if not comparisons:
        return "No tasks evaluated — inconclusive."

    avg_concern_delta = statistics.mean([c.concern_delta for c in comparisons])
    avg_only_default = statistics.mean([c.concern_only_default for c in comparisons])
    avg_only_randomized = statistics.mean([c.concern_only_randomized for c in comparisons])
    order_changed_count = sum(1 for c in comparisons if c.order_changed)

    lines = [
        "## Order-Effect Conclusion",
        "",
        f"Tasks evaluated: {len(comparisons)}",
        f"Tasks where order actually changed: {order_changed_count}",
        f"Average concern delta (randomized - default): {avg_concern_delta:+.1f}",
        f"Average concerns only in default: {avg_only_default:.1f}",
        f"Average concerns only in randomized: {avg_only_randomized:.1f}",
        "",
    ]

    # If order didn't actually change (e.g., only 1 dev role), we can't conclude
    if order_changed_count == 0:
        lines.append("**Conclusion: Order test was not meaningful.**")
        lines.append(
            "The dev-adjacent tier had only one agent, so randomization had no effect. "
            "Test with a squad that has 2+ dev roles."
        )
        return "\n".join(lines)

    # If the delta is small relative to the total, order doesn't matter much
    total_concerns = sum(c.default.total_concerns for c in comparisons)
    if total_concerns == 0:
        lines.append("**Conclusion: Inconclusive — no concerns raised in any run.**")
        return "\n".join(lines)

    delta_ratio = abs(avg_concern_delta) / max(total_concerns / len(comparisons), 1)
    unique_concern_ratio = (avg_only_default + avg_only_randomized) / max(
        total_concerns / len(comparisons), 1
    )

    if delta_ratio < 0.15 and unique_concern_ratio < 0.3:
        lines.append("**Conclusion: Order does not meaningfully matter for the dev-adjacent subset.**")
        lines.append(
            "Randomizing the order of Backend/Frontend/DevOps/Data/AI within their tier "
            "does not significantly change the number or content of concerns raised. "
            "PO and Architect should remain fixed at the front (they are deliberately ordered). "
            "The current default order can be kept, with a code comment noting it was tested."
        )
    else:
        lines.append("**Conclusion: Order appears to matter for the dev-adjacent subset.**")
        lines.append(
            f"Randomizing order changed concern counts by {avg_concern_delta:+.1f} on average "
            f"and introduced {avg_only_default + avg_only_randomized:.1f} unique concerns per task. "
            "Consider keeping PO/Architect fixed at the front but randomizing or reconsidering "
            "the rest of the dev-adjacent ordering."
        )

    return "\n".join(lines)


def run_order_effect_eval(
    tasks: list[EvalTask],
    *,
    llm_call: Callable[[str], str] | None,
    metrics_dir: str | None = None,
    seed: int | None = 42,
) -> OrderEffectReport:
    """Run the full order-effect experiment across all tasks.

    Args:
        tasks: List of tasks to evaluate.
        llm_call: LLM callable for debate (same model for both conditions).
        metrics_dir: Directory for metrics persistence.
        seed: Random seed for reproducible shuffling.
    """
    comparisons: list[OrderComparison] = []

    for task in tasks:
        default = _run_order_condition(
            task,
            condition="default_order",
            randomize_dev_order=False,
            llm_call=llm_call,
            metrics_dir=metrics_dir,
        )

        randomized = _run_order_condition(
            task,
            condition="randomized_order",
            randomize_dev_order=True,
            llm_call=llm_call,
            seed=seed,
            metrics_dir=metrics_dir,
        )

        comparisons.append(compare_orderings(default, randomized, task.task_id))

    conclusion = make_conclusion(comparisons)

    return OrderEffectReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_count=len(tasks),
        comparisons=comparisons,
        conclusion=conclusion,
    )


def format_order_effect_report(report: OrderEffectReport) -> str:
    """Render the order-effect report as markdown."""
    lines = [
        "# Order-Effect Experiment Report",
        f"**Generated:** {report.timestamp}",
        f"**Tasks evaluated:** {report.task_count}",
        "",
        "## Per-task comparison",
        "| Task | Order changed | Default concerns | Randomized concerns | Δ | Overlap | Only default | Only randomized |",
        "|------|---------------|-----------------|--------------------|---|---------|-------------|-----------------|",
    ]
    for c in report.comparisons:
        lines.append(
            f"| {c.task_id} | {'yes' if c.order_changed else 'no'} | "
            f"{c.default.total_concerns} | {c.randomized.total_concerns} | "
            f"{c.concern_delta:+d} | {c.concern_overlap} | "
            f"{c.concern_only_default} | {c.concern_only_randomized} |"
        )
    lines.append("")
    lines.append("## Dev-tier orderings")
    for c in report.comparisons:
        lines.append(f"### {c.task_id}")
        lines.append(f"- Default: {' → '.join(c.default.dev_tier_order)}")
        lines.append(f"- Randomized: {' → '.join(c.randomized.dev_tier_order)}")
        lines.append("")
    lines.append(report.conclusion)
    return "\n".join(lines)


def save_order_effect_report(report: OrderEffectReport, output_path: str) -> str:
    """Save report as JSON and markdown, return the markdown path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(format_order_effect_report(report), encoding="utf-8")
    return str(md_path)