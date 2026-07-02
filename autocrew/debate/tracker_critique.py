"""Deterministic Progress Tracker (Avery) critique for the debate phase.

Avery's debate seat aggregates prior agent critiques and project state instead of
calling an LLM. Build-phase codebase scanning lives in tracker.progress_tracker.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from autocrew.analyzer.project_model import ProjectContext
from autocrew.debate.critique_schema import attach_structured_fields
from autocrew.debate.debate_model import AgentCritique
from autocrew.squad.squad_model import AgentConfig, AgentRole


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = _normalize_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def _features_by_status(context: ProjectContext, *statuses: str) -> list[str]:
    return [f.name for f in context.features if f.status in statuses]


def _plan_covers(plan_text: str, keywords: list[str]) -> bool:
    lower = plan_text.lower()
    return any(kw.lower() in lower for kw in keywords if kw)


def _aggregate_prior_critiques(
    prior_critiques: list[AgentCritique],
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Merge blockers, concerns, and suggestions from agents before Avery."""
    blockers: list[str] = []
    concerns: list[str] = []
    suggestions: list[str] = []
    counts = {"blockers": 0, "concerns": 0, "suggestions": 0, "agents": 0}

    for critique in prior_critiques:
        if critique.agent_role == AgentRole.PROGRESS_TRACKER.value:
            continue
        counts["agents"] += 1
        counts["blockers"] += len(critique.blockers)
        counts["concerns"] += len(critique.concerns)
        counts["suggestions"] += len(critique.suggestions)

        for blocker in critique.blockers:
            blockers.append(f"[{critique.agent_role}] {blocker}")
        for concern in critique.concerns:
            concerns.append(f"[{critique.agent_role}] {concern}")
        for suggestion in critique.suggestions:
            suggestions.append(f"[{critique.agent_role}] {suggestion}")

    return (
        _dedupe_preserve_order(blockers),
        _dedupe_preserve_order(concerns),
        _dedupe_preserve_order(suggestions),
        counts,
    )


def _feature_status_summary(context: ProjectContext) -> tuple[str, list[str]]:
    done = _features_by_status(context, "done")
    partial = _features_by_status(context, "partial")
    missing = _features_by_status(context, "not_started")
    total = len(context.features) or 1
    completion = (len(done) + 0.5 * len(partial)) / total * 100

    summary = (
        f"{context.project_name}: {completion:.0f}% spec complete "
        f"({len(done)} done, {len(partial)} partial, {len(missing)} not started)."
    )
    concerns: list[str] = []
    if partial:
        concerns.append(
            f"Partial features need completion plan: {', '.join(partial[:5])}"
            + ("..." if len(partial) > 5 else "")
        )
    if missing:
        concerns.append(
            f"Not-started features: {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
        )
    high_open = [
        f.name
        for f in context.features
        if f.priority == "high" and f.status != "done"
    ]
    if high_open:
        concerns.append(f"High-priority gaps remain: {', '.join(high_open[:5])}")
    return summary, concerns


def _plan_gap_concerns(context: ProjectContext, plan_text: str) -> list[str]:
    concerns: list[str] = []
    lower_plan = plan_text.lower()

    if "product.md" in lower_plan or "--- docs/product.md ---" in lower_plan:
        product_section = plan_text
        if "--- docs/product.md ---" in plan_text:
            product_section = plan_text.split("--- docs/product.md ---", 1)[1]
            if "---" in product_section:
                product_section = product_section.split("---", 1)[0]
        if len(product_section.strip()) < 400 and "acceptance criteria" not in lower_plan:
            concerns.append(
                "product.md lacks acceptance criteria or detailed requirements — "
                "implementation and QA cannot define done"
            )

    if "architecture.md" in lower_plan or "--- docs/architecture.md ---" in lower_plan:
        if not _plan_covers(plan_text, ["api contract", "schema", "folder structure", "service"]):
            concerns.append(
                "architecture.md missing service boundaries, API contracts, or folder structure"
            )

    for part in context.missing_parts:
        token = part.split()[0] if part.split() else part
        if token and not _plan_covers(plan_text, [part, token]):
            concerns.append(f"Missing part not covered in plan: {part}")

    return _dedupe_preserve_order(concerns)


def _consensus_narratives(
    prior_critiques: list[AgentCritique],
    aggregated_blockers: list[str],
) -> list[str]:
    """Pure aggregation: describe whether prior agents agree on blockers."""
    if not prior_critiques:
        return []

    agents_with_blockers = [
        c.agent_name for c in prior_critiques if c.blockers and c.agent_role != AgentRole.PROGRESS_TRACKER.value
    ]
    if not agents_with_blockers:
        return ["No blockers raised by prior agents this round."]

    unique_blocker_texts = _dedupe_preserve_order(
        blocker for c in prior_critiques for blocker in c.blockers
    )
    if len(agents_with_blockers) >= 2 and len(unique_blocker_texts) <= max(3, len(agents_with_blockers)):
        return [
            f"Prior agents ({len(agents_with_blockers)}) converge on {len(aggregated_blockers)} "
            f"distinct blocker(s) — unanimous confirmation that deliverables are incomplete."
        ]
    return [
        f"Prior agents raised {len(aggregated_blockers)} distinct blocker(s) "
        f"across {len(agents_with_blockers)} agent(s)."
    ]


def _round_summary_concern(
    round_number: int,
    prior_counts: dict[str, int],
    aggregated_blockers: list[str],
    aggregated_concerns: list[str],
) -> str:
    return (
        f"Round {round_number} tracker summary: {prior_counts['agents']} agents reviewed; "
        f"{prior_counts['blockers']} raw blockers → {len(aggregated_blockers)} unique; "
        f"{prior_counts['concerns']} raw concerns → {len(aggregated_concerns)} unique."
    )


def _tracker_suggestions(context: ProjectContext, plan_text: str) -> list[str]:
    suggestions: list[str] = []
    not_started = _features_by_status(context, "not_started")
    partial = _features_by_status(context, "partial")

    if not_started or partial:
        remaining = not_started + partial
        if not _plan_covers(plan_text, ["priority", "next", "sprint"]):
            suggestions.append(
                f"Prioritize remaining work: {', '.join(remaining[:5])}"
                + ("..." if len(remaining) > 5 else "")
            )

    if context.missing_parts:
        suggestions.append(
            "Address missing parts in plan order: " + "; ".join(context.missing_parts[:5])
        )

    return suggestions


def generate_tracker_critique(
    agent: AgentConfig,
    context: ProjectContext,
    plan_text: str,
    round_number: int,
    prior_critiques: list[AgentCritique],
) -> AgentCritique:
    """Produce Avery's debate critique deterministically from state and prior agents."""
    aggregated_blockers, aggregated_concerns, aggregated_suggestions, prior_counts = (
        _aggregate_prior_critiques(prior_critiques)
    )

    status_summary, feature_concerns = _feature_status_summary(context)
    plan_gaps = _plan_gap_concerns(context, plan_text)
    consensus = _consensus_narratives(prior_critiques, aggregated_blockers)

    concerns = _dedupe_preserve_order([
        _round_summary_concern(
            round_number, prior_counts, aggregated_blockers, aggregated_concerns
        ),
        status_summary,
        *consensus,
        *feature_concerns,
        *plan_gaps,
        *aggregated_concerns,
    ])
    blockers = list(aggregated_blockers)
    suggestions = _dedupe_preserve_order([
        *aggregated_suggestions,
        *_tracker_suggestions(context, plan_text),
    ])

    approved = len(blockers) == 0 and len(concerns) == 0

    return attach_structured_fields(
        AgentCritique(
            agent_role=agent.role.value,
            agent_name=agent.name,
            round_number=round_number,
            approved=approved,
            concerns=concerns,
            suggestions=suggestions,
            blockers=blockers,
            model_used="deterministic",
        )
    )
