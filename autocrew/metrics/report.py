"""Query persisted metrics and compute summary statistics."""

from __future__ import annotations

import statistics
from typing import Any

from autocrew.metrics.store import MetricsStore


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def query_metrics(metrics_dir: str) -> dict[str, Any]:
    store = MetricsStore(metrics_dir)
    debate_summaries = store.fetch_phase_summaries("debate")
    build_summaries = store.fetch_phase_summaries("build")
    debate_calls = store.fetch_agent_calls("debate")
    build_calls = store.fetch_agent_calls("build")

    debate_rounds = [float(s["debate_rounds"] or s["total_rounds"] or 0) for s in debate_summaries]
    debate_tokens = [
        float((s["total_input_tokens"] or 0) + (s["total_output_tokens"] or 0))
        for s in debate_summaries
    ]
    debate_latency = [float(s["total_latency_ms"] or 0) for s in debate_summaries]

    build_tokens = [
        float((s["total_input_tokens"] or 0) + (s["total_output_tokens"] or 0))
        for s in build_summaries
    ]
    build_latency = [float(s["total_latency_ms"] or 0) for s in build_summaries]

    def agent_totals(calls: list[dict[str, Any]], field: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        for call in calls:
            key = call.get("agent_name") or call.get("agent_role") or "unknown"
            totals[key] = totals.get(key, 0.0) + float(call.get(field) or 0)
        return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    def agent_token_totals(calls: list[dict[str, Any]]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for call in calls:
            key = call.get("agent_name") or call.get("agent_role") or "unknown"
            tokens = float((call.get("input_tokens") or 0) + (call.get("output_tokens") or 0))
            totals[key] = totals.get(key, 0.0) + tokens
        return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    debate_token_by_agent = agent_token_totals(debate_calls)
    debate_latency_by_agent = agent_totals(debate_calls, "latency_ms")
    build_token_by_agent = agent_token_totals(build_calls)
    build_latency_by_agent = agent_totals(build_calls, "latency_ms")

    return {
        "sessions": {
            "debate": len(debate_summaries),
            "build": len(build_summaries),
        },
        "debate": {
            "rounds_median": _median(debate_rounds),
            "rounds_p95": _percentile(debate_rounds, 95),
            "tokens_median": _median(debate_tokens),
            "tokens_p95": _percentile(debate_tokens, 95),
            "latency_ms_median": _median(debate_latency),
            "latency_ms_p95": _percentile(debate_latency, 95),
            "most_expensive_by_tokens": list(debate_token_by_agent.items())[:5],
            "most_expensive_by_latency_ms": list(debate_latency_by_agent.items())[:5],
        },
        "build": {
            "tokens_median": _median(build_tokens),
            "tokens_p95": _percentile(build_tokens, 95),
            "latency_ms_median": _median(build_latency),
            "latency_ms_p95": _percentile(build_latency, 95),
            "most_expensive_by_tokens": list(build_token_by_agent.items())[:5],
            "most_expensive_by_latency_ms": list(build_latency_by_agent.items())[:5],
        },
        "raw": {
            "debate_summaries": debate_summaries,
            "build_summaries": build_summaries,
        },
    }


def format_metrics_report(metrics: dict[str, Any]) -> str:
    lines = ["# Session Metrics Report", ""]
    sessions = metrics["sessions"]
    lines.append(f"Debate sessions logged: {sessions['debate']}")
    lines.append(f"Build sessions logged: {sessions['build']}")
    lines.append("")

    debate = metrics["debate"]
    lines.extend([
        "## Debate phase",
        f"- Rounds per debate — median: {debate['rounds_median']:.1f}, p95: {debate['rounds_p95']:.1f}",
        f"- Total tokens per debate — median: {debate['tokens_median']:,.0f}, p95: {debate['tokens_p95']:,.0f}",
        f"- Wall-clock latency per debate (ms) — median: {debate['latency_ms_median']:,.0f}, "
        f"p95: {debate['latency_ms_p95']:,.0f}",
        "- Top agents by tokens:",
    ])
    for name, total in debate["most_expensive_by_tokens"]:
        lines.append(f"  - {name}: {total:,.0f} tokens")
    lines.append("- Top agents by latency (ms):")
    for name, total in debate["most_expensive_by_latency_ms"]:
        lines.append(f"  - {name}: {total:,.0f} ms")
    lines.append("")

    build = metrics["build"]
    lines.extend([
        "## Build phase",
        f"- Total tokens per build — median: {build['tokens_median']:,.0f}, p95: {build['tokens_p95']:,.0f}",
        f"- Wall-clock latency per build (ms) — median: {build['latency_ms_median']:,.0f}, "
        f"p95: {build['latency_ms_p95']:,.0f}",
        "- Top agents by tokens:",
    ])
    for name, total in build["most_expensive_by_tokens"]:
        lines.append(f"  - {name}: {total:,.0f} tokens")
    lines.append("- Top agents by latency (ms):")
    for name, total in build["most_expensive_by_latency_ms"]:
        lines.append(f"  - {name}: {total:,.0f} ms")

    return "\n".join(lines)
