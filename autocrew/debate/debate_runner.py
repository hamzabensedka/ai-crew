"""Orchestrates multi-round squad debate until consensus."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rich.console import Console

from autocrew.analyzer.llm_client import LLMClient, call_with_json_retry
from autocrew.analyzer.project_model import ProjectContext
from autocrew.config import settings
from autocrew.debate.convergence import diff_rounds, log_early_exit_event, should_early_exit
from autocrew.debate.critique_context import FULL_CONTEXT_ROLES, build_critique_context
from autocrew.debate.debate_tiers import build_debate_tiers, is_parallel_tier
from autocrew.debate.critique_schema import (
    STRUCTURED_CRITIQUE_PROMPT_SCHEMA,
    attach_structured_fields,
    parse_critique_response,
)
from autocrew.debate.debate_model import AgentCritique, DebateResult, DebateRound
from autocrew.debate.heuristic_critique import generate_heuristic_critique
from autocrew.debate.model_router import DualModelRouter
from autocrew.debate.tracker_critique import generate_tracker_critique
from autocrew.metrics import begin_session, end_session
from autocrew.metrics.instrumentation import instrument_llm_call, record_non_llm_agent_call
from autocrew.planner import render_product_doc
from autocrew.progress_log import progress_log
from autocrew.squad.squad_model import AgentConfig, AgentRole, Squad
from autocrew.tasks.task_model import TaskConfig

CRITIQUE_PROMPT = """You are {agent_name}, a {agent_role} on the {project_name} project.

Your goal: {goal}
Your expertise: {backstory}

Current plan (product + architecture + tasks):
\"\"\"
{plan_text}
\"\"\"

Project gaps:
- Missing parts: {missing_parts}
- Not started features: {not_started}
- Partial features: {partial}

Other agents' critiques this round:
{other_critiques}

Review the plan from YOUR role's perspective.
{schema_instructions}
Be honest and specific.
"""

LEGACY_CRITIQUE_SCHEMA = """Return JSON:
{{
  "approved": true or false,
  "concerns": ["things wrong or missing"],
  "suggestions": ["improvements to add"],
  "blockers": ["must-fix before implementation"]
}}

Return only valid JSON.
"""

_debate_console = Console()


def _log_debate(msg: str) -> None:
    _debate_console.print(msg)


def _timed_llm_call(
    llm_call: Callable[[str], str],
    *,
    model_name: str,
    agent_name: str,
    agent_role: str,
    round_number: int,
) -> Callable[[str], str]:
    short_model = model_name.split("/")[-1]
    measured = instrument_llm_call(
        llm_call,
        phase="debate",
        agent_name=agent_name,
        agent_role=agent_role,
        model_name=model_name,
        round_number=round_number,
    )

    def wrapped(prompt: str) -> str:
        progress_log(
            f"  → {agent_name}: calling {short_model} "
            f"({len(prompt):,} char prompt)...",
        )
        try:
            result = measured(prompt)
        except Exception as exc:
            progress_log(f"  ✗ {agent_name}: failed — {exc}")
            raise
        progress_log(
            f"  ← {agent_name}: {short_model} replied "
            f"({len(result):,} chars)",
        )
        return result

    return wrapped


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def load_plan_text(project_root: str, context: ProjectContext) -> str:
    root = Path(project_root)
    parts: list[str] = []

    for rel in ("docs/product.md", "docs/architecture.md", "docs/tasks.md"):
        path = root / rel
        if path.is_file():
            parts.append(f"--- {rel} ---\n{path.read_text(encoding='utf-8', errors='ignore')}")

    if not parts:
        parts.append("--- generated product spec ---\n" + render_product_doc(context))

    return "\n\n".join(parts)


def _format_other_critiques(
    critiques: list[AgentCritique],
    exclude_role: str,
    receiver_role: str,
    receiver_index: int,
) -> str:
    prior = [c for c in critiques if c.agent_role != exclude_role]
    if settings.debate_structured_critiques:
        return build_critique_context(
            prior,
            receiver_role,
            receiver_index=receiver_index,
        )
    lines = []
    for c in prior:
        items = c.blockers + c.concerns + c.suggestions
        if items:
            lines.append(f"[{c.agent_name}]: " + "; ".join(items[:5]))
    return "\n".join(lines) if lines else "(none yet)"


def _context_critiques_for_agent(
    agent: AgentConfig,
    *,
    tier: list[AgentConfig],
    round_critiques: list[AgentCritique],
    previous_tier_critiques: list[AgentCritique],
    parallel_tiers_enabled: bool,
) -> tuple[list[AgentCritique], int]:
    """Choose prior critiques visible to this agent (tier vs sequential rules)."""
    if not parallel_tiers_enabled:
        return round_critiques, len(round_critiques)

    if agent.role.value in FULL_CONTEXT_ROLES:
        return list(round_critiques), len(round_critiques)

    if is_parallel_tier(tier):
        return list(previous_tier_critiques), len(previous_tier_critiques)

    return list(previous_tier_critiques), len(previous_tier_critiques)


def _generate_critique_for_agent(
    agent: AgentConfig,
    *,
    context: ProjectContext,
    plan_text: str,
    round_number: int,
    context_critiques: list[AgentCritique],
    round_critiques: list[AgentCritique],
    receiver_index: int,
    llm: LLMClient | None,
    llm_call: Callable[[str], str] | None,
    dual_router: DualModelRouter | None,
    slot_label: str,
) -> AgentCritique:
    if (
        agent.role == AgentRole.PROGRESS_TRACKER
        and settings.debate_deterministic_tracker
    ):
        _log_debate(f"  {slot_label} {agent.name} (deterministic) — aggregating state...")
        critique = generate_tracker_critique(
            agent,
            context,
            plan_text,
            round_number,
            round_critiques,
        )
        record_non_llm_agent_call(
            phase="debate",
            agent_name=agent.name,
            agent_role=agent.role.value,
            model_used="deterministic",
            round_number=round_number,
        )
        return critique

    if dual_router is not None:
        agent_llm, model_name = dual_router.for_agent(agent)
        short_model = model_name.split("/")[-1]
        _log_debate(f"  {slot_label} {agent.name} ({short_model}) — waiting for API...")
        return _llm_critique(
            agent,
            context,
            plan_text,
            round_number,
            context_critiques,
            _timed_llm_call(
                agent_llm.complete,
                model_name=model_name,
                agent_name=agent.name,
                agent_role=agent.role.value,
                round_number=round_number,
            ),
            model_used=model_name,
            receiver_index=receiver_index,
        )

    if llm_call is not None:
        _log_debate(f"  {slot_label} {agent.name} — waiting for API...")
        return _llm_critique(
            agent,
            context,
            plan_text,
            round_number,
            context_critiques,
            _timed_llm_call(
                llm_call,
                model_name="LLM",
                agent_name=agent.name,
                agent_role=agent.role.value,
                round_number=round_number,
            ),
            receiver_index=receiver_index,
        )

    if llm is not None:
        _log_debate(f"  {slot_label} {agent.name} — waiting for API...")
        return _llm_critique(
            agent,
            context,
            plan_text,
            round_number,
            context_critiques,
            _timed_llm_call(
                llm.complete,
                model_name=getattr(llm, "label", getattr(llm, "model", "LLM")),
                agent_name=agent.name,
                agent_role=agent.role.value,
                round_number=round_number,
            ),
            receiver_index=receiver_index,
        )

    return generate_heuristic_critique(agent, context, plan_text, round_number)


def _sort_critiques_by_tier(critiques: list[AgentCritique], tier: list[AgentConfig]) -> list[AgentCritique]:
    order = {agent.role.value: index for index, agent in enumerate(tier)}
    return sorted(critiques, key=lambda critique: order.get(critique.agent_role, 999))


def _run_debate_round(
    *,
    round_num: int,
    max_rounds: int,
    squad: Squad,
    context: ProjectContext,
    plan_text: str,
    llm: LLMClient | None,
    llm_call: Callable[[str], str] | None,
    dual_router: DualModelRouter | None,
) -> list[AgentCritique]:
    round_critiques: list[AgentCritique] = []
    previous_tier_critiques: list[AgentCritique] = []
    parallel_enabled = settings.debate_parallel_tiers
    tiers = build_debate_tiers(squad) if parallel_enabled else [[agent] for agent in squad.agents]

    _log_debate(f"\n[bold cyan]Round {round_num}/{max_rounds}[/bold cyan]")
    if parallel_enabled:
        parallel_tier_count = sum(1 for tier in tiers if is_parallel_tier(tier))
        if parallel_tier_count:
            _log_debate(
                f"[dim]Parallel tiers enabled — {parallel_tier_count} multi-agent tier(s)[/dim]"
            )

    agent_counter = 0
    for tier in tiers:
        context_critiques_base = previous_tier_critiques
        run_parallel = parallel_enabled and is_parallel_tier(tier)

        if run_parallel:
            tier_label = f"[tier parallel x{len(tier)}]"
            tier_critiques: list[AgentCritique] = []

            def _run_one(agent: AgentConfig) -> AgentCritique:
                ctx_critiques, receiver_index = _context_critiques_for_agent(
                    agent,
                    tier=tier,
                    round_critiques=round_critiques,
                    previous_tier_critiques=context_critiques_base,
                    parallel_tiers_enabled=parallel_enabled,
                )
                return _generate_critique_for_agent(
                    agent,
                    context=context,
                    plan_text=plan_text,
                    round_number=round_num,
                    context_critiques=ctx_critiques,
                    round_critiques=round_critiques,
                    receiver_index=receiver_index,
                    llm=llm,
                    llm_call=llm_call,
                    dual_router=dual_router,
                    slot_label=tier_label,
                )

            with ThreadPoolExecutor(max_workers=len(tier)) as pool:
                futures = {pool.submit(_run_one, agent): agent for agent in tier}
                for future in as_completed(futures):
                    critique = future.result()
                    tier_critiques.append(critique)
                    agent_counter += 1
                    status = "approved" if critique.approved else f"{len(critique.blockers)} blocker(s)"
                    _log_debate(f"  [green]done[/green] {critique.agent_name} — {status}")

            tier_critiques = _sort_critiques_by_tier(tier_critiques, tier)
        else:
            tier_critiques = []
            for agent in tier:
                agent_counter += 1
                ctx_critiques, receiver_index = _context_critiques_for_agent(
                    agent,
                    tier=tier,
                    round_critiques=round_critiques,
                    previous_tier_critiques=context_critiques_base,
                    parallel_tiers_enabled=parallel_enabled,
                )
                critique = _generate_critique_for_agent(
                    agent,
                    context=context,
                    plan_text=plan_text,
                    round_number=round_num,
                    context_critiques=ctx_critiques,
                    round_critiques=round_critiques,
                    receiver_index=receiver_index,
                    llm=llm,
                    llm_call=llm_call,
                    dual_router=dual_router,
                    slot_label=f"[{agent_counter}/{len(squad.agents)}]",
                )
                status = "approved" if critique.approved else f"{len(critique.blockers)} blocker(s)"
                _log_debate(f"  [green]done[/green] — {status}")
                tier_critiques.append(critique)

        round_critiques.extend(tier_critiques)
        previous_tier_critiques = tier_critiques

    return round_critiques


def _llm_critique(
    agent: AgentConfig,
    context: ProjectContext,
    plan_text: str,
    round_number: int,
    prior_in_round: list[AgentCritique],
    llm_call: Callable[[str], str],
    model_used: str = "",
    receiver_index: int | None = None,
) -> AgentCritique:
    not_started = [f.name for f in context.features if f.status == "not_started"]
    partial = [f.name for f in context.features if f.status == "partial"]
    if receiver_index is None:
        receiver_index = len(prior_in_round)

    schema_instructions = (
        STRUCTURED_CRITIQUE_PROMPT_SCHEMA
        if settings.debate_structured_critiques
        else LEGACY_CRITIQUE_SCHEMA
    )

    prompt = CRITIQUE_PROMPT.format(
        agent_name=agent.name,
        agent_role=agent.role.value,
        project_name=context.project_name,
        goal=agent.goal,
        backstory=agent.backstory,
        plan_text=plan_text[:8000],
        missing_parts=context.missing_parts,
        not_started=not_started,
        partial=partial,
        other_critiques=_format_other_critiques(
            prior_in_round,
            agent.role.value,
            agent.role.value,
            receiver_index,
        ),
        schema_instructions=schema_instructions,
    )

    data = call_with_json_retry(llm_call, prompt)
    if settings.debate_structured_critiques:
        return parse_critique_response(
            data, agent, round_number, model_used=model_used
        )
    return attach_structured_fields(
        AgentCritique(
            agent_role=agent.role.value,
            agent_name=agent.name,
            round_number=round_number,
            approved=bool(data.get("approved", False)),
            concerns=list(data.get("concerns", [])),
            suggestions=list(data.get("suggestions", [])),
            blockers=list(data.get("blockers", [])),
            model_used=model_used,
        )
    )


def _merge_plan_revision(
    plan_text: str,
    critiques: list[AgentCritique],
    context: ProjectContext,
    round_number: int,
) -> tuple[str, list[str]]:
    """Product Owner synthesizes critiques into plan revisions."""
    action_items: list[str] = []

    for c in critiques:
        for blocker in c.blockers:
            action_items.append(f"[{c.agent_role}] BLOCKER: {blocker}")
        for concern in c.concerns:
            action_items.append(f"[{c.agent_role}] CONCERN: {concern}")
        for suggestion in c.suggestions:
            action_items.append(f"[{c.agent_role}] SUGGEST: {suggestion}")

    for part in context.missing_parts:
        item = f"Implement missing part: {part}"
        if item not in action_items:
            action_items.append(item)

    for f in context.features:
        if f.status in ("not_started", "partial") and f.priority == "high":
            item = f"Complete feature ({f.status}): {f.name} — {f.description}"
            if not any(f.name in a for a in action_items):
                action_items.append(item)

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in action_items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    revision = (
        f"\n\n## Debate Revision — Round {round_number}\n\n"
        f"*Synthesized by Product Owner from squad critiques.*\n\n"
    )
    for i, item in enumerate(unique_items, 1):
        revision += f"{i}. {item}\n"

    revision += (
        "\n### Consensus criteria\n"
        "- All blockers addressed in action items above\n"
        "- Each dev agent confirms their scope is covered\n"
        "- Tester confirms test plan exists for new work\n"
    )

    return plan_text + revision, unique_items


def _assign_role_for_action(action: str, squad: Squad) -> str:
    lower = action.lower()
    role_map = [
        (("payment", "stripe", "webhook", "api", "backend", "nestjs", "prisma", "migration", "notification", "review", "bullmq", "job"), AgentRole.BACKEND_DEV),
        (("admin", "dashboard", "next.js", "frontend", "mobile", "screen", "ui", "expo", "portal"), AgentRole.FRONTEND_DEV),
        (("docker", "ci", "deploy", "github actions", "infra"), AgentRole.DEVOPS),
        (("test", "jest", "e2e", "qa"), AgentRole.TESTER),
        (("schema", "database", "etl", "pipeline"), AgentRole.DATA_ENGINEER),
        (("llm", "ai", "embedding"), AgentRole.AI_ENGINEER),
    ]
    squad_roles = {a.role for a in squad.agents}
    for keywords, role in role_map:
        if any(k in lower for k in keywords) and role in squad_roles:
            return role.value
    if AgentRole.FULLSTACK_DEV in squad_roles:
        return AgentRole.FULLSTACK_DEV.value
    if AgentRole.BACKEND_DEV in squad_roles:
        return AgentRole.BACKEND_DEV.value
    return AgentRole.PRODUCT_OWNER.value


def build_tasks_from_debate(
    debate: DebateResult,
    squad: Squad,
    context: ProjectContext,
) -> list[TaskConfig]:
    """Create implementation tasks from debate action items."""
    tasks: list[TaskConfig] = []
    task_index = 0
    for action in debate.action_items:
        if action.startswith("[") and ("CONCERN:" in action or "SUGGEST:" in action):
            continue
        if not any(k in action for k in ("BLOCKER:", "Implement missing", "Complete feature")):
            continue
        role = _assign_role_for_action(action, squad)
        task_index += 1
        slug = re.sub(r"[^a-z0-9]+", "_", action[:40].lower()).strip("_")
        task_id = f"debate_{task_index}_{slug}"[:60]
        tasks.append(
            TaskConfig(
                task_id=task_id,
                title=action[:80],
                description=(
                    f"Implement per squad debate consensus:\n{action}\n\n"
                    f"Refer to docs/product.md and the locked debate plan at {debate.final_plan_path}"
                ),
                assigned_agent_role=role,
                depends_on=["arch_design"],
                output_format="code",
                output_path=None,
                expected_output=f"Working implementation for: {action[:100]}",
                context_files=["docs/product.md", "docs/architecture.md"],
            )
        )
    return tasks


def render_round_summary(round_data: DebateRound) -> str:
    lines = [f"# Debate Round {round_data.round_number}", ""]
    for c in round_data.critiques:
        status = "APPROVED" if c.approved else "NOT APPROVED"
        model = f" [{c.model_used}]" if c.model_used else ""
        lines.append(f"## {c.agent_name}{model} ({status})")
        if c.blockers:
            lines.append("**Blockers:**")
            lines.extend(f"- {b}" for b in c.blockers)
        if c.concerns:
            lines.append("**Concerns:**")
            lines.extend(f"- {x}" for x in c.concerns)
        if c.suggestions:
            lines.append("**Suggestions:**")
            lines.extend(f"- {s}" for s in c.suggestions)
        lines.append("")
    lines.append(f"**All approved:** {round_data.all_approved}")
    lines.append(f"**Total blockers:** {round_data.total_blockers}")
    return "\n".join(lines)


def run_debate(
    context: ProjectContext,
    squad: Squad,
    project_root: str,
    output_dir: str,
    max_rounds: int = 3,
    llm: LLMClient | None = None,
    llm_call: Callable[[str], str] | None = None,
    dual_router: DualModelRouter | None = None,
) -> DebateResult:
    debate_dir = Path(output_dir) / "debate" / _slug(context.project_name)
    debate_dir.mkdir(parents=True, exist_ok=True)

    begin_session(context.project_name, phase="debate")
    plan_text = load_plan_text(project_root, context)
    rounds: list[DebateRound] = []
    all_action_items: list[str] = []
    consensus_reached = False
    converged_early = False
    early_exit_round: int | None = None
    early_exit_log_path = debate_dir / "early_exit_log.jsonl"
    models_used: dict[str, str] = {}
    if dual_router:
        models_used = {
            "planning": dual_router.planning_model,
            "implementation": dual_router.implementation_model,
        }

    for round_num in range(1, max_rounds + 1):
        round_critiques = _run_debate_round(
            round_num=round_num,
            max_rounds=max_rounds,
            squad=squad,
            context=context,
            plan_text=plan_text,
            llm=llm,
            llm_call=llm_call,
            dual_router=dual_router,
        )

        total_blockers = sum(len(c.blockers) for c in round_critiques)
        all_approved = all(c.approved for c in round_critiques)

        plan_text, action_items = _merge_plan_revision(plan_text, round_critiques, context, round_num)
        all_action_items = action_items

        round_dir = debate_dir / f"round-{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        round_data = DebateRound(
            round_number=round_num,
            critiques=round_critiques,
            revised_plan_excerpt=plan_text[-3000:],
            all_approved=all_approved,
            total_blockers=total_blockers,
        )

        if all_approved:
            consensus_reached = True
            rounds.append(round_data)
            (round_dir / "critiques.json").write_text(
                json.dumps([c.to_dict() for c in round_critiques], indent=2),
                encoding="utf-8",
            )
            (round_dir / "summary.md").write_text(render_round_summary(round_data), encoding="utf-8")
            (round_dir / "revised_plan.md").write_text(plan_text, encoding="utf-8")
            break

        if (
            settings.debate_early_exit
            and round_num >= 2
            and len(rounds) >= 1
        ):
            convergence = diff_rounds(
                rounds[-1].critiques,
                round_critiques,
                previous_round=rounds[-1].round_number,
                current_round=round_num,
            )
            if should_early_exit(
                convergence,
                round_number=round_num,
                min_rounds=settings.debate_min_rounds,
            ):
                converged_early = True
                consensus_reached = True
                early_exit_round = round_num
                round_data.converged_early = True
                log_early_exit_event(
                    early_exit_log_path,
                    project_name=context.project_name,
                    task_id=_slug(context.project_name),
                    round_number=round_num,
                    diff=convergence,
                )
                _log_debate(
                    f"[yellow]Early exit[/yellow] — round {round_num} raised no new concerns "
                    f"or open questions (logged to {early_exit_log_path.name})"
                )
                rounds.append(round_data)
                (round_dir / "critiques.json").write_text(
                    json.dumps([c.to_dict() for c in round_critiques], indent=2),
                    encoding="utf-8",
                )
                (round_dir / "summary.md").write_text(
                    render_round_summary(round_data), encoding="utf-8"
                )
                (round_dir / "revised_plan.md").write_text(plan_text, encoding="utf-8")
                break

        rounds.append(round_data)

        (round_dir / "critiques.json").write_text(
            json.dumps([c.to_dict() for c in round_critiques], indent=2),
            encoding="utf-8",
        )
        (round_dir / "summary.md").write_text(render_round_summary(round_data), encoding="utf-8")
        (round_dir / "revised_plan.md").write_text(plan_text, encoding="utf-8")

        if total_blockers == 0 and round_num >= 2:
            consensus_reached = True
            break

    timestamp = datetime.now(timezone.utc).isoformat()
    final_plan_path = debate_dir / "final_plan.md"
    final_plan_path.write_text(plan_text, encoding="utf-8")

    result = DebateResult(
        project_name=context.project_name,
        timestamp=timestamp,
        rounds=rounds,
        consensus_reached=consensus_reached,
        final_plan_path=str(final_plan_path),
        debate_dir=str(debate_dir),
        action_items=all_action_items,
        models_used=models_used,
        converged_early=converged_early,
        early_exit_round=early_exit_round,
        early_exit_log_path=str(early_exit_log_path) if converged_early else None,
    )

    (debate_dir / "debate_result.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )

    product_doc = Path(project_root) / "docs" / "product.md"
    product_doc.parent.mkdir(parents=True, exist_ok=True)
    if "## Debate Revision" in plan_text:
        revision_part = plan_text[plan_text.index("## Debate Revision") :]
        if product_doc.is_file():
            existing = product_doc.read_text(encoding="utf-8")
            if revision_part.strip() not in existing:
                product_doc.write_text(existing.rstrip() + "\n\n" + revision_part, encoding="utf-8")
        else:
            product_doc.write_text(plan_text, encoding="utf-8")
    elif not product_doc.is_file():
        product_doc.write_text(plan_text, encoding="utf-8")

    end_session(
        phase="debate",
        debate_rounds=len(rounds),
        extra={
            "consensus_reached": consensus_reached,
            "converged_early": converged_early,
            "early_exit_round": early_exit_round,
            "debate_dir": str(debate_dir),
            "debate_parallel_tiers": settings.debate_parallel_tiers,
        },
    )

    return result
