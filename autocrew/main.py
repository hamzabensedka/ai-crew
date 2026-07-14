"""AutoCrew CLI — AI Project Orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autocrew.autopilot import run_autopilot
from autocrew.analyzer.idea_analyzer import analyze_idea
from autocrew.analyzer.llm_client import LLMClient, create_llm_client, create_model_client
from autocrew.analyzer.project_model import ProjectContext
from autocrew.config import settings
from autocrew.crew.crew_runner import run_crew
from autocrew.cursor_mode import CURSOR_WORKFLOW_HINT, write_scout_report
from autocrew.debate.debate_runner import (
    build_tasks_from_debate,
    render_round_summary,
    run_debate,
)
from autocrew.debate.model_router import (
    DualModelRouter,
    ModelRouter,
    PerAgentModelRouter,
    RoleModelRouter,
    parse_per_agent_models,
)
from autocrew.planner import write_plan_docs
from autocrew.progress_log import ProgressLogger, set_progress_logger
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import Squad
from autocrew.storage import (
    find_latest_context,
    find_latest_debate,
    find_latest_report,
    find_latest_squad,
    find_latest_tasks,
    load_context,
    load_squad,
    load_tasks,
    save_context,
    save_squad,
    save_tasks,
)
from autocrew.tasks.dependency_resolver import resolve_dependencies
from autocrew.tasks.task_builder import build_tasks, merge_foundation_tasks
from autocrew.tracker.progress_tracker import generate_progress_report, render_report_markdown, save_report
from autocrew.tracker.report_model import ProgressReport

app = typer.Typer(name="autocrew", help="AI Project Orchestrator — idea to built project")
console = Console()


def _get_llm():
    if not settings.has_api_keys():
        raise RuntimeError("No LLM API keys configured")
    return create_llm_client(
        anthropic_key=settings.anthropic_api_key,
        openai_key=settings.openai_api_key,
        nvidia_key=settings.nvidia_api_key,
        zenmux_key=settings.zenmux_api_key,
        openrouter_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        default_model=settings.default_llm,
        fallback_model=settings.fallback_llm,
        llm_provider=settings.llm_provider,
        nvidia_base_url=settings.nvidia_base_url,
        zenmux_base_url=settings.zenmux_base_url,
        nvidia_enable_thinking=settings.nvidia_enable_thinking,
        nvidia_max_tokens=settings.nvidia_max_tokens,
        nvidia_temperature=settings.nvidia_temperature,
        nvidia_top_p=settings.nvidia_top_p,
        nvidia_reasoning_budget=settings.nvidia_reasoning_budget,
        llm_max_retries=settings.llm_max_retries,
        llm_retry_backoff_seconds=settings.llm_retry_backoff_seconds,
        llm_request_timeout_seconds=settings.llm_request_timeout_seconds,
    )


def _llm_settings_kwargs() -> dict:
    return {
        "anthropic_key": settings.anthropic_api_key,
        "openai_key": settings.openai_api_key,
        "nvidia_key": settings.nvidia_api_key,
        "zenmux_key": settings.zenmux_api_key,
        "openrouter_key": settings.openrouter_api_key,
        "openrouter_base_url": settings.openrouter_base_url,
        "llm_provider": settings.llm_provider,
        "nvidia_base_url": settings.nvidia_base_url,
        "zenmux_base_url": settings.zenmux_base_url,
        "nvidia_enable_thinking": settings.nvidia_enable_thinking,
        "nvidia_max_tokens": settings.nvidia_max_tokens,
        "nvidia_temperature": settings.nvidia_temperature,
        "nvidia_top_p": settings.nvidia_top_p,
        "nvidia_reasoning_budget": settings.nvidia_reasoning_budget,
        "llm_max_retries": settings.llm_max_retries,
        "llm_retry_backoff_seconds": settings.llm_retry_backoff_seconds,
        "llm_request_timeout_seconds": settings.llm_request_timeout_seconds,
    }


def _start_progress_logging(*, verbose: bool, prefix: str) -> str:
    settings.ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = str(Path(settings.logs_dir) / f"{prefix}_{timestamp}.log")
    set_progress_logger(ProgressLogger(log_path=log_path, verbose=verbose))
    return log_path


def _get_debate_router(dual_model: bool) -> DualModelRouter | None:
    if not dual_model or not settings.has_api_keys():
        return None
    planning_model = settings.debate_planning_model or settings.fallback_llm
    implementation_model = settings.debate_implementation_model or settings.default_llm
    if planning_model == implementation_model:
        return None
    kwargs = _llm_settings_kwargs()
    planning_base = create_model_client(planning_model, **kwargs)
    implementation_base = create_model_client(implementation_model, **kwargs)
    return DualModelRouter(
        planning_llm=create_model_client(
            planning_model,
            **kwargs,
            fallback=implementation_base,
        ),
        implementation_llm=create_model_client(
            implementation_model,
            **kwargs,
            fallback=planning_base,
        ),
        planning_model=planning_model,
        implementation_model=implementation_model,
    )




def _get_per_agent_router() -> PerAgentModelRouter | None:
    """Build a PerAgentModelRouter from the debate_per_agent_models config.

    Returns None if the config is empty or invalid.
    """
    if not settings.has_api_keys():
        return None
    role_models = parse_per_agent_models(settings.debate_per_agent_models)
    if not role_models:
        return None

    kwargs = _llm_settings_kwargs()
    default_model = settings.default_llm
    fallback_model = settings.fallback_llm
    reasoning_roles = frozenset({"product_owner", "architect", "code_reviewer"})

    if default_model == fallback_model:
        reasoning_client = coder_client = create_model_client(default_model, **kwargs)
    else:
        reasoning_client = create_model_client(default_model, **kwargs)
        coder_client = create_model_client(fallback_model, **kwargs)
        reasoning_client = create_model_client(
            default_model, **kwargs, fallback=coder_client,
        )
        coder_client = create_model_client(
            fallback_model, **kwargs, fallback=reasoning_client,
        )

    default_client = reasoning_client
    client_cache: dict[str, LLMClient] = {
        default_model: reasoning_client,
        fallback_model: coder_client,
    }
    role_model_map: dict[str, tuple[LLMClient, str]] = {}

    for role, model_name in role_models.items():
        if model_name == "deterministic":
            continue
        if model_name not in client_cache:
            cross_fallback = (
                coder_client if role in reasoning_roles else reasoning_client
            )
            fb = cross_fallback if model_name not in {default_model, fallback_model} else None
            client_cache[model_name] = create_model_client(
                model_name,
                **kwargs,
                fallback=fb,
            )
        role_model_map[role] = (client_cache[model_name], model_name)

    if not role_model_map:
        return None

    return PerAgentModelRouter(
        role_model_map=role_model_map,
        default_llm=default_client,
        default_model=default_model,
    )


def _get_role_router() -> RoleModelRouter | None:
    """Free-tier provider chain with per-role model assignment."""
    if not settings.llm_free_tier_chain or not settings.has_api_keys():
        return None
    settings.sync_provider_env()
    return RoleModelRouter()


def _get_router(use_dual: bool) -> DualModelRouter | PerAgentModelRouter | RoleModelRouter | None:
    """Get the appropriate router: per-agent JSON > free-tier role > dual-model."""
    per_agent = _get_per_agent_router()
    if per_agent is not None:
        return per_agent
    role_router = _get_role_router()
    if role_router is not None:
        return role_router
    return _get_debate_router(use_dual)


def _router_mode_label(router: ModelRouter | None) -> str:
    if isinstance(router, RoleModelRouter):
        return "free-tier role routing (NIM -> Groq -> Cerebras -> OpenRouter)"
    if isinstance(router, PerAgentModelRouter):
        return "per-agent JSON routing"
    if isinstance(router, DualModelRouter):
        return "dual-model routing"
    if settings.llm_free_tier_chain:
        return "LEGACY single model (free-tier chain not active — upgrade autocrew or check .env)"
    return f"legacy single model ({settings.default_llm})"


def _resolve_llm_routing(
    use_dual: bool,
) -> tuple[DualModelRouter | PerAgentModelRouter | RoleModelRouter | None, LLMClient | None, str]:
    """Return (router, fallback_llm, mode_label). Router takes precedence over llm."""
    settings.sync_provider_env()
    router = _get_router(use_dual)
    if router is not None:
        return router, None, _router_mode_label(router)
    if not settings.has_api_keys():
        raise RuntimeError("No LLM API keys configured")
    if settings.llm_free_tier_chain:
        console.print(
            "[yellow]Warning:[/yellow] LLM_FREE_TIER_CHAIN=true but role router did not start. "
            "Install the latest autocrew + litellm, or set LLM_FREE_TIER_CHAIN=false."
        )
    return None, _get_llm(), _router_mode_label(None)

def _build_tasks_no_llm(squad: Squad, context: ProjectContext):
    return build_tasks(squad, context, llm_call=lambda _: "[]")


def _display_context(context: ProjectContext) -> None:
    table = Table(title=f"Project: {context.project_name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Type", context.project_type.value)
    table.add_row("Domain", context.domain.value)
    table.add_row("Description", context.description[:200])
    table.add_row("Frontend", ", ".join(context.tech_stack.frontend) or "—")
    table.add_row("Backend", ", ".join(context.tech_stack.backend) or "—")
    table.add_row("DevOps", ", ".join(context.tech_stack.devops) or "—")
    console.print(table)

    feat_table = Table(title="Features")
    feat_table.add_column("Name")
    feat_table.add_column("Status")
    feat_table.add_column("Priority")
    for f in context.features:
        status_style = {"done": "green", "partial": "yellow", "not_started": "red"}.get(
            f.status, "white"
        )
        feat_table.add_row(f.name, f"[{status_style}]{f.status}[/{status_style}]", f.priority)
    console.print(feat_table)

    if context.missing_parts:
        console.print(Panel("\n".join(f"• {p}" for p in context.missing_parts), title="Missing Parts"))


def _display_squad(squad: Squad) -> None:
    table = Table(title=f"Squad for {squad.project_name}")
    table.add_column("Role", style="cyan")
    table.add_column("Name")
    table.add_column("Tools")
    for agent in squad.agents:
        table.add_row(agent.role.value, agent.name, ", ".join(agent.tools))
    console.print(table)

    if squad.parallel_groups:
        console.print(f"Parallel groups: {squad.parallel_groups}")
    console.print(f"Execution order: {' -> '.join(squad.execution_order)}")


def _confirm(message: str) -> bool:
    if not settings.require_confirmation:
        return True
    return typer.confirm(message, default=True)


def _finalize_analysis(context: ProjectContext) -> tuple[ProjectContext, Squad]:
    settings.ensure_dirs()
    _display_context(context)
    squad = build_squad(context)
    _display_squad(squad)
    if _confirm("Save this context and squad?"):
        ctx_path = save_context(context, settings.contexts_dir)
        squad_path = save_squad(squad, settings.squads_dir)
        console.print(f"[green]Saved context:[/green] {ctx_path}")
        console.print(f"[green]Saved squad:[/green] {squad_path}")
        console.print("[bold]Ready. Run[/bold] [cyan]autocrew plan[/cyan] then [cyan]autocrew debate[/cyan] then [cyan]autocrew build[/cyan]")
    else:
        console.print("[yellow]Not saved.[/yellow]")
    return context, squad


@app.command()
def scout(
    path: str = typer.Argument(..., help="Path to existing project folder"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write scout JSON to file"),
) -> None:
    """Export file tree + key files for Cursor Composer to analyze (no API keys)."""
    settings.ensure_dirs()
    if not Path(path).is_dir():
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(1)
    try:
        result = write_scout_report(path, output)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if output:
        console.print(f"[green]Scout report saved:[/green] {result}")
        console.print(
            "[bold]Next:[/bold] Ask Cursor to read this file and create context JSON, "
            "then run [cyan]autocrew import-context context.json[/cyan]"
        )
    else:
        console.print(result)


@app.command("import-context")
def import_context(
    context_file: str = typer.Argument(..., help="Path to ProjectContext JSON"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Import a Cursor-generated context JSON and build + save the squad (no API keys)."""
    settings.ensure_dirs()
    path = Path(context_file)
    if not path.is_file():
        console.print(f"[red]File not found:[/red] {context_file}")
        raise typer.Exit(1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        context = ProjectContext.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        console.print(f"[red]Invalid context JSON:[/red] {exc}")
        console.print("See docs/templates/context.example.json for the expected format.")
        raise typer.Exit(1) from exc

    _display_context(context)
    squad = build_squad(context)
    _display_squad(squad)

    if not yes and settings.require_confirmation:
        if not typer.confirm("Save this context and squad?", default=True):
            console.print("[yellow]Not saved.[/yellow]")
            raise typer.Exit(0)

    ctx_path = save_context(context, settings.contexts_dir)
    squad_path = save_squad(squad, settings.squads_dir)
    console.print(f"[green]Saved context:[/green] {ctx_path}")
    console.print(f"[green]Saved squad:[/green] {squad_path}")
    console.print("[bold]Ready. Run[/bold] [cyan]autocrew plan[/cyan] then [cyan]autocrew debate[/cyan] then [cyan]autocrew build[/cyan]")


@app.command()
def new(
    idea: str = typer.Argument(..., help="Plain text project idea"),
) -> None:
    """Analyze a new project idea and generate a tailored squad."""
    settings.ensure_dirs()
    if not settings.has_api_keys():
        console.print(CURSOR_WORKFLOW_HINT)
        raise typer.Exit(1)
    console.print("[bold]Analyzing idea...[/bold]")
    try:
        llm = _get_llm()
        context = analyze_idea(idea, llm=llm)
    except Exception as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _finalize_analysis(context)


@app.command()
def analyze(
    path: str = typer.Argument(..., help="Path to existing project folder"),
) -> None:
    """Analyze an existing codebase and generate a tailored squad."""
    settings.ensure_dirs()
    if not Path(path).is_dir():
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(1)
    if not settings.has_api_keys():
        console.print(CURSOR_WORKFLOW_HINT)
        console.print(f"\n[bold]Tip:[/bold] Run [cyan]autocrew scout {path} --output output/scout.json[/cyan] first.")
        raise typer.Exit(1)
    console.print("[bold]Analyzing codebase...[/bold]")
    try:
        llm = _get_llm()
        context = analyze_codebase(path, llm=llm)
    except Exception as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _finalize_analysis(context)


def _merge_task_lists(base: list, extra: list) -> list:
    seen = {t.task_id for t in base}
    merged = list(base)
    for task in extra:
        if task.task_id not in seen:
            merged.append(task)
            seen.add(task.task_id)
    return merged


def _display_debate_round(round_data) -> None:
    table = Table(title=f"Debate Round {round_data.round_number}")
    table.add_column("Agent", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Status")
    table.add_column("Blockers", style="red")
    table.add_column("Concerns", style="yellow")
    for c in round_data.critiques:
        status = "[green]Approved[/green]" if c.approved else "[red]Not approved[/red]"
        model_label = c.model_used.split("/")[-1] if c.model_used else "-"
        table.add_row(
            c.agent_name,
            model_label,
            status,
            str(len(c.blockers)),
            str(len(c.concerns)),
        )
    console.print(table)
    if round_data.all_approved:
        console.print("[green]All agents approved this round.[/green]")
    else:
        console.print(
            f"[yellow]{round_data.total_blockers} blocker(s) remaining — "
            "PO will revise the plan for the next round.[/yellow]"
        )


@app.command()
def debate(
    project_root: Optional[str] = typer.Option(None, "--root", help="Target project root"),
    context_path: Optional[str] = typer.Option(None, "--context", help="Path to context JSON"),
    squad_path: Optional[str] = typer.Option(None, "--squad", help="Path to squad JSON"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Maximum debate rounds"),
    dual_model: Optional[bool] = typer.Option(
        None,
        "--dual-model/--single-model",
        help="Use two LLMs: planning agents vs implementation agents",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Extra LLM detail in logs"),
) -> None:
    """Run multi-agent debate: critique plan until squad consensus, then save implementation tasks."""
    settings.ensure_dirs()
    log_path = _start_progress_logging(verbose=verbose, prefix="debate")
    console.print(f"[dim]Progress log: {log_path}[/dim]")
    ctx_file = context_path or find_latest_context(settings.contexts_dir)
    sq_file = squad_path or find_latest_squad(settings.squads_dir)

    if not ctx_file or not sq_file:
        console.print("[red]No saved context/squad. Run autocrew import-context first.[/red]")
        raise typer.Exit(1)

    context = load_context(ctx_file)
    squad = load_squad(sq_file)
    root = project_root or context.codebase_path or "."

    use_dual = settings.debate_dual_model if dual_model is None else dual_model

    console.print(Panel(
        f"Squad debate for [bold]{context.project_name}[/bold]\n"
        f"{len(squad.agents)} agents, up to {rounds} rounds",
        title="AutoCrew Debate",
    ))
    _display_squad(squad)

    if use_dual and settings.has_api_keys():
        router = _get_router(True)
        if router:
            console.print(Panel(router.summary(), title="Model routing"))
        else:
            console.print("[yellow]Dual-model requested but both models are identical — using single model.[/yellow]")

    if not yes and settings.require_confirmation:
        if not typer.confirm("Start squad debate?", default=True):
            raise typer.Exit(0)

    if settings.has_api_keys():
        console.print("[bold]Running LLM-powered debate...[/bold]")
        try:
            router = _get_router(use_dual)
            if router:
                result = run_debate(
                    context, squad, root, settings.output_dir,
                    max_rounds=rounds, dual_router=router,
                )
            else:
                llm = _get_llm()
                result = run_debate(
                    context, squad, root, settings.output_dir,
                    max_rounds=rounds, llm=llm,
                )
        except Exception as exc:
            console.print(f"[red]Debate failed:[/red] {exc}")
            raise typer.Exit(1) from exc
    else:
        console.print("[dim]No API keys — heuristic debate (Cursor mode)[/dim]")
        result = run_debate(context, squad, root, settings.output_dir, max_rounds=rounds)

    for round_data in result.rounds:
        _display_debate_round(round_data)
        console.print(Panel(render_round_summary(round_data)[:2000], title=f"Round {round_data.round_number} details"))

    consensus_msg = (
        "[green]Consensus reached — squad is happy with the plan.[/green]"
        if result.consensus_reached
        else f"[yellow]Max rounds reached with {result.rounds[-1].total_blockers} blocker(s) remaining.[/yellow]"
    )
    console.print(consensus_msg)

    debate_tasks = build_tasks_from_debate(result, squad, context)
    all_tasks = merge_foundation_tasks(squad, context, debate_tasks)
    tasks_path = save_tasks(all_tasks, settings.output_dir, context.project_name)

    console.print(f"[green]Progress log:[/green] {log_path}")
    console.print(f"[green]Final plan:[/green] {result.final_plan_path}")
    console.print(f"[green]Debate log:[/green] {result.debate_dir}")
    console.print(f"[green]Tasks ({len(all_tasks)}, incl. {len(debate_tasks)} from debate):[/green] {tasks_path}")
    console.print(f"[green]Action items:[/green] {len(result.action_items)}")
    console.print("[bold]Next:[/bold] [cyan]autocrew build --root {root} --yes[/cyan]".format(root=root))


@app.command()
def plan(
    context_path: Optional[str] = typer.Option(None, "--context", help="Path to saved context JSON"),
    squad_path: Optional[str] = typer.Option(None, "--squad", help="Path to saved squad JSON"),
    project_root: Optional[str] = typer.Option(None, "--root", help="Target project root"),
) -> None:
    """Generate product, architecture, and task plan documents for review."""
    settings.ensure_dirs()
    ctx_file = context_path or find_latest_context(settings.contexts_dir)
    sq_file = squad_path or find_latest_squad(settings.squads_dir)

    if not ctx_file or not sq_file:
        console.print("[red]No saved context/squad found. Run autocrew new or autocrew analyze first.[/red]")
        raise typer.Exit(1)

    context = load_context(ctx_file)
    squad = load_squad(sq_file)
    root = project_root or context.codebase_path or "."

    console.print("[bold]Building task plan...[/bold]")
    if settings.has_api_keys():
        try:
            llm = _get_llm()
            tasks = build_tasks(squad, context, llm=llm)
        except Exception as exc:
            console.print(f"[yellow]LLM task generation failed, using standard tasks:[/yellow] {exc}")
            tasks = _build_tasks_no_llm(squad, context)
    else:
        console.print("[dim]No API keys — using standard task templates (Cursor mode)[/dim]")
        tasks = _build_tasks_no_llm(squad, context)

    tasks_path = save_tasks(tasks, settings.output_dir, context.project_name)
    doc_paths = write_plan_docs(context, squad, tasks, root)

    console.print(f"[green]Tasks saved:[/green] {tasks_path}")
    for name, path in doc_paths.items():
        console.print(f"[green]{name} doc:[/green] {path}")
    console.print("[bold]Review the plan, then run[/bold] [cyan]autocrew debate[/cyan] or [cyan]autocrew build[/cyan]")


@app.command()
def build(
    squad_path: Optional[str] = typer.Option(None, "--squad", help="Path to squad JSON"),
    tasks_path: Optional[str] = typer.Option(None, "--tasks", help="Path to tasks JSON"),
    context_path: Optional[str] = typer.Option(None, "--context", help="Path to context JSON"),
    project_root: Optional[str] = typer.Option(None, "--root", help="Target project root"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    limit: int = typer.Option(0, "--limit", "-n", help="Max tasks to run (0 = all)"),
    parallel_git: Optional[bool] = typer.Option(
        None,
        "--parallel-git/--no-parallel-git",
        help="Parallel devs work on separate branches, reviewed then merged",
    ),
    git_push: Optional[bool] = typer.Option(
        None,
        "--push/--no-push",
        help="Push approved branches to origin after review",
    ),
    recover_worktrees: Optional[bool] = typer.Option(
        None,
        "--recover-worktrees/--no-recover-worktrees",
        help="Before dev phase, commit and merge orphaned .autocrew/worktrees",
    ),
    simulation: bool = typer.Option(
        False,
        "--simulation",
        help="Stub mode — write placeholders without calling the LLM",
    ),
    dual_model: Optional[bool] = typer.Option(
        None,
        "--dual-model/--single-model",
        help="Planning agents use Kimi, dev agents use DeepSeek (NVIDIA)",
    ),
) -> None:
    """Run the full crew build (LLM-powered when API keys are configured)."""
    settings.ensure_dirs()
    ctx_file = context_path or find_latest_context(settings.contexts_dir)
    sq_file = squad_path or find_latest_squad(settings.squads_dir)

    if not ctx_file or not sq_file:
        console.print("[red]No saved context/squad. Run autocrew new/analyze first.[/red]")
        raise typer.Exit(1)

    context = load_context(ctx_file)
    squad = load_squad(sq_file)
    root = project_root or context.codebase_path or "."
    tk_file = tasks_path or find_latest_tasks(settings.output_dir, context.project_name)

    if tk_file:
        tasks = load_tasks(tk_file)
    else:
        console.print("[yellow]No saved tasks. Generating...[/yellow]")
        if settings.has_api_keys():
            try:
                llm = _get_llm()
                tasks = build_tasks(squad, context, llm=llm)
            except Exception:
                tasks = _build_tasks_no_llm(squad, context)
        else:
            tasks = _build_tasks_no_llm(squad, context)
        save_tasks(tasks, settings.output_dir, context.project_name)

    use_llm = settings.has_api_keys() and not simulation
    use_dual = settings.debate_dual_model if dual_model is None else dual_model
    router = _get_router(use_dual) if use_llm else None
    effective_limit = limit
    task_count = len(tasks if limit <= 0 else tasks[:limit])

    if use_llm:
        if router:
            console.print(Panel(router.summary(), title="LLM build (dual-model)"))
        else:
            console.print(f"[bold]LLM build[/bold] using {settings.default_llm}")
        if len(tasks) > 20 and limit <= 0:
            console.print(
                f"[yellow]Warning: {len(tasks)} tasks — this may take hours. "
                "Use --limit 5 to test first.[/yellow]"
            )
    else:
        console.print("[dim]Simulation mode — placeholder files only (no LLM)[/dim]")

    use_parallel_git = settings.parallel_git if parallel_git is None else parallel_git
    do_push = settings.git_push if git_push is None else git_push
    do_recover = settings.worktree_recovery if recover_worktrees is None else recover_worktrees
    if use_parallel_git:
        console.print("[dim]Parallel git: each dev → branch → code review → merge[/dim]")
    if do_recover:
        console.print("[dim]Worktree recovery: scan .autocrew/worktrees before dev phase[/dim]")

    if not yes and settings.require_confirmation:
        mode = "LLM-powered" if use_llm else "simulation"
        count_msg = f"{task_count}" if limit <= 0 else f"{task_count} of {len(tasks)}"
        if not typer.confirm(f"Run {mode} build for '{context.project_name}' ({count_msg} tasks)?", default=True):
            raise typer.Exit(0)

    def _on_task_start(agent, task, model_label: str) -> None:
        model_part = f" ({model_label})" if model_label else ""
        console.print(f"  [cyan]{agent.name}[/cyan]{model_part} — [dim]{task.title[:60]}[/dim] — waiting for API...")

    def _on_task_done(agent, task, result: str) -> None:
        preview = result[:80] + ("..." if len(result) > 80 else "")
        console.print(f"  [green]done[/green] — {preview}")

    console.print("[bold green]Starting crew build...[/bold green]")
    try:
        crew_kwargs = dict(
            task_limit=limit,
            on_task_start=_on_task_start,
            on_task_done=_on_task_done,
            parallel_git=use_parallel_git,
            git_push=do_push,
            worktree_recovery=do_recover,
        )
        if use_llm and router:
            result = run_crew(
                squad,
                tasks,
                context,
                project_root=root,
                use_llm=True,
                dual_router=router,
                **crew_kwargs,
            )
        elif use_llm:
            llm = _get_llm()
            result = run_crew(
                squad,
                tasks,
                context,
                project_root=root,
                use_llm=True,
                llm_call=llm.complete,
                **crew_kwargs,
            )
        else:
            result = run_crew(squad, tasks, context, project_root=root, **crew_kwargs)
    except Exception as exc:
        console.print(f"[red]Build failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(Panel(result, title="Build Complete"))
    console.print(f"[dim]Log:[/dim] {settings.logs_dir}")


@app.command("recover-worktrees")
def recover_worktrees_cmd(
    project_root: Optional[str] = typer.Option(None, "--root", help="Target project root"),
    max_merges: int = typer.Option(
        0,
        "--max-merges",
        help="Max branches to merge (0 = use settings.worktree_recovery_max_merges)",
    ),
    min_insertions: int = typer.Option(
        0,
        "--min-insertions",
        help="Min diff insertions to merge (0 = use settings default)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List mergeable branches without merging",
    ),
) -> None:
    """Commit and merge unmerged agent work from .autocrew/worktrees."""
    from autocrew.crew.crew_logger import CrewLogger
    from autocrew.tools.worktree_recovery import recover_worktrees

    settings.ensure_dirs()
    root = project_root or "."
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = CrewLogger(log_path=str(Path(settings.logs_dir) / f"recover_{timestamp}.log"))

    console.print(f"[bold]Scanning worktrees in[/bold] {Path(root).resolve()}")
    result = recover_worktrees(
        root,
        logger,
        max_merges=max_merges or settings.worktree_recovery_max_merges,
        min_insertions=min_insertions or settings.worktree_recovery_min_insertions,
        merge=not dry_run,
    )
    logger.flush()

    if result is None:
        console.print("[red]Recovery failed — not a git repository?[/red]")
        raise typer.Exit(1)

    table = Table(title="Worktree recovery")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Base branch", result.base_branch)
    table.add_row("Discovered", str(result.discovered))
    table.add_row("Committed", str(len(result.committed)))
    table.add_row("Merged", str(sum(1 for m in result.merged if m.merged)))
    table.add_row("Conflicts", str(len(result.conflicts)))
    table.add_row("Skipped", str(len(result.skipped)))
    console.print(table)

    if result.committed:
        console.print("\n[green]Committed:[/green]")
        for item in result.committed:
            console.print(f"  • {item}")
    if result.merged:
        console.print("\n[green]Merged:[/green]")
        for attempt in result.merged:
            if attempt.merged:
                console.print(f"  • {attempt.branch} ({attempt.role})")
    if result.conflicts:
        console.print("\n[yellow]Conflicts (resolve manually):[/yellow]")
        for branch in result.conflicts:
            console.print(f"  • {branch}")
    if dry_run and result.skipped:
        console.print("\n[dim]Skipped / dry-run details in log[/dim]")

    console.print(f"\n[dim]Log:[/dim] {settings.logs_dir}")


@app.command()
def autopilot(
    project_root: Optional[str] = typer.Option(None, "--root", help="Target project root"),
    context_path: Optional[str] = typer.Option(None, "--context", help="Path to context JSON"),
    squad_path: Optional[str] = typer.Option(None, "--squad", help="Path to squad JSON"),
    tasks_path: Optional[str] = typer.Option(
        None,
        "--tasks",
        help="Fixed task list JSON (skips debate; use for remaining/partial features only)",
    ),
    max_cycles: int = typer.Option(50, "--max-cycles", help="Max debate→build loops (safety cap)"),
    debate_rounds: int = typer.Option(1, "--debate-rounds", "-r", help="Debate rounds per cycle"),
    build_limit: int = typer.Option(5, "--build-limit", "-n", help="LLM build tasks per cycle"),
    min_completion: float = typer.Option(
        100.0,
        "--min-completion",
        help="Required completion %% (default 100 = fully built)",
    ),
    run_tests: bool = typer.Option(True, "--test/--no-test", help="Run pnpm/npm test each cycle"),
    run_security: bool = typer.Option(True, "--security/--no-security", help="Security audit each cycle"),
    llm_security: bool = typer.Option(True, "--llm-security/--static-security-only", help="LLM security review"),
    dual_model: Optional[bool] = typer.Option(
        None,
        "--dual-model/--single-model",
        help="Planning agents (Kimi) vs dev agents (DeepSeek)",
    ),
    parallel_git: Optional[bool] = typer.Option(
        None,
        "--parallel-git/--no-parallel-git",
        help="Each parallel dev gets a branch + worktree, review, merge",
    ),
    git_push: Optional[bool] = typer.Option(
        None,
        "--push/--no-push",
        help="Push approved branches after code review",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Extra LLM detail in logs"),
) -> None:
    """Run until the app is fully built, secured, tested, and the whole crew approves."""
    settings.ensure_dirs()
    log_path = _start_progress_logging(verbose=verbose, prefix="autopilot")
    console.print(f"[dim]Progress log: {log_path}[/dim]")
    ctx_file = context_path or find_latest_context(settings.contexts_dir)
    sq_file = squad_path or find_latest_squad(settings.squads_dir)

    if not ctx_file or not sq_file:
        console.print("[red]No saved context/squad. Run autocrew import-context first.[/red]")
        raise typer.Exit(1)

    if not settings.has_api_keys():
        console.print("[red]Autopilot requires LLM API keys (NVIDIA/Anthropic/OpenAI).[/red]")
        raise typer.Exit(1)

    context = load_context(ctx_file)
    squad = load_squad(sq_file)
    root = project_root or context.codebase_path or "."
    fixed_tasks = load_tasks(tasks_path) if tasks_path else None
    if fixed_tasks:
        console.print(
            f"[green]Remaining-work mode:[/green] {len(fixed_tasks)} fixed tasks from {tasks_path}"
        )
    use_dual = settings.debate_dual_model if dual_model is None else dual_model
    router, llm, routing_mode = _resolve_llm_routing(use_dual)

    import autocrew

    console.print(
        f"[dim]Package: {Path(autocrew.__file__).resolve().parent}[/dim]"
    )
    console.print(Panel(
        f"Autopilot for [bold]{context.project_name}[/bold]\n\n"
        f"Crew stops only when ALL of these are true:\n"
        f"  • [green]Every agent approves[/green] (zero debate blockers)\n"
        f"  • [green]App fully built[/green] (no high-priority gaps, {min_completion:.0f}% complete)\n"
        f"  • [green]Security audit passes[/green] (no critical/high issues)\n"
        f"  • [green]Tests pass[/green]" + ("" if run_tests else " [dim](disabled)[/dim]") + "\n\n"
        f"Max cycles: {max_cycles} | Build {build_limit}/cycle | Debate {debate_rounds} round(s)/cycle\n"
        "[dim]Press Ctrl+C to stop manually.[/dim]",
        title="AutoCrew Autopilot",
    ))

    console.print(Panel(router.summary() if router else routing_mode, title=f"LLM routing — {routing_mode}"))

    if not yes and settings.require_confirmation:
        if not typer.confirm("Start autopilot loop?", default=True):
            raise typer.Exit(0)

    def _on_cycle(n: int) -> None:
        console.print(f"\n[bold cyan]=== Cycle {n}/{max_cycles} ===[/bold cyan]")

    def _on_phase(phase: str) -> None:
        console.print(f"  [bold]-> {phase}[/bold]")

    try:
        result = run_autopilot(
            context,
            squad,
            root,
            settings.output_dir,
            max_cycles=max_cycles,
            debate_rounds=debate_rounds,
            build_limit=build_limit,
            min_completion=min_completion,
            run_tests=run_tests,
            run_security=run_security,
            llm_security=llm_security,
            dual_router=router,
            llm=llm,
            use_llm_build=True,
            parallel_git=settings.parallel_git if parallel_git is None else parallel_git,
            git_push=settings.git_push if git_push is None else git_push,
            fixed_tasks=fixed_tasks,
            on_cycle_start=_on_cycle,
            on_phase=_on_phase,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Autopilot stopped by user (Ctrl+C).[/yellow]")
        raise typer.Exit(130) from None
    except Exception as exc:
        console.print(f"[red]Autopilot failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="Autopilot cycles")
    table.add_column("Cycle", style="cyan")
    table.add_column("Consensus")
    table.add_column("Blockers", style="red")
    table.add_column("Built")
    table.add_column("Secure")
    table.add_column("Tests")
    for c in result.cycles:
        table.add_row(
            str(c.cycle_number),
            "[green]yes[/green]" if c.consensus_reached else "no",
            str(c.total_blockers),
            str(c.tasks_built),
            "[green]yes[/green]" if c.build_complete else "no",
            "[green]yes[/green]" if c.security_passed else "[red]no[/red]",
            "pass" if c.tests_passed else ("fail" if c.tests_passed is False else "—"),
        )
    console.print(table)

    if result.consensus_reached and result.build_complete and result.security_passed:
        console.print(Panel(
            f"[green]Mission complete — crew is satisfied.[/green]\n{result.stopped_reason}\n"
            f"Completion: {result.final_completion:.0f}%",
            title="Autopilot finished",
        ))
    else:
        console.print(Panel(
            f"[yellow]Stopped: {result.stopped_reason}[/yellow]\n"
            f"Completion: {result.final_completion:.0f}%",
            title="Autopilot stopped",
        ))

    log_dir = Path(settings.output_dir) / "autopilot"
    console.print(f"[dim]Autopilot logs:[/dim] {log_dir}")
    console.print(f"[dim]Progress log:[/dim] {log_path}")


@app.command()
def track(
    context_path: Optional[str] = typer.Option(None, "--context", help="Path to context JSON"),
    project_root: Optional[str] = typer.Option(None, "--root", help="Project root to scan"),
) -> None:
    """Run progress tracker on the current codebase."""
    settings.ensure_dirs()
    ctx_file = context_path or find_latest_context(settings.contexts_dir)
    if not ctx_file:
        console.print("[red]No saved context found.[/red]")
        raise typer.Exit(1)

    context = load_context(ctx_file)
    root = project_root or context.codebase_path or "."
    report = generate_progress_report(context, root)
    json_path, md_path = save_report(report, settings.reports_dir)

    console.print(Panel(render_report_markdown(report), title="Progress Report"))
    console.print(f"[green]Saved:[/green] {json_path}")
    console.print(f"[green]Saved:[/green] {md_path}")


@app.command()
def squad(
    squad_path: Optional[str] = typer.Option(None, "--squad", help="Path to squad JSON"),
) -> None:
    """Show the saved agent squad."""
    settings.ensure_dirs()
    sq_file = squad_path or find_latest_squad(settings.squads_dir)
    if not sq_file:
        console.print("[red]No squad found. Run autocrew import-context first.[/red]")
        raise typer.Exit(1)

    loaded = load_squad(sq_file)
    console.print(f"[dim]Squad file:[/dim] {sq_file}\n")
    _display_squad(loaded)


@app.command()
def metrics(
    backfill: bool = typer.Option(
        False,
        "--backfill",
        help="Import round/token estimates from saved debate_result.json files",
    ),
    metrics_dir: Optional[str] = typer.Option(
        None, "--metrics-dir", help="Metrics database directory"
    ),
) -> None:
    """Show debate/build cost, latency, and round-count statistics from persisted logs."""
    from autocrew.metrics.backfill import backfill_debate_from_results
    from autocrew.metrics.report import format_metrics_report, query_metrics

    settings.ensure_dirs()
    target_dir = metrics_dir or settings.metrics_dir

    if backfill:
        debate_files = list(Path(settings.output_dir).glob("debate/*/debate_result.json"))
        if not debate_files:
            console.print("[yellow]No debate_result.json files found to backfill.[/yellow]")
        else:
            ids = backfill_debate_from_results(debate_files, metrics_dir=target_dir)
            console.print(f"[green]Backfilled {len(ids)} debate session(s).[/green]")

    report_data = query_metrics(target_dir)
    console.print(format_metrics_report(report_data))
    console.print(f"\n[dim]Database:[/dim] {Path(target_dir) / 'session_metrics.db'}")
    console.print(f"[dim]JSONL:[/dim] {Path(target_dir) / 'agent_calls.jsonl'}")


@app.command()
def doctor() -> None:
    """Verify install, .env, and LLM routing (run before autopilot)."""
    import autocrew

    pkg = Path(autocrew.__file__).resolve().parent
    console.print(Panel(f"[bold]{pkg}[/bold]", title="autocrew package"))

    checks: list[tuple[str, bool, str]] = []

    try:
        import litellm  # noqa: F401

        checks.append(("litellm installed", True, "ok"))
    except ImportError:
        checks.append(("litellm installed", False, "pip install litellm"))

    checks.append(("LLM_FREE_TIER_CHAIN", settings.llm_free_tier_chain, str(settings.llm_free_tier_chain)))
    checks.append(("NVIDIA_API_KEY", bool(settings.nvidia_api_key.strip()), "set" if settings.nvidia_api_key.strip() else "missing"))
    checks.append(("GROQ_API_KEY", bool(settings.groq_api_key.strip()), "set" if settings.groq_api_key.strip() else "missing"))
    checks.append(("CEREBRAS_API_KEY", bool(settings.cerebras_api_key.strip()), "set" if settings.cerebras_api_key.strip() else "missing"))
    checks.append(("DEBATE_PER_AGENT_MODELS empty", not parse_per_agent_models(settings.debate_per_agent_models), "must be empty for role tiers"))

    try:
        router, _, mode = _resolve_llm_routing(settings.debate_dual_model)
        checks.append(("RoleModelRouter active", isinstance(router, RoleModelRouter), mode))
        if router:
            console.print(Panel(router.summary(), title=f"Routing: {mode}"))
    except Exception as exc:
        checks.append(("RoleModelRouter active", False, str(exc)))

    has_debate_v2 = (pkg / "analyzer" / "litellm_chain.py").is_file()
    checks.append(("free-tier chain code present", has_debate_v2, "litellm_chain.py" if has_debate_v2 else "OLD PACKAGE — reinstall"))

    table = Table(title="Doctor checks")
    table.add_column("Check")
    table.add_column("OK")
    table.add_column("Detail")
    all_ok = True
    for name, ok, detail in checks:
        if not ok:
            all_ok = False
        table.add_row(name, "[green]yes[/green]" if ok else "[red]no[/red]", detail)
    console.print(table)

    if all_ok:
        console.print("[green]Ready. Expect: Consultant phase -> 3 core debaters -> plan review.[/green]")
        console.print("[green]PO/Architect should use Kimi (reasoning), not Qwen coder.[/green]")
    else:
        console.print("[red]Fix failures above, then re-run autocrew doctor.[/red]")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show the latest progress report."""
    settings.ensure_dirs()
    report_path = find_latest_report(settings.reports_dir)
    if not report_path:
        console.print("[yellow]No progress reports found. Run autocrew track first.[/yellow]")
        raise typer.Exit(0)

    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    report = ProgressReport.from_dict(data)
    console.print(Panel(render_report_markdown(report), title="Latest Status"))


if __name__ == "__main__":
    app()
