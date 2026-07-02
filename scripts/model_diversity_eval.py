#!/usr/bin/env python3
"""Step 6 CLI — Run the model diversity experiment.

Usage:
    python scripts/model_diversity_eval.py [--tasks-dir output/contexts] [--output output/eval]

Requires API keys for both model families. Loads saved contexts/squads and runs
debate twice per task: once all-Kimi (baseline), once with Drew+Jordan on a
different model family (cross-model).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import typer

from autocrew.analyzer.llm_client import create_llm_client, create_model_client
from autocrew.analyzer.project_model import ProjectContext
from autocrew.config import settings
from autocrew.debate.model_diversity_eval import (
    EvalTask,
    run_model_diversity_eval,
    save_eval_report,
)
from autocrew.debate.model_router import DualModelRouter
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import Squad
from autocrew.storage import find_latest_context, find_latest_squad, load_context, load_squad

app = typer.Typer(name="model-diversity-eval", help="Step 6: cross-model critique experiment")


def _load_eval_tasks(tasks_dir: str, limit: int) -> list[EvalTask]:
    """Load contexts + squads from saved files as eval tasks."""
    tasks: list[EvalTask] = []
    contexts_dir = Path(tasks_dir) / "contexts"
    squads_dir = Path(tasks_dir) / "squads"

    ctx_files = sorted(contexts_dir.glob("*.json")) if contexts_dir.is_dir() else []
    sq_files = sorted(squads_dir.glob("*.json")) if squads_dir.is_dir() else []

    if not ctx_files or not sq_files:
        # Fallback: use latest single context/squad
        ctx_file = find_latest_context(settings.contexts_dir)
        sq_file = find_latest_squad(settings.squads_dir)
        if ctx_file and sq_file:
            ctx_files = [Path(ctx_file)]
            sq_files = [Path(sq_file)]

    for i, (ctx_file, sq_file) in enumerate(zip(ctx_files, sq_files)):
        if i >= limit:
            break
        try:
            context = load_context(str(ctx_file))
            squad = load_squad(str(sq_file))
            tasks.append(
                EvalTask(
                    task_id=f"task_{i+1}_{context.project_name.replace(' ', '_').lower()[:30]}",
                    context=context,
                    squad=squad,
                    project_root=context.codebase_path or ".",
                    max_rounds=2,
                )
            )
        except Exception as exc:
            typer.echo(f"  ⚠ Failed to load {ctx_file.name}: {exc}", err=True)

    return tasks


def _get_baseline_llm():
    """Create the all-Kimi baseline LLM client."""
    kwargs = {
        "anthropic_key": settings.anthropic_api_key,
        "openai_key": settings.openai_api_key,
        "nvidia_key": settings.nvidia_api_key,
        "zenmux_key": settings.zenmux_api_key,
        "default_model": settings.fallback_llm,
        "fallback_model": settings.fallback_llm,
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
    llm = create_llm_client(**kwargs)
    return llm


def _get_cross_model_router() -> DualModelRouter | None:
    """Create the cross-model router (Drew+Jordan on different model)."""
    planning_model = settings.debate_planning_model or settings.fallback_llm
    implementation_model = settings.debate_implementation_model or settings.default_llm
    if planning_model == implementation_model:
        typer.echo("  ⚠ Planning and implementation models are identical — no cross-model test.", err=True)
        return None

    kwargs = {
        "anthropic_key": settings.anthropic_api_key,
        "openai_key": settings.openai_api_key,
        "nvidia_key": settings.nvidia_api_key,
        "zenmux_key": settings.zenmux_api_key,
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
    planning_base = create_model_client(planning_model, **kwargs)
    implementation_base = create_model_client(implementation_model, **kwargs)
    return DualModelRouter(
        planning_llm=create_model_client(planning_model, **kwargs, fallback=implementation_base),
        implementation_llm=create_model_client(implementation_model, **kwargs, fallback=planning_base),
        planning_model=planning_model,
        implementation_model=implementation_model,
    )


@app.command()
def run(
    tasks_dir: str = typer.Option("./output", "--tasks-dir", help="Directory with contexts/ and squads/ subdirs"),
    output: str = typer.Option("./output/eval/model_diversity", "--output", "-o", help="Output path for report"),
    limit: int = typer.Option(15, "--limit", "-n", help="Max tasks to evaluate"),
    rounds: int = typer.Option(2, "--rounds", "-r", help="Max debate rounds per task"),
) -> None:
    """Run the model diversity experiment: all-Kimi baseline vs cross-model Drew+Jordan."""
    settings.ensure_dirs()

    if not settings.has_api_keys():
        typer.echo("Error: No LLM API keys configured. Set .env variables.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading up to {limit} tasks from {tasks_dir}...")
    tasks = _load_eval_tasks(tasks_dir, limit)
    if not tasks:
        typer.echo("Error: No tasks found. Run autocrew import-context first.", err=True)
        raise typer.Exit(1)

    for t in tasks:
        t.max_rounds = rounds

    typer.echo(f"Loaded {len(tasks)} task(s).")
    typer.echo("Setting up baseline (all-Kimi) LLM...")
    baseline_llm = _get_baseline_llm()

    typer.echo("Setting up cross-model router (Drew+Jordan on different model)...")
    router = _get_cross_model_router()
    if router:
        typer.echo(f"  Planning model: {router.planning_model}")
        typer.echo(f"  Implementation model: {router.implementation_model}")
    else:
        typer.echo("  ⚠ No router — both conditions will use the same model.", err=True)

    typer.echo(f"\nRunning {len(tasks)} task(s) x 2 conditions = {len(tasks) * 2} debate runs...")
    report = run_model_diversity_eval(
        tasks,
        baseline_llm_call=baseline_llm.complete,
        cross_model_router=router,
        cross_model_llm_call=baseline_llm.complete if not router else None,
        metrics_dir=settings.metrics_dir,
    )

    md_path = save_eval_report(report, output)
    typer.echo(f"\n✓ Report saved: {md_path}")
    typer.echo(f"  JSON: {Path(output).with_suffix('.json')}")
    typer.echo(f"\n{report.recommendation}")


if __name__ == "__main__":
    app()