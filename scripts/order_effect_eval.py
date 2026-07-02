#!/usr/bin/env python3
"""Step 7 CLI — Run the order-effect experiment.

Usage:
    python scripts/order_effect_eval.py [--tasks-dir output] [--output output/eval/order_effect]

Runs debate on the same tasks with default dev-role order vs randomized order
within the dev-adjacent tier, and compares concern counts and content.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import typer

from autocrew.config import settings
from autocrew.debate.model_diversity_eval import EvalTask
from autocrew.debate.order_effect_eval import (
    run_order_effect_eval,
    save_order_effect_report,
)
from autocrew.storage import find_latest_context, find_latest_squad, load_context, load_squad

app = typer.Typer(name="order-effect-eval", help="Step 7: order-effect experiment")


def _load_eval_tasks(tasks_dir: str, limit: int) -> list[EvalTask]:
    """Load contexts + squads from saved files as eval tasks."""
    tasks: list[EvalTask] = []
    contexts_dir = Path(tasks_dir) / "contexts"
    squads_dir = Path(tasks_dir) / "squads"

    ctx_files = sorted(contexts_dir.glob("*.json")) if contexts_dir.is_dir() else []
    sq_files = sorted(squads_dir.glob("*.json")) if squads_dir.is_dir() else []

    if not ctx_files or not sq_files:
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


@app.command()
def run(
    tasks_dir: str = typer.Option("./output", "--tasks-dir", help="Directory with contexts/ and squads/ subdirs"),
    output: str = typer.Option("./output/eval/order_effect", "--output", "-o", help="Output path for report"),
    limit: int = typer.Option(15, "--limit", "-n", help="Max tasks to evaluate"),
    rounds: int = typer.Option(2, "--rounds", "-r", help="Max debate rounds per task"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible shuffling"),
) -> None:
    """Run the order-effect experiment: default order vs randomized dev-role order."""
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
    typer.echo(f"Running {len(tasks)} task(s) x 2 orderings = {len(tasks) * 2} debate runs...")

    from autocrew.analyzer.llm_client import create_llm_client

    llm = create_llm_client(
        anthropic_key=settings.anthropic_api_key,
        openai_key=settings.openai_api_key,
        nvidia_key=settings.nvidia_api_key,
        zenmux_key=settings.zenmux_api_key,
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

    report = run_order_effect_eval(
        tasks,
        llm_call=llm.complete,
        metrics_dir=settings.metrics_dir,
        seed=seed,
    )

    md_path = save_order_effect_report(report, output)
    typer.echo(f"\n✓ Report saved: {md_path}")
    typer.echo(f"  JSON: {Path(output).with_suffix('.json')}")
    typer.echo(f"\n{report.conclusion}")


if __name__ == "__main__":
    app()