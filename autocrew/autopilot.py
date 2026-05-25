"""Autopilot loop — debate → build → test → secure → repeat until done."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from autocrew.analyzer.project_model import ProjectContext
from autocrew.crew.crew_runner import run_crew
from autocrew.debate.debate_runner import build_tasks_from_debate, run_debate
from autocrew.debate.model_router import DualModelRouter
from autocrew.security_audit import run_llm_security_review, run_security_audit
from autocrew.squad.squad_model import Squad
from autocrew.storage import save_tasks
from autocrew.tasks.task_builder import merge_foundation_tasks
from autocrew.tasks.task_model import TaskConfig
from autocrew.tracker.progress_tracker import generate_progress_report


@dataclass
class AutopilotCycle:
    cycle_number: int
    consensus_reached: bool
    total_blockers: int
    tasks_built: int
    completion_pct: float
    tests_passed: bool | None
    security_passed: bool = True
    build_complete: bool = False
    debate_dir: str = ""


@dataclass
class AutopilotResult:
    project_name: str
    cycles: list[AutopilotCycle] = field(default_factory=list)
    consensus_reached: bool = False
    build_complete: bool = False
    security_passed: bool = False
    final_completion: float = 0.0
    tests_passed: bool | None = None
    stopped_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "consensus_reached": self.consensus_reached,
            "build_complete": self.build_complete,
            "security_passed": self.security_passed,
            "final_completion": self.final_completion,
            "tests_passed": self.tests_passed,
            "stopped_reason": self.stopped_reason,
            "cycles": [
                {
                    "cycle_number": c.cycle_number,
                    "consensus_reached": c.consensus_reached,
                    "total_blockers": c.total_blockers,
                    "tasks_built": c.tasks_built,
                    "completion_pct": c.completion_pct,
                    "tests_passed": c.tests_passed,
                    "security_passed": c.security_passed,
                    "build_complete": c.build_complete,
                    "debate_dir": c.debate_dir,
                }
                for c in self.cycles
            ],
        }


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def run_project_tests(project_root: str, timeout_seconds: int = 600) -> tuple[bool, str]:
    """Run pnpm/npm test if available. Returns (passed, message)."""
    root = Path(project_root)
    package_json = root / "package.json"
    if not package_json.is_file():
        return True, "skipped (no package.json)"

    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True, "skipped (invalid package.json)"

    scripts = data.get("scripts") or {}
    if "test" not in scripts:
        return True, "skipped (no test script)"

    for cmd in (["pnpm", "test"], ["npm", "test"], ["npx", "pnpm", "test"]):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-1000:]
            if proc.returncode == 0:
                return True, f"passed ({' '.join(cmd)})"
            return False, f"failed ({' '.join(cmd)}): {output[-500:]}"
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_seconds}s"

    return True, "skipped (pnpm/npm not found)"


def is_build_complete(context: ProjectContext, report_completion: float, min_completion: float) -> tuple[bool, str]:
    """True when MVP/high-priority scope is implemented."""
    high_open = [
        f.name
        for f in context.features
        if f.priority == "high" and f.status in ("not_started", "partial")
    ]
    if high_open:
        return False, f"high-priority features incomplete: {', '.join(high_open[:5])}"
    if context.missing_parts:
        return False, f"missing parts remain: {', '.join(context.missing_parts[:5])}"
    if min_completion > 0 and report_completion < min_completion:
        return False, f"completion {report_completion:.1f}% < {min_completion}%"
    return True, "MVP build complete"


def _is_mission_complete(
    *,
    consensus: bool,
    build_ok: bool,
    security_ok: bool,
    tests_passed: bool | None,
    require_tests: bool,
) -> tuple[bool, str]:
    """Crew stops only when built, secured, tested, and every agent approves."""
    if not consensus:
        return False, "crew not satisfied — debate blockers remain"
    if not build_ok:
        return False, "app not fully built yet"
    if not security_ok:
        return False, "security issues remain"
    if require_tests and tests_passed is False:
        return False, "tests failing"
    return True, "crew approved + app built + secured + tests OK"


def run_autopilot(
    context: ProjectContext,
    squad: Squad,
    project_root: str,
    output_dir: str,
    *,
    max_cycles: int = 50,
    debate_rounds: int = 1,
    build_limit: int = 5,
    min_completion: float = 100.0,
    run_tests: bool = True,
    run_security: bool = True,
    llm_security: bool = True,
    dual_router: DualModelRouter | None = None,
    llm=None,
    use_llm_build: bool = True,
    parallel_git: bool = True,
    git_push: bool = False,
    on_cycle_start: Callable[[int], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> AutopilotResult:
    """Run until the app is fully built, secured, tested, and the crew approves."""
    root = project_root or context.codebase_path or "."
    log_dir = Path(output_dir) / "autopilot" / _slug(context.project_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    result = AutopilotResult(project_name=context.project_name)

    for cycle in range(1, max_cycles + 1):
        if on_cycle_start:
            on_cycle_start(cycle)

        if on_phase:
            on_phase("debate (crew review)")
        if dual_router is not None:
            debate = run_debate(
                context,
                squad,
                root,
                output_dir,
                max_rounds=debate_rounds,
                dual_router=dual_router,
            )
        elif llm is not None:
            debate = run_debate(
                context,
                squad,
                root,
                output_dir,
                max_rounds=debate_rounds,
                llm=llm,
            )
        else:
            debate = run_debate(
                context,
                squad,
                root,
                output_dir,
                max_rounds=debate_rounds,
            )

        blockers = debate.rounds[-1].total_blockers if debate.rounds else 999
        debate_tasks = build_tasks_from_debate(debate, squad, context)
        tasks: list[TaskConfig] = merge_foundation_tasks(squad, context, debate_tasks)
        save_tasks(tasks, output_dir, context.project_name)

        tasks_built = 0
        if debate_tasks and not debate.consensus_reached:
            if on_phase:
                on_phase(f"build ({min(build_limit, len(tasks))} tasks)")
            if use_llm_build and (dual_router is not None or llm is not None):
                if dual_router is not None:
                    run_crew(
                        squad,
                        tasks,
                        context,
                        project_root=root,
                        use_llm=True,
                        dual_router=dual_router,
                        task_limit=build_limit,
                        parallel_git=parallel_git,
                        git_push=git_push,
                    )
                else:
                    run_crew(
                        squad,
                        tasks,
                        context,
                        project_root=root,
                        use_llm=True,
                        llm_call=llm.complete,
                        task_limit=build_limit,
                        parallel_git=parallel_git,
                        git_push=git_push,
                    )
            else:
                run_crew(
                    squad,
                    tasks,
                    context,
                    project_root=root,
                    task_limit=build_limit,
                    parallel_git=parallel_git,
                    git_push=git_push,
                )
            tasks_built = min(build_limit, len(tasks))

        report = generate_progress_report(context, root)

        tests_ok: bool | None = None
        test_msg = ""
        if run_tests:
            if on_phase:
                on_phase("tests")
            tests_ok, test_msg = run_project_tests(root)

        security_ok = True
        security_msg = "skipped"
        security_report = None
        if run_security:
            if on_phase:
                on_phase("security audit")
            security_report = run_security_audit(root)
            if llm_security and llm is not None:
                security_report = run_llm_security_review(
                    root, context.project_name, security_report, llm.complete
                )
            elif llm_security and dual_router is not None:
                reviewer = next((a for a in squad.agents if a.role.value == "code_reviewer"), squad.agents[0])
                llm_client, _ = dual_router.for_agent(reviewer)
                security_report = run_llm_security_review(
                    root, context.project_name, security_report, llm_client.complete
                )
            security_ok = security_report.passed
            security_msg = security_report.summary
            sec_path = log_dir / f"cycle-{cycle}-security.json"
            sec_path.write_text(json.dumps(security_report.to_dict(), indent=2), encoding="utf-8")

        build_ok, build_msg = is_build_complete(context, report.completion_percentage, min_completion)

        cycle_record = AutopilotCycle(
            cycle_number=cycle,
            consensus_reached=debate.consensus_reached,
            total_blockers=blockers,
            tasks_built=tasks_built,
            completion_pct=report.completion_percentage,
            tests_passed=tests_ok,
            security_passed=security_ok,
            build_complete=build_ok,
            debate_dir=debate.debate_dir,
        )
        result.cycles.append(cycle_record)

        cycle_log = log_dir / f"cycle-{cycle}.json"
        cycle_log.write_text(
            json.dumps(
                {
                    **cycle_record.__dict__,
                    "test_message": test_msg,
                    "security_message": security_msg,
                    "build_message": build_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        complete, reason = _is_mission_complete(
            consensus=debate.consensus_reached,
            build_ok=build_ok,
            security_ok=security_ok,
            tests_passed=tests_ok,
            require_tests=run_tests,
        )
        if complete:
            result.consensus_reached = True
            result.build_complete = True
            result.security_passed = True
            result.final_completion = report.completion_percentage
            result.tests_passed = tests_ok
            result.stopped_reason = reason
            break

        if cycle >= max_cycles:
            result.final_completion = report.completion_percentage
            result.tests_passed = tests_ok
            result.security_passed = security_ok
            result.build_complete = build_ok
            result.stopped_reason = (
                f"max cycles ({max_cycles}) — blockers={blockers}, "
                f"build={build_msg}, security={security_msg}"
            )
            break
    else:
        result.stopped_reason = "max cycles exhausted"

    if result.cycles and not result.final_completion:
        last = result.cycles[-1]
        result.final_completion = last.completion_pct
        result.tests_passed = last.tests_passed
        result.security_passed = last.security_passed
        result.build_complete = last.build_complete

    summary_path = log_dir / "autopilot_result.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
