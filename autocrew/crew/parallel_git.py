"""Parallel dev workflow: branch per agent → review → merge → conflict fix."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from autocrew.analyzer.llm_client import call_with_json_retry
from autocrew.analyzer.project_model import ProjectContext
from autocrew.crew.crew_logger import CrewLogger
from autocrew.crew.llm_task_executor import execute_task_with_llm
from autocrew.debate.model_router import DualModelRouter
from autocrew.squad.squad_model import AgentConfig, AgentRole, Squad
from autocrew.tasks.task_model import TaskConfig
from autocrew.tools.git_tools import (
    GitError,
    MergeBatchResult,
    git_branch_diff_stat,
    git_commit,
    git_create_worktree,
    git_ensure_initial_commit,
    git_init,
    git_merge_branch,
    git_push_branch,
    git_remove_worktree,
    git_resolve_base_branch,
)


REVIEW_PROMPT = """You are {reviewer_name}, code reviewer for {project_name}.

Review this developer branch before merge.

Developer role: {role}
Branch: {branch}

Diff stat (base...feature):
{diff_stat}

Return JSON:
{{
  "approved": true or false,
  "blockers": ["must-fix before merge"],
  "summary": "one sentence"
}}

Approve only if the change set looks safe, scoped, and ready to merge.
Return only valid JSON.
"""

CONFLICT_FIX_PROMPT = """You are {agent_name} ({role}) resolving git merge conflicts for {project_name}.

Branches being merged: {branches}

Conflicted files (with markers):
{conflict_content}

Return JSON:
{{
  "files": [
    {{"path": "relative/path", "content": "resolved full file content without conflict markers"}}
  ],
  "summary": "how conflicts were resolved"
}}

Remove all <<<<<<< ======= >>>>>>> markers. Preserve correct combined logic.
Return only valid JSON.
"""


@dataclass
class DevBranch:
    role: str
    branch: str
    worktree_path: str
    agent: AgentConfig


def _session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _branch_name(session: str, role: str) -> str:
    return f"autocrew/{session}/{role.replace('_', '-')}"


def _worktree_path(project_root: str, session: str, role: str) -> str:
    return str(Path(project_root) / ".autocrew" / "worktrees" / session / role)


def _review_branch(
    reviewer: AgentConfig,
    context: ProjectContext,
    project_root: str,
    base_branch: str,
    dev: DevBranch,
    llm_call: Callable[[str], str] | None,
) -> tuple[bool, str]:
    diff_stat = git_branch_diff_stat(project_root, base_branch, dev.branch)
    if not diff_stat.strip() or diff_stat.startswith("(diff unavailable"):
        return True, "no changes to review"

    if llm_call is None:
        return True, "approved (no LLM reviewer)"

    prompt = REVIEW_PROMPT.format(
        reviewer_name=reviewer.name,
        project_name=context.project_name,
        role=dev.role,
        branch=dev.branch,
        diff_stat=diff_stat[:8000],
    )
    try:
        data = call_with_json_retry(llm_call, prompt)
        approved = bool(data.get("approved", False)) and not data.get("blockers")
        summary = str(data.get("summary", ""))
        if data.get("blockers"):
            summary += " | blockers: " + "; ".join(str(b) for b in data["blockers"][:3])
        return approved, summary
    except Exception as exc:
        return False, f"review failed: {exc}"


def _pick_conflict_fixer(squad: Squad, roles: list[str]) -> AgentConfig:
    priority = [
        AgentRole.BACKEND_DEV,
        AgentRole.FULLSTACK_DEV,
        AgentRole.FRONTEND_DEV,
        AgentRole.DEVOPS,
        AgentRole.DATA_ENGINEER,
        AgentRole.AI_ENGINEER,
    ]
    by_role = {a.role: a for a in squad.agents}
    for role in priority:
        if role.value in roles and role in by_role:
            return by_role[role]
    for role in roles:
        agent = next((a for a in squad.agents if a.role.value == role), None)
        if agent:
            return agent
    return squad.agents[0]


def _find_conflict_files(project_root: str) -> list[str]:
    root = Path(project_root)
    found: list[str] = []
    for rel in root.rglob("*"):
        if not rel.is_file():
            continue
        if ".git" in rel.parts or "node_modules" in rel.parts or ".autocrew" in rel.parts:
            continue
        try:
            if "<<<<<<<" in rel.read_text(encoding="utf-8", errors="ignore"):
                found.append(rel.relative_to(root).as_posix())
        except OSError:
            continue
    return found


def _resolve_conflicts(
    project_root: str,
    fixer: AgentConfig,
    context: ProjectContext,
    conflict_files: list[str],
    branches: list[str],
    logger: CrewLogger,
    llm_call: Callable[[str], str] | None,
) -> bool:
    if not conflict_files or llm_call is None:
        return False

    parts: list[str] = []
    for rel in conflict_files[:10]:
        path = Path(project_root) / rel
        if path.is_file():
            parts.append(f"--- {rel} ---\n{path.read_text(encoding='utf-8', errors='ignore')[:12000]}")

    prompt = CONFLICT_FIX_PROMPT.format(
        agent_name=fixer.name,
        role=fixer.role.value,
        project_name=context.project_name,
        branches=", ".join(branches),
        conflict_content="\n\n".join(parts)[:20000],
    )

    task = TaskConfig(
        task_id="merge_conflict_fix",
        title="Resolve merge conflicts",
        description=prompt,
        assigned_agent_role=fixer.role.value,
        output_format="code",
    )
    try:
        execute_task_with_llm(task, fixer, context, project_root, logger, llm_call)
        git_commit(project_root, f"[autocrew] {fixer.role.value}: resolve merge conflicts")
        return True
    except Exception as exc:
        logger.log(f"Conflict resolution failed: {exc}")
        return False


async def run_parallel_group_with_git(
    roles: list[str],
    squad: Squad,
    tasks: list[TaskConfig],
    context: ProjectContext,
    project_root: str,
    logger: CrewLogger,
    max_retries: int,
    *,
    run_phase_sequential: Callable[..., list[str]],
    use_llm: bool,
    dual_router: DualModelRouter | None,
    llm_call: Callable[[str], str] | None,
    task_filter: set[str] | None,
    on_task_start: Callable | None = None,
    on_task_done: Callable | None = None,
    git_push: bool = False,
) -> tuple[list[str], MergeBatchResult | None]:
    """Run dev roles in parallel on isolated git worktrees, then review and merge."""
    from autocrew.crew.crew_runner import _tasks_for_role

    root = Path(project_root).resolve()
    git_init(str(root))

    try:
        git_ensure_initial_commit(str(root))
        base_branch = git_resolve_base_branch(str(root))
    except GitError as exc:
        logger.log(f"Parallel git unavailable: {exc}")
        return [], None

    session = _session_id()
    dev_branches: list[DevBranch] = []

    for role in roles:
        agent = next((a for a in squad.agents if a.role.value == role), None)
        if not agent:
            continue
        branch = _branch_name(session, role)
        wt = _worktree_path(str(root), session, role)
        try:
            git_create_worktree(str(root), branch, wt, base_branch)
            dev_branches.append(DevBranch(role=role, branch=branch, worktree_path=wt, agent=agent))
            logger.log(f"Branch {branch} → worktree {wt}")
        except Exception as exc:
            logger.log(f"Worktree failed for {role}: {exc}")

    if not dev_branches:
        return [], None

    async def run_on_worktree(dev: DevBranch) -> list[str]:
        role_tasks = _tasks_for_role(tasks, dev.role)
        if task_filter is not None:
            role_tasks = [t for t in role_tasks if t.task_id in task_filter]
        if not role_tasks:
            return []

        results = await asyncio.to_thread(
            run_phase_sequential,
            [dev.role],
            squad,
            tasks,
            context,
            dev.worktree_path,
            logger,
            max_retries,
            use_llm=use_llm,
            dual_router=dual_router,
            llm_call=llm_call,
            task_filter=task_filter,
            on_task_start=on_task_start,
            on_task_done=on_task_done,
            skip_git_commit=True,
        )
        git_commit(dev.worktree_path, f"[autocrew] {dev.role}: parallel work")
        return results

    all_results: list[str] = []
    parallel_results = await asyncio.gather(*[run_on_worktree(d) for d in dev_branches])
    for batch in parallel_results:
        all_results.extend(batch)

    reviewer = next((a for a in squad.agents if a.role == AgentRole.CODE_REVIEWER), None)
    reviewer_llm = llm_call
    if dual_router and reviewer:
        client, _ = dual_router.for_agent(reviewer)
        reviewer_llm = client.complete

    merge_result = MergeBatchResult(base_branch=base_branch)
    approved_devs: list[DevBranch] = []

    for dev in dev_branches:
        if reviewer is None:
            approved, summary = True, "auto-approved (no reviewer)"
        else:
            approved, summary = _review_branch(
                reviewer, context, str(root), base_branch, dev, reviewer_llm
            )
        logger.log(f"Review {dev.branch}: {'APPROVED' if approved else 'REJECTED'} — {summary}")
        if approved:
            merge_result.approved.append(dev.branch)
            approved_devs.append(dev)

    for dev in approved_devs:
        if git_push:
            msg = git_push_branch(str(root), dev.branch)
            merge_result.push_messages.append(msg)
            logger.log(msg)

    conflict_branches: list[str] = []
    for dev in approved_devs:
        attempt = git_merge_branch(str(root), base_branch, dev.branch)
        attempt.role = dev.role
        merge_result.merged.append(attempt)
        logger.log(attempt.message)
        if attempt.had_conflicts:
            conflict_branches.append(dev.branch)
            merge_result.conflicts_on.append(dev.branch)

    if conflict_branches:
        fixer = _pick_conflict_fixer(squad, roles)
        merge_result.conflict_fixer_role = fixer.role.value
        logger.log(f"Conflicts on {conflict_branches} — assigned to {fixer.name} only")
        conflict_files = _find_conflict_files(str(root))
        fixer_llm = llm_call
        if dual_router:
            client, _ = dual_router.for_agent(fixer)
            fixer_llm = client.complete
        if _resolve_conflicts(
            str(root), fixer, context, conflict_files, conflict_branches, logger, fixer_llm
        ):
            if git_push:
                merge_result.push_messages.append(git_push_branch(str(root), base_branch))

    for dev in dev_branches:
        try:
            git_remove_worktree(str(root), dev.worktree_path)
        except Exception as exc:
            logger.log(f"Worktree cleanup: {exc}")

    return all_results, merge_result
