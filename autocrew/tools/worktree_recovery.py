"""Recover unmerged agent work from orphaned .autocrew/worktrees directories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from autocrew.crew.crew_logger import CrewLogger
from autocrew.tools.git_tools import (
    GitError,
    MergeAttempt,
    git_branch_diff_stat,
    git_commit,
    git_commit_succeeded,
    git_commits_ahead,
    git_diff_shortstat,
    git_merge_branch,
    git_resolve_base_branch,
    git_worktree_branch,
    git_worktree_porcelain,
)


@dataclass
class WorktreeCandidate:
    session_id: str
    agent_role: str
    path: str
    branch: str
    dirty: bool = False
    ahead: int = 0
    insertions: int = 0
    diff_stat: str = ""


@dataclass
class WorktreeRecoveryResult:
    base_branch: str
    discovered: int = 0
    committed: list[str] = field(default_factory=list)
    merged: list[MergeAttempt] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def _parse_insertions(diff_stat: str) -> int:
    total = 0
    for line in diff_stat.splitlines():
        match = re.search(r"(\d+)\s+insertion", line)
        if match:
            total += int(match.group(1))
    return total


def _is_worktree(path: Path) -> bool:
    return path.is_dir() and (path / ".git").exists()


def discover_worktrees(project_root: str) -> list[WorktreeCandidate]:
    """List agent worktrees under .autocrew/worktrees/{session}/{role}."""
    root = Path(project_root).resolve()
    wt_root = root / ".autocrew" / "worktrees"
    if not wt_root.is_dir():
        return []

    candidates: list[WorktreeCandidate] = []
    for session_dir in sorted(wt_root.iterdir()):
        if not session_dir.is_dir():
            continue
        for agent_dir in sorted(session_dir.iterdir()):
            if not agent_dir.is_dir() or not _is_worktree(agent_dir):
                continue
            branch = git_worktree_branch(str(agent_dir))
            if not branch or branch == "HEAD":
                continue
            candidates.append(
                WorktreeCandidate(
                    session_id=session_dir.name,
                    agent_role=agent_dir.name,
                    path=str(agent_dir.resolve()),
                    branch=branch,
                    dirty=bool(git_worktree_porcelain(str(agent_dir))),
                )
            )
    return candidates


def _refresh_candidate_stats(
    candidate: WorktreeCandidate,
    project_root: str,
    base_branch: str,
) -> WorktreeCandidate:
    candidate.dirty = bool(git_worktree_porcelain(candidate.path))
    candidate.ahead = git_commits_ahead(candidate.path, base_branch)
    candidate.diff_stat = git_branch_diff_stat(project_root, base_branch, candidate.branch)
    shortstat = git_diff_shortstat(project_root, base_branch, candidate.branch)
    candidate.insertions = _parse_insertions(f"{candidate.diff_stat}\n{shortstat}")
    return candidate


def recover_worktrees(
    project_root: str,
    logger: CrewLogger,
    *,
    max_merges: int = 10,
    min_insertions: int = 5,
    merge: bool = True,
) -> WorktreeRecoveryResult | None:
    """
    Commit pending changes in orphaned worktrees and merge valuable branches into base.

    Scans .autocrew/worktrees for dirty trees or branches ahead of base, commits
    uncommitted work, then merges the largest diffs first (up to max_merges).

    When merge=False (dry run), nothing is written — only discovery and logging.
    """
    root = str(Path(project_root).resolve())
    try:
        base_branch = git_resolve_base_branch(root)
    except GitError as exc:
        logger.log(f"Worktree recovery skipped: {exc}")
        return None

    result = WorktreeRecoveryResult(base_branch=base_branch)
    candidates = discover_worktrees(root)
    result.discovered = len(candidates)
    if not candidates:
        logger.log("Worktree recovery: no orphaned worktrees found")
        return result

    logger.log(f"Worktree recovery: scanning {len(candidates)} worktree(s)")

    for candidate in candidates:
        if not candidate.dirty:
            continue
        if not merge:
            result.skipped.append(
                f"{candidate.branch}: uncommitted changes (dry run — not committed)"
            )
            logger.log(
                f"Recovery dry run: would commit {candidate.branch} "
                f"({candidate.session_id}/{candidate.agent_role})"
            )
            continue
        message = (
            f"[autocrew] recover: {candidate.session_id}/{candidate.agent_role}"
        )
        commit_msg = git_commit(candidate.path, message)
        if git_commit_succeeded(commit_msg):
            result.committed.append(f"{candidate.branch} ({candidate.session_id}/{candidate.agent_role})")
            logger.log(f"Recovery commit: {candidate.branch} — {commit_msg}")
        else:
            result.skipped.append(f"{candidate.branch}: commit failed ({commit_msg})")
            logger.log(f"Recovery commit skipped for {candidate.branch}: {commit_msg}")

    mergeable: list[WorktreeCandidate] = []
    seen_branches: set[str] = set()
    for candidate in candidates:
        refreshed = _refresh_candidate_stats(candidate, root, base_branch)
        if refreshed.branch in seen_branches:
            continue
        seen_branches.add(refreshed.branch)

        # In dry run, include dirty trees that would gain a commit in the diff estimate
        effective_insertions = refreshed.insertions
        if not merge and refreshed.dirty and refreshed.insertions < min_insertions:
            effective_insertions = min_insertions

        if not refreshed.diff_stat.strip() or refreshed.diff_stat.startswith("(diff unavailable"):
            if refreshed.ahead <= 0 and not refreshed.dirty:
                continue
            if not merge and refreshed.dirty:
                mergeable.append(refreshed)
                continue
            result.skipped.append(f"{refreshed.branch}: no diff vs {base_branch}")
            continue

        if effective_insertions < min_insertions:
            result.skipped.append(
                f"{refreshed.branch}: below min insertions ({refreshed.insertions} < {min_insertions})"
            )
            continue

        mergeable.append(refreshed)

    mergeable.sort(key=lambda c: c.insertions, reverse=True)

    if not merge:
        logger.log(f"Worktree recovery: {len(mergeable)} branch(es) mergeable (dry run)")
        for candidate in mergeable[:max_merges]:
            logger.log(
                f"  would merge {candidate.branch} (+{candidate.insertions} lines, "
                f"{candidate.session_id}/{candidate.agent_role})"
            )
        return result

    for candidate in mergeable[:max_merges]:
        attempt = git_merge_branch(root, base_branch, candidate.branch)
        attempt.role = candidate.agent_role
        result.merged.append(attempt)
        if attempt.merged:
            logger.log(
                f"Recovery merge: {attempt.message} "
                f"(+{candidate.insertions} lines from {candidate.session_id}/{candidate.agent_role})"
            )
        elif attempt.had_conflicts:
            result.conflicts.append(candidate.branch)
            logger.log(f"Recovery merge conflict: {candidate.branch} — skipped")
        else:
            result.skipped.append(f"{candidate.branch}: {attempt.message}")
            logger.log(f"Recovery merge failed: {candidate.branch} — {attempt.message}")

    logger.log(
        "Worktree recovery complete: "
        f"committed={len(result.committed)}, merged={sum(1 for m in result.merged if m.merged)}, "
        f"conflicts={len(result.conflicts)}, skipped={len(result.skipped)}"
    )
    return result
