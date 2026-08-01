"""Headless recovery: merge latest autocrew agent branches from origin."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from autocrew.tools.git_tools import (
    GitError,
    MergeAttempt,
    _run_git,
    git_commit,
    git_commit_succeeded,
    git_merge_branch,
    git_push_branch,
    git_resolve_base_branch,
)
from autocrew.tools.secret_sanitizer import sanitize_project

SESSION_RE = re.compile(r"autocrew/(\d{8}_\d{6})/([^/]+)$")


@dataclass
class BranchRecoveryResult:
    base_branch: str
    merged: list[MergeAttempt] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    sanitized: list[str] = field(default_factory=list)
    push_messages: list[str] = field(default_factory=list)


def _fetch(project_root: str, remote: str = "origin") -> str:
    proc = _run_git(project_root, "fetch", remote, "--prune")
    if proc.returncode != 0:
        raise GitError((proc.stderr or proc.stdout or "git fetch failed").strip())
    return f"Fetched {remote}"


def list_remote_autocrew_branches(project_root: str, remote: str = "origin") -> dict[str, list[str]]:
    """Map agent role slug -> branch names sorted newest session first."""
    proc = _run_git(project_root, "branch", "-r", "--list", f"{remote}/autocrew/*/*")
    if proc.returncode != 0:
        return {}

    by_role: dict[str, list[tuple[str, str]]] = {}
    for line in proc.stdout.splitlines():
        ref = line.strip().removeprefix(f"{remote}/")
        match = SESSION_RE.match(ref)
        if not match:
            continue
        session, role = match.group(1), match.group(2)
        by_role.setdefault(role, []).append((session, ref))

    return {
        role: [branch for _, branch in sorted(entries, reverse=True)]
        for role, entries in by_role.items()
    }


def pick_branches(
    catalog: dict[str, list[str]],
    roles: list[str],
    *,
    session: str | None = None,
) -> list[str]:
    """Pick one branch per role (latest session unless session is set)."""
    picked: list[str] = []
    for role in roles:
        role_slug = role.replace("_", "-")
        branches = catalog.get(role_slug, [])
        if not branches:
            continue
        if session:
            match = next((b for b in branches if f"/{session}/" in b), None)
            if match:
                picked.append(match)
            continue
        picked.append(branches[0])
    return picked


def recover_agent_branches(
    project_root: str,
    roles: list[str],
    *,
    session: str | None = None,
    remote: str = "origin",
    merge: bool = True,
    push: bool = False,
    sanitize_secrets: bool = True,
    prefer_incoming: bool = True,
) -> BranchRecoveryResult:
    """
    Fetch origin, merge latest autocrew branches for the given roles into base,
    sanitize .env.example secrets, optionally push base branch.
    """
    root = str(Path(project_root).resolve())
    _fetch(root, remote)
    base = git_resolve_base_branch(root)
    result = BranchRecoveryResult(base_branch=base)

    catalog = list_remote_autocrew_branches(root, remote)
    branches = pick_branches(catalog, roles, session=session)
    if not branches:
        result.skipped.append("No matching autocrew branches on remote")
        return result

    checkout = _run_git(root, "checkout", base)
    if checkout.returncode != 0:
        raise GitError((checkout.stderr or checkout.stdout or "checkout failed").strip())

    for branch in branches:
        if not merge:
            result.skipped.append(f"dry-run would merge {branch}")
            continue
        if prefer_incoming:
            attempt = _merge_prefer_theirs(root, base, branch)
        else:
            attempt = git_merge_branch(root, base, branch)
        result.merged.append(attempt)
        if not attempt.merged and attempt.had_conflicts:
            _run_git(root, "merge", "--abort")

    if sanitize_secrets:
        result.sanitized = sanitize_project(root)
        if result.sanitized:
            msg = git_commit(root, "[autocrew] chore: sanitize placeholder secrets in env examples")
            if git_commit_succeeded(msg):
                result.sanitized.append(msg)

    if push:
        result.push_messages.append(git_push_branch(root, base, remote=remote))
        for branch in branches:
            # Re-push agent branches after sanitizing worktrees isn't needed; base is what matters.
            pass

    return result


def _merge_prefer_theirs(project_root: str, base_branch: str, feature_branch: str) -> MergeAttempt:
    """Merge feature branch, preferring incoming changes on conflict."""
    checkout = _run_git(project_root, "checkout", base_branch)
    if checkout.returncode != 0:
        return MergeAttempt(
            branch=feature_branch,
            role="",
            merged=False,
            had_conflicts=False,
            message=checkout.stderr or checkout.stdout or "checkout failed",
        )

    proc = _run_git(
        project_root,
        "merge",
        feature_branch,
        "--no-ff",
        "-X",
        "theirs",
        "-m",
        f"[autocrew] recover merge {feature_branch}",
    )
    if proc.returncode == 0:
        return MergeAttempt(
            branch=feature_branch,
            role="",
            merged=True,
            had_conflicts=False,
            message=f"Merged {feature_branch} into {base_branch} (-X theirs)",
        )

    err = (proc.stderr or proc.stdout or "").lower()
    had_conflicts = "conflict" in err
    _run_git(project_root, "merge", "--abort")
    return MergeAttempt(
        branch=feature_branch,
        role="",
        merged=False,
        had_conflicts=had_conflicts,
        message=(proc.stderr or proc.stdout or "merge failed").strip(),
    )
