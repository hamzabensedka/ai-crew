"""Git operations for agent checkpoints, branches, worktrees, and merges."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class GitError(Exception):
    pass


@dataclass
class MergeAttempt:
    branch: str
    role: str
    merged: bool
    had_conflicts: bool
    message: str = ""


@dataclass
class MergeBatchResult:
    base_branch: str
    approved: list[str] = field(default_factory=list)
    merged: list[MergeAttempt] = field(default_factory=list)
    conflicts_on: list[str] = field(default_factory=list)
    conflict_fixer_role: str = ""
    push_messages: list[str] = field(default_factory=list)


def _repo(project_root: str):
    try:
        import git
    except ImportError as exc:
        raise GitError("GitPython is required for parallel git workflow") from exc
    path = Path(project_root).resolve()
    if not (path / ".git").exists():
        raise GitError(f"No git repository at {path}")
    return git.Repo(str(path))


def git_init(project_root: str) -> str:
    try:
        import git

        repo_path = Path(project_root)
        if (repo_path / ".git").exists():
            return "Git repo already initialized"
        git.Repo.init(repo_path)
        return "Git repo initialized"
    except Exception as exc:
        return f"Git init failed: {exc}"


def git_default_branch(project_root: str) -> str:
    try:
        repo = _repo(project_root)
        if repo.head.is_detached:
            return "main"
        return repo.active_branch.name
    except GitError:
        return "main"


def git_commit(project_root: str, message: str) -> str:
    try:
        repo = _repo(project_root)
        repo.git.add(A=True)
        if not repo.is_dirty(untracked_files=True):
            return "Nothing to commit"
        repo.index.commit(message)
        return f"Committed: {message}"
    except Exception as exc:
        return f"Git commit failed: {exc}"


def git_branch(project_root: str, branch_name: str) -> str:
    try:
        repo = _repo(project_root)
        if branch_name in [h.name for h in repo.heads]:
            repo.heads[branch_name].checkout()
            return f"Switched to existing branch: {branch_name}"
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        return f"Created and checked out branch: {branch_name}"
    except Exception as exc:
        return f"Git branch failed: {exc}"


def git_branch_diff_stat(project_root: str, base_branch: str, feature_branch: str) -> str:
    try:
        repo = _repo(project_root)
        return repo.git.diff("--stat", f"{base_branch}...{feature_branch}")
    except Exception as exc:
        return f"(diff unavailable: {exc})"


def git_create_worktree(
    project_root: str,
    branch: str,
    worktree_path: str,
    base_branch: str,
) -> str:
    """Create an isolated worktree on a new branch from base."""
    repo = _repo(project_root)
    wt = Path(worktree_path).resolve()
    wt.parent.mkdir(parents=True, exist_ok=True)
    if wt.exists():
        return f"Worktree already exists: {wt}"
    repo.git.worktree("add", "-B", branch, str(wt), base_branch)
    return f"Worktree {wt} on branch {branch}"


def git_remove_worktree(project_root: str, worktree_path: str) -> str:
    repo = _repo(project_root)
    wt = str(Path(worktree_path).resolve())
    try:
        repo.git.worktree("remove", wt, force=True)
    except Exception:
        subprocess.run(
            ["git", "worktree", "remove", "--force", wt],
            cwd=project_root,
            check=False,
        )
    return f"Removed worktree {wt}"


def git_push_branch(project_root: str, branch: str, remote: str = "origin") -> str:
    repo = _repo(project_root)
    try:
        repo.git.push(remote, branch)
        return f"Pushed {branch} to {remote}"
    except Exception as exc:
        return f"Push skipped/failed for {branch}: {exc}"


def git_merge_branch(project_root: str, base_branch: str, feature_branch: str) -> MergeAttempt:
    repo = _repo(project_root)
    repo.heads[base_branch].checkout()
    try:
        repo.git.merge(feature_branch, "--no-ff", m=f"[autocrew] merge {feature_branch}")
        if repo.index.unmerged_blobs():
            repo.git.merge("--abort")
            return MergeAttempt(
                branch=feature_branch,
                role="",
                merged=False,
                had_conflicts=True,
                message=f"Conflicts merging {feature_branch}",
            )
        return MergeAttempt(
            branch=feature_branch,
            role="",
            merged=True,
            had_conflicts=False,
            message=f"Merged {feature_branch} into {base_branch}",
        )
    except Exception as exc:
        try:
            repo.git.merge("--abort")
        except Exception:
            pass
        msg = str(exc).lower()
        had_conflicts = "conflict" in msg
        return MergeAttempt(
            branch=feature_branch,
            role="",
            merged=False,
            had_conflicts=had_conflicts,
            message=str(exc),
        )
