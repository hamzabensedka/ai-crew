"""Git operations for agent checkpoints, branches, worktrees, and merges."""

from __future__ import annotations

import os
import shutil
import stat
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


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(project_root: str, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=_git_env(),
        check=check,
    )


def _force_remove_directory(path: Path) -> None:
    """Remove a directory without interactive prompts (Windows-safe)."""

    def _on_rm_error(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.exists():
        shutil.rmtree(path, onerror=_on_rm_error)


def git_init(project_root: str) -> str:
    try:
        import git

        repo_path = Path(project_root)
        if (repo_path / ".git").exists():
            return "Git repo already initialized"
        git.Repo.init(repo_path, initial_branch="main")
        return "Git repo initialized"
    except Exception as exc:
        return f"Git init failed: {exc}"


def git_ensure_initial_commit(project_root: str, branch: str = "main") -> str:
    """Ensure repo has at least one commit on a named branch."""
    repo = _repo(project_root)
    if repo.heads:
        return f"Repo already has branch(es): {', '.join(h.name for h in repo.heads)}"

    repo.git.checkout("-b", branch)
    marker = Path(project_root) / ".gitkeep"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    repo.index.add([str(marker)])
    repo.index.commit("Initial commit")
    return f"Created initial commit on {branch}"


def git_resolve_base_branch(project_root: str) -> str:
    """Pick main/master or create an initial branch — never assume master exists."""
    repo = _repo(project_root)
    for name in ("main", "master"):
        if name in [h.name for h in repo.heads]:
            return name
    if not repo.heads:
        git_ensure_initial_commit(project_root, "main")
        return "main"
    if repo.head.is_detached:
        return "main"
    return repo.active_branch.name


def git_default_branch(project_root: str) -> str:
    try:
        return git_resolve_base_branch(project_root)
    except GitError:
        return "main"


def git_has_remote(project_root: str, remote: str = "origin") -> bool:
    try:
        repo = _repo(project_root)
        return remote in [r.name for r in repo.remotes]
    except GitError:
        return False


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
    root = str(Path(project_root).resolve())
    git_ensure_initial_commit(root, base_branch if base_branch in ("main", "master") else "main")
    base_branch = git_resolve_base_branch(root)
    wt = Path(worktree_path).resolve()
    wt.parent.mkdir(parents=True, exist_ok=True)
    if wt.exists():
        git_remove_worktree(root, str(wt))
    proc = _run_git(root, "worktree", "add", "-B", branch, str(wt), base_branch)
    if proc.returncode != 0:
        raise GitError(proc.stderr.strip() or proc.stdout.strip() or "worktree add failed")
    return f"Worktree {wt} on branch {branch}"


def git_remove_worktree(project_root: str, worktree_path: str) -> str:
    """Remove worktree without interactive prompts (safe for unattended runs)."""
    root = str(Path(project_root).resolve())
    wt = Path(worktree_path).resolve()
    wt_str = str(wt)

    _run_git(root, "worktree", "remove", "--force", wt_str)
    _run_git(root, "worktree", "prune")
    _force_remove_directory(wt)

    return f"Removed worktree {wt_str}"


def git_push_branch(project_root: str, branch: str, remote: str = "origin") -> str:
    if not git_has_remote(project_root, remote):
        return f"Push skipped for {branch}: no '{remote}' remote configured"
    proc = _run_git(project_root, "push", remote, branch)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return f"Push skipped/failed for {branch}: {err}"
    return f"Pushed {branch} to {remote}"


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
