"""Tests for git_tools unattended cleanup."""

from pathlib import Path

from autocrew.tools.git_tools import (
    git_create_worktree,
    git_ensure_initial_commit,
    git_init,
    git_push_branch,
    git_remove_worktree,
    git_resolve_base_branch,
)


class TestGitTools:
    def test_resolve_base_branch_uses_main(self, tmp_path):
        git_init(str(tmp_path))
        assert git_resolve_base_branch(str(tmp_path)) == "main"

    def test_worktree_create_and_remove_noninteractive(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        git_init(str(root))
        git_ensure_initial_commit(str(root))
        base = git_resolve_base_branch(str(root))
        wt = root / ".autocrew" / "worktrees" / "s1" / "backend_developer"
        git_create_worktree(str(root), "autocrew/s1/backend", str(wt), base)
        assert wt.is_dir()
        msg = git_remove_worktree(str(root), str(wt))
        assert "Removed worktree" in msg
        assert not wt.exists()

    def test_push_skips_without_origin(self, tmp_path):
        root = tmp_path / "repo2"
        root.mkdir()
        git_init(str(root))
        git_ensure_initial_commit(str(root))
        msg = git_push_branch(str(root), "main")
        assert "no 'origin' remote" in msg
