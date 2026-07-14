"""Tests for orphaned worktree recovery."""

from pathlib import Path

from autocrew.crew.crew_logger import CrewLogger
from autocrew.tools.git_tools import (
    git_create_worktree,
    git_ensure_initial_commit,
    git_init,
    git_resolve_base_branch,
)
from autocrew.tools.worktree_recovery import discover_worktrees, recover_worktrees


class TestWorktreeRecovery:
    def test_discover_worktrees_finds_agent_dirs(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        git_init(str(root))
        git_ensure_initial_commit(str(root))
        base = git_resolve_base_branch(str(root))
        wt = root / ".autocrew" / "worktrees" / "20260629_135557" / "backend_developer"
        git_create_worktree(str(root), "autocrew/20260629_135557/backend-developer", str(wt), base)

        found = discover_worktrees(str(root))
        assert len(found) == 1
        assert found[0].session_id == "20260629_135557"
        assert found[0].agent_role == "backend_developer"
        assert found[0].branch == "autocrew/20260629_135557/backend-developer"

    def test_recovers_uncommitted_changes_into_base(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        git_init(str(root))
        git_ensure_initial_commit(str(root))
        base = git_resolve_base_branch(str(root))

        payment_dir = root / "backend" / "src" / "payment"
        payment_dir.mkdir(parents=True)
        (payment_dir / "payment.service.ts").write_text("// stub\n", encoding="utf-8")
        repo = __import__("git").Repo(str(root))
        repo.index.add([str(payment_dir / "payment.service.ts")])
        repo.index.commit("add payment stub on base")

        wt = root / ".autocrew" / "worktrees" / "20260629_135557" / "backend_developer"
        git_create_worktree(
            str(root),
            "autocrew/20260629_135557/backend-developer",
            str(wt),
            base,
        )

        service = wt / "backend" / "src" / "payment" / "payment.service.ts"
        service.parent.mkdir(parents=True, exist_ok=True)
        service.write_text("// stripe implementation\n" * 40, encoding="utf-8")

        logger = CrewLogger(log_path=str(tmp_path / "recover.log"))
        result = recover_worktrees(
            str(root),
            logger,
            max_merges=5,
            min_insertions=1,
        )

        assert result is not None
        assert result.discovered == 1
        assert len(result.committed) == 1
        assert sum(1 for m in result.merged if m.merged) == 1

        merged_content = (root / "backend" / "src" / "payment" / "payment.service.ts").read_text(
            encoding="utf-8"
        )
        assert "stripe implementation" in merged_content

    def test_dry_run_does_not_merge(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        git_init(str(root))
        git_ensure_initial_commit(str(root))
        base = git_resolve_base_branch(str(root))

        wt = root / ".autocrew" / "worktrees" / "s1" / "backend_developer"
        git_create_worktree(str(root), "autocrew/s1/backend-developer", str(wt), base)
        target = wt / "backend" / "src" / "feature.ts"
        target.parent.mkdir(parents=True)
        target.write_text("export const x = 1;\n" * 20, encoding="utf-8")

        logger = CrewLogger(log_path=str(tmp_path / "recover_dry.log"))
        result = recover_worktrees(
            str(root),
            logger,
            max_merges=5,
            min_insertions=1,
            merge=False,
        )

        assert result is not None
        assert len(result.committed) == 1
        assert not any(m.merged for m in result.merged)
        assert not (root / "backend" / "src" / "feature.ts").exists()
