import json

import pytest
from typer.testing import CliRunner

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.main import app
from autocrew.squad.squad_builder import build_squad
from autocrew.storage import save_context, save_squad
from autocrew.tasks.dependency_resolver import DependencyError, resolve_dependencies
from autocrew.tasks.task_builder import build_tasks, merge_foundation_tasks
from autocrew.tasks.task_model import TaskConfig

runner = CliRunner()


def _sample_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.NEW_IDEA,
        project_name="CLI Test Project",
        domain=ProjectDomain.SAAS,
        description="Test project for CLI",
        tech_stack=TechStack(frontend=["React"], backend=["FastAPI"]),
        features=[FeatureItem(name="Auth", description="Login", priority="high")],
    )


class TestDependencyResolver:
    def test_topological_order(self):
        tasks = [
            TaskConfig("c", "C", "d", "architect", depends_on=["a", "b"]),
            TaskConfig("a", "A", "d", "po", depends_on=[]),
            TaskConfig("b", "B", "d", "architect", depends_on=["a"]),
        ]
        ordered = resolve_dependencies(tasks)
        ids = [t.task_id for t in ordered]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_cycle_raises(self):
        tasks = [
            TaskConfig("a", "A", "d", "po", depends_on=["b"]),
            TaskConfig("b", "B", "d", "arch", depends_on=["a"]),
        ]
        with pytest.raises(DependencyError):
            resolve_dependencies(tasks)


class TestTaskBuilder:
    def test_includes_standard_tasks_without_llm(self):
        context = _sample_context()
        squad = build_squad(context)
        tasks = build_tasks(squad, context, llm_call=lambda _: "[]")
        task_ids = {t.task_id for t in tasks}
        assert "po_product_spec" in task_ids
        assert "arch_design" in task_ids
        assert "review_code" in task_ids
        assert "track_progress" in task_ids

    def test_merge_foundation_tasks_resolves_arch_design(self):
        context = _sample_context()
        squad = build_squad(context)
        debate_tasks = [
            TaskConfig(
                task_id="debate_1_fix_auth",
                title="Fix auth",
                description="Implement auth",
                assigned_agent_role="fullstack_dev",
                depends_on=["arch_design"],
            )
        ]
        tasks = merge_foundation_tasks(squad, context, debate_tasks)
        task_ids = [t.task_id for t in tasks]
        assert "arch_design" in task_ids
        assert "debate_1_fix_auth" in task_ids
        assert task_ids.index("arch_design") < task_ids.index("debate_1_fix_auth")


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "autocrew" in result.stdout.lower() or "AI Project" in result.stdout

    def test_new_with_mocked_llm(self, sample_idea_json, monkeypatch):
        from autocrew.config import settings

        monkeypatch.setattr(settings, "require_confirmation", False)
        monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

        def mock_analyze(raw_text, llm=None, llm_call=None):
            from autocrew.analyzer.idea_analyzer import _parse_idea_response
            return _parse_idea_response(sample_idea_json, raw_text)

        monkeypatch.setattr("autocrew.main.analyze_idea", mock_analyze)

        result = runner.invoke(app, ["new", "Build a CRM"])
        assert result.exit_code == 0
        assert "TaskFlow CRM" in result.stdout or "Saved" in result.stdout

    def test_plan_command(self, isolated_output_dirs, monkeypatch):
        context = _sample_context()
        squad = build_squad(context)
        save_context(context, str(isolated_output_dirs / "contexts"))
        save_squad(squad, str(isolated_output_dirs / "squads"))

        monkeypatch.setattr(
            "autocrew.config.settings.contexts_dir",
            str(isolated_output_dirs / "contexts"),
        )
        monkeypatch.setattr(
            "autocrew.config.settings.squads_dir",
            str(isolated_output_dirs / "squads"),
        )
        monkeypatch.setattr(
            "autocrew.config.settings.output_dir",
            str(isolated_output_dirs),
        )

        def mock_build_tasks(squad, context, llm=None, llm_call=None):
            return build_tasks(squad, context, llm_call=lambda _: "[]")

        monkeypatch.setattr("autocrew.main.build_tasks", mock_build_tasks)

        result = runner.invoke(app, ["plan", "--root", str(isolated_output_dirs / "project")])
        assert result.exit_code == 0

    def test_track_command(self, isolated_output_dirs, fixture_project, monkeypatch):
        context = ProjectContext(
            project_type=ProjectType.EXISTING_CODE,
            project_name="Fixture",
            domain=ProjectDomain.API,
            description="Fixture project",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[
                FeatureItem(name="Health", description="Health check", status="done", priority="low"),
            ],
            codebase_path=str(fixture_project),
        )
        ctx_dir = isolated_output_dirs / "contexts"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        save_context(context, str(ctx_dir))

        monkeypatch.setattr("autocrew.config.settings.contexts_dir", str(ctx_dir))
        monkeypatch.setattr("autocrew.config.settings.reports_dir", str(isolated_output_dirs / "reports"))

        result = runner.invoke(app, ["track"])
        assert result.exit_code == 0
        assert "Progress Report" in result.stdout or "complete" in result.stdout.lower()

    def test_build_command(self, isolated_output_dirs, tmp_path, monkeypatch):
        context = _sample_context()
        squad = build_squad(context)
        ctx_dir = isolated_output_dirs / "contexts"
        sq_dir = isolated_output_dirs / "squads"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        sq_dir.mkdir(parents=True, exist_ok=True)
        save_context(context, str(ctx_dir))
        save_squad(squad, str(sq_dir))

        monkeypatch.setattr("autocrew.config.settings.contexts_dir", str(ctx_dir))
        monkeypatch.setattr("autocrew.config.settings.squads_dir", str(sq_dir))
        monkeypatch.setattr("autocrew.config.settings.output_dir", str(isolated_output_dirs))
        monkeypatch.setattr("autocrew.config.settings.logs_dir", str(isolated_output_dirs / "logs"))
        monkeypatch.setattr("autocrew.config.settings.require_confirmation", False)

        project_root = tmp_path / "build-target"
        result = runner.invoke(
            app,
            ["build", "--root", str(project_root), "--yes", "--simulation", "--no-parallel-git"],
        )
        assert result.exit_code == 0
        assert "Build Complete" in result.stdout or "complete" in result.stdout.lower()

    def test_status_command(self, isolated_output_dirs, monkeypatch):
        from autocrew.tracker.progress_tracker import generate_progress_report, save_report

        context = _sample_context()
        report = generate_progress_report(context, str(isolated_output_dirs))
        reports_dir = isolated_output_dirs / "reports"
        save_report(report, str(reports_dir))

        monkeypatch.setattr("autocrew.config.settings.reports_dir", str(reports_dir))

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Latest Status" in result.stdout or context.project_name in result.stdout
