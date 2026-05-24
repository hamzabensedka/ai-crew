import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.planner import render_product_doc, render_tasks_doc
from autocrew.squad.squad_builder import build_squad
from autocrew.tasks.task_builder import build_tasks
from autocrew.tools.file_tools import ScopeError, read_file, write_file
from autocrew.tools.code_tools import run_command
from autocrew.tracker.progress_tracker import generate_progress_report, render_report_markdown


class TestFileTools:
    def test_write_and_read(self, tmp_path):
        root = str(tmp_path)
        write_file("docs/test.md", "# Hello", root, ["/docs"], enforce_scope=True)
        content = read_file("docs/test.md", root, ["*"], enforce_scope=True)
        assert "# Hello" in content

    def test_write_scope_denied(self, tmp_path):
        with pytest.raises(ScopeError):
            write_file("secret.txt", "data", str(tmp_path), ["/docs"], enforce_scope=True)


class TestCodeTools:
    def test_run_command_timeout(self, tmp_path):
        result = run_command("echo hello", str(tmp_path), timeout=5)
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]


class TestPlanner:
    def test_render_product_doc(self):
        ctx = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="PlannerTest",
            domain=ProjectDomain.API,
            description="API project",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[FeatureItem(name="Auth", description="JWT auth", priority="high")],
        )
        doc = render_product_doc(ctx)
        assert "PlannerTest" in doc
        assert "Auth" in doc

    def test_render_tasks_doc(self):
        ctx = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="TasksTest",
            domain=ProjectDomain.API,
            description="Test",
            tech_stack=TechStack(),
        )
        squad = build_squad(ctx)
        tasks = build_tasks(squad, ctx, llm_call=lambda _: "[]")
        doc = render_tasks_doc(squad, tasks)
        assert "TasksTest" in doc
        assert "po_product_spec" in doc


class TestTracker:
    def test_generates_report(self, fixture_project):
        ctx = ProjectContext(
            project_type=ProjectType.EXISTING_CODE,
            project_name="Fixture",
            domain=ProjectDomain.API,
            description="Fixture",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[
                FeatureItem(name="Health", description="Health endpoint", status="done", priority="low"),
                FeatureItem(name="Auth", description="Authentication", status="not_started", priority="high"),
            ],
            codebase_path=str(fixture_project),
        )
        report = generate_progress_report(ctx, str(fixture_project))
        assert 0 <= report.completion_percentage <= 100
        md = render_report_markdown(report)
        assert "Fixture" in md
        assert "Next Priorities" in md
