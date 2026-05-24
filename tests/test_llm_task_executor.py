"""Tests for LLM-powered build task execution."""

import json

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.crew.crew_logger import CrewLogger
from autocrew.crew.crew_runner import run_crew
from autocrew.crew.llm_task_executor import _parse_files_payload, execute_task_with_llm
from autocrew.squad.squad_builder import build_squad
from autocrew.tasks.task_builder import build_tasks
from autocrew.tasks.task_model import TaskConfig


class TestLlmTaskExecutor:
    def test_parse_files_payload_from_files_array(self):
        task = TaskConfig(
            task_id="t1",
            title="Test",
            description="d",
            assigned_agent_role="backend_developer",
            output_path="docs/out.md",
        )
        data = {"files": [{"path": "docs/out.md", "content": "# Hello"}], "summary": "done"}
        files = _parse_files_payload(data, task)
        assert files[0]["path"] == "docs/out.md"
        assert files[0]["content"] == "# Hello"

    def test_parse_files_payload_fallback_to_output_path(self):
        task = TaskConfig(
            task_id="t2",
            title="Test",
            description="d",
            assigned_agent_role="backend_developer",
            output_path="docs/report.md",
        )
        data = {"content": "# Report body", "summary": "ok"}
        files = _parse_files_payload(data, task)
        assert files[0]["path"] == "docs/report.md"

    def test_execute_task_with_llm_writes_file(self, tmp_path):
        context = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="LLMBuild",
            domain=ProjectDomain.API,
            description="Test",
            tech_stack=TechStack(backend=["NestJS"]),
            features=[FeatureItem(name="API", description="REST", priority="high")],
        )
        squad = build_squad(context)
        agent = squad.agents[0]
        task = TaskConfig(
            task_id="po_product_spec",
            title="Write Product Specification",
            description="Write product spec",
            assigned_agent_role=agent.role.value,
            output_path="docs/product.md",
            output_format="markdown",
            expected_output="product.md",
        )
        payload = {
            "files": [{"path": "docs/product.md", "content": "# Real Product Spec\n\nDetails here."}],
            "summary": "Wrote product spec",
        }

        def fake_llm(_prompt: str) -> str:
            return json.dumps(payload)

        logger = CrewLogger()
        result = execute_task_with_llm(
            task,
            agent,
            context,
            str(tmp_path),
            logger,
            fake_llm,
        )
        assert "Real Product Spec" in (tmp_path / "docs" / "product.md").read_text(encoding="utf-8")
        assert "Wrote product spec" in result

    def test_run_crew_with_llm(self, tmp_path, isolated_output_dirs, monkeypatch):
        context = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="LLMCrew",
            domain=ProjectDomain.API,
            description="Test LLM crew",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[FeatureItem(name="API", description="REST API", priority="high")],
        )
        squad = build_squad(context)
        tasks = build_tasks(squad, context, llm_call=lambda _: "[]")
        project_root = tmp_path / "project"
        project_root.mkdir()

        payload = {
            "files": [{"path": "docs/product.md", "content": "# LLM generated product doc"}],
            "summary": "done",
        }

        def fake_llm(_prompt: str) -> str:
            return json.dumps(payload)

        monkeypatch.setattr(
            "autocrew.crew.crew_runner.git_commit",
            lambda *_args, **_kwargs: None,
        )

        result = run_crew(
            squad,
            tasks,
            context,
            project_root=str(project_root),
            use_llm=True,
            llm_call=fake_llm,
            task_limit=1,
        )
        assert "LLM" in result
        assert (project_root / "docs" / "product.md").exists()
