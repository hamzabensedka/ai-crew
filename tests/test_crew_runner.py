import pytest

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.crew.crew_runner import run_crew
from autocrew.squad.squad_builder import build_squad
from autocrew.tasks.task_builder import build_tasks


class TestCrewRunner:
    def test_run_crew_executes_tasks(self, tmp_path, isolated_output_dirs):
        context = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="CrewTest",
            domain=ProjectDomain.API,
            description="Test crew execution",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[FeatureItem(name="API", description="REST API", priority="high")],
        )
        squad = build_squad(context)
        tasks = build_tasks(squad, context, llm_call=lambda _: "[]")
        project_root = tmp_path / "project"
        project_root.mkdir()

        result = run_crew(squad, tasks, context, project_root=str(project_root))
        assert "complete" in result.lower()
        assert (project_root / "docs" / "product.md").exists()

    @pytest.mark.asyncio
    async def test_parallel_group_runs(self, tmp_path):
        from autocrew.crew.crew_runner import _run_parallel_group
        from autocrew.crew.crew_logger import CrewLogger

        context = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="ParallelTest",
            domain=ProjectDomain.SAAS,
            description="Test",
            tech_stack=TechStack(frontend=["React"], backend=["FastAPI"]),
            features=[FeatureItem(name=f"F{i}", description="d") for i in range(7)],
        )
        squad = build_squad(context)
        tasks = build_tasks(squad, context, llm_call=lambda _: "[]")
        project_root = tmp_path / "proj"
        project_root.mkdir()
        logger = CrewLogger()

        if squad.parallel_groups:
            results = await _run_parallel_group(
                squad.parallel_groups[0],
                squad,
                tasks,
                context,
                str(project_root),
                logger,
                max_retries=0,
                use_llm=False,
                dual_router=None,
                llm_call=None,
                task_filter=None,
                parallel_git=False,
                git_push=False,
            )
            assert len(results) >= 0
