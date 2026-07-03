import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.debate.debate_runner import build_tasks_from_debate, run_debate
from autocrew.debate.heuristic_critique import generate_heuristic_critique
from autocrew.main import app
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole
from autocrew.storage import save_context, save_squad

runner = CliRunner()


def _planity_context() -> ProjectContext:
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name="Debate Test",
        domain=ProjectDomain.MOBILE_APP,
        description="Salon booking app",
        tech_stack=TechStack(
            frontend=["Expo", "React Native"],
            backend=["NestJS", "Prisma"],
            devops=["Docker"],
        ),
        features=[
            FeatureItem(name="User Auth", description="JWT auth", status="done", priority="high"),
            FeatureItem(name="Payment Integration", description="Stripe", status="not_started", priority="high"),
            FeatureItem(name="Admin Dashboard", description="Next.js admin", status="not_started", priority="medium"),
        ],
        missing_parts=["Stripe checkout", "Admin dashboard"],
        codebase_path=".",
    )


class TestHeuristicCritique:
    def test_backend_flags_payment_gap(self):
        context = _planity_context()
        context.features.append(
            FeatureItem(name="Booking Flow", description="Book", status="done", priority="high")
        )
        context.features.append(
            FeatureItem(name="Search", description="Search", status="done", priority="high")
        )
        context.features.append(
            FeatureItem(name="Favorites", description="Fav", status="done", priority="medium")
        )
        squad = build_squad(context)
        dev = next(
            (a for a in squad.agents if a.role in (AgentRole.BACKEND_DEV, AgentRole.FULLSTACK_DEV)),
            None,
        )
        assert dev is not None
        critique = generate_heuristic_critique(dev, context, "Basic plan", round_number=1)
        assert not critique.approved
        assert critique.blockers or critique.concerns

    def test_po_flags_missing_features(self):
        context = _planity_context()
        squad = build_squad(context)
        po = next(a for a in squad.agents if a.role == AgentRole.PRODUCT_OWNER)
        critique = generate_heuristic_critique(po, context, "Basic plan", round_number=1)
        assert not critique.approved


class TestDebateRunner:
    def test_run_debate_reaches_consensus_or_max_rounds(self, tmp_path, isolated_output_dirs):
        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Product\nInitial plan", encoding="utf-8")

        result = run_debate(
            context, squad, str(project_root), str(isolated_output_dirs), max_rounds=3
        )
        assert len(result.rounds) >= 1
        assert Path(result.final_plan_path).is_file()
        assert Path(result.debate_dir, "debate_result.json").is_file()
        assert result.action_items

    def test_build_tasks_from_debate(self, tmp_path, isolated_output_dirs):
        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()

        result = run_debate(
            context, squad, str(project_root), str(isolated_output_dirs), max_rounds=2
        )
        tasks = build_tasks_from_debate(result, squad, context)
        assert len(tasks) >= 1
        roles = {t.assigned_agent_role for t in tasks}
        assert roles  # dev role assigned (fullstack or backend or frontend)


class TestDebateCLI:
    def test_debate_command(self, tmp_path, isolated_output_dirs, monkeypatch):
        context = _planity_context()
        squad = build_squad(context)
        ctx_dir = isolated_output_dirs / "contexts"
        sq_dir = isolated_output_dirs / "squads"
        save_context(context, str(ctx_dir))
        save_squad(squad, str(sq_dir))

        project_root = tmp_path / "planity"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Plan", encoding="utf-8")

        monkeypatch.setattr(settings, "contexts_dir", str(ctx_dir))
        monkeypatch.setattr(settings, "squads_dir", str(sq_dir))
        monkeypatch.setattr(settings, "output_dir", str(isolated_output_dirs))
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(settings, "nvidia_api_key", "")
        monkeypatch.setattr(settings, "zenmux_api_key", "")
        monkeypatch.setattr(settings, "openrouter_api_key", "")
        monkeypatch.setattr(settings, "require_confirmation", False)

        result = runner.invoke(
            app, ["debate", "--root", str(project_root), "--rounds", "2", "--yes"]
        )
        assert result.exit_code == 0
        assert "Debate Round" in result.stdout or "debate" in result.stdout.lower()
        assert "autocrew build" in result.stdout


class TestDualModelRouter:
    def test_routes_planning_vs_implementation(self):
        from autocrew.debate.model_router import DualModelRouter
        from autocrew.squad.squad_builder import build_squad
        from autocrew.squad.squad_model import AgentRole

        context = _planity_context()
        context.features.extend([
            FeatureItem(name="Booking", description="Book", status="done", priority="high"),
            FeatureItem(name="Search", description="Search", status="done", priority="high"),
            FeatureItem(name="Favorites", description="Fav", status="done", priority="medium"),
        ])
        squad = build_squad(context)

        class FakeLLM:
            def __init__(self, name):
                self.name = name

            def complete(self, prompt: str) -> str:
                return self.name

        router = DualModelRouter(
            FakeLLM("kimi"), FakeLLM("deepseek"),
            "moonshotai/kimi-k2.6", "deepseek-ai/deepseek-v4-flash",
        )

        po = next(a for a in squad.agents if a.role == AgentRole.PRODUCT_OWNER)
        backend = next(
            a for a in squad.agents
            if a.role in (AgentRole.BACKEND_DEV, AgentRole.FULLSTACK_DEV)
        )

        _, po_model = router.for_agent(po)
        _, dev_model = router.for_agent(backend)
        assert "kimi" in po_model
        assert "deepseek" in dev_model

    def test_dual_model_debate_uses_both_models(self, tmp_path, isolated_output_dirs, monkeypatch):
        from autocrew.debate.model_router import DualModelRouter

        context = _planity_context()
        squad = build_squad(context)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "docs").mkdir()
        (project_root / "docs" / "product.md").write_text("# Plan", encoding="utf-8")

        calls: list[str] = []

        class FakeLLM:
            def __init__(self, model):
                self.model = model

            def complete(self, prompt: str) -> str:
                calls.append(self.model)
                return json.dumps({
                    "approved": True,
                    "concerns": [],
                    "suggestions": [],
                    "blockers": [],
                })

        router = DualModelRouter(
            FakeLLM("moonshotai/kimi-k2.6"),
            FakeLLM("deepseek-ai/deepseek-v4-flash"),
            "moonshotai/kimi-k2.6",
            "deepseek-ai/deepseek-v4-flash",
        )

        result = run_debate(
            context, squad, str(project_root), str(isolated_output_dirs),
            max_rounds=1, dual_router=router,
        )
        assert "kimi" in result.models_used["planning"]
        assert "deepseek" in result.models_used["implementation"]
        assert "moonshotai/kimi-k2.6" in calls
        assert "deepseek-ai/deepseek-v4-flash" in calls
