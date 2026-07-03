import json

import pytest
from typer.testing import CliRunner

from autocrew.analyzer.project_model import FeatureItem, ProjectContext, ProjectDomain, ProjectType, TechStack
from autocrew.config import settings
from autocrew.cursor_mode import scout_codebase
from autocrew.main import app

runner = CliRunner()


def _sample_context_dict() -> dict:
    return {
        "project_type": "new_idea",
        "project_name": "Cursor Mode App",
        "domain": "saas",
        "description": "A CRM built via Cursor workflow",
        "tech_stack": {
            "frontend": ["Next.js"],
            "backend": ["FastAPI"],
            "devops": [],
            "other": [],
        },
        "features": [
            {
                "name": "Auth",
                "description": "User login",
                "status": "not_started",
                "priority": "high",
            }
        ],
        "existing_files": [],
        "missing_parts": [],
        "special_requirements": ["auth"],
    }


class TestScout:
    def test_scout_codebase(self, fixture_project):
        report = scout_codebase(str(fixture_project))
        assert report["file_count"] >= 2
        assert "main.py" in report["file_map"]
        assert "README.md" in report["key_files"]

    def test_scout_cli(self, fixture_project, isolated_output_dirs):
        out = isolated_output_dirs / "scout.json"
        result = runner.invoke(app, ["scout", str(fixture_project), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "file_map" in data


class TestImportContext:
    def test_import_context_saves_files(self, tmp_path, isolated_output_dirs, monkeypatch):
        ctx_file = tmp_path / "context.json"
        ctx_file.write_text(json.dumps(_sample_context_dict()), encoding="utf-8")

        monkeypatch.setattr(settings, "contexts_dir", str(isolated_output_dirs / "contexts"))
        monkeypatch.setattr(settings, "squads_dir", str(isolated_output_dirs / "squads"))
        monkeypatch.setattr(settings, "require_confirmation", False)

        result = runner.invoke(app, ["import-context", str(ctx_file), "--yes"])
        assert result.exit_code == 0
        assert "Saved context" in result.stdout
        assert "Saved squad" in result.stdout
        assert list((isolated_output_dirs / "contexts").glob("*_context.json"))

    def test_import_invalid_json_fails(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid", encoding="utf-8")
        result = runner.invoke(app, ["import-context", str(bad)])
        assert result.exit_code == 1


class TestNoApiKeys:
    def test_new_without_keys_shows_cursor_hint(self, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(settings, "nvidia_api_key", "")
        monkeypatch.setattr(settings, "zenmux_api_key", "")
        monkeypatch.setattr(settings, "openrouter_api_key", "")
        result = runner.invoke(app, ["new", "Build a CRM"])
        assert result.exit_code == 1
        assert "Cursor" in result.stdout or "cursor-workflow" in result.stdout

    def test_plan_without_keys_uses_standard_tasks(self, isolated_output_dirs, monkeypatch):
        from autocrew.squad.squad_builder import build_squad
        from autocrew.storage import save_context, save_squad

        context = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="NoKeyPlan",
            domain=ProjectDomain.API,
            description="Test",
            tech_stack=TechStack(backend=["FastAPI"]),
            features=[FeatureItem(name="API", description="REST", priority="high")],
        )
        squad = build_squad(context)
        ctx_dir = isolated_output_dirs / "contexts"
        sq_dir = isolated_output_dirs / "squads"
        save_context(context, str(ctx_dir))
        save_squad(squad, str(sq_dir))

        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(settings, "nvidia_api_key", "")
        monkeypatch.setattr(settings, "zenmux_api_key", "")
        monkeypatch.setattr(settings, "contexts_dir", str(ctx_dir))
        monkeypatch.setattr(settings, "squads_dir", str(sq_dir))
        monkeypatch.setattr(settings, "output_dir", str(isolated_output_dirs))

        result = runner.invoke(app, ["plan", "--root", str(isolated_output_dirs / "proj")])
        assert result.exit_code == 0
        assert "standard task" in result.stdout.lower() or "Tasks saved" in result.stdout
