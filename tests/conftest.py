import json
from pathlib import Path

import pytest

from autocrew.config import settings


@pytest.fixture(autouse=True)
def isolated_output_dirs(tmp_path, monkeypatch):
    """Redirect all output directories to a temp folder for each test."""
    output = tmp_path / "output"
    monkeypatch.setattr(settings, "output_dir", str(output))
    monkeypatch.setattr(settings, "squads_dir", str(output / "squads"))
    monkeypatch.setattr(settings, "reports_dir", str(output / "reports"))
    monkeypatch.setattr(settings, "logs_dir", str(output / "logs"))
    monkeypatch.setattr(settings, "contexts_dir", str(output / "contexts"))
    monkeypatch.setattr(settings, "metrics_dir", str(output / "metrics"))
    (output / "contexts").mkdir(parents=True, exist_ok=True)
    (output / "squads").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    yield output


@pytest.fixture
def sample_idea_json() -> dict:
    return {
        "project_name": "TaskFlow CRM",
        "domain": "saas",
        "description": "A lightweight CRM for small teams",
        "tech_stack": {
            "frontend": ["Next.js", "Tailwind"],
            "backend": ["FastAPI", "PostgreSQL"],
            "devops": ["Docker"],
            "other": [],
        },
        "features": [
            {
                "name": "User Authentication",
                "description": "Email/password login with JWT",
                "priority": "high",
            },
            {
                "name": "Contact Management",
                "description": "CRUD for contacts and companies",
                "priority": "high",
            },
        ],
        "special_requirements": ["auth"],
    }


@pytest.fixture
def sample_codebase_json() -> dict:
    return {
        "project_name": "ExistingApp",
        "domain": "api",
        "description": "Partial FastAPI backend",
        "tech_stack": {
            "frontend": [],
            "backend": ["FastAPI"],
            "devops": [],
            "other": [],
        },
        "features": [
            {
                "name": "Health Check",
                "description": "GET /health endpoint",
                "status": "done",
                "priority": "low",
                "evidence": "main.py contains /health route",
            },
            {
                "name": "User Auth",
                "description": "JWT authentication",
                "status": "not_started",
                "priority": "high",
                "evidence": "no auth module found",
            },
        ],
        "missing_parts": ["authentication", "database migrations"],
        "special_requirements": ["auth"],
    }


@pytest.fixture
def fixture_project(tmp_path) -> Path:
    """Minimal existing project for codebase analysis tests."""
    root = tmp_path / "sample-project"
    root.mkdir()
    (root / "README.md").write_text("# Sample Project\nA FastAPI backend.", encoding="utf-8")
    (root / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    (root / "main.py").write_text(
        'from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/health")\ndef health():\n    return {"status": "ok"}\n',
        encoding="utf-8",
    )
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.js").write_text("// skip", encoding="utf-8")
    return root


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
