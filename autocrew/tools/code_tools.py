"""Command execution tools with timeout."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_command(
    command: str,
    project_root: str,
    timeout: int = 30,
) -> dict[str, str | int]:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1}


def detect_test_command(project_root: str) -> str:
    root = Path(project_root)
    if (root / "package.json").exists():
        return "npm test"
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        return "pytest"
    if (root / "requirements.txt").exists():
        return "pytest"
    return "echo 'No test command detected'"


def run_tests(project_root: str, timeout: int = 60) -> dict[str, str | int]:
    cmd = detect_test_command(project_root)
    return run_command(cmd, project_root, timeout=timeout)


def run_lint(project_root: str, timeout: int = 30) -> dict[str, str | int]:
    root = Path(project_root)
    if (root / "package.json").exists():
        return run_command("npm run lint", project_root, timeout=timeout)
    if (root / "pyproject.toml").exists():
        return run_command("ruff check .", project_root, timeout=timeout)
    return {"stdout": "", "stderr": "No linter configured", "returncode": 0}
