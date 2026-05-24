"""Helpers for Cursor Composer workflow (no external LLM API keys)."""

from __future__ import annotations

import json
from pathlib import Path

from autocrew.analyzer.codebase_analyzer import _build_file_map, _read_key_files

CURSOR_WORKFLOW_HINT = (
    "[yellow]No API keys configured.[/yellow] Use the Cursor Composer workflow:\n"
    "  1. Ask Cursor to analyze your idea or run [cyan]autocrew scout ./path[/cyan]\n"
    "  2. Cursor saves context JSON → [cyan]autocrew import-context context.json[/cyan]\n"
    "  3. Then [cyan]autocrew plan[/cyan] → [cyan]autocrew build[/cyan]\n"
    "See [bold]docs/cursor-workflow.md[/bold] for the full guide."
)


def scout_codebase(folder_path: str) -> dict:
    """Collect file tree and key file contents for Cursor Composer to analyze."""
    root = Path(folder_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path not found: {folder_path}")

    file_map = _build_file_map(folder_path)
    key_contents = _read_key_files(folder_path, file_map)
    return {
        "codebase_path": str(root),
        "file_count": len(file_map),
        "file_map": file_map,
        "key_files": key_contents,
        "instructions_for_cursor": (
            "Analyze this codebase and produce a ProjectContext JSON file. "
            "Save it and run: autocrew import-context <path>. "
            "See docs/cursor-workflow.md for the exact JSON schema."
        ),
    }


def write_scout_report(folder_path: str, output_path: str | None = None) -> str:
    report = scout_codebase(folder_path)
    text = json.dumps(report, indent=2)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)
    return text
