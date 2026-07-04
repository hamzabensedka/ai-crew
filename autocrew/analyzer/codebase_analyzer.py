"""Analyzes existing codebases into ProjectContext."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pathspec

from autocrew.analyzer.llm_client import LLMClient, call_with_json_retry
from autocrew.analyzer.project_model import (
    FeatureItem,
    ProjectContext,
    ProjectDomain,
    ProjectType,
    TechStack,
)

from autocrew.context.path_filter import EXCLUDED_DIRS, is_scannable_path

SKIP_DIRS = EXCLUDED_DIRS | {
    ".next",
    ".mypy_cache",
    ".pytest_cache",
    "output",
    "__pycache__",
    "venv",
    ".venv",
}

INCLUDE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".env.example",
}

KEY_FILES = [
    "README.md",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
]

MAX_FILE_CHARS = 4000
TRUNCATE_HEAD = 2000
TRUNCATE_TAIL = 2000

CODEBASE_PROMPT = """You are a senior software architect reviewing an existing project.

File tree:
{file_map}

Key file contents:
{key_contents}

Analyze this codebase and return a JSON object:
{{
  "project_name": "...",
  "domain": "saas|mobile_app|api|data_pipeline|ecommerce|ai_tool|other",
  "description": "...",
  "tech_stack": {{
    "frontend": [...],
    "backend": [...],
    "devops": [...],
    "other": [...]
  }},
  "features": [
    {{
      "name": "...",
      "description": "...",
      "status": "done|partial|not_started",
      "priority": "high|medium|low",
      "evidence": "which files suggest this status"
    }}
  ],
  "missing_parts": ["list of clearly absent features or components"],
  "special_requirements": [...]
}}

Be conservative: mark as "done" only if you see real implementation, not just a stub or empty file.
Return only valid JSON.
"""


def _load_gitignore(folder_path: Path) -> pathspec.PathSpec | None:
    gitignore = folder_path / ".gitignore"
    if not gitignore.exists():
        return None
    patterns = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _should_skip(relative: str, gitignore_spec: pathspec.PathSpec | None) -> bool:
    parts = Path(relative).parts
    if any(part in SKIP_DIRS for part in parts):
        return True
    if gitignore_spec and gitignore_spec.match_file(relative):
        return True
    return False


def _build_file_map(folder_path: str) -> list[str]:
    root = Path(folder_path).resolve()
    gitignore_spec = _load_gitignore(root)
    file_map: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            full = Path(dirpath) / filename
            relative = str(full.relative_to(root)).replace("\\", "/")
            if _should_skip(relative, gitignore_spec):
                continue
            suffix = full.suffix.lower()
            if suffix in INCLUDE_EXTENSIONS or filename in KEY_FILES:
                if is_scannable_path(relative, folder_path):
                    file_map.append(relative)

    return sorted(file_map)


def _truncate_content(content: str) -> str:
    if len(content) <= MAX_FILE_CHARS:
        return content
    return content[:TRUNCATE_HEAD] + "\n...[truncated]...\n" + content[-TRUNCATE_TAIL:]


def _read_key_files(folder_path: str, file_map: list[str]) -> dict[str, str]:
    root = Path(folder_path).resolve()
    contents: dict[str, str] = {}

    candidates = set(KEY_FILES) & set(file_map)
    for rel in sorted(candidates):
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="ignore")
            contents[rel] = _truncate_content(text)
        except OSError:
            continue

    for rel in file_map:
        if rel.endswith(("main.py", "app.py", "index.ts", "index.tsx")) and rel not in contents:
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="ignore")
                contents[rel] = _truncate_content(text)
            except OSError:
                continue

    return contents


def _parse_codebase_response(
    data: dict, folder_path: str, file_map: list[str]
) -> ProjectContext:
    features = [FeatureItem.from_dict(f) for f in data.get("features", [])]
    return ProjectContext(
        project_type=ProjectType.EXISTING_CODE,
        project_name=data["project_name"],
        domain=ProjectDomain(data["domain"]),
        description=data["description"],
        tech_stack=TechStack.from_dict(data.get("tech_stack")),
        features=features,
        existing_files=file_map,
        missing_parts=list(data.get("missing_parts", [])),
        special_requirements=list(data.get("special_requirements", [])),
        codebase_path=str(Path(folder_path).resolve()),
    )


def analyze_codebase(
    folder_path: str,
    llm: LLMClient | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> ProjectContext:
    root = Path(folder_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path not found: {folder_path}")

    file_map = _build_file_map(folder_path)
    key_contents = _read_key_files(folder_path, file_map)

    file_map_str = "\n".join(file_map[:500])
    if len(file_map) > 500:
        file_map_str += f"\n... and {len(file_map) - 500} more files"

    key_contents_str = "\n\n".join(f"--- {k} ---\n{v}" for k, v in key_contents.items())

    prompt = CODEBASE_PROMPT.format(
        file_map=file_map_str,
        key_contents=key_contents_str or "(no key files found)",
    )

    if llm_call is not None:
        data = call_with_json_retry(llm_call, prompt)
    elif llm is not None:
        data = call_with_json_retry(llm.complete, prompt)
    else:
        raise ValueError("Either llm or llm_call must be provided")

    return _parse_codebase_response(data, folder_path, file_map)
