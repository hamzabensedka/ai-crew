"""File tools with scope enforcement."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from autocrew.context.path_filter import is_scannable_path


class ScopeError(PermissionError):
    pass


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _path_allowed(path: str, scopes: list[str]) -> bool:
    normalized = _normalize_path(path)
    if "*" in scopes:
        return True
    for scope in scopes:
        scope_norm = _normalize_path(scope)
        if scope_norm == "/":
            return True
        if normalized == scope_norm.lstrip("/"):
            return True
        if normalized.startswith(scope_norm.lstrip("/") + "/"):
            return True
        if fnmatch.fnmatch(normalized, scope_norm.lstrip("/")):
            return True
    return False


def check_read_scope(path: str, can_read: list[str], enforce: bool = True) -> None:
    if enforce and not _path_allowed(path, can_read):
        raise ScopeError(f"Read denied for '{path}'. Allowed: {can_read}")


def check_write_scope(path: str, can_write_to: list[str], enforce: bool = True) -> None:
    if enforce and not _path_allowed(path, can_write_to):
        raise ScopeError(f"Write denied for '{path}'. Allowed: {can_write_to}")


def read_file(path: str, project_root: str, can_read: list[str], enforce_scope: bool = True) -> str:
    check_read_scope(path, can_read, enforce_scope)
    if not is_scannable_path(path, project_root):
        raise ScopeError(f"Path excluded from model context: '{path}'")
    full = Path(project_root) / path
    return full.read_text(encoding="utf-8", errors="ignore")


def write_file(
    path: str,
    content: str,
    project_root: str,
    can_write_to: list[str],
    enforce_scope: bool = True,
) -> str:
    check_write_scope(path, can_write_to, enforce_scope)
    full = Path(project_root) / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


def list_directory(
    path: str,
    project_root: str,
    can_read: list[str],
    enforce_scope: bool = True,
) -> list[str]:
    check_read_scope(path, can_read, enforce_scope)
    full = Path(project_root) / path
    if not full.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return [
        str(p.relative_to(Path(project_root))).replace("\\", "/")
        for p in full.iterdir()
        if is_scannable_path(str(p.relative_to(Path(project_root))).replace("\\", "/"), project_root)
    ]


def create_folder(
    path: str,
    project_root: str,
    can_write_to: list[str],
    enforce_scope: bool = True,
) -> str:
    check_write_scope(path, can_write_to, enforce_scope)
    full = Path(project_root) / path
    full.mkdir(parents=True, exist_ok=True)
    return f"Created folder {path}"
