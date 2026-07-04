"""Exclude build artifacts and non-source paths from model context."""

from __future__ import annotations

from pathlib import Path

EXCLUDED_DIRS = frozenset({
    "dist",
    "build",
    "node_modules",
    ".expo",
    "coverage",
    "android/app/build",
    "ios/build",
    "ios/Pods",
    ".git",
    "__pycache__",
    ".next",
    ".nx",
    "venv",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
})

EXCLUDED_EXTENSIONS = frozenset({
    ".so",
    ".map",
    ".apk",
    ".aab",
    ".jar",
    ".class",
})

ALLOWED_SOURCE_PREFIXES = (
    "apps/api/src/",
    "apps/mobile/src/",
    "apps/mobile/app/",
    "packages/",
    "docs/",
)

# When no monorepo layout exists, fall back to common source roots.
FALLBACK_SOURCE_DIRS = frozenset({
    "src",
    "app",
    "lib",
    "docs",
    "packages",
    "apps",
})


def _normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(_normalize(path).split("/"))


def is_excluded_dir_part(part: str) -> bool:
    return part in EXCLUDED_DIRS or part.startswith(".")


def path_has_excluded_segment(path: str) -> bool:
    normalized = _normalize(path)
    parts = _path_parts(normalized)
    for i, part in enumerate(parts):
        if is_excluded_dir_part(part):
            return True
        segment = "/".join(parts[: i + 1])
        if segment in EXCLUDED_DIRS:
            return True
    return False


def is_excluded_extension(path: str) -> bool:
    return Path(_normalize(path)).suffix.lower() in EXCLUDED_EXTENSIONS


def is_allowed_source_path(path: str, *, monorepo_layout: bool | None = None) -> bool:
    """True when path is under an allowed source prefix (or fallback roots)."""
    normalized = _normalize(path)
    if monorepo_layout is False:
        parts = _path_parts(normalized)
        if parts and parts[0] in FALLBACK_SOURCE_DIRS:
            return True
        if normalized.startswith("docs/"):
            return True
        return False

    for prefix in ALLOWED_SOURCE_PREFIXES:
        if prefix.endswith("/"):
            if normalized.startswith(prefix):
                return True
        elif normalized == prefix.rstrip("/") or normalized.startswith(prefix + "/"):
            return True

    if "packages/" in normalized and "/src/" in normalized:
        return True
    return False


def is_scannable_path(path: str, project_root: str | None = None) -> bool:
    """Return False for build artifacts; True for scannable source paths."""
    if is_excluded_extension(path):
        return False
    if path_has_excluded_segment(path):
        return False

    monorepo = None
    if project_root:
        root = Path(project_root)
        monorepo = (root / "apps" / "api" / "src").is_dir() or (
            root / "packages"
        ).is_dir()

    if monorepo:
        return is_allowed_source_path(path, monorepo_layout=True)
    return is_allowed_source_path(path, monorepo_layout=False) or not path_has_excluded_segment(
        path
    )


def filter_scannable_paths(paths: list[str], project_root: str | None = None) -> list[str]:
    return [p for p in paths if is_scannable_path(p, project_root)]


def detect_monorepo_layout(project_root: str) -> bool:
    root = Path(project_root)
    return (root / "apps" / "api" / "src").is_dir() or (root / "packages").is_dir()


def iter_source_files(
    project_root: str,
    *,
    max_entries: int | None = None,
) -> list[str]:
    """Walk project and return relative paths limited to allowed source dirs."""
    root = Path(project_root)
    if not root.is_dir():
        return []

    monorepo = detect_monorepo_layout(project_root)
    results: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not is_scannable_path(rel, project_root):
            continue
        if monorepo and not is_allowed_source_path(rel, monorepo_layout=True):
            continue
        results.append(rel)
        if max_entries is not None and len(results) >= max_entries:
            break
    return results
