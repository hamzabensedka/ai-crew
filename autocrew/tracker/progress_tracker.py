"""Progress Tracker — compares product spec against codebase."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from autocrew.analyzer.project_model import FeatureItem, ProjectContext
from autocrew.tracker.report_model import FeatureStatus, ProgressReport

STUB_PATTERNS = [
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"pass\s*#\s*stub", re.IGNORECASE),
    re.compile(r"raise NotImplementedError"),
    re.compile(r"return None\s*#\s*placeholder", re.IGNORECASE),
]


def _scan_file_for_stubs(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    issues = []
    for i, line in enumerate(content.splitlines(), 1):
        for pattern in STUB_PATTERNS:
            if pattern.search(line):
                issues.append(f"{path.name}:{i}: {line.strip()[:80]}")
    return issues


def _find_related_files(feature_name: str, project_root: Path) -> list[str]:
    keyword = feature_name.lower().replace(" ", "_").split("_")[0]
    related: list[str] = []
    skip = {".git", "node_modules", "__pycache__", "venv", ".venv", "output"}
    for path in project_root.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file() and keyword in path.name.lower():
            related.append(str(path.relative_to(project_root)).replace("\\", "/"))
    return related[:10]


def _assess_feature(feature: FeatureItem, project_root: Path) -> FeatureStatus:
    related = _find_related_files(feature.name, project_root)
    bugs: list[str] = []

    for rel in related:
        full = project_root / rel
        if full.is_file():
            bugs.extend(_scan_file_for_stubs(full))

    if feature.status == "done":
        status = "done"
        details = f"Marked done with evidence in {len(related)} related files"
    elif feature.status == "partial" or (related and bugs):
        status = "partial"
        details = f"Partial implementation in {related[:3]}" if related else "Partial per analysis"
    elif related:
        status = "partial"
        details = f"Found related files: {related[:3]}"
    else:
        status = "missing"
        details = "No implementation files found"

    if bugs and status == "done":
        status = "partial"
        details += f"; {len(bugs)} stub/TODO indicators found"

    return FeatureStatus(
        name=feature.name,
        status=status,
        details=details,
        files_involved=related,
    )


def generate_progress_report(context: ProjectContext, project_root: str | None = None) -> ProgressReport:
    root = Path(project_root or context.codebase_path or ".").resolve()
    done: list[FeatureStatus] = []
    partial: list[FeatureStatus] = []
    missing: list[FeatureStatus] = []
    bugs: list[FeatureStatus] = []

    for feature in context.features:
        assessment = _assess_feature(feature, root)
        if assessment.status == "done":
            done.append(assessment)
        elif assessment.status == "partial":
            partial.append(assessment)
        else:
            missing.append(assessment)

        for rel in assessment.files_involved:
            stub_issues = _scan_file_for_stubs(root / rel)
            if stub_issues:
                bugs.append(
                    FeatureStatus(
                        name=f"Stubs in {rel}",
                        status="bug",
                        details="; ".join(stub_issues[:5]),
                        files_involved=[rel],
                    )
                )

    total = len(context.features) or 1
    completion = (len(done) + 0.5 * len(partial)) / total * 100

    next_priorities = [
        f.name for f in context.features if f.priority == "high" and f.status != "done"
    ]
    next_priorities.extend(context.missing_parts[:5])

    summary = (
        f"{context.project_name}: {completion:.0f}% complete. "
        f"{len(done)} done, {len(partial)} partial, {len(missing)} missing, {len(bugs)} bug indicators."
    )

    return ProgressReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        project_name=context.project_name,
        completion_percentage=round(completion, 1),
        done=done,
        partial=partial,
        missing=missing,
        bugs=bugs,
        next_priorities=list(dict.fromkeys(next_priorities)),
        raw_summary=summary,
    )


def save_report(report: ProgressReport, reports_dir: str) -> tuple[str, str]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace("+", "_")
    json_path = reports_path / f"progress_{ts}.json"
    md_path = reports_path / f"progress_{ts}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_report_markdown(report), encoding="utf-8")
    return str(json_path), str(md_path)


def render_report_markdown(report: ProgressReport) -> str:
    lines = [
        f"# Progress Report — {report.project_name}",
        f"**Generated:** {report.timestamp}",
        f"**Completion:** {report.completion_percentage}%",
        "",
        report.raw_summary,
        "",
        "## Done",
    ]
    for f in report.done:
        lines.append(f"- **{f.name}:** {f.details}")
    lines.append("\n## Partial")
    for f in report.partial:
        lines.append(f"- **{f.name}:** {f.details}")
    lines.append("\n## Missing")
    for f in report.missing:
        lines.append(f"- **{f.name}:** {f.details}")
    if report.bugs:
        lines.append("\n## Bug Indicators")
        for f in report.bugs:
            lines.append(f"- **{f.name}:** {f.details}")
    lines.append("\n## Next Priorities")
    for p in report.next_priorities:
        lines.append(f"- {p}")
    return "\n".join(lines)
