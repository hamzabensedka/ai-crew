"""Security audit for autopilot — static checks + optional LLM review."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SECRET_PATTERNS = [
    (re.compile(r"sk_live_[a-zA-Z0-9]{20,}"), "critical", "Stripe live secret in source"),
    (re.compile(r"nvapi-[a-zA-Z0-9_-]{20,}"), "critical", "NVIDIA API key in source"),
    (re.compile(r"(?i)(api[_-]?key|secret|password)\s*=\s*['\"][^'\"]{8,}['\"]"), "high", "Hardcoded credential"),
    (re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"), "critical", "Private key in repository"),
]

SECURITY_FILE_GLOBS = ("*.ts", "*.tsx", "*.js", "*.jsx", "*.py", "*.env", "*.json")
SKIP_DIRS = {".git", "node_modules", "dist", "build", ".nx", "coverage", "__pycache__"}


@dataclass
class SecurityFinding:
    severity: str
    title: str
    detail: str
    file: str = ""

    def to_dict(self) -> dict:
        return {"severity": self.severity, "title": self.title, "detail": self.detail, "file": self.file}


@dataclass
class SecurityReport:
    passed: bool
    findings: list[SecurityFinding] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }


def _scan_file(path: Path, project_root: Path) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    rel = path.relative_to(project_root).as_posix()

    if path.name == ".env" and ".git" not in path.parts:
        findings.append(
            SecurityFinding(
                "critical",
                "Environment file in tree",
                ".env should not be committed; use .env.example only",
                rel,
            )
        )

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings

    for pattern, severity, title in SECRET_PATTERNS:
        if pattern.search(content):
            findings.append(SecurityFinding(severity, title, f"Pattern matched in {rel}", rel))

    if "apps/api" in rel and path.suffix == ".ts":
        lower = content.lower()
        if "controller" in path.name.lower() and "@public" not in content and "guard" not in lower:
            if any(k in path.name.lower() for k in ("user", "appointment", "payment", "admin")):
                findings.append(
                    SecurityFinding(
                        "high",
                        "Possible unguarded controller",
                        "Sensitive controller may lack auth guard",
                        rel,
                    )
                )
        if "webhook" in path.name.lower() and "signature" not in lower and "constructevent" not in lower:
            findings.append(
                SecurityFinding(
                    "high",
                    "Webhook without signature verification",
                    "Payment webhooks must verify provider signatures",
                    rel,
                )
            )

    return findings


def run_security_audit(project_root: str, max_files: int = 500) -> SecurityReport:
    """Static security scan. Passes when no critical/high findings."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        return SecurityReport(passed=True, summary="skipped (no project root)")

    findings: list[SecurityFinding] = []
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        gi = gitignore.read_text(encoding="utf-8", errors="ignore")
        if ".env" not in gi:
            findings.append(
                SecurityFinding(
                    "high",
                    ".env not gitignored",
                    "Add .env to .gitignore to prevent credential leaks",
                    ".gitignore",
                )
            )

    scanned = 0
    for path in root.rglob("*"):
        if scanned >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx", ".py", ".env", ".json"}:
            continue
        if path.name.endswith(".example") or "lock" in path.name:
            continue
        findings.extend(_scan_file(path, root))
        scanned += 1

    blocking = [f for f in findings if f.severity in ("critical", "high")]
    passed = len(blocking) == 0
    summary = (
        f"security OK ({len(findings)} notes)"
        if passed
        else f"{len(blocking)} critical/high security issue(s)"
    )
    return SecurityReport(passed=passed, findings=findings, summary=summary)


SECURITY_LLM_PROMPT = """You are a security-focused code reviewer for {project_name}.

Review these project docs and the security scan summary:
{context}

Static scan findings:
{findings}

Return JSON:
{{
  "approved": true or false,
  "blockers": ["must-fix security issues before production"],
  "concerns": ["medium issues"],
  "summary": "one sentence"
}}

Approve only if there are zero blockers and no critical unresolved risks.
Return only valid JSON.
"""


def run_llm_security_review(
    project_root: str,
    context_name: str,
    static_report: SecurityReport,
    llm_call: Callable[[str], str],
) -> SecurityReport:
    """Optional LLM pass — merges with static findings."""
    from autocrew.analyzer.llm_client import call_with_json_retry

    parts: list[str] = []
    for rel in ("docs/product.md", "docs/architecture.md"):
        path = Path(project_root) / rel
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="ignore")[:6000])

    findings_text = "\n".join(
        f"- [{f.severity}] {f.title}: {f.detail} ({f.file})" for f in static_report.findings[:20]
    ) or "(none)"

    prompt = SECURITY_LLM_PROMPT.format(
        project_name=context_name,
        context="\n\n".join(parts)[:8000] or "(no docs)",
        findings=findings_text,
    )

    try:
        data = call_with_json_retry(llm_call, prompt)
    except Exception as exc:
        return SecurityReport(
            passed=static_report.passed,
            findings=static_report.findings,
            summary=f"{static_report.summary}; LLM review skipped ({exc})",
        )

    llm_findings = list(static_report.findings)
    for blocker in data.get("blockers", []):
        llm_findings.append(
            SecurityFinding("high", "LLM security blocker", str(blocker), "llm-review")
        )

    approved = bool(data.get("approved", False)) and static_report.passed
    blocking = [f for f in llm_findings if f.severity in ("critical", "high")]
    if not approved and not any(f.file == "llm-review" for f in blocking):
        llm_findings.append(
            SecurityFinding(
                "high",
                "LLM security review not approved",
                str(data.get("summary", "unresolved risks")),
                "llm-review",
            )
        )
        blocking = [f for f in llm_findings if f.severity in ("critical", "high")]

    passed = len(blocking) == 0 and approved
    return SecurityReport(
        passed=passed,
        findings=llm_findings,
        summary=str(data.get("summary", static_report.summary)),
    )
