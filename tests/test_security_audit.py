"""Tests for security audit."""

from autocrew.security_audit import run_security_audit


class TestSecurityAudit:
    def test_detects_hardcoded_secret(self, tmp_path):
        (tmp_path / "app.ts").write_text('const key = "nvapi-abcdefghijklmnopqrstuvwxyz123456"', encoding="utf-8")
        report = run_security_audit(str(tmp_path))
        assert not report.passed
        assert any("NVIDIA" in f.title for f in report.findings)

    def test_clean_project_passes(self, tmp_path):
        (tmp_path / "app.ts").write_text("export const x = 1;\n", encoding="utf-8")
        (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
        report = run_security_audit(str(tmp_path))
        assert report.passed
