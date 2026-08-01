"""Tests for env example secret sanitization."""

from pathlib import Path

from autocrew.tools.secret_sanitizer import sanitize_file, sanitize_project, sanitize_text


class TestSecretSanitizer:
    def test_replaces_stripe_test_key(self):
        content = "STRIPE_SECRET_KEY=sk_test_51AbCdEfGhIjKlMnOpQrStUvWxYz1234567890\n"
        updated, count = sanitize_text(content)
        assert count == 1
        assert "sk_test_your_key_here" in updated
        assert "51AbCdEf" not in updated

    def test_leaves_placeholder_keys(self):
        content = "STRIPE_SECRET_KEY=sk_test_your_key_here\n"
        updated, count = sanitize_text(content)
        assert count == 0
        assert updated == content

    def test_sanitize_project_finds_env_examples(self, tmp_path):
        env_path = tmp_path / "apps" / "api" / ".env.example"
        env_path.parent.mkdir(parents=True)
        env_path.write_text("STRIPE_WEBHOOK_SECRET=whsec_realSecretValue123456789012345\n", encoding="utf-8")
        messages = sanitize_project(str(tmp_path))
        assert len(messages) == 1
        assert "whsec_your_webhook_secret_here" in env_path.read_text(encoding="utf-8")

    def test_sanitize_file_root_env_example(self, tmp_path):
        path = tmp_path / "env.example"
        path.write_text("KEY=sk_test_abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8")
        count = sanitize_file(path)
        assert count == 1
