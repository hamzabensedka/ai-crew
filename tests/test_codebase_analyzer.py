import json

import pytest

from autocrew.analyzer.codebase_analyzer import (
    _build_file_map,
    _read_key_files,
    _truncate_content,
    analyze_codebase,
)


class TestFileMap:
    def test_skips_node_modules(self, fixture_project):
        file_map = _build_file_map(str(fixture_project))
        assert "main.py" in file_map
        assert "README.md" in file_map
        assert not any("node_modules" in f for f in file_map)

    def test_includes_key_extensions(self, fixture_project):
        file_map = _build_file_map(str(fixture_project))
        assert "requirements.txt" in file_map


class TestKeyFiles:
    def test_reads_readme(self, fixture_project):
        file_map = _build_file_map(str(fixture_project))
        contents = _read_key_files(str(fixture_project), file_map)
        assert "README.md" in contents
        assert "FastAPI" in contents["README.md"]


class TestTruncate:
    def test_short_content_unchanged(self):
        assert _truncate_content("hello") == "hello"

    def test_long_content_truncated(self):
        content = "a" * 5000
        result = _truncate_content(content)
        assert "truncated" in result
        assert len(result) < len(content)


class TestAnalyzeCodebase:
    def test_parses_llm_response(self, fixture_project, sample_codebase_json):
        def mock_llm(_prompt):
            return json.dumps(sample_codebase_json)

        ctx = analyze_codebase(str(fixture_project), llm_call=mock_llm)
        assert ctx.project_name == "ExistingApp"
        assert ctx.project_type.value == "existing_code"
        assert "main.py" in ctx.existing_files
        assert "authentication" in ctx.missing_parts

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_codebase("/nonexistent/path", llm_call=lambda _: "{}")
