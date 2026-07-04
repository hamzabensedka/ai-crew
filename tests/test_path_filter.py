"""Tests for context path filtering."""

from __future__ import annotations

from autocrew.context.path_filter import (
    filter_scannable_paths,
    is_excluded_extension,
    is_scannable_path,
)


class TestPathFilter:
    def test_excludes_build_artifacts(self):
        assert not is_scannable_path("dist/bundle.js")
        assert not is_scannable_path("node_modules/pkg/index.js")
        assert not is_scannable_path("android/app/build/output.apk")
        assert not is_excluded_extension("app/main.ts")
        assert is_excluded_extension("app/bundle.map")

    def test_allows_monorepo_source_paths(self):
        assert is_scannable_path("apps/api/src/businesses.controller.ts", "/proj")
        assert is_scannable_path("apps/mobile/src/screens/Home.tsx", "/proj")
        assert is_scannable_path("packages/shared/src/index.ts", "/proj")
        assert is_scannable_path("docs/architecture.md", "/proj")

    def test_filter_scannable_paths(self):
        paths = [
            "apps/api/src/main.ts",
            "dist/out.js",
            "node_modules/x.js",
        ]
        filtered = filter_scannable_paths(paths)
        assert filtered == ["apps/api/src/main.ts"]
