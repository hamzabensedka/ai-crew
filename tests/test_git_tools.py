"""Tests for git diff helpers used in parallel review."""

from autocrew.tools.git_tools import count_changed_files_from_diff_stat


class TestGitDiffHelpers:
    def test_count_changed_files_from_diff_stat(self):
        stat = """ apps/api/src/payment/payment.service.ts | 120 +++++++++
 apps/mobile/src/screens/PayScreen.tsx   |  45 +++--
 2 files changed, 150 insertions(+), 15 deletions(-)
"""
        assert count_changed_files_from_diff_stat(stat) == 2

    def test_count_zero_for_empty(self):
        assert count_changed_files_from_diff_stat("") == 0
        assert count_changed_files_from_diff_stat("(diff unavailable: x)") == 0
