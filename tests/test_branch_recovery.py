"""Tests for remote branch recovery helpers."""

from autocrew.tools.branch_recovery import pick_branches


class TestBranchRecovery:
    def test_pick_latest_per_role(self):
        catalog = {
            "backend-developer": [
                "autocrew/20260801_112515/backend-developer",
                "autocrew/20260801_103236/backend-developer",
            ],
            "devops-engineer": [
                "autocrew/20260801_112515/devops-engineer",
            ],
        }
        picked = pick_branches(catalog, ["backend-developer", "devops-engineer"])
        assert picked == [
            "autocrew/20260801_112515/backend-developer",
            "autocrew/20260801_112515/devops-engineer",
        ]

    def test_pick_specific_session(self):
        catalog = {
            "backend-developer": [
                "autocrew/20260801_112515/backend-developer",
                "autocrew/20260801_103236/backend-developer",
            ],
        }
        picked = pick_branches(catalog, ["backend-developer"], session="20260801_103236")
        assert picked == ["autocrew/20260801_103236/backend-developer"]
