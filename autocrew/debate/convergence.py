"""Debate convergence detection for early-exit between rounds."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from autocrew.debate.critique_types import StructuredConcern, StructuredOpenQuestion
from autocrew.debate.debate_model import AgentCritique
from autocrew.squad.squad_model import AgentRole

EXCLUDED_ROLES = {AgentRole.PROGRESS_TRACKER.value}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True)
class TrackedItem:
    kind: str
    item_id: str
    text: str
    agent_role: str

    @property
    def match_key(self) -> str:
        if self.item_id:
            return f"{self.kind}:id:{self.item_id}"
        return f"{self.kind}:text:{_normalize_text(self.text)}"


@dataclass
class ConvergenceDiff:
    previous_round: int
    current_round: int
    previous_concerns: list[TrackedItem] = field(default_factory=list)
    previous_open_questions: list[TrackedItem] = field(default_factory=list)
    current_concerns: list[TrackedItem] = field(default_factory=list)
    current_open_questions: list[TrackedItem] = field(default_factory=list)
    net_new_concerns: list[TrackedItem] = field(default_factory=list)
    net_new_open_questions: list[TrackedItem] = field(default_factory=list)

    @property
    def has_net_new(self) -> bool:
        return bool(self.net_new_concerns or self.net_new_open_questions)

    def to_dict(self) -> dict:
        return {
            "previous_round": self.previous_round,
            "current_round": self.current_round,
            "net_new_concern_count": len(self.net_new_concerns),
            "net_new_open_question_count": len(self.net_new_open_questions),
            "net_new_concerns": [asdict(i) for i in self.net_new_concerns],
            "net_new_open_questions": [asdict(i) for i in self.net_new_open_questions],
        }


def _items_from_concerns(
    concerns: list[StructuredConcern],
    *,
    agent_role: str,
) -> list[TrackedItem]:
    return [
        TrackedItem(
            kind="concern",
            item_id=c.id,
            text=c.text,
            agent_role=agent_role,
        )
        for c in concerns
        if c.text.strip()
    ]


def _items_from_questions(
    questions: list[StructuredOpenQuestion],
    *,
    agent_role: str,
) -> list[TrackedItem]:
    return [
        TrackedItem(
            kind="open_question",
            item_id=q.id,
            text=q.text,
            agent_role=agent_role,
        )
        for q in questions
        if q.text.strip()
    ]


def collect_round_items(critiques: list[AgentCritique]) -> tuple[list[TrackedItem], list[TrackedItem]]:
    """Aggregate concerns and open_questions from a round (excludes Progress Tracker)."""
    concerns: list[TrackedItem] = []
    questions: list[TrackedItem] = []
    for critique in critiques:
        if critique.agent_role in EXCLUDED_ROLES:
            continue
        if critique.structured_concerns or critique.structured_open_questions:
            concerns.extend(
                _items_from_concerns(critique.structured_concerns, agent_role=critique.agent_role)
            )
            questions.extend(
                _items_from_questions(
                    critique.structured_open_questions,
                    agent_role=critique.agent_role,
                )
            )
        else:
            for index, text in enumerate(critique.concerns, 1):
                if text.strip():
                    concerns.append(
                        TrackedItem(
                            kind="concern",
                            item_id=f"c{index}",
                            text=text,
                            agent_role=critique.agent_role,
                        )
                    )
    return concerns, questions


def _net_new(current: list[TrackedItem], previous: list[TrackedItem]) -> list[TrackedItem]:
    prior_keys = {item.match_key for item in previous}
    prior_texts = {_normalize_text(item.text) for item in previous if item.text.strip()}
    result: list[TrackedItem] = []
    seen: set[str] = set()
    for item in current:
        key = item.match_key
        if key in seen:
            continue
        seen.add(key)
        if key in prior_keys:
            continue
        if _normalize_text(item.text) in prior_texts:
            continue
        result.append(item)
    return result


def diff_rounds(
    previous_critiques: list[AgentCritique],
    current_critiques: list[AgentCritique],
    *,
    previous_round: int,
    current_round: int,
) -> ConvergenceDiff:
    prev_concerns, prev_questions = collect_round_items(previous_critiques)
    cur_concerns, cur_questions = collect_round_items(current_critiques)
    return ConvergenceDiff(
        previous_round=previous_round,
        current_round=current_round,
        previous_concerns=prev_concerns,
        previous_open_questions=prev_questions,
        current_concerns=cur_concerns,
        current_open_questions=cur_questions,
        net_new_concerns=_net_new(cur_concerns, prev_concerns),
        net_new_open_questions=_net_new(cur_questions, prev_questions),
    )


def should_early_exit(
    diff: ConvergenceDiff,
    *,
    round_number: int,
    min_rounds: int,
    consecutive_stable_rounds: int = 1,
    stable_rounds_required: int = 2,
) -> bool:
    """True when debate positions stabilized for ``stable_rounds_required`` rounds."""
    if round_number < min_rounds:
        return False
    if round_number < 2:
        return False
    if diff.has_net_new:
        return False
    return consecutive_stable_rounds >= stable_rounds_required


@dataclass
class ConvergenceTracker:
    """Track consecutive rounds with no meaningful position change."""

    consecutive_stable_rounds: int = 0

    def update(self, diff: ConvergenceDiff) -> None:
        if diff.has_net_new:
            self.consecutive_stable_rounds = 0
        else:
            self.consecutive_stable_rounds += 1

    def should_exit(
        self,
        *,
        round_number: int,
        min_rounds: int,
        stable_rounds_required: int = 2,
    ) -> bool:
        if round_number < min_rounds or round_number < 2:
            return False
        return self.consecutive_stable_rounds >= stable_rounds_required


def log_early_exit_event(
    log_path: Path,
    *,
    project_name: str,
    task_id: str,
    round_number: int,
    diff: ConvergenceDiff,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "debate_early_exit",
        "project_name": project_name,
        "task_id": task_id,
        "round_number": round_number,
        "reason": "zero_net_new_concerns_and_open_questions",
        "diff": diff.to_dict(),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=True) + "\n")
