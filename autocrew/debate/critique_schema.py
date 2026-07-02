"""Structured debate critique schema — parse, serialize, and legacy migration."""

from __future__ import annotations

import re
from typing import Any

from autocrew.debate.critique_types import (
    StructuredConcern,
    StructuredDecision,
    StructuredOpenQuestion,
)
from autocrew.debate.debate_model import AgentCritique
from autocrew.squad.squad_model import AgentConfig


def _slug_id(prefix: str, index: int, text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", text.lower()[:24]).strip("_") or "item"
    return f"{prefix}{index}_{token}"


def _parse_concern_list(
    items: list[Any],
    *,
    prefix: str,
    default_severity: str = "medium",
) -> list[StructuredConcern]:
    result: list[StructuredConcern] = []
    for index, item in enumerate(items, 1):
        if isinstance(item, str) and item.strip():
            result.append(
                StructuredConcern(
                    id=_slug_id(prefix, index, item),
                    severity=default_severity,
                    text=item.strip(),
                    targets=[],
                )
            )
        elif isinstance(item, dict) and item.get("text"):
            result.append(
                StructuredConcern(
                    id=str(item.get("id") or _slug_id(prefix, index, str(item["text"]))),
                    severity=str(item.get("severity", default_severity)),
                    text=str(item["text"]),
                    targets=[str(t) for t in item.get("targets", [])],
                )
            )
    return result


def _parse_decision_list(items: list[Any], *, prefix: str) -> list[StructuredDecision]:
    result: list[StructuredDecision] = []
    for index, item in enumerate(items, 1):
        if isinstance(item, str) and item.strip():
            result.append(
                StructuredDecision(id=_slug_id(prefix, index, item), text=item.strip())
            )
        elif isinstance(item, dict) and item.get("text"):
            result.append(
                StructuredDecision(
                    id=str(item.get("id") or _slug_id(prefix, index, str(item["text"]))),
                    text=str(item["text"]),
                )
            )
    return result


def _parse_open_questions(items: list[Any]) -> list[StructuredOpenQuestion]:
    result: list[StructuredOpenQuestion] = []
    for index, item in enumerate(items, 1):
        if isinstance(item, str) and item.strip():
            result.append(
                StructuredOpenQuestion(
                    id=_slug_id("q", index, item),
                    text=item.strip(),
                    for_roles=[],
                )
            )
        elif isinstance(item, dict) and item.get("text"):
            result.append(StructuredOpenQuestion.from_dict(item))
    return result


def structured_payload_dict(critique: AgentCritique) -> dict[str, Any]:
    return {
        "agent": critique.agent_name,
        "role": critique.agent_role,
        "approved": critique.approved,
        "concerns": [c.to_dict() for c in critique.structured_concerns],
        "decisions": [d.to_dict() for d in critique.structured_decisions],
        "open_questions": [q.to_dict() for q in critique.structured_open_questions],
        "blockers": [b.to_dict() for b in critique.structured_blockers],
    }


def attach_structured_fields(
    critique: AgentCritique,
    *,
    concerns: list[StructuredConcern] | None = None,
    decisions: list[StructuredDecision] | None = None,
    open_questions: list[StructuredOpenQuestion] | None = None,
    blockers: list[StructuredConcern] | None = None,
) -> AgentCritique:
    if concerns is not None:
        critique.structured_concerns = concerns
    if decisions is not None:
        critique.structured_decisions = decisions
    if open_questions is not None:
        critique.structured_open_questions = open_questions
    if blockers is not None:
        critique.structured_blockers = blockers

    if not critique.structured_concerns and critique.concerns:
        critique.structured_concerns = _parse_concern_list(critique.concerns, prefix="c")
    if not critique.structured_decisions and critique.suggestions:
        critique.structured_decisions = _parse_decision_list(critique.suggestions, prefix="d")
    if not critique.structured_blockers and critique.blockers:
        critique.structured_blockers = _parse_concern_list(
            critique.blockers, prefix="b", default_severity="high"
        )

    critique.concerns = [c.text for c in critique.structured_concerns]
    critique.suggestions = [d.text for d in critique.structured_decisions]
    critique.blockers = [b.text for b in critique.structured_blockers]
    return critique


def parse_critique_response(
    data: dict[str, Any],
    agent: AgentConfig,
    round_number: int,
    *,
    model_used: str = "",
) -> AgentCritique:
    """Parse LLM JSON into AgentCritique with structured fields as source of truth."""
    concerns = _parse_concern_list(list(data.get("concerns", [])), prefix="c")
    blockers = _parse_concern_list(
        list(data.get("blockers", [])), prefix="b", default_severity="high"
    )
    decisions = _parse_decision_list(
        list(data.get("decisions", data.get("suggestions", []))),
        prefix="d",
    )
    open_questions = _parse_open_questions(list(data.get("open_questions", [])))

    critique = AgentCritique(
        agent_role=agent.role.value,
        agent_name=agent.name,
        round_number=round_number,
        approved=bool(data.get("approved", False)),
        concerns=[],
        suggestions=[],
        blockers=[],
        model_used=model_used,
        structured_concerns=concerns,
        structured_decisions=decisions,
        structured_open_questions=open_questions,
        structured_blockers=blockers,
    )
    return attach_structured_fields(critique)


STRUCTURED_CRITIQUE_PROMPT_SCHEMA = """Return JSON only (no markdown fences):
{{
  "approved": true or false,
  "concerns": [
    {{"id": "c1", "severity": "high|medium|low", "text": "specific issue", "targets": ["role_or_name"]}}
  ],
  "decisions": [
    {{"id": "d1", "text": "concrete recommendation"}}
  ],
  "open_questions": [
    {{"id": "q1", "text": "unresolved question", "for": ["role_to_answer"]}}
  ],
  "blockers": [
    {{"id": "b1", "severity": "high", "text": "must-fix before implementation", "targets": []}}
  ]
}}

Use unique ids (c1, c2, b1, ...). Approve only with zero blockers and no major concerns.
"""
