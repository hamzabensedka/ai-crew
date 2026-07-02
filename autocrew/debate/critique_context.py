"""Build per-agent debate context from structured prior critiques."""

from __future__ import annotations

import json

from autocrew.config import settings
from autocrew.debate.critique_schema import structured_payload_dict
from autocrew.debate.debate_model import AgentCritique
from autocrew.squad.squad_model import AgentRole


FULL_CONTEXT_ROLES = {AgentRole.CODE_REVIEWER.value, AgentRole.TESTER.value}


def _summarize_critique(critique: AgentCritique) -> dict:
    """Compact summary preserving concern ids and first-line text."""
    return {
        "agent": critique.agent_name,
        "role": critique.agent_role,
        "approved": critique.approved,
        "concerns": [
            {"id": c.id, "severity": c.severity, "text": c.text[:160]}
            for c in critique.structured_concerns[:8]
        ],
        "blockers": [
            {"id": b.id, "text": b.text[:160]}
            for b in critique.structured_blockers[:5]
        ],
        "open_questions": [
            {"id": q.id, "text": q.text[:160]}
            for q in critique.structured_open_questions[:5]
        ],
        "decisions_count": len(critique.structured_decisions),
        "_summary": True,
    }


def _full_critique_payload(critique: AgentCritique, *, max_items: int | None = None) -> dict:
    payload = structured_payload_dict(critique)
    if max_items is None:
        return payload
    payload["concerns"] = payload["concerns"][:max_items]
    payload["blockers"] = payload["blockers"][:max_items]
    payload["open_questions"] = payload["open_questions"][:max_items]
    payload["decisions"] = payload["decisions"][:max_items]
    return payload


def _turns_back(receiver_index: int, critique_index: int) -> int:
    return receiver_index - critique_index


def build_critique_context(
    prior_critiques: list[AgentCritique],
    receiver_role: str,
    *,
    receiver_index: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    Build context for the receiving agent.

    - Immediate predecessor: full structured JSON.
    - Drew/Jamie: full structured JSON for every prior agent in the round.
    - Other agents: summary for critiques 2+ turns back.
    """
    if not prior_critiques:
        return "(none yet)"

    if receiver_index is None:
        receiver_index = len(prior_critiques)

    budget = max_chars if max_chars is not None else settings.debate_context_max_chars
    use_structured = settings.debate_structured_critiques
    full_for_all_prior = receiver_role in FULL_CONTEXT_ROLES

    sections: list[dict] = []
    for idx, critique in enumerate(prior_critiques):
        turns = _turns_back(receiver_index, idx)
        if not use_structured:
            items = critique.blockers + critique.concerns + critique.suggestions
            if items:
                sections.append({
                    "agent": critique.agent_name,
                    "legacy_summary": "; ".join(items[:5]),
                })
            continue

        if full_for_all_prior or turns <= 1:
            sections.append(_full_critique_payload(critique))
        else:
            sections.append(_summarize_critique(critique))

    rendered = json.dumps(sections, indent=2, ensure_ascii=True)

    if len(rendered) <= budget:
        return rendered

    compact = json.dumps(sections, ensure_ascii=True, separators=(",", ":"))
    if len(compact) <= budget:
        return compact

    # Shrink: truncate list items in summarized sections first, then trim full sections.
    for trim_items in (5, 3, 1):
        shrunk: list[dict] = []
        for idx, critique in enumerate(prior_critiques):
            turns = _turns_back(receiver_index, idx)
            if full_for_all_prior or turns <= 1:
                shrunk.append(_full_critique_payload(critique, max_items=trim_items))
            else:
                shrunk.append(_summarize_critique(critique))
        rendered = json.dumps(shrunk, ensure_ascii=True, separators=(",", ":"))
        if len(rendered) <= budget:
            return rendered

    return compact[:budget] + "...[truncated]"


def context_contains_concern_ids(context: str, concern_ids: list[str]) -> bool:
    return all(cid in context for cid in concern_ids)


def context_contains_texts(context: str, phrases: list[str]) -> bool:
    lower = context.lower()
    return all(phrase.lower() in lower for phrase in phrases)
