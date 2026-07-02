from dataclasses import asdict, dataclass, field
from typing import Any

from autocrew.debate.critique_types import (
    StructuredConcern,
    StructuredDecision,
    StructuredOpenQuestion,
)


@dataclass
class AgentCritique:
    agent_role: str
    agent_name: str
    round_number: int
    approved: bool
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    model_used: str = ""
    structured_concerns: list[StructuredConcern] = field(default_factory=list)
    structured_decisions: list[StructuredDecision] = field(default_factory=list)
    structured_open_questions: list[StructuredOpenQuestion] = field(default_factory=list)
    structured_blockers: list[StructuredConcern] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["structured_concerns"] = [c.to_dict() for c in self.structured_concerns]
        data["structured_decisions"] = [d.to_dict() for d in self.structured_decisions]
        data["structured_open_questions"] = [q.to_dict() for q in self.structured_open_questions]
        data["structured_blockers"] = [b.to_dict() for b in self.structured_blockers]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCritique":
        critique = cls(
            agent_role=data["agent_role"],
            agent_name=data["agent_name"],
            round_number=data["round_number"],
            approved=data["approved"],
            concerns=list(data.get("concerns", [])),
            suggestions=list(data.get("suggestions", [])),
            blockers=list(data.get("blockers", [])),
            model_used=data.get("model_used", ""),
            structured_concerns=[
                StructuredConcern.from_dict(c) for c in data.get("structured_concerns", [])
            ],
            structured_decisions=[
                StructuredDecision.from_dict(d) for d in data.get("structured_decisions", [])
            ],
            structured_open_questions=[
                StructuredOpenQuestion.from_dict(q)
                for q in data.get("structured_open_questions", [])
            ],
            structured_blockers=[
                StructuredConcern.from_dict(b) for b in data.get("structured_blockers", [])
            ],
        )
        from autocrew.debate.critique_schema import attach_structured_fields

        return attach_structured_fields(critique)


@dataclass
class DebateRound:
    round_number: int
    critiques: list[AgentCritique]
    revised_plan_excerpt: str
    all_approved: bool
    total_blockers: int
    converged_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "critiques": [c.to_dict() for c in self.critiques],
            "revised_plan_excerpt": self.revised_plan_excerpt,
            "all_approved": self.all_approved,
            "total_blockers": self.total_blockers,
            "converged_early": self.converged_early,
        }


@dataclass
class DebateResult:
    project_name: str
    timestamp: str
    rounds: list[DebateRound]
    consensus_reached: bool
    final_plan_path: str
    debate_dir: str
    action_items: list[str] = field(default_factory=list)
    models_used: dict[str, str] = field(default_factory=dict)
    converged_early: bool = False
    early_exit_round: int | None = None
    early_exit_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "timestamp": self.timestamp,
            "rounds": [r.to_dict() for r in self.rounds],
            "consensus_reached": self.consensus_reached,
            "converged_early": self.converged_early,
            "early_exit_round": self.early_exit_round,
            "early_exit_log_path": self.early_exit_log_path,
            "final_plan_path": self.final_plan_path,
            "debate_dir": self.debate_dir,
            "action_items": self.action_items,
            "models_used": self.models_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebateResult":
        return cls(
            project_name=data["project_name"],
            timestamp=data["timestamp"],
            rounds=[
                DebateRound(
                    round_number=r["round_number"],
                    critiques=[AgentCritique.from_dict(c) for c in r["critiques"]],
                    revised_plan_excerpt=r["revised_plan_excerpt"],
                    all_approved=r["all_approved"],
                    total_blockers=r["total_blockers"],
                    converged_early=bool(r.get("converged_early", False)),
                )
                for r in data["rounds"]
            ],
            consensus_reached=data["consensus_reached"],
            final_plan_path=data["final_plan_path"],
            debate_dir=data["debate_dir"],
            action_items=list(data.get("action_items", [])),
            models_used=dict(data.get("models_used", {})),
            converged_early=bool(data.get("converged_early", False)),
            early_exit_round=data.get("early_exit_round"),
            early_exit_log_path=data.get("early_exit_log_path"),
        )
