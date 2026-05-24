from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCritique":
        return cls(
            agent_role=data["agent_role"],
            agent_name=data["agent_name"],
            round_number=data["round_number"],
            approved=data["approved"],
            concerns=list(data.get("concerns", [])),
            suggestions=list(data.get("suggestions", [])),
            blockers=list(data.get("blockers", [])),
            model_used=data.get("model_used", ""),
        )


@dataclass
class DebateRound:
    round_number: int
    critiques: list[AgentCritique]
    revised_plan_excerpt: str
    all_approved: bool
    total_blockers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "critiques": [c.to_dict() for c in self.critiques],
            "revised_plan_excerpt": self.revised_plan_excerpt,
            "all_approved": self.all_approved,
            "total_blockers": self.total_blockers,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "timestamp": self.timestamp,
            "rounds": [r.to_dict() for r in self.rounds],
            "consensus_reached": self.consensus_reached,
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
                )
                for r in data["rounds"]
            ],
            consensus_reached=data["consensus_reached"],
            final_plan_path=data["final_plan_path"],
            debate_dir=data["debate_dir"],
            action_items=list(data.get("action_items", [])),
            models_used=dict(data.get("models_used", {})),
        )
