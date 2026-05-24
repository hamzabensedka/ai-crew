from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FeatureStatus:
    name: str
    status: str
    details: str
    files_involved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressReport:
    timestamp: str
    project_name: str
    completion_percentage: float
    done: list[FeatureStatus] = field(default_factory=list)
    partial: list[FeatureStatus] = field(default_factory=list)
    missing: list[FeatureStatus] = field(default_factory=list)
    bugs: list[FeatureStatus] = field(default_factory=list)
    next_priorities: list[str] = field(default_factory=list)
    raw_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "project_name": self.project_name,
            "completion_percentage": self.completion_percentage,
            "done": [f.to_dict() for f in self.done],
            "partial": [f.to_dict() for f in self.partial],
            "missing": [f.to_dict() for f in self.missing],
            "bugs": [f.to_dict() for f in self.bugs],
            "next_priorities": self.next_priorities,
            "raw_summary": self.raw_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressReport":
        def parse_features(key: str) -> list[FeatureStatus]:
            return [FeatureStatus(**f) for f in data.get(key, [])]

        return cls(
            timestamp=data["timestamp"],
            project_name=data["project_name"],
            completion_percentage=data["completion_percentage"],
            done=parse_features("done"),
            partial=parse_features("partial"),
            missing=parse_features("missing"),
            bugs=parse_features("bugs"),
            next_priorities=list(data.get("next_priorities", [])),
            raw_summary=data.get("raw_summary", ""),
        )
