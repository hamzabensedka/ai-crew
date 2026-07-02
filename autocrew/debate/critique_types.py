"""Structured critique item types (no AgentCritique dependency)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StructuredConcern:
    id: str
    severity: str
    text: str
    targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredConcern:
        return cls(
            id=str(data.get("id", "")),
            severity=str(data.get("severity", "medium")),
            text=str(data.get("text", "")),
            targets=[str(t) for t in data.get("targets", [])],
        )


@dataclass
class StructuredDecision:
    id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredDecision:
        return cls(id=str(data.get("id", "")), text=str(data.get("text", "")))


@dataclass
class StructuredOpenQuestion:
    id: str
    text: str
    for_roles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "for": self.for_roles}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredOpenQuestion:
        for_roles = data.get("for", data.get("for_roles", []))
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            for_roles=[str(r) for r in for_roles],
        )
