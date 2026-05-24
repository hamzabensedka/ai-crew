from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    PRODUCT_OWNER = "product_owner"
    ARCHITECT = "architect"
    BACKEND_DEV = "backend_developer"
    FRONTEND_DEV = "frontend_developer"
    FULLSTACK_DEV = "fullstack_developer"
    DEVOPS = "devops_engineer"
    DATA_ENGINEER = "data_engineer"
    AI_ENGINEER = "ai_engineer"
    CODE_REVIEWER = "code_reviewer"
    PROGRESS_TRACKER = "progress_tracker"
    TESTER = "tester"


@dataclass
class AgentConfig:
    role: AgentRole
    name: str
    goal: str
    backstory: str
    tools: list[str]
    can_write_to: list[str]
    can_read: list[str]
    verbose: bool = True
    allow_delegation: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        return cls(
            role=AgentRole(data["role"]),
            name=data["name"],
            goal=data["goal"],
            backstory=data["backstory"],
            tools=list(data.get("tools", [])),
            can_write_to=list(data.get("can_write_to", [])),
            can_read=list(data.get("can_read", [])),
            verbose=data.get("verbose", True),
            allow_delegation=data.get("allow_delegation", False),
        )


@dataclass
class Squad:
    project_name: str
    agents: list[AgentConfig]
    execution_order: list[str]
    parallel_groups: list[list[str]]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "agents": [a.to_dict() for a in self.agents],
            "execution_order": self.execution_order,
            "parallel_groups": self.parallel_groups,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Squad":
        return cls(
            project_name=data["project_name"],
            agents=[AgentConfig.from_dict(a) for a in data["agents"]],
            execution_order=list(data.get("execution_order", [])),
            parallel_groups=[list(g) for g in data.get("parallel_groups", [])],
            created_at=data["created_at"],
        )
