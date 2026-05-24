from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class ProjectType(str, Enum):
    NEW_IDEA = "new_idea"
    EXISTING_CODE = "existing_code"


class ProjectDomain(str, Enum):
    SAAS = "saas"
    MOBILE_APP = "mobile_app"
    API = "api"
    DATA_PIPELINE = "data_pipeline"
    ECOMMERCE = "ecommerce"
    AI_TOOL = "ai_tool"
    OTHER = "other"


@dataclass
class TechStack:
    frontend: list[str] = field(default_factory=list)
    backend: list[str] = field(default_factory=list)
    devops: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TechStack":
        data = data or {}
        return cls(
            frontend=list(data.get("frontend", [])),
            backend=list(data.get("backend", [])),
            devops=list(data.get("devops", [])),
            other=list(data.get("other", [])),
        )


@dataclass
class FeatureItem:
    name: str
    description: str
    status: str = "not_started"
    priority: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureItem":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            status=data.get("status", "not_started"),
            priority=data.get("priority", "medium"),
        )


@dataclass
class ProjectContext:
    project_type: ProjectType
    project_name: str
    domain: ProjectDomain
    description: str
    tech_stack: TechStack
    features: list[FeatureItem] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)
    missing_parts: list[str] = field(default_factory=list)
    special_requirements: list[str] = field(default_factory=list)
    raw_idea: Optional[str] = None
    codebase_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["project_type"] = self.project_type.value
        data["domain"] = self.domain.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectContext":
        return cls(
            project_type=ProjectType(data["project_type"]),
            project_name=data["project_name"],
            domain=ProjectDomain(data["domain"]),
            description=data["description"],
            tech_stack=TechStack.from_dict(data.get("tech_stack")),
            features=[FeatureItem.from_dict(f) for f in data.get("features", [])],
            existing_files=list(data.get("existing_files", [])),
            missing_parts=list(data.get("missing_parts", [])),
            special_requirements=list(data.get("special_requirements", [])),
            raw_idea=data.get("raw_idea"),
            codebase_path=data.get("codebase_path"),
        )
