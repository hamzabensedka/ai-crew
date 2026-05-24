from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class TaskConfig:
    task_id: str
    title: str
    description: str
    assigned_agent_role: str
    depends_on: list[str] = field(default_factory=list)
    output_format: str = "markdown"
    output_path: Optional[str] = None
    expected_output: str = ""
    context_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        return cls(
            task_id=data["task_id"],
            title=data["title"],
            description=data["description"],
            assigned_agent_role=data["assigned_agent_role"],
            depends_on=list(data.get("depends_on", [])),
            output_format=data.get("output_format", "markdown"),
            output_path=data.get("output_path"),
            expected_output=data.get("expected_output", ""),
            context_files=list(data.get("context_files", [])),
        )
