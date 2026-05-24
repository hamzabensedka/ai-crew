"""Analyzes raw project ideas into ProjectContext."""

from __future__ import annotations

from typing import Callable

from autocrew.analyzer.llm_client import LLMClient, call_with_json_retry
from autocrew.analyzer.project_model import (
    FeatureItem,
    ProjectContext,
    ProjectDomain,
    ProjectType,
    TechStack,
)

IDEA_PROMPT = """You are a senior software architect.
A user described a project idea. Extract structured information.

User idea:
\"\"\"
{raw_text}
\"\"\"

Return a JSON object with this exact structure:
{{
  "project_name": "...",
  "domain": "saas|mobile_app|api|data_pipeline|ecommerce|ai_tool|other",
  "description": "...",
  "tech_stack": {{
    "frontend": [...],
    "backend": [...],
    "devops": [...],
    "other": [...]
  }},
  "features": [
    {{"name": "...", "description": "...", "priority": "high|medium|low"}}
  ],
  "special_requirements": ["auth", "billing", "real-time", ...]
}}

If the user didn't specify a tech stack, infer a sensible modern default.
Return only valid JSON. No explanation.
"""


def _parse_idea_response(data: dict, raw_text: str) -> ProjectContext:
    features = [
        FeatureItem(
            name=f["name"],
            description=f.get("description", ""),
            status="not_started",
            priority=f.get("priority", "medium"),
        )
        for f in data.get("features", [])
    ]
    return ProjectContext(
        project_type=ProjectType.NEW_IDEA,
        project_name=data["project_name"],
        domain=ProjectDomain(data["domain"]),
        description=data["description"],
        tech_stack=TechStack.from_dict(data.get("tech_stack")),
        features=features,
        special_requirements=list(data.get("special_requirements", [])),
        raw_idea=raw_text,
    )


def analyze_idea(
    raw_text: str,
    llm: LLMClient | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> ProjectContext:
    prompt = IDEA_PROMPT.format(raw_text=raw_text)
    if llm_call is not None:
        data = call_with_json_retry(llm_call, prompt)
    elif llm is not None:
        data = call_with_json_retry(llm.complete, prompt)
    else:
        raise ValueError("Either llm or llm_call must be provided")
    return _parse_idea_response(data, raw_text)
