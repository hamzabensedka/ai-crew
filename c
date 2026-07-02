"""Tests for PerAgentModelRouter and per-agent model routing."""

from __future__ import annotations

import json

import pytest

from autocrew.debate.model_router import (
    DualModelRouter,
    PerAgentModelRouter,
    parse_per_agent_models,
)
from autocrew.squad.squad_model import AgentConfig, AgentRole


class _FakeLLM:
    """Fake LLM client for testing."""

    def __init__(self, label: str) -> None:
        self.label = label

    def complete(self, prompt: str) -> str:
        return json.dumps({"approved": True, "concerns": [], "suggestions": [], "blockers": []})


def _agent(role: AgentRole, name: str = "Test") -> AgentConfig:
    return AgentConfig(
        role=role,
        name=name,
        goal="g",
        backstory="b",
        tools=[],
        can_write_to=[],
        can_read=[],
    )


class TestParsePerAgentModels:
    def test_empty_string_returns_empty(self):
        assert parse_per_agent_models("") == {}

    def test_valid_json(self):
        config = '{"product_owner": "claude-3-5-sonnet", "backend_developer": "deepseek-v4"}'
        result = parse_per_agent_models(config)
        assert result == {
            "product_owner": "claude-3-5-sonnet",
            "backend_developer": "deepseek-v4",
        }

    def test_invalid_json_returns_empty(self):
        assert parse_per_agent_models("not json") == {}

    def test_non_dict_returns_empty(self):
        assert parse_per_agent_models('["a", "b"]') == {}

    def test_filters_empty_values(self):
        config = '{"product_owner": "claude", "backend_developer": ""}'
        result = parse_per_agent_models(config)
        assert result == {"product_owner": "claude"}


class TestPerAgentModelRouter:
    def test_routes_to_specific_model(self):
        llm_a = _FakeLLM("model_a")
        llm_b = _FakeLLM("model_b")
        router = PerAgentModelRouter(
            role_model_map={
                "product_owner": (llm_a, "model_a"),
                "backend_developer": (llm_b, "model_b"),
            },
            default_llm=_FakeLLM("default"),
            default_model="default_model",
        )
        po = _agent(AgentRole.PRODUCT_OWNER)
        backend = _agent(AgentRole.BACKEND_DEV)
        assert router.for_agent(po) == (llm_a, "model_a")
        assert router.for_agent(backend) == (llm_b, "model_b")

    def test_falls_back_to_default(self):
        default_llm = _FakeLLM("default")
        router = PerAgentModelRouter(
            role_model_map={"product_owner": (_FakeLLM("a"), "model_a")},
            default_llm=default_llm,
            default_model="default_model",
        )
        tester = _agent(AgentRole.TESTER)
        assert router.for_agent(tester) == (default_llm, "default_model")

    def test_for_build_task_same_as_for_agent(self):
        llm_a = _FakeLLM("model_a")
        router = PerAgentModelRouter(
            role_model_map={"product_owner": (llm_a, "model_a")},
            default_llm=_FakeLLM("default"),
            default_model="default_model",
        )
        po = _agent(AgentRole.PRODUCT_OWNER)
        # Per-agent routing doesn't change for build tasks
        assert router.for_build_task(po, None) == router.for_agent(po)

    def test_summary_lists_all_roles(self):
        router = PerAgentModelRouter(
            role_model_map={"product_owner": (_FakeLLM("a"), "model_a")},
            default_llm=_FakeLLM("default"),
            default_model="default_model",
        )
        summary = router.summary()
        assert "Per-agent model routing:" in summary
        assert "product_owner -> model_a" in summary
        assert "default_model (default)" in summary

    def test_models_used_returns_all_roles(self):
        router = PerAgentModelRouter(
            role_model_map={
                "product_owner": (_FakeLLM("a"), "model_a"),
                "backend_developer": (_FakeLLM("b"), "model_b"),
            },
            default_llm=_FakeLLM("default"),
            default_model="default_model",
        )
        models = router.models_used()
        assert models["product_owner"] == "model_a"
        assert models["backend_developer"] == "model_b"
        assert models["tester"] == "default_model"
        assert len(models) == 11  # all AgentRole values

    def test_deterministic_role_skipped_in_routing(self):
        """The 'deterministic' model name should not get an LLM client."""
        default_llm = _FakeLLM("default")
        router = PerAgentModelRouter(
            role_model_map={
                "product_owner": (_FakeLLM("a"), "model_a"),
                # progress_tracker would be "deterministic" but it's not in the map
            },
            default_llm=default_llm,
            default_model="default_model",
        )
        tracker = _agent(AgentRole.PROGRESS_TRACKER)
        # Tracker falls back to default (deterministic handling is in debate_runner)
        assert router.for_agent(tracker) == (default_llm, "default_model")


class TestDualModelRouterBackwardCompat:
    """Ensure DualModelRouter still works alongside PerAgentModelRouter."""

    def test_dual_router_still_routes(self):
        planning_llm = _FakeLLM("kimi")
        impl_llm = _FakeLLM("deepseek")
        router = DualModelRouter(
            planning_llm=planning_llm,
            implementation_llm=impl_llm,
            planning_model="kimi",
            implementation_model="deepseek",
        )
        po = _agent(AgentRole.PRODUCT_OWNER)
        backend = _agent(AgentRole.BACKEND_DEV)
        assert router.for_agent(po) == (planning_llm, "kimi")
        assert router.for_agent(backend) == (impl_llm, "deepseek")