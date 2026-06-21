import json

import pytest

from autocrew.analyzer.llm_client import (
    LLMError,
    ResilientLLMClient,
    call_with_json_retry,
    extract_json,
    is_retryable_error,
)
from autocrew.analyzer.idea_analyzer import analyze_idea
from autocrew.analyzer.project_model import ProjectContext, ProjectDomain, ProjectType, TechStack


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        text = '```json\n{"b": 2}\n```'
        assert extract_json(text) == {"b": 2}

    def test_invalid_json_raises(self):
        with pytest.raises(LLMError):
            extract_json("not json")


class TestJsonRetry:
    def test_succeeds_first_try(self):
        result = call_with_json_retry(lambda p: '{"ok": true}', "prompt")
        assert result == {"ok": True}

    def test_retries_on_invalid_json(self):
        calls = []

        def flaky(_prompt):
            calls.append(1)
            return "{invalid}" if len(calls) == 1 else '{"fixed": true}'

        result = call_with_json_retry(flaky, "prompt", max_retries=1)
        assert result == {"fixed": True}
        assert len(calls) == 2


class TestProjectContext:
    def test_roundtrip_serialization(self):
        ctx = ProjectContext(
            project_type=ProjectType.NEW_IDEA,
            project_name="Test",
            domain=ProjectDomain.SAAS,
            description="A test project",
            tech_stack=TechStack(frontend=["React"], backend=["FastAPI"]),
            raw_idea="build a crm",
        )
        restored = ProjectContext.from_dict(ctx.to_dict())
        assert restored.project_name == "Test"
        assert restored.tech_stack.frontend == ["React"]


class TestAnalyzeIdea:
    def test_parses_llm_response(self, sample_idea_json):
        def mock_llm(_prompt):
            return json.dumps(sample_idea_json)

        ctx = analyze_idea("Build a CRM", llm_call=mock_llm)
        assert ctx.project_name == "TaskFlow CRM"
        assert ctx.project_type == ProjectType.NEW_IDEA
        assert all(f.status == "not_started" for f in ctx.features)
        assert ctx.domain == ProjectDomain.SAAS


class TestResilience:
    def test_is_retryable_error_detects_504(self):
        assert is_retryable_error(LLMError("NVIDIA API error: Error code: 504"))

    def test_resilient_client_retries_then_succeeds(self, monkeypatch):
        calls = {"count": 0}

        class FlakyClient:
            def complete(self, prompt: str) -> str:
                calls["count"] += 1
                if calls["count"] < 3:
                    raise LLMError("NVIDIA API error: Error code: 504")
                return '{"ok": true}'

        monkeypatch.setattr("autocrew.analyzer.llm_client.time.sleep", lambda *_: None)
        client = ResilientLLMClient(FlakyClient(), max_retries=4, backoff_seconds=1.0)
        assert client.complete("hi") == '{"ok": true}'
        assert calls["count"] == 3


class TestNvidiaClient:
    def test_create_nvidia_client(self):
        from autocrew.analyzer.llm_client import NvidiaClient, create_llm_client

        from autocrew.analyzer.llm_client import ResilientLLMClient

        client = create_llm_client(
            nvidia_key="test-nvapi-key",
            default_model="deepseek-ai/deepseek-v4-pro",
            fallback_model="moonshotai/kimi-k2.6",
            llm_provider="nvidia",
        )
        assert isinstance(client, ResilientLLMClient)
        assert isinstance(client.primary, NvidiaClient)

    def test_nvidia_deepseek_v4_pro_request(self, monkeypatch):
        from autocrew.analyzer.llm_client import NvidiaClient

        captured: dict = {}

        class FakeMessage:
            content = '{"ok": true}'
            reasoning_content = None

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = FakeChat()

        monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

        client = NvidiaClient("nvapi-test", "deepseek-ai/deepseek-v4-pro")
        result = client.complete("Return JSON")
        assert result == '{"ok": true}'
        assert captured["model"] == "deepseek-ai/deepseek-v4-pro"
        assert captured["stream"] is False
        assert captured["temperature"] == 1.0
        assert captured["top_p"] == 0.95
        assert captured["extra_body"] == {"chat_template_kwargs": {"thinking": False}}

    def test_nvidia_nemotron_streams_with_thinking(self, monkeypatch):
        from autocrew.analyzer.llm_client import NvidiaClient

        captured: dict = {}

        class FakeDelta:
            content = "answer"
            reasoning_content = None

        class FakeChunkChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChunkChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return iter([FakeChunk()])

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = FakeChat()

        monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

        client = NvidiaClient("nvapi-test", "nvidia/nemotron-3-ultra-550b-a55b")
        result = client.complete("Plan this")
        assert result == "answer"
        assert captured["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert captured["stream"] is True
        assert captured["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 4096,
        }

    def test_nvidia_complete_legacy(self, monkeypatch):
        from autocrew.analyzer.llm_client import NvidiaClient

        class FakeMessage:
            content = '{"ok": true}'
            reasoning_content = None

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                assert kwargs["model"] == "deepseek-ai/deepseek-v4-flash"
                assert kwargs["stream"] is False
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                assert kwargs["api_key"] == "nvapi-test"
                assert kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
                self.chat = FakeChat()

        monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

        client = NvidiaClient("nvapi-test", "deepseek-ai/deepseek-v4-flash")
        result = client.complete("Return JSON")
        assert result == '{"ok": true}'

    def test_gateway_timeout_switches_to_fallback(self, monkeypatch):
        from autocrew.analyzer.llm_client import LLMError, ResilientLLMClient

        monkeypatch.setattr("autocrew.analyzer.llm_client.time.sleep", lambda *_: None)

        class Primary:
            def complete(self, prompt: str) -> str:
                raise LLMError("NVIDIA API error: Error code: 504")

        class Fallback:
            def complete(self, prompt: str) -> str:
                return "fallback ok"

        client = ResilientLLMClient(
            Primary(),
            Fallback(),
            max_retries=6,
            label="nvidia/nemotron-3-ultra-550b-a55b",
        )
        assert client.complete("hi") == "fallback ok"

    def test_has_api_keys_includes_nvidia(self, monkeypatch):
        from autocrew.config import Settings

        s = Settings(nvidia_api_key="nvapi-test")
        assert s.has_api_keys()


class TestZenMuxClient:
    def test_create_zenmux_client(self):
        from autocrew.analyzer.llm_client import ResilientLLMClient, ZenMuxClient, create_llm_client

        client = create_llm_client(
            zenmux_key="zenmux-test-key",
            default_model="z-ai/glm-5.2-free",
            fallback_model="moonshotai/kimi-k2.7-code-free",
            llm_provider="zenmux",
        )
        assert isinstance(client, ResilientLLMClient)
        assert isinstance(client.primary, ZenMuxClient)

    def test_zenmux_complete(self, monkeypatch):
        from autocrew.analyzer.llm_client import ZenMuxClient

        class FakeMessage:
            content = '{"ok": true}'
            reasoning_content = None

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                assert kwargs["model"] == "z-ai/glm-5.2-free"
                assert kwargs["stream"] is False
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                assert kwargs["api_key"] == "zenmux-test"
                assert kwargs["base_url"] == "https://zenmux.ai/api/v1"
                self.chat = FakeChat()

        monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

        client = ZenMuxClient("zenmux-test", "z-ai/glm-5.2-free")
        result = client.complete("Return JSON")
        assert result == '{"ok": true}'

    def test_has_api_keys_includes_zenmux(self):
        from autocrew.config import Settings

        s = Settings(zenmux_api_key="zenmux-test")
        assert s.has_api_keys()
