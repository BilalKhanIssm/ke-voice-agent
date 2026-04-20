import types

import pytest

from app.config import Settings
from app.telephony import llm_warmup


class _FakeStream:
    def __init__(self) -> None:
        self._sent = False

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> str:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return "0"


class _FakeLLM:
    def __init__(self) -> None:
        self.prewarm_called = False
        self.chat_calls: list[dict[object, object]] = []

    def prewarm(self) -> None:
        self.prewarm_called = True

    def chat(self, **kwargs: object) -> _FakeStream:
        self.chat_calls.append(kwargs)
        return _FakeStream()


@pytest.mark.asyncio
async def test_warmup_runs_tiny_chat_when_enabled() -> None:
    env = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "a",
        "LIVEKIT_API_SECRET": "b",
        "LIVEKIT_AGENT_NAME": "x",
        "DEEPGRAM_API_KEY": "d",
        "UPLIFTAI_API_KEY": "u",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "k",
        "LLM_WARMUP_ENABLED": "true",
    }
    settings = Settings(**env)
    fake_llm = _FakeLLM()
    session = types.SimpleNamespace(llm=fake_llm)
    await llm_warmup.warmup_llm_if_enabled(settings, session)
    assert fake_llm.prewarm_called
    assert len(fake_llm.chat_calls) == 1


@pytest.mark.asyncio
async def test_warmup_skipped_when_disabled() -> None:
    env = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "a",
        "LIVEKIT_API_SECRET": "b",
        "LIVEKIT_AGENT_NAME": "x",
        "DEEPGRAM_API_KEY": "d",
        "UPLIFTAI_API_KEY": "u",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "k",
        "LLM_WARMUP_ENABLED": "false",
    }
    settings = Settings(**env)
    fake_llm = _FakeLLM()
    session = types.SimpleNamespace(llm=fake_llm)
    await llm_warmup.warmup_llm_if_enabled(settings, session)
    assert not fake_llm.chat_calls
