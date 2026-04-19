import os

import pytest

from app.config import Settings


def _base_env():
    return {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "abc",
        "LIVEKIT_API_SECRET": "def",
        "LIVEKIT_AGENT_NAME": "telephony-agent",
        "DEEPGRAM_API_KEY": "dg",
        "UPLIFTAI_API_KEY": "up",
    }


def test_config_openai_path():
    env = _base_env() | {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "ok"}
    settings = Settings(**env)
    assert settings.llm_provider == "openai"


def test_config_openrouter_requires_key():
    env = _base_env() | {"LLM_PROVIDER": "openrouter"}
    with pytest.raises(Exception):
        Settings(**env)
