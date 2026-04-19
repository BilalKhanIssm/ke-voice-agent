import importlib
import types

import pytest


def _set_min_env(monkeypatch):
    monkeypatch.delenv("TEST_START_LANGUAGE", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "abc")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "def")
    monkeypatch.setenv("LIVEKIT_AGENT_NAME", "telephony-agent")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("UPLIFTAI_API_KEY", "up")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")


def _reload_entrypoint_fresh(monkeypatch):
    _set_min_env(monkeypatch)
    import app.config as config_mod

    config_mod.get_settings.cache_clear()
    entrypoint_mod = importlib.import_module("app.telephony.entrypoint")
    importlib.reload(entrypoint_mod)
    return entrypoint_mod


class DummySession:
    def __init__(self):
        self.started = False

    def on(self, *_args, **_kwargs):
        return None

    async def start(self, room, agent):
        self.started = True

    async def say(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_session_starts_with_ivr_gate(monkeypatch):
    entrypoint_mod = _reload_entrypoint_fresh(monkeypatch)

    session = DummySession()

    async def fake_on_ivr_failed():
        return None

    def fake_build_ivr_session(_settings, _voice_agent_holder, _on_ivr_failed):
        return object(), session

    monkeypatch.setattr(entrypoint_mod, "build_ivr_session", fake_build_ivr_session)

    ctx = types.SimpleNamespace(room=types.SimpleNamespace(name="room-x"))
    await entrypoint_mod.entrypoint(ctx)
    assert session.started


@pytest.mark.asyncio
async def test_build_ivr_session_invocation(monkeypatch):
    entrypoint_mod = _reload_entrypoint_fresh(monkeypatch)

    session = DummySession()
    called = {"holder": None, "has_cb": False}

    def fake_build_ivr_session(_settings, voice_agent_holder, on_ivr_failed):
        called["holder"] = voice_agent_holder
        called["has_cb"] = callable(on_ivr_failed)
        return object(), session

    monkeypatch.setattr(entrypoint_mod, "build_ivr_session", fake_build_ivr_session)

    ctx = types.SimpleNamespace(room=types.SimpleNamespace(name="room-x"))
    await entrypoint_mod.entrypoint(ctx)
    assert session.started
    assert called["holder"] is not None
    assert called["has_cb"] is True
