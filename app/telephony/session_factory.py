from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Literal

from livekit import agents
from livekit.agents import AgentSession
from livekit.plugins import silero

from app.config import Settings
from app.core.agent_core import VoiceAgent, build_system_prompt
from app.telephony.language_gate import LanguageGateAgent
from app.integrations.providers import get_llm, get_stt, get_tts


def _build_turn_handling_options() -> object | None:
    turn_handling_cls = getattr(agents, "TurnHandlingOptions", None)
    if turn_handling_cls is None:
        return None
    kwargs: dict[str, object] = {}
    try:
        sig = inspect.signature(turn_handling_cls)
        param_names = set(sig.parameters.keys())
        if "mode" in param_names:
            kwargs["mode"] = "vad"
        if "min_endpointing_delay" in param_names:
            kwargs["min_endpointing_delay"] = 0.02
        if "max_endpointing_delay" in param_names:
            kwargs["max_endpointing_delay"] = 0.10
    except (TypeError, ValueError):
        pass
    return turn_handling_cls(**kwargs)


def build_session(
    settings: Settings,
    selected_language: Literal["en", "ur"],
) -> tuple[VoiceAgent, AgentSession]:
    """Voice agent with fixed language (tests or direct use)."""
    agent = VoiceAgent(
        instructions=build_system_prompt(selected_language),
        preferred_language=selected_language,
        on_language_locked=None,
    )
    turn_handling = _build_turn_handling_options()
    session_kwargs: dict[str, object] = {
        "stt": get_stt(settings, selected_language),
        "llm": get_llm(settings),
        "tts": get_tts(settings, selected_language),
        "vad": silero.VAD.load(min_silence_duration=0.10, prefix_padding_duration=0.05),
        "preemptive_generation": True,
    }
    if turn_handling is not None:
        session_kwargs["turn_handling"] = turn_handling

    session = AgentSession(**session_kwargs)
    return agent, session


def build_ivr_session(
    settings: Settings,
    voice_agent_holder: dict[str, VoiceAgent | None],
    on_ivr_failed: Callable[[], Awaitable[None]],
) -> tuple[LanguageGateAgent, AgentSession]:
    """IVR language gate first (English STT for menu); voice agent + STT/TTS applied after DTMF."""
    agent = LanguageGateAgent(
        settings,
        voice_agent_holder,
        on_ivr_failed,
    )
    turn_handling = _build_turn_handling_options()
    session_kwargs: dict[str, object] = {
        "stt": get_stt(settings, "en"),
        "llm": get_llm(settings),
        "tts": get_tts(settings, "ur"),
        "vad": silero.VAD.load(min_silence_duration=0.10, prefix_padding_duration=0.05),
        "preemptive_generation": True,
    }
    if turn_handling is not None:
        session_kwargs["turn_handling"] = turn_handling

    session = AgentSession(**session_kwargs)
    return agent, session
