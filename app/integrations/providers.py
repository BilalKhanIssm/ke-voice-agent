from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from livekit.plugins import deepgram, openai as lk_openai

from app.config import Settings

if TYPE_CHECKING:
    from livekit.agents import AgentSession


def get_llm(settings: Settings) -> lk_openai.LLM:
    if settings.llm_provider == "openrouter":
        return lk_openai.LLM(
            model=settings.openrouter_llm_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            max_completion_tokens=settings.llm_max_completion_tokens,
            temperature=settings.llm_temperature,
        )
    return lk_openai.LLM(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        max_completion_tokens=settings.llm_max_completion_tokens,
        temperature=settings.llm_temperature,
    )


def get_stt(settings: Settings, language: Literal["en", "ur"] | None) -> deepgram.STT:
    deepgram_language = "multi" if language is None else (
        settings.deepgram_stt_language_ur if language == "ur" else settings.deepgram_stt_language_en
    )
    # Multi-language detection is more stable with nova-2 than nova-3 in telephony.
    stt_model = "nova-2" if language is None else settings.deepgram_stt_model
    return deepgram.STT(
        api_key=settings.deepgram_api_key,
        model=stt_model,
        language=deepgram_language,
        interim_results=True,
        punctuate=True,
        smart_format=True,
        no_delay=True,
        filler_words=True,
        endpointing_ms=70,
    )


def get_tts(settings: Settings, language: Literal["en", "ur"] | None) -> object:
    if language == "ur":
        instructions = """
        You are a K-Electric (Karachi) customer support agent.

        Speak in clear Pakistani Urdu. Use a Karachi neutral accent.
        Stay warm, calm, and professional.
        """

    else:
        instructions = """
        You are a professional customer support agent.

        Speak clearly, confidently, and politely.
        Maintain a neutral and helpful tone.
        """
    return lk_openai.TTS(
        model=settings.openai_tts_model,
        voice=settings.openai_tts_voice,
        api_key=settings.openai_api_key,
        instructions=instructions,
    )


def apply_language_providers(session: "AgentSession", settings: Settings, language: Literal["en", "ur"]) -> None:
    """Hot-swap STT/TTS on the session after IVR language selection."""
    session._stt = get_stt(settings, language)
    session._tts = get_tts(settings, language)
