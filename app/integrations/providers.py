from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from livekit.plugins import cartesia, deepgram, openai as lk_openai, upliftai

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
    if language == "en" and settings.cartesia_api_key:
        return cartesia.TTS(
            api_key=settings.cartesia_api_key,
            model=settings.cartesia_tts_model,
            language="en",
            voice=settings.cartesia_tts_voice_id_en,
            sample_rate=settings.cartesia_tts_sample_rate,
            encoding="pcm_s16le",
            word_timestamps=False,
        )
    return upliftai.TTS(
        api_key=settings.upliftai_api_key,
        voice_id=settings.upliftai_voice_id,
        output_format=settings.upliftai_output_format,
    )


def apply_language_providers(session: "AgentSession", settings: Settings, language: Literal["en", "ur"]) -> None:
    """Hot-swap STT/TTS on the session after IVR language selection."""
    session._stt = get_stt(settings, language)
    session._tts = get_tts(settings, language)
