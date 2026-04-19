from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    livekit_url: str = Field(alias="LIVEKIT_URL")
    livekit_api_key: str = Field(alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(alias="LIVEKIT_API_SECRET")
    livekit_agent_name: str = Field(alias="LIVEKIT_AGENT_NAME")

    deepgram_api_key: str = Field(alias="DEEPGRAM_API_KEY")
    deepgram_stt_model: str = Field(default="nova-3", alias="DEEPGRAM_STT_MODEL")
    deepgram_stt_language_en: str = Field(default="en-US", alias="DEEPGRAM_STT_LANGUAGE_EN")
    deepgram_stt_language_ur: str = Field(default="ur", alias="DEEPGRAM_STT_LANGUAGE_UR")
    test_start_language: Literal["en", "ur"] | None = Field(default=None, alias="TEST_START_LANGUAGE")

    llm_provider: Literal["openai", "openrouter"] = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_llm_model: str = Field(default="gpt-4o-mini", alias="OPENAI_LLM_MODEL")
    openai_tts_model: str = Field(default="tts-1", alias="OPENAI_TTS_MODEL")
    openai_tts_voice: str = Field(default="onyx", alias="OPENAI_TTS_VOICE")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_llm_model: str = Field(default="x-ai/grok-3-mini-beta", alias="OPENROUTER_LLM_MODEL")
    # Voice + tool calls need headroom; 80 often drops tool_calls after a short filler line.
    llm_max_completion_tokens: int = Field(default=512, alias="LLM_MAX_COMPLETION_TOKENS")
    llm_temperature: float = Field(default=0.05, alias="LLM_TEMPERATURE")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = []
        for err in exc.errors():
            if err.get("type") == "missing":
                loc = err.get("loc", [])
                if loc:
                    missing.append(str(loc[0]))
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(sorted(missing))}") from exc
        raise
