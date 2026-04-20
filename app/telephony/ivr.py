from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.shared.observability import log_marker

Lang = Literal["en", "ur"]

PROMPT_1 = "Press 1 for English, or اردو کے لیے 2 دبائیں۔"
PROMPT_RETRY = "We didn't catch that - press 1 for English, یا اردو کے لیے 2 دبائیں۔"
PROMPT_END = "We're sorry, we weren't able to connect you right now. Please call us back at 118. Allah Hafiz."


GREETING_CONVERSATION_READY_EN = "Assalamualaikum! Thanks for choosing K-Electric - what can I help you with?"
GREETING_CONVERSATION_READY_UR = "السلام علیکم! کے-الیکٹرک کو منتخب کرنے کا شکریہ - بتائیں، میں آپ کی کیا مدد کر سکتی ہوں؟"


@dataclass(frozen=True)
class IVRConfig:
    first_wait_seconds: int = 8
    retry_wait_seconds: int = 6
    retries: int = 1


class IVRIO(Protocol):
    async def play(self, text: str) -> None: ...
    async def wait_for_dtmf(self, timeout_seconds: int) -> str | None: ...
    async def hangup(self) -> None: ...


def _map_key(key: str | None) -> Lang | None:
    if key == "1":
        return "en"
    if key == "2":
        return "ur"
    return None


async def select_language(io: IVRIO, config: IVRConfig) -> Lang | None:
    await io.play(PROMPT_1)
    log_marker("ivr.prompt", stage="initial")
    key = await io.wait_for_dtmf(config.first_wait_seconds)
    lang = _map_key(key)
    log_marker("ivr.input", stage="initial", key=key)
    if lang:
        log_marker("lang.selected", value=lang)
        return lang

    for attempt in range(config.retries):
        await io.play(PROMPT_RETRY)
        log_marker("ivr.retry", attempt=attempt + 1)
        key = await io.wait_for_dtmf(config.retry_wait_seconds)
        lang = _map_key(key)
        log_marker("ivr.input", stage="retry", key=key)
        if lang:
            log_marker("lang.selected", value=lang)
            return lang

    await io.play(PROMPT_END)
    log_marker("ivr.hangup", reason="invalid_or_timeout")
    await io.hangup()
    return None
