from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.shared.observability import log_marker

Lang = Literal["en", "ur"]

PROMPT_1 = "Press one for English. Urdu key liey 2 dabain"
PROMPT_RETRY = "Invalid input. Press one for English. Urdu key liey 2 dabain"
PROMPT_END = "Sorry, we could not get your selection. Goodbye."


GREETING_CONVERSATION_READY_EN = (
    "Thank you. I'm ready to help you. How may I assist you today?"
)
GREETING_CONVERSATION_READY_UR = (
    "شکریہ۔ میں آپ کی مدد کے لیے تیار ہوں۔ بتائیں، میں آپ کی کیسے مدد کر سکتا ہوں؟"
)


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
