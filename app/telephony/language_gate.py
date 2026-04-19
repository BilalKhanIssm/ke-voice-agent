from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.job import get_job_context

from app.config import Settings
from app.core.agent_core import VoiceAgent, build_system_prompt
from app.telephony.ivr import (
    GREETING_CONVERSATION_READY_EN,
    GREETING_CONVERSATION_READY_UR,
    PROMPT_1,
    PROMPT_END,
    PROMPT_RETRY,
)
from app.shared.observability import log_marker
from app.integrations.providers import apply_language_providers
from app.tools.llm_tools import LlmTools


def map_dtmf_to_language(user_input: str) -> Literal["en", "ur"] | None:
    raw = (user_input or "").strip().lower()
    if raw in ("1", "one", "english", "en"):
        return "en"
    if raw in ("2", "two", "urdu", "ur"):
        return "ur"
    return None


async def await_menu_digit(timeout_seconds: float) -> str | None:
    """
    Wait for the first keypad digit 1 or 2 on sip_dtmf_received.

    Unlike GetDtmfTask, this completes immediately on a valid key. GetDtmfTask debounces
    and cancels that debounce when the user speaks, which caused a single '2' to never
    finalize and a second press to become '2 2'.
    """
    ctx = get_job_context()
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[str] = loop.create_future()

    def on_dtmf(ev: rtc.SipDTMF) -> None:
        if fut.done():
            return
        digit = (getattr(ev, "digit", None) or "").strip()
        if not digit and getattr(ev, "code", None) is not None:
            code = int(ev.code)
            code_to_digit = {i: str(i) for i in range(10)} | {10: "*", 11: "#"}
            digit = (code_to_digit.get(code) or "").strip()
        if digit not in ("1", "2"):
            log_marker("ivr.dtmf_ignored", digit=digit or None)
            return
        fut.set_result(digit)

    ctx.room.on("sip_dtmf_received", on_dtmf)
    try:
        return await asyncio.wait_for(fut, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return None
    finally:
        ctx.room.off("sip_dtmf_received", on_dtmf)


def make_voice_agent(settings: Settings, language: Literal["en", "ur"]) -> VoiceAgent:
    return VoiceAgent(
        tools=LlmTools(),
        instructions=build_system_prompt(language),
        preferred_language=language,
        on_language_locked=None,
    )


class LanguageGateAgent(Agent):
    def __init__(
        self,
        settings: Settings,
        voice_agent_holder: dict[str, VoiceAgent | None],
        on_ivr_failed: Callable[[], Awaitable[None]],
        *,
        dtmf_menu_timeout_seconds: float = 15.0,
    ) -> None:
        self._settings = settings
        self._voice_agent_holder = voice_agent_holder
        self._on_ivr_failed = on_ivr_failed
        self._dtmf_menu_timeout_seconds = dtmf_menu_timeout_seconds
        super().__init__(
            instructions=(
                "Internal routing step only. Do not chat. The session will transfer to the voice agent after language selection."
            ),
        )

    async def _safe_say_and_wait(
        self,
        text: str,
        *,
        add_to_chat_ctx: bool,
        allow_interruptions: bool,
    ) -> bool:
        session = self.session
        try:
            handle = session.say(
                text,
                add_to_chat_ctx=add_to_chat_ctx,
                allow_interruptions=allow_interruptions,
            )
            await handle
            return True
        except RuntimeError as exc:
            if "closing" in str(exc).lower():
                log_marker("ivr.say_skipped", reason="session_closing")
                return False
            raise

    async def _collect_one_digit(self) -> str | None:
        digit = await await_menu_digit(self._dtmf_menu_timeout_seconds)
        if digit:
            log_marker("ivr.dtmf_digit", digit=digit)
        return digit

    async def on_enter(self) -> None:
        session = self.session
        if not await self._safe_say_and_wait(
            PROMPT_1,
            add_to_chat_ctx=False,
            allow_interruptions=True,
        ):
            return
        log_marker("ivr.prompt", stage="initial")

        lang = await self._try_select_language()
        if lang is None:
            if not await self._safe_say_and_wait(
                PROMPT_RETRY,
                add_to_chat_ctx=False,
                allow_interruptions=True,
            ):
                return
            log_marker("ivr.retry", attempt=1)
            lang = await self._try_select_language()

        if lang is None:
            await self._safe_say_and_wait(
                PROMPT_END,
                add_to_chat_ctx=False,
                allow_interruptions=False,
            )
            log_marker("ivr.hangup", reason="invalid_or_timeout")
            await self._on_ivr_failed()
            return

        log_marker("lang.selected", value=lang)
        apply_language_providers(session, self._settings, lang)
        voice_agent = make_voice_agent(self._settings, lang)
        self._voice_agent_holder["agent"] = voice_agent
        session.update_agent(voice_agent)
        greeting = GREETING_CONVERSATION_READY_EN if lang == "en" else GREETING_CONVERSATION_READY_UR
        try:
            session.say(greeting, add_to_chat_ctx=False, allow_interruptions=True)
            log_marker("ivr.conversation_ready_greeting", language=lang)
        except RuntimeError as exc:
            if "closing" in str(exc).lower():
                log_marker("ivr.say_skipped", reason="session_closing")
                return
            raise

    async def _try_select_language(self) -> Literal["en", "ur"] | None:
        digit = await self._collect_one_digit()
        if digit is None:
            return None
        return map_dtmf_to_language(digit)
