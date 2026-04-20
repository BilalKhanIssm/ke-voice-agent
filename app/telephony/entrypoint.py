from __future__ import annotations

import asyncio
import logging

from livekit import agents

from app.config import get_settings
from app.core.agent_core import VoiceAgent, trim_chat_context
from app.shared.observability import attach_latency_logging, log_marker
from app.telephony.ivr import GREETING_CONVERSATION_READY_EN, GREETING_CONVERSATION_READY_UR
from app.telephony.llm_warmup import warmup_llm_if_enabled
from app.telephony.session_factory import build_ivr_session, build_session

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def _safe_ctx_connect(ctx: agents.JobContext) -> None:
    connect = getattr(ctx, "connect", None)
    if callable(connect):
        result = connect()
        if asyncio.iscoroutine(result):
            await result


async def _safe_ctx_shutdown(ctx: agents.JobContext, reason: str) -> None:
    shutdown = getattr(ctx, "shutdown", None)
    if callable(shutdown):
        result = shutdown(reason=reason)
        if asyncio.iscoroutine(result):
            await result


async def entrypoint(ctx: agents.JobContext) -> None:
    await _safe_ctx_connect(ctx)
    voice_agent_holder: dict[str, VoiceAgent | None] = {"agent": None}

    async def trim_when_voice_agent_ready() -> None:
        agent = voice_agent_holder.get("agent")
        if agent is not None:
            await trim_chat_context(agent)

    def current_phase() -> str:
        return "conversation" if voice_agent_holder.get("agent") is not None else "ivr"

    try:
        forced_language = settings.test_start_language
        if forced_language in ("en", "ur"):
            agent, session = build_session(settings, forced_language)
            voice_agent_holder["agent"] = agent
            attach_latency_logging(session, trim_when_voice_agent_ready, current_phase)
            await warmup_llm_if_enabled(settings, session)
            await session.start(room=ctx.room, agent=agent)
            greeting = GREETING_CONVERSATION_READY_EN if forced_language == "en" else GREETING_CONVERSATION_READY_UR
            session.say(greeting, add_to_chat_ctx=False, allow_interruptions=True)
            log_marker("session.started", room=ctx.room.name, phase="conversation", forced_language=forced_language)
            return


        async def on_ivr_failed() -> None:
            await _safe_ctx_shutdown(ctx, "ivr_failed")

        agent, session = build_ivr_session(settings, voice_agent_holder, on_ivr_failed)
        attach_latency_logging(session, trim_when_voice_agent_ready, current_phase)
        await warmup_llm_if_enabled(settings, session)
        await session.start(room=ctx.room, agent=agent)
        log_marker("session.started", room=ctx.room.name, phase="ivr")
    finally:
        pass


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=settings.livekit_agent_name,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )
