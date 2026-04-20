from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livekit.agents import llm as lk_llm
from livekit.agents.types import APIConnectOptions

from app.shared.observability import log_marker

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from app.config import Settings

logger = logging.getLogger(__name__)

_WARM_SYSTEM = (
    "Connection warmup only. Reply with exactly one character: 0. No other text, no punctuation."
)


async def warmup_llm_if_enabled(settings: "Settings", session: "AgentSession") -> None:
    """Prime the configured chat LLM before the first user turn (TLS/connection pool + provider cold start)."""
    if not settings.llm_warmup_enabled:
        return

    llm = getattr(session, "llm", None)
    if llm is None:
        log_marker("llm.warmup.skip", reason="no_llm")
        return

    chat = getattr(llm, "chat", None)
    if not callable(chat):
        log_marker("llm.warmup.skip", reason="no_chat_method")
        return

    try:
        llm.prewarm()
    except Exception:
        pass

    ctx = lk_llm.ChatContext()
    ctx.items = [
        lk_llm.ChatMessage(role="system", content=[_WARM_SYSTEM]),
        lk_llm.ChatMessage(role="user", content=["ping"]),
    ]
    conn = APIConnectOptions(max_retry=1, retry_interval=0.5, timeout=settings.llm_warmup_timeout_seconds)

    try:
        stream = llm.chat(chat_ctx=ctx, tools=None, conn_options=conn)
        async with stream:
            async for _ in stream:
                break
        log_marker("llm.warmup.ok")
    except Exception as exc:
        logger.warning("llm warmup failed (call continues): %s", exc)
        log_marker("llm.warmup.failed", error=str(exc))
