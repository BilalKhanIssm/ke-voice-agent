from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livekit.agents import llm as lk_llm
from livekit.agents.types import APIConnectOptions

from app.shared.observability import log_marker
from app.tools.llm_tools import LlmTools

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from app.config import Settings

logger = logging.getLogger(__name__)

_WARM_SYSTEM = (
    "Connection warmup only. Reply with exactly one character: 0. No other text, no punctuation."
)
_warmup_tools = LlmTools()
_WARMUP_TOOLS = [
    _warmup_tools.get_outage_status,
    _warmup_tools.get_complaint_reference,
]


async def warmup_llm_if_enabled(settings: "Settings", session: "AgentSession") -> None:
    """Prime chat LLM/network path before first user turn."""
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

    ctx = lk_llm.ChatContext()
    ctx.items = [
        lk_llm.ChatMessage(role="system", content=[_WARM_SYSTEM]),
        lk_llm.ChatMessage(role="user", content=["ping"]),
    ]
    conn = APIConnectOptions(max_retry=0, timeout=settings.llm_warmup_timeout_seconds)

    try:
        stream = llm.chat(chat_ctx=ctx, tools=_WARMUP_TOOLS, conn_options=conn)
        async with stream:
            async for _ in stream:
                break
        log_marker("llm.warmup.ok", mode="chat_with_tools")
    except Exception as exc:
        logger.warning("llm warmup failed (call continues): %s", exc)
        log_marker("llm.warmup.failed", error=str(exc))
