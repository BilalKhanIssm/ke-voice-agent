from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm as lk_llm
from livekit.agents import AgentSession
from livekit.agents.voice.events import ConversationItemAddedEvent

logger = logging.getLogger(__name__)

# Merge consecutive user ChatMessages with no assistant in between when they arrive
# within this many seconds (endpointing / STT splits one spoken turn into two items).
_STT_MERGE_MAX_GAP_S = 4.0


def log_marker(event: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("%s %s", event, payload)


@dataclass
class _TurnBucket:
    stt_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    llm_request_count: int = 0
    anchor_created_at: float = 0.0


def _user_stt_ms(metrics: lk_llm.MetricsReport) -> int:
    td = float(metrics.get("transcription_delay") or 0.0)
    eod = float(metrics.get("end_of_turn_delay") or 0.0)
    ou = float(metrics.get("on_user_turn_completed_delay") or 0.0)
    return int(round((td + eod + ou) * 1000))


def attach_latency_logging(
    session: AgentSession,
    trim_cb: Any,
    phase_getter: Callable[[], str] | None = None,
) -> None:
    """
    Per-turn latency from ChatMessage.metrics.

    Buckets STT with the following assistant segment(s): we only flush a bucket to the log
    when a new user turn begins *after* at least one assistant reply carried LLM/TTS metrics,
    or when we merge rapid STT-only user items (see _STT_MERGE_MAX_GAP_S). Values are TTFT/TTFB
    style fields from LiveKit, not full model/audio duration.
    """

    state: dict[str, Any] = {
        "turn_seq": 0,
        "pending": None,  # _TurnBucket | None
    }

    def _in_conversation() -> bool:
        return not phase_getter or phase_getter() == "conversation"

    def _emit_completed_turn(bucket: _TurnBucket) -> None:
        state["turn_seq"] = int(state["turn_seq"]) + 1
        seq = state["turn_seq"]
        stt, llm, tts = bucket.stt_ms, bucket.llm_ms, bucket.tts_ms
        total = stt + llm + tts
        n_llm = bucket.llm_request_count
        if n_llm > 1:
            logger.info(
                "turn %s complete: stt=%sms llm_ttft_sum=%sms (llm_streams=%s) tts_ttfb_sum=%sms total=%sms",
                seq,
                stt,
                llm,
                n_llm,
                tts,
                total,
            )
        else:
            logger.info(
                "turn %s complete: stt=%sms llm_ttft=%sms tts_ttfb=%sms total=%sms",
                seq,
                stt,
                llm,
                tts,
                total,
            )

    def _flush_pending() -> None:
        bucket = state.get("pending")
        if bucket is None:
            return
        _emit_completed_turn(bucket)
        state["pending"] = None

    def on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        if not _in_conversation():
            return
        item = ev.item
        if not isinstance(item, lk_llm.ChatMessage):
            return
        if item.role == "user":
            stt_new = _user_stt_ms(item.metrics)
            ts = float(getattr(item, "created_at", 0.0) or 0.0)
            prev = state.get("pending")
            if prev is not None and (prev.llm_ms > 0 or prev.tts_ms > 0):
                _emit_completed_turn(prev)
                prev = None
            if prev is not None and prev.llm_ms == 0 and prev.tts_ms == 0:
                if prev.anchor_created_at and ts - prev.anchor_created_at <= _STT_MERGE_MAX_GAP_S:
                    prev.stt_ms += stt_new
                    prev.anchor_created_at = ts
                    return
                if prev.stt_ms:
                    _emit_completed_turn(prev)
            nb = _TurnBucket()
            nb.stt_ms = stt_new
            nb.anchor_created_at = ts
            state["pending"] = nb
        elif item.role == "assistant":
            b = state.get("pending")
            if b is None:
                b = _TurnBucket()
                b.anchor_created_at = float(getattr(item, "created_at", 0.0) or 0.0)
                state["pending"] = b
            m = item.metrics
            ttft = float(m.get("llm_node_ttft") or 0.0)
            ttfb = float(m.get("tts_node_ttfb") or 0.0)
            b.llm_ms += int(round(ttft * 1000))
            b.tts_ms += int(round(ttfb * 1000))
            b.llm_request_count += 1

    def on_close(_: Any = None) -> None:
        _flush_pending()

    def on_agent_state_changed(ev: object) -> None:
        if not _in_conversation():
            return
        if getattr(ev, "new_state", None) == "thinking":
            asyncio.ensure_future(trim_cb())

    session.on("conversation_item_added", on_conversation_item_added)
    session.on("close", on_close)
    session.on("agent_state_changed", on_agent_state_changed)
