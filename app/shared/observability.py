from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from livekit.agents import AgentSession
from livekit.agents.metrics import EOUMetrics, LLMMetrics, TTSMetrics
from livekit.agents.voice.events import MetricsCollectedEvent

logger = logging.getLogger(__name__)


def log_marker(event: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("%s %s", event, payload)


@dataclass
class _TurnBucket:
    stt_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    # One user end-of-utterance can span multiple LLM completions depending on pipeline behaviour.
    llm_request_count: int = 0


def attach_latency_logging(
    session: AgentSession,
    trim_cb: Any,
    phase_getter: Callable[[], str] | None = None,
) -> None:
    """
    One INFO line per completed user→agent turn with STT / LLM / TTS breakdown (from LiveKit metrics).
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
                "turn %s complete: stt=%sms llm=%sms (llm_requests=%s) tts=%sms total=%sms",
                seq,
                stt,
                llm,
                n_llm,
                tts,
                total,
            )
        else:
            logger.info("turn %s complete: stt=%sms llm=%sms tts=%sms total=%sms", seq, stt, llm, tts, total)

    def _flush_pending() -> None:
        bucket = state.get("pending")
        if bucket is None:
            return
        _emit_completed_turn(bucket)
        state["pending"] = None

    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        if not _in_conversation():
            return
        m = ev.metrics
        if isinstance(m, EOUMetrics):
            _flush_pending()
            nb = _TurnBucket()
            nb.stt_ms = int(
                round(
                    (m.end_of_utterance_delay + m.transcription_delay + m.on_user_turn_completed_delay) * 1000
                )
            )
            state["pending"] = nb
        elif isinstance(m, LLMMetrics):
            b = state["pending"]
            if b is not None:
                b.llm_ms += int(round(m.duration * 1000))
                b.llm_request_count += 1
        elif isinstance(m, TTSMetrics):
            b = state["pending"]
            if b is not None:
                b.tts_ms += int(round(m.duration * 1000))

    def on_close(_: Any = None) -> None:
        _flush_pending()

    def on_agent_state_changed(ev: object) -> None:
        if not _in_conversation():
            return
        if getattr(ev, "new_state", None) == "thinking":
            asyncio.ensure_future(trim_cb())

    session.on("metrics_collected", on_metrics_collected)
    session.on("close", on_close)
    session.on("agent_state_changed", on_agent_state_changed)
