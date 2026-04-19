from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from typing import Any, Callable, Literal

from livekit import agents
from livekit.agents import Agent, ModelSettings
from livekit.agents import llm as lk_llm

from app.core.knowledge_base import classify_intent, retrieve
from app.tools.llm_tools import LlmTools
from app.core.transcript_utils import (
    normalize_digits,
    normalize_english_currency_for_tts,
    normalize_roman_urdu_for_tts,
)

logger = logging.getLogger(__name__)
MAX_CHAT_CONTEXT_MESSAGES = 8

BASE_POLICY_PROMPT = """You are a professional, friendly male K-Electric (KE) call-centre voice agent for outage calls.

Sound human and clear; avoid generic “we are resolving it” only. Spoken style: short natural sentences, no bullet lists or numbered lists. After outage tool data: give a concise story (cause → who is affected → field work → realistic delays if any → what to expect next) in roughly 20–35 seconds of speech, not a long monologue.

Never invent feeder state, fault, crew, or ETA — call get_outage_status once you have an area or 13-digit account; if missing, ask once briefly.

When area/account is known: same assistant turn = one short thanks/ack line plus get_outage_status (never ack-only without the tool call — that causes dead air)."""


class VoiceAgent(Agent):
    def __init__(
        self,
        tools: LlmTools,
        instructions: str,
        preferred_language: Literal["ur", "en"] | None,
        on_language_locked: Callable[[Literal["ur", "en"]], None] | None = None,
    ) -> None:
        super().__init__(instructions=instructions, tools=[tools.get_outage_status])
        self.preferred_language = preferred_language
        self._language_locked = preferred_language in ("en", "ur")
        self._on_language_locked = on_language_locked

    @staticmethod
    def _chunk_has_tool_call(chunk: Any) -> bool:
        if chunk is None:
            return False
        for attr in ("tool_calls", "function_calls"):
            val = getattr(chunk, attr, None)
            if isinstance(val, list) and val:
                return True
        delta = getattr(chunk, "delta", None)
        if delta is not None:
            for attr in ("tool_calls", "function_calls"):
                val = getattr(delta, attr, None)
                if isinstance(val, list) and val:
                    return True
        return False

    def _filler_phrase(self) -> str:
        return "جی، ایک لمحہ دیں، چیک کر رہا ہوں۔" if self.preferred_language == "ur" else "One moment, let me check that for you."

    def _detect_lang_from_event(self, item: Any) -> Literal["ur", "en"] | None:
        for attr in ("language", "detected_language", "language_code", "lang"):
            val = getattr(item, attr, None)
            if not val:
                continue
            text = str(val).strip().lower()
            if text.startswith("ur"):
                return "ur"
            if text.startswith("en"):
                return "en"
        return None

    def _lock_language_if_needed(self, chat_ctx: lk_llm.ChatContext) -> None:
        if self._language_locked:
            return
        detected: Literal["ur", "en"] | None = None
        for item in reversed(chat_ctx.items):
            if getattr(item, "role", None) == "user":
                detected = self._detect_lang_from_event(item)
                if detected is None:
                    txt = str(getattr(item, "content", "")).lower()
                    detected = "ur" if any("\u0600" <= ch <= "\u06ff" for ch in txt) else "en"
                break
        if detected is None:
            return
        self.preferred_language = detected
        self._language_locked = True
        if self._on_language_locked:
            self._on_language_locked(detected)
        logger.info("lang.selected value=%s", detected)

    async def llm_node(self, chat_ctx: lk_llm.ChatContext, tools: list[lk_llm.Tool], model_settings: agents.ModelSettings):
        self._lock_language_if_needed(chat_ctx)
        chat_ctx_for_llm = preprocess_chat_context(chat_ctx=chat_ctx, preferred_language=self.preferred_language)
        async for chunk in Agent.default.llm_node(self, chat_ctx_for_llm, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        if self.preferred_language == "ur":

            async def _urdu_tts():
                async for chunk in text:
                    yield normalize_roman_urdu_for_tts(chunk) if isinstance(chunk, str) else chunk

            return Agent.default.tts_node(self, _urdu_tts(), model_settings)

        if self.preferred_language == "en":

            async def _currency_normalized():
                async for chunk in text:
                    yield normalize_english_currency_for_tts(chunk) if isinstance(chunk, str) else chunk

            return Agent.default.tts_node(self, _currency_normalized(), model_settings)

        return Agent.default.tts_node(self, text, model_settings)


def build_system_prompt(preferred_language: Literal["ur", "en"] | None) -> str:
    if preferred_language == "ur":
        language_prompt = (
            "LANGUAGE LOCK: Always respond ONLY in Urdu"
            "Never reply in English words or phrases (no “okay”, “let me check”, “sure”) — use Urdu equivalents. "
            "Keep 13-digit K-Electric account numbers as digits when repeating them; speak times and ETAs in natural spoken Urdu/Roman Urdu. "
            "Avoid Hindi register; prefer Pakistani Urdu vocabulary."
        )
    elif preferred_language == "en":
        language_prompt = "LANGUAGE LOCK: Always respond ONLY in English. Never reply in Urdu."
    else:
        language_prompt = (
            "Language policy: detect the caller language from first utterance, then respond only in that language "
            "for the remainder of the call."
        )
    return f"{BASE_POLICY_PROMPT}\n{language_prompt}"


def preprocess_chat_context(*, chat_ctx: lk_llm.ChatContext, preferred_language: Literal["ur", "en"] | None) -> lk_llm.ChatContext:
    user_text = ""
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) != "user":
            continue
        content = item.content if isinstance(item.content, str) else " ".join(str(x) for x in item.content)
        user_text = normalize_digits(content)
        item.content = user_text
        break
    if not user_text:
        return chat_ctx

    intent = classify_intent(user_text)
    injections: list[str] = []
    if intent in ("rag", "mixed"):
        retrieval_lang: Literal["en", "ur"] = preferred_language or "en"
        results = retrieve(user_text, language=retrieval_lang, top_k=1)
        if results:
            injections.append("[Relevant domain policy]\n" + "\n".join(f"• {r.content}" for r in results))
    if intent in ("tool", "mixed"):
        injections.append(
            "[Turn] Same turn: brief thanks + get_outage_status. After tool JSON: one concise spoken summary (cause, impact, crew, delays, next step); no bullets."
        )
    if not injections:
        return chat_ctx

    items = list(chat_ctx.items)
    last_user_idx = next((i for i in range(len(items) - 1, -1, -1) if getattr(items[i], "role", None) == "user"), len(items))
    items.insert(last_user_idx, lk_llm.ChatMessage(role="system", content=["\n\n".join(injections)]))
    cloned = chat_ctx.copy()
    cloned.items = items
    return cloned


def _is_trim_billable_message(item: Any) -> bool:
    return getattr(item, "type", None) == "message" and getattr(item, "role", None) in ("user", "assistant", "developer")


async def trim_chat_context(agent: VoiceAgent) -> None:
    try:
        chat_ctx_copy = agent.chat_ctx.copy()
        items = list(chat_ctx_copy.items)
        system_items = [msg for msg in items if getattr(msg, "role", None) == "system"]
        non_system_items = [msg for msg in items if getattr(msg, "role", None) != "system"]
        billable = [m for m in non_system_items if _is_trim_billable_message(m)]
        if len(billable) <= MAX_CHAT_CONTEXT_MESSAGES:
            return
        # Keep only tail section with the latest budgeted billable messages.
        kept = []
        seen = 0
        for msg in reversed(non_system_items):
            kept.append(msg)
            if _is_trim_billable_message(msg):
                seen += 1
            if seen >= MAX_CHAT_CONTEXT_MESSAGES:
                break
        chat_ctx_copy.items = [*system_items, *reversed(kept)]
        await agent.update_chat_ctx(chat_ctx_copy)
    except Exception:
        logger.warning("chat context trimming skipped", exc_info=True)
