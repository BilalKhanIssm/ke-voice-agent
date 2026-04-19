from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterable
from typing import Any, Callable, Literal

from livekit import agents
from livekit.agents import Agent, ModelSettings
from livekit.agents import llm as lk_llm

from app.core.knowledge_base import classify_intent, retrieve
from app.tools.llm_tools import build_outage_snapshot
from app.core.transcript_utils import (
    normalize_digits,
    normalize_english_currency_for_tts,
    normalize_roman_urdu_for_tts,
)

logger = logging.getLogger(__name__)
MAX_CHAT_CONTEXT_MESSAGES = 8

BASE_POLICY_PROMPT = """You are a professional, friendly female K-Electric (KE) call-centre voice agent for K-Electric supply and outage calls.

Sound human and clear; avoid generic “we are resolving it” only. Spoken style: short natural sentences, no bullet lists or numbered lists.

Conversation (not a form): acknowledge greetings and what the caller just said; show brief empathy when they describe loss of supply or stress. Let them explain the problem before you sound procedural. Do **not** open with an immediate demand for account number unless they jumped straight to “check my account”.

Location for **feeder-level** facts: to narrate fault/crew/restoration from an injected outage snapshot you need **either** a clear neighbourhood / block / landmark **or** a 13-digit KE account — **one is enough, not both**. If they already named an area or gave an account earlier in this chat, **reuse it** and do **not** ask again. Ask at most **one** concise clarifying question if truly nothing usable has been said yet; never repeat the same ask twice in a row. If they complain that you repeated yourself, apologise once briefly and continue with what they already gave.

When an “[Authoritative outage snapshot]” JSON block is in context for this turn: use **only** that JSON for feeder name, fault, crew, delays, ETA line, and `complaint_reference`. If they only asked for the complaint/reference number, read **only** that field from JSON first (digit-by-digit in Urdu per language rules), then at most one short helpful sentence — do not re-demand area/account. If there is **no** snapshot this turn, do **not** invent feeder/crew/reference numbers; stay conversational and generic, or ask one focused question.

Complaint / reference numbers for text-to-speech: never output a bare compact digit string for a reference (e.g. 12345) where TTS may read it as a full number (“twelve thousand…”). English: digit-by-digit or clear short-code phrasing. Urdu: spoken Urdu or Roman Urdu digit-by-digit / words so the synthesiser (e.g. Uplift) does not misread the reference."""


class VoiceAgent(Agent):
    def __init__(
        self,
        instructions: str,
        preferred_language: Literal["ur", "en"] | None,
        on_language_locked: Callable[[Literal["ur", "en"]], None] | None = None,
    ) -> None:
        super().__init__(instructions=instructions, tools=[])
        self.preferred_language = preferred_language
        self._language_locked = preferred_language in ("en", "ur")
        self._on_language_locked = on_language_locked

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
            "LANGUAGE LOCK: Always respond ONLY in Urdu. "
            "Never reply in English words or phrases (no “okay”, “let me check”, “sure”) — use Urdu equivalents. "
            "Keep 13-digit K-Electric account numbers as digits when repeating them; speak times and ETAs in natural spoken Urdu/Roman Urdu. "
            "For complaint or reference IDs from the outage snapshot, never write a bare multi-digit numeral like 12345 — say digits in Urdu/Roman Urdu so TTS (e.g. Uplift) does not read it as thousands. "
            "Avoid Hindi register; prefer Pakistani Urdu vocabulary."
        )
    elif preferred_language == "en":
        language_prompt = (
            "LANGUAGE LOCK: Always respond ONLY in English. Never reply in Urdu. "
            "For complaint or reference IDs, read digits clearly (e.g. digit-by-digit) so TTS does not merge them into one large number."
        )
    else:
        language_prompt = (
            "Language policy: detect the caller language from first utterance, then respond only in that language "
            "for the remainder of the call."
        )
    return f"{BASE_POLICY_PROMPT}\n{language_prompt}"


_AREA_HINTS_EN: frozenset[str] = frozenset(
    ("block", "sector", "phase", "scheme", "society", "gul", "dha", "pechs", "korangi", "landhi", "clifton", "nazimabad")
)
_AREA_HINTS_UR: frozenset[str] = frozenset(("بلاک", "سیکٹر", "علاق", "محلہ", "مُحلّہ", "گل", "سائٹ", "جوہر"))


def _user_item_text(item: Any) -> str:
    raw = item.content if isinstance(item.content, str) else " ".join(str(x) for x in item.content)
    return str(raw).strip()


def _recent_user_messages(chat_ctx: lk_llm.ChatContext, *, max_messages: int = 6) -> tuple[str, list[str]]:
    """Latest user text (normalized) plus recent user turns, newest first."""
    texts: list[str] = []
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) != "user":
            continue
        norm = normalize_digits(_user_item_text(item))
        texts.append(norm)
        if len(texts) >= max_messages:
            break
    if not texts:
        return "", []
    return texts[0], texts


def _rich_enough_for_snapshot_lookup(text: str) -> bool:
    if len(re.sub(r"\D", "", text)) >= 10:
        return True
    low = text.lower()
    if any(h in low for h in _AREA_HINTS_EN):
        return True
    if any(h in text for h in _AREA_HINTS_UR):
        return True
    urdu_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    if urdu_chars >= 10 and len(text.split()) >= 4:
        return True
    return False


def _snapshot_lookup_key(current_turn: str, recent_newest_first: list[str]) -> str:
    if _rich_enough_for_snapshot_lookup(current_turn):
        return current_turn
    for past in recent_newest_first[1:]:
        if _rich_enough_for_snapshot_lookup(past):
            return past
    return current_turn


def preprocess_chat_context(*, chat_ctx: lk_llm.ChatContext, preferred_language: Literal["ur", "en"] | None) -> lk_llm.ChatContext:
    user_text, recent_users = _recent_user_messages(chat_ctx)
    if not user_text:
        return chat_ctx

    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) != "user":
            continue
        item.content = user_text
        break

    intent = classify_intent(user_text)
    lookup = _snapshot_lookup_key(user_text, recent_users)
    # Continue outage context when the caller follows up (e.g. reference / frustration) but the area was in an earlier turn.
    inject_snapshot = intent in ("tool", "mixed") or (
        intent == "rag"
        and _rich_enough_for_snapshot_lookup(lookup)
        and lookup.strip() != user_text.strip()
    )
    injections: list[str] = []
    if intent in ("rag", "mixed"):
        retrieval_lang: Literal["en", "ur"] = preferred_language or "en"
        results = retrieve(user_text, language=retrieval_lang, top_k=1)
        if results:
            injections.append("[Relevant domain policy]\n" + "\n".join(f"• {r.content}" for r in results))
    if inject_snapshot:
        snap = build_outage_snapshot(lookup)
        injections.append(
            "[Authoritative outage snapshot for this user turn — summarise only these facts in speech; do not invent extra grid details.]\n"
            + json.dumps(snap)
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
