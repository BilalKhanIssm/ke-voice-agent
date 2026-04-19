from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterable
from typing import Any, Callable, Literal

from livekit import agents
from livekit.agents import Agent, ModelSettings
from livekit.agents import llm as lk_llm

from app.core.knowledge_base import (
    classify_intent,
    is_closing_utterance,
    normalize_query_for_retrieval,
    retrieve,
)
from app.tools.llm_tools import LlmTools
from app.core.transcript_utils import (
    expand_ke_reference_for_english_tts,
    normalize_digits,
    normalize_english_currency_for_tts,
    normalize_roman_urdu_for_tts,
)

logger = logging.getLogger(__name__)
MAX_CHAT_CONTEXT_MESSAGES = 8

BASE_POLICY_PROMPT = """You are a professional, friendly female K-Electric (KE) call-centre voice agent for outage calls.

Sound human and clear; avoid generic “we are resolving it” only. Spoken style: short natural sentences, no bullet lists or numbered lists.

Strict routing (follow exactly):
• Branch A — The conversation already contains a usable location (neighbourhood or landmark **plus** block/scheme/plot if they gave one, e.g. “Johar block 18”, “جوہر بلاک 18”) **or** a 13-digit K-Electric account number: call get_outage_status immediately in the same assistant turn, with NO spoken preamble before the tool runs. After the tool returns, give one flowing spoken summary in at most 2–3 sentences; do not restate every JSON field; never invent feeder state, fault, crew, or ETA. **Do not** ask for a 13-digit account only to explain outage, cause, or feeder status once neighbourhood (+ block if any) is already known — account is optional for finer billing detail, not required for outage answers.
• Branch B — Neither a usable area description nor a 13-digit account is in the conversation: ask once in one short sentence for neighbourhood (and block/scheme if applicable) **or** the 13-digit account. Do NOT call get_outage_status or any other tool. Do NOT speculate about the fault or outage.

Account numbers from speech: a KE account is **exactly 13 digits**. If the caller reads digits in broken groups or mixed words, do **not** concatenate groups into one long number — ask them to repeat slowly or send the number by SMS; never invent digits.

If you have already asked for the area or account number and the caller still has not provided it, do NOT ask again. Briefly say you can check once they share their area or 13-digit account number, and wait.

Mock complaint reference (demo only): call get_complaint_reference with the same area or account string you use for outage lookup. **Speak the entire complaint_reference_spoken_ur (or _en) verbatim in one go** — all digit words, no truncation. Do not invent a different code. The same area/account key yields a consistent demo reference.

**Outage story before reference:** Whenever area (or account) is known and the caller cares about no supply, cause, timeline, complaint, or a reference number, you must **first** call get_outage_status (again if needed for this area) and briefly explain fault_summary, crew_status, and eta_summary so they hear what is wrong and that field work is under way. **Only after that outage summary**, if they need a demo reference, call get_complaint_reference and read the spoken reference. The only exception is if they ask **only** for the digits of a reference with no outage question — still prefer one quick get_outage_status pass first when area is known so the answer stays consistent.

Never say you cannot give reason or restoration guidance if get_outage_status JSON includes fault_summary, crew_status, or eta_summary — paraphrase those fields briefly instead of refusing.

Never claim a complaint was registered, logged, or closed unless get_outage_status JSON explicitly supports it or you have just called get_complaint_reference for a demo reference. Never invent a reference number; only read values returned by tools.

When the caller clearly says goodbye or thanks and is ending the call, reply once with a short polite closing (thank them, optional helpline 118). Do not ask for area, account, or reference again."""


_TOOL_HINT_HAS_CONTEXT = (
    "[Turn] Usable location or account is already in the conversation: call get_outage_status this turn with NO "
    "spoken preamble before the tool. Pass neighbourhood + block in Urdu/Roman as the caller said (e.g. "
    "سکیم 33). Do NOT demand a 13-digit account for outage/cause questions if block+area are known. "
    "After JSON: first 2–3 short sentences on fault/crew/ETA from the tool; only then, if they need a demo "
    "reference, call get_complaint_reference and read complaint_reference_spoken_ur / _en in full."
)

_TOOL_HINT_OUTAGE_BEFORE_REFERENCE = (
    "[Turn] Caller is asking why power is off, when it may return, and/or a reference: call get_outage_status "
    "this turn (same area string) before get_complaint_reference. Summarise fault_summary, crew_status, and "
    "eta_summary for reassurance first; then read the spoken reference verbatim if still needed."
)

_TOOL_HINT_NEED_LOCATION = (
    "[Turn] Area or account is NOT in the conversation yet: ask for the caller's area or 13-digit "
    "K-Electric account number in one short sentence. Do NOT call get_outage_status or any other tool; "
    "do not speculate about the fault."
)

_CLOSING_TURN_HINT = (
    "[Turn] Caller is ending the call (goodbye or thanks). Reply once in one or two short sentences: "
    "thank them, optional helpline 118, wish them well. Do NOT ask for area, account, or reference; "
    "do NOT call get_outage_status, get_complaint_reference, or any other tool."
)

# Karachi / KE telephony: common localities (normalized lowercase) plus Roman Urdu variants.
_AREA_KEYWORDS: tuple[str, ...] = (
    "dha",
    "defence",
    "defense",
    "clifton",
    "gulshan",
    "nazimabad",
    "north nazimabad",
    "korangi",
    "malir",
    "lyari",
    "saddar",
    "fb area",
    "federal b area",
    "surjani",
    "orangi",
    "site",
    "baldia",
    "kemari",
    "gulistan-e-jauhar",
    "johar",
    "pechs",
    "bahadurabad",
    "scheme 33",
    "landhi",
    "shah faisal",
    "liaquatabad",
    "garden",
    "karsaz",
    "buffer zone",
    "sohrab goth",
    "steel town",
    "bin qasim",
    "پاکستان چورنگی",
    "گلشن",
    "ناظم آباد",
    "کورنگی",
    "ملیر",
    "لیاقت آباد",
    "ڈیفنس",
    "جوہر",
    "گلستان",
    "بلاک",
    "سکیم",
    "scheme",
)

_ASSISTANT_ASKED_AREA_MARKERS: tuple[str, ...] = (
    "your area",
    "which area",
    "what area",
    "locality",
    "neighbourhood",
    "neighborhood",
    "where are you calling",
    "علاقہ",
    "علاقے",
    "کون سا علاقہ",
    "کس علاقے",
    "کہاں سے",
    "account number",
    "13-digit",
    "13 digit",
    "thirteen digit",
    "اکاؤنٹ",
    "تیرہ ہندسوں",
)


def _item_text_content(item: Any) -> str:
    role = getattr(item, "role", None)
    if role not in ("user", "assistant"):
        return ""
    content = getattr(item, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(x) for x in content)
    return str(content or "")


def _user_message_has_location_candidate(t: str, low: str) -> bool:
    """True if a user turn plausibly names a block, scheme, plot, or 'I live at N' style hint."""
    if "بلاک" in t and re.search(r"\d", t):
        return True
    if re.search(
        r"\b(?:block|plot|phase|sector|scheme)\b.{0,12}\d|\b\d{2,4}\b.{0,20}\b(?:block|plot|phase|sector|scheme)\b",
        low,
    ):
        return True
    if re.search(
        r"\b(?:main|mein|men|may|rehta|rehti|rahata|rahati)\b.{0,16}\b\d{2,4}\b|\b\d{2,4}\b.{0,16}\b(?:main|mein|men|may)\b",
        low,
    ):
        return True
    if re.search(r"(?:میں|سے)\s*\d{2,4}|\d{2,4}\s*(?:میں|سے|پر)", t):
        return True
    if "علاق" in t and ("میں" in t or "سے" in t):
        return True
    return False


def _conversation_has_area_or_account(chat_ctx: lk_llm.ChatContext) -> bool:
    for it in chat_ctx.items:
        if getattr(it, "role", None) != "user":
            continue
        t = normalize_digits(_item_text_content(it))
        digits = re.sub(r"\D", "", t)
        if len(digits) >= 13:
            return True
        low = t.lower()
        for kw in _AREA_KEYWORDS:
            if kw in low:
                return True
        if _user_message_has_location_candidate(t, low):
            return True
    return False


def _user_wants_outage_explanation(user_text: str) -> bool:
    """Cause, timing, or 'what is wrong' — should trigger get_outage_status summary, not a refusal."""
    n = normalize_query_for_retrieval(user_text).lower()
    t = normalize_digits(user_text)
    ascii_markers = (
        "why ",
        " why",
        "reason",
        "cause",
        "when will",
        "what wrong",
        "what happened",
        "how long",
        "restore",
        "restoration",
        "outage",
        "fault",
        "feeder",
        "kab tak",
        "waju",
        "wajah",
        "masla",
        "light when",
        "power when",
    )
    ur_markers = ("وجہ", "کیوں", "کب تک", "مسئلہ", "بحال", "بحالی", "سلائیڈ", "بجلی کب", "کیا مسئلہ")
    return any(m in n for m in ascii_markers) or any(m in t for m in ur_markers)


def _user_asks_demo_reference(user_text: str) -> bool:
    n = normalize_query_for_retrieval(user_text).lower()
    t = normalize_digits(user_text)
    return any(
        m in n or m in t
        for m in (
            "reference",
            "ریفرنس",
            "شکایت نمبر",
            "complaint number",
            "complaint ref",
            "ref number",
        )
    )


def _assistant_already_asked_area_or_account(chat_ctx: lk_llm.ChatContext) -> bool:
    for it in chat_ctx.items:
        if getattr(it, "role", None) != "assistant":
            continue
        low = _item_text_content(it).lower()
        if any(m in low for m in _ASSISTANT_ASKED_AREA_MARKERS):
            return True
    return False


class VoiceAgent(Agent):
    def __init__(
        self,
        tools: LlmTools,
        instructions: str,
        preferred_language: Literal["ur", "en"] | None,
        on_language_locked: Callable[[Literal["ur", "en"]], None] | None = None,
    ) -> None:
        super().__init__(instructions=instructions, tools=[tools.get_outage_status, tools.get_complaint_reference])
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
        yielded_tts_text = False
        filler_emitted = False
        async for chunk in Agent.default.llm_node(self, chat_ctx_for_llm, tools, model_settings):
            if isinstance(chunk, str):
                if str(chunk).strip():
                    yielded_tts_text = True
            elif isinstance(chunk, lk_llm.ChatChunk) and chunk.delta:
                if chunk.delta.content and str(chunk.delta.content).strip():
                    yielded_tts_text = True

            if self._chunk_has_tool_call(chunk) and not filler_emitted and not yielded_tts_text:
                filler_emitted = True
                yield self._filler_phrase()

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
                    if isinstance(chunk, str):
                        c = normalize_english_currency_for_tts(chunk)
                        yield expand_ke_reference_for_english_tts(c)
                    else:
                        yield chunk

            return Agent.default.tts_node(self, _currency_normalized(), model_settings)

        return Agent.default.tts_node(self, text, model_settings)


def build_system_prompt(preferred_language: Literal["ur", "en"] | None) -> str:
    if preferred_language == "ur":
        language_prompt = (
            "LANGUAGE LOCK: Always respond ONLY in Urdu"
            "Never reply in English words or phrases (no “okay”, “let me check”, “sure”) — use Urdu equivalents. "
            "Keep 13-digit K-Electric account numbers as digits when repeating them, except mock complaint codes: "
            "use complaint_reference_spoken_ur from tools verbatim for those. Speak times and ETAs in natural spoken Urdu/Roman Urdu. "
            "Avoid Hindi register; prefer Pakistani Urdu vocabulary."
        )
    elif preferred_language == "en":
        language_prompt = (
            "LANGUAGE LOCK: Always respond ONLY in English. Never reply in Urdu. "
            "For mock complaint reference codes from tools, read complaint_reference_spoken_en verbatim."
        )
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

    if is_closing_utterance(user_text):
        items = list(chat_ctx.items)
        last_user_idx = next(
            (i for i in range(len(items) - 1, -1, -1) if getattr(items[i], "role", None) == "user"), len(items)
        )
        items.insert(last_user_idx, lk_llm.ChatMessage(role="system", content=[_CLOSING_TURN_HINT]))
        cloned = chat_ctx.copy()
        cloned.items = items
        return cloned

    intent = classify_intent(user_text)
    injections: list[str] = []
    if intent in ("rag", "mixed"):
        retrieval_lang: Literal["en", "ur"] = preferred_language or "en"
        results = retrieve(user_text, language=retrieval_lang, top_k=1)
        if results:
            injections.append("[Relevant domain policy]\n" + "\n".join(f"• {r.content}" for r in results))
    if intent in ("tool", "mixed"):
        if _assistant_already_asked_area_or_account(chat_ctx) and not _conversation_has_area_or_account(chat_ctx):
            pass
        elif _conversation_has_area_or_account(chat_ctx):
            injections.append(_TOOL_HINT_HAS_CONTEXT)
            if _user_wants_outage_explanation(user_text) or _user_asks_demo_reference(user_text):
                injections.append(_TOOL_HINT_OUTAGE_BEFORE_REFERENCE)
        else:
            injections.append(_TOOL_HINT_NEED_LOCATION)
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
