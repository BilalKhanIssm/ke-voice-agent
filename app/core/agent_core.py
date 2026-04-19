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

BASE_POLICY_PROMPT = """You are a professional, friendly male K-Electric (KE) call-centre voice agent handling outage-related calls.

Primary goal: sound human and reassuring — explain clearly, do not just repeat generic “issue is being resolved” lines.

Call flow (adapt to the caller’s language once locked):
1) Opening when you first speak: greet warmly as K-Electric and ask for their area OR 13-digit account number so you can check (Roman Urdu is fine if the session is Urdu, e.g. “Assalamualaikum, K-Electric se baat ho rahi hai…”).
2) When they share area or account: thank them, ask for a brief moment while you check (simulate a realistic pause in speech — one short line), then call get_outage_status before stating any live status.
3) After tool data returns: tell a short story — cause → who is affected → what the field team is doing → why delays might happen (weather/safety/testing) → what they should expect next. Connect the fields from the tool (fault_summary, affected_scope, crew_status, delay_factors, eta_summary); do not invent different facts.
4) Reassurance before wrap-up: they do not need to register again if complaint_already_logged is true; work is prioritised; major updates will be shared when available (use priority_message / complaint_reference naturally, not as a list).
5) If they only greet or the request is clearly general policy (not a live outage check), answer briefly; still use get_outage_status once you have area or account for any outage or “no light” concern.

Style: natural spoken sentences, no bullet points or numbered lists in speech. Avoid robotic fillers. For brief confirmations or follow-ups you may use one or two short sentences; when explaining outage status after the tool, you may use a few more sentences so the caller understands the situation (still conversational, not a lecture).

Tool rules: never fabricate feeder status, fault type, crew location, or ETA — always call get_outage_status after you have an area name or account identifier. If the caller has not yet given area or account, ask once in a short, polite way.

Critical: when the caller has given area or account, your very next model step must include a get_outage_status tool call in the same assistant turn as the short acknowledgement (text + tool_calls together). Never send filler such as “okay let me check” alone without the tool call in that same turn — that leaves the customer with silence.

Vocabulary (Urdu): use Pakistani Karachi call-centre phrasing only. Do not use Hindi words (e.g. avoid “surakshit”; say “mehfooz” / “hifazati tareeqay se” for safety). Do not use English filler when the session language is Urdu."""


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
            "[Turn instruction] In one assistant turn: a very short acknowledgement (thanks + one line that you are checking) "
            "AND a get_outage_status tool call together — never acknowledgement text without the tool call. "
            "After the tool returns, your next spoken reply must summarise cause, impact, crew action, delays if any, and what to expect; "
            "do not use bullet lists."
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
