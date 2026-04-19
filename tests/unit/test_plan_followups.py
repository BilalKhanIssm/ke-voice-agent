from livekit.agents import llm as lk_llm

from app.core.agent_core import (
    _conversation_has_area_or_account,
    _user_asks_demo_reference,
    _user_wants_outage_explanation,
)
from app.core.knowledge_base import is_closing_utterance


def test_is_closing_urdu_allah_hafiz():
    assert is_closing_utterance("اللہ حافظ") is True
    assert is_closing_utterance("چلیں شکریہ آپ کی کال کا اللہ حافظ") is True


def test_is_closing_english_goodbye():
    assert is_closing_utterance("Goodbye, thank you for your help.") is True


def test_is_closing_not_rant():
    long = "no light " * 30 + "allah hafiz"
    assert is_closing_utterance(long) is False


def test_block_style_location_mein_333():
    ctx = lk_llm.ChatContext()
    ctx.items = [lk_llm.ChatMessage(role="user", content=["جی میں 333 میں رہتا ہوں"])]
    assert _conversation_has_area_or_account(ctx) is True


def test_scheme_block_pattern():
    ctx = lk_llm.ChatContext()
    ctx.items = [lk_llm.ChatMessage(role="user", content=["I am in block 5 scheme 33"])]
    assert _conversation_has_area_or_account(ctx) is True


def test_urdu_johar_block_18():
    ctx = lk_llm.ChatContext()
    ctx.items = [lk_llm.ChatMessage(role="user", content=["آ جی میں جوہر بلاک 18 میں رہتا ہوں"])]
    assert _conversation_has_area_or_account(ctx) is True


def test_user_wants_outage_explanation_urdu():
    assert _user_wants_outage_explanation("مسئلہ کیا ہے بجلی کیوں نہیں آ رہی") is True
    assert _user_wants_outage_explanation("کب تک آ جائے گی") is True


def test_user_asks_demo_reference():
    assert _user_asks_demo_reference("ریفرنس نمبر چاہیے") is True
    assert _user_asks_demo_reference("just saying hi") is False
