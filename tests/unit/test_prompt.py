from app.core.agent_core import MAX_CHAT_CONTEXT_MESSAGES, VoiceAgent, build_system_prompt


def test_prompt_selection_english():
    prompt = build_system_prompt("en")
    assert "ONLY in English" in prompt


def test_prompt_selection_urdu():
    prompt = build_system_prompt("ur")
    assert "ONLY in Urdu" in prompt
    assert "ONLY in Urdu. Never reply in English words" in prompt


def test_context_budget_increased():
    assert MAX_CHAT_CONTEXT_MESSAGES == 12


def test_filler_phrase_varies_without_immediate_repeat():
    agent = VoiceAgent.__new__(VoiceAgent)
    agent.preferred_language = "en"
    agent._rng = __import__("random").Random(7)
    agent._last_filler_phrase = None

    first = VoiceAgent._filler_phrase(agent)
    second = VoiceAgent._filler_phrase(agent)
    assert first != second
