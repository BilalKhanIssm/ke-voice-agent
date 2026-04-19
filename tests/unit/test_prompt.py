from app.core.agent_core import build_system_prompt


def test_prompt_selection_english():
    prompt = build_system_prompt("en")
    assert "ONLY in English" in prompt


def test_prompt_selection_urdu():
    prompt = build_system_prompt("ur")
    assert "ONLY in Urdu" in prompt
