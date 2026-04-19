from app.core.transcript_utils import (
    expand_ke_reference_for_english_tts,
    expand_ke_reference_for_urdu_tts,
    format_ke_reference_spoken_en,
    format_ke_reference_spoken_urdu,
    normalize_roman_urdu_for_tts,
)


def test_format_ke_reference_spoken_urdu_digit_by_digit():
    ref = "KE-0000000007"
    out = format_ke_reference_spoken_urdu(ref)
    assert out.startswith("کے ای،")
    assert "سات" in out
    assert "صفر" in out


def test_expand_ke_reference_in_sentence_urdu():
    s = "آپ کا نمبر KE-0000000007 ہے۔"
    out = expand_ke_reference_for_urdu_tts(s)
    assert "KE-0000000007" not in out
    assert "کے ای،" in out


def test_format_ke_reference_spoken_en():
    ref = "KE-0000000007"
    out = format_ke_reference_spoken_en(ref)
    assert out.startswith("K E,")
    assert "seven" in out


def test_normalize_roman_urdu_for_tts_expands_ke_reference():
    s = "ریفرنس KE-1234567890"
    out = normalize_roman_urdu_for_tts(s)
    assert "KE-1234567890" not in out
    assert "کے ای،" in out


def test_expand_ke_reference_english():
    s = "Your reference is KE-1234567890."
    out = expand_ke_reference_for_english_tts(s)
    assert "KE-1234567890" not in out
    assert "one" in out.lower()
