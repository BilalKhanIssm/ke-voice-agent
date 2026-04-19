import re
from typing import Final

_WORD_TO_DIGIT: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ایک": "1",
    "دو": "2",
    "تین": "3",
    "چار": "4",
    "پانچ": "5",
    "چھ": "6",
    "سات": "7",
    "آٹھ": "8",
    "نو": "9",
}
_SORTED_WORDS = sorted(_WORD_TO_DIGIT.keys(), key=len, reverse=True)
_PATTERN = re.compile(r"(?<!\S)(" + "|".join(re.escape(w) for w in _SORTED_WORDS) + r")(?!\S)", re.IGNORECASE)

_RS_BEFORE_DIGIT = re.compile(r"(?i)(?<![A-Za-z0-9])rs\.?\s*(?=\d)")
_PKR_BEFORE_DIGIT = re.compile(r"(?i)(?<![A-Za-z0-9])pkr\.?\s*(?=\d)")

# Mock complaint references (KE-##########): expand for TTS so engines read digit-by-digit, not as one integer.
_URDU_DIGIT_WORD: Final[tuple[str, ...]] = (
    "صفر",
    "ایک",
    "دو",
    "تین",
    "چار",
    "پانچ",
    "چھ",
    "سات",
    "آٹھ",
    "نو",
)
_EN_DIGIT_WORD: Final[tuple[str, ...]] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
_KE_REF_RE: Final[re.Pattern[str]] = re.compile(r"\bKE-(\d{10})\b", re.IGNORECASE)


def normalize_digits(text: str) -> str:
    result = _PATTERN.sub(lambda m: _WORD_TO_DIGIT.get(m.group(0).lower(), m.group(0)), text)
    result = re.sub(
        r"(?<!\d)(\d)([ \u00a0]\d)+(?!\d)",
        lambda m: m.group(0).replace(" ", "").replace("\u00a0", ""),
        result,
    )
    return result.strip()


def normalize_english_currency_for_tts(text: str) -> str:
    if not text:
        return text
    out = _RS_BEFORE_DIGIT.sub("rupees ", text)
    out = _PKR_BEFORE_DIGIT.sub("rupees ", out)
    return out


# TTS often picks a Hindi lexicon for Devanagari-looking tokens from the LLM; normalise a few frequent glitches.
_URDU_TTS_LEXICON: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bsurakshit\b"), "mehfooz"),
    (re.compile(r"(?i)\bsurakṣit\b"), "mehfooz"),
]


def format_ke_reference_spoken_urdu(ref: str) -> str:
    """Spoken Urdu for a mock KE complaint reference (digit-by-digit for UpliftAI / telephony TTS)."""
    m = _KE_REF_RE.search((ref or "").strip())
    if not m:
        return ref or ""
    digits = m.group(1)
    spoken_digits = "، ".join(_URDU_DIGIT_WORD[int(d)] for d in digits)
    return f"کے ای، {spoken_digits}"


def format_ke_reference_spoken_en(ref: str) -> str:
    """Spoken English digit names for the same mock reference shape."""
    m = _KE_REF_RE.search((ref or "").strip())
    if not m:
        return ref or ""
    digits = m.group(1)
    spoken_digits = ", ".join(_EN_DIGIT_WORD[int(d)] for d in digits)
    return f"K E, {spoken_digits}"


def expand_ke_reference_for_urdu_tts(text: str) -> str:
    """Replace KE-########## in assistant text so Urdu TTS does not read it as a single large number."""

    def _repl(match: re.Match[str]) -> str:
        return format_ke_reference_spoken_urdu(match.group(0))

    if not text:
        return text
    return _KE_REF_RE.sub(_repl, text)


def expand_ke_reference_for_english_tts(text: str) -> str:
    if not text:
        return text

    def _repl(match: re.Match[str]) -> str:
        return format_ke_reference_spoken_en(match.group(0))

    return _KE_REF_RE.sub(_repl, text)


def normalize_roman_urdu_for_tts(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, replacement in _URDU_TTS_LEXICON:
        out = pattern.sub(replacement, out)
    out = expand_ke_reference_for_urdu_tts(out)
    return out
