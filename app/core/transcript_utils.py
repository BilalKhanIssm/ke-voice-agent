import re

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


def normalize_roman_urdu_for_tts(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, replacement in _URDU_TTS_LEXICON:
        out = pattern.sub(replacement, out)
    return out
