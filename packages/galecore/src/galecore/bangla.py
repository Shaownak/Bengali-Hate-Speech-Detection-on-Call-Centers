"""Bangla text normalisation, phonetic keys and tokenisation.

The same ``normalize()`` must be used for training targets, inference, lexicon matching and
WER scoring (docs/02_DATA_PLAN.md §2.5).

Project rules applied after Unicode NFC:
- nukta letters are recomposed: ড+় → ড় (U+09DC), ঢ+় → ঢ় (U+09DD), য+় → য় (U+09DF).
  NFC itself decomposes these (they are composition exclusions); we pick the precomposed
  form because that is what keyboards and most corpora produce.
- legacy khanda-ta ত্‍ (ত + hasanta + ZWJ) → ৎ (U+09CE).
- ZWJ, ZWNJ, ZWSP, BOM, soft hyphen and bidi marks are removed.
- digits are unified (ASCII by default, or Bangla).
- whitespace is collapsed; Latin (Banglish) is lower-cased.
- optional canonical-form map for loanword spelling variants (configs/bangla_canonical.yaml).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

NUKTA = "়"
HASANTA = "্"
ZWJ = "‍"
ZWNJ = "‌"

_NUKTA_COMPOSE = {
    "ড" + NUKTA: "ড়",  # ড় RRA
    "ঢ" + NUKTA: "ঢ়",  # ঢ় RHA
    "য" + NUKTA: "য়",  # য় YYA
}
_KHANDA_TA_LEGACY = "ত" + HASANTA + ZWJ  # ত্‍

# zero-width and invisible formatting characters that only add inconsistency
_INVISIBLE = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­‎‏‪‫‬‭‮"),
    None,
)

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
_TO_ASCII_DIGITS = str.maketrans(BN_DIGITS, ASCII_DIGITS)
_TO_BN_DIGITS = str.maketrans(ASCII_DIGITS, BN_DIGITS)

_WS = re.compile(r"\s+")
TAG = re.compile(r"<[A-Z][A-Z0-9_]*>")  # placeholder tags: <PHONE>, <NAME>, <GALI_1> …


def _lower_latin_outside_tags(s: str) -> str:
    parts: list[str] = []
    pos = 0
    for m in TAG.finditer(s):
        parts.append(s[pos : m.start()].translate(_LATIN_LOWER))
        parts.append(m.group())
        pos = m.end()
    parts.append(s[pos:].translate(_LATIN_LOWER))
    return "".join(parts)


_LATIN_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")

DigitStyle = Literal["ascii", "bangla", "keep"]


def to_ascii_digits(text: str) -> str:
    """Bangla digits → ASCII. Length-preserving (1 char → 1 char)."""
    return text.translate(_TO_ASCII_DIGITS)


def to_bangla_digits(text: str) -> str:
    """ASCII digits → Bangla. Length-preserving."""
    return text.translate(_TO_BN_DIGITS)


def normalize(
    text: str,
    *,
    digits: DigitStyle = "ascii",
    canonical: Mapping[str, str] | None = None,
    lowercase_latin: bool = True,
) -> str:
    """Normalise Bangla/Banglish text. Idempotent: ``normalize(normalize(x)) == normalize(x)``."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = s.replace(_KHANDA_TA_LEGACY, "ৎ")
    s = s.translate(_INVISIBLE)
    for decomposed, composed in _NUKTA_COMPOSE.items():
        s = s.replace(decomposed, composed)
    if digits == "ascii":
        s = to_ascii_digits(s)
    elif digits == "bangla":
        s = to_bangla_digits(s)
    if lowercase_latin:
        s = _lower_latin_outside_tags(s)
    s = _WS.sub(" ", s).strip()
    if canonical:
        s = " ".join(canonical.get(tok, tok) for tok in s.split(" "))
    return s


def unicode_canonical(text: str) -> str:
    """The Unicode part of ``normalize()`` only (NFC, nukta recomposition, khanda-ta, invisible
    characters removed): no digit, case or whitespace changes."""
    s = unicodedata.normalize("NFC", text).replace(_KHANDA_TA_LEGACY, "ৎ").translate(_INVISIBLE)
    for decomposed, composed in _NUKTA_COMPOSE.items():
        s = s.replace(decomposed, composed)
    return s


def find_canonical(text: str, needle: str) -> tuple[int, int] | None:
    """``(start, end)`` of ``needle`` in ``text``: exact if possible, else the shortest span of
    ``text`` equal to ``needle`` under ``unicode_canonical``. Model tokenizers normalise Unicode
    (Qwen applies NFC, which decomposes ড় / ঢ় / য়), so text a model copies verbatim can come back
    in another form; the span returned always indexes the original ``text``."""
    if not needle.strip():
        return None
    i = text.find(needle)
    if i >= 0:
        return i, i + len(needle)
    want = unicode_canonical(needle)
    canon = unicode_canonical(text)
    if want not in canon:
        return None
    slack = len(needle) // 2 + 2
    for start in range(len(text)):
        if not unicode_canonical(text[start:]).startswith(want):
            continue
        lo, hi = max(start + 1, start + len(want) - slack), start + len(needle) + slack
        for end in range(lo, min(len(text), hi) + 1):
            if unicode_canonical(text[start:end]) == want:
                return start, end
    return None


# ---------------------------------------------------------------------------
# Phonetic key (docs/02 §4): merges letters that ASR and writers confuse
# ---------------------------------------------------------------------------

_PHONETIC = str.maketrans(
    {
        "শ": "স",  # শ → স
        "ষ": "স",  # ষ → স
        "য": "জ",  # য → জ
        "ণ": "ন",  # ণ → ন
        "ঈ": "ই",  # ঈ → ই
        "ী": "ি",  # vowel sign ী → ি
        "ঊ": "উ",  # ঊ → উ
        "ূ": "ু",  # vowel sign ূ → ু
        "ড়": "র",  # ড় → র
        "ঢ়": "র",  # ঢ় → র
    }
)


def phonetic_key(text: str) -> str:
    """Bangla-aware phonetic key used for fuzzy lexicon matching.

    Merges শ/ষ/স, জ/য, ন/ণ, ই/ঈ (and ি/ী), উ/ঊ (and ু/ূ), র/ড়/ঢ়. Hasanta and
    chandrabindu are dropped since speakers and ASR realise them inconsistently.
    """
    s = normalize(text)
    s = s.translate(_PHONETIC)
    return s.replace(HASANTA, "").replace("ঁ", "")


# ---------------------------------------------------------------------------
# Tokeniser with character offsets (evidence spans must be exact substrings)
# ---------------------------------------------------------------------------

# Placeholder tags (<PHONE>, <GALI_1>) stay whole; Bangla block chars incl. combining marks.
_TOKEN = re.compile(r"<[A-Z][A-Z0-9_]*>|[ঀ-৿\w]+(?:['’][ঀ-৿\w]+)*")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Split into word tokens with offsets into ``text``. Punctuation (। , ? !) is dropped."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def words(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]
