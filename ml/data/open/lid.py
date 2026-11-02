"""Script detection and a light language-ID filter (docs/09 §6.3).

Rows written in Devanagari (Hindi) are dropped; all-Latin rows are dropped when they read as
English (English function words dominate and no Banglish markers appear). Everything else is kept
and tagged bangla / roman / mixed.
"""

from __future__ import annotations

import re
from typing import Literal

from galecore.bangla import TAG

Lang = Literal["bn", "banglish", "en", "hi", "other"]
BENGALI = re.compile(r"[ঀ-৿]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[a-z']+")
EN_FUNCTION = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "at",
        "for",
        "with",
        "this",
        "that",
        "it",
        "you",
        "your",
        "he",
        "she",
        "they",
        "we",
        "i",
        "my",
        "me",
        "not",
        "no",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "what",
        "which",
        "who",
        "how",
        "why",
        "from",
        "by",
        "as",
        "if",
        "but",
        "so",
        "just",
        "very",
    ]
)
BANGLISH = frozenset(
    [
        "ami",
        "tumi",
        "apni",
        "tui",
        "amar",
        "tomar",
        "apnar",
        "tor",
        "ki",
        "keno",
        "kno",
        "na",
        "nai",
        "nah",
        "ar",
        "r",
        "ei",
        "oi",
        "koro",
        "koren",
        "kor",
        "bolo",
        "bolen",
        "bol",
        "kotha",
        "bhai",
        "vai",
        "bhalo",
        "valo",
        "kemon",
        "kmn",
        "ache",
        "ase",
        "hobe",
        "hbe",
        "korbo",
        "kore",
        "korse",
        "korchi",
        "chai",
        "lagbe",
        "taka",
        "sala",
        "shala",
        "mama",
        "apu",
        "dada",
        "bou",
        "baba",
        "ma",
        "onek",
        "onk",
        "khub",
        "kintu",
        "tai",
        "jodi",
        "tahole",
        "thik",
        "ase",
    ]
)


def script_of(text: str) -> Literal["bangla", "roman", "mixed", "other"]:
    text = TAG.sub(" ", text)  # placeholder tags (<GALI_1>, <PHONE>) are not Latin-script content
    bn, lat = len(BENGALI.findall(text)), len(LATIN.findall(text))
    if bn and lat:
        return "bangla" if bn >= 4 * lat else "roman" if lat >= 4 * bn else "mixed"
    if bn:
        return "bangla"
    if lat:
        return "roman"
    return "other"


def language_of(text: str) -> Lang:
    if DEVANAGARI.search(text) and not BENGALI.search(text):
        return "hi"
    text = TAG.sub(" ", text)
    sc = script_of(text)
    if sc in ("bangla", "mixed"):
        return "bn"
    if sc == "other":
        return "other"
    words = WORD.findall(text.lower())
    if not words:
        return "other"
    en = sum(w in EN_FUNCTION for w in words) / len(words)
    bl = sum(w in BANGLISH for w in words) / len(words)
    return "banglish" if bl > 0 and bl >= en / 3 else "en" if en >= 0.25 else "banglish"


def keep(text: str) -> bool:
    """docs/09 §6.3: drop Hindi and English-only rows."""
    return language_of(text) in ("bn", "banglish")
