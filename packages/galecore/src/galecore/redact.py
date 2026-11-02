"""PII redaction (docs/02_DATA_PLAN.md §2.6, ADR-006).

Replaces personal data with tags: ``<PHONE>``, ``<NID>``, ``<ACCT>``, ``<EMAIL>``, and, when an
NER backend is plugged in, ``<NAME>`` / ``<ADDR>``.

Covers BD mobile numbers in ASCII or Bangla digits, with or without +880 and separators;
NID numbers (10/13/17 digits); emails; other long digit runs (account / card / wallet numbers);
and spoken-digit sequences ("শূন্য এক সাত ...", "zero one seven ...", "ডাবল ছয়").

Input should already be ``galecore.bangla.normalize``-d; offsets in the result refer to the input.
The original text is never logged or stored by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from galecore.bangla import normalize, to_ascii_digits, tokenize

PHONE = "<PHONE>"
NID = "<NID>"
ACCT = "<ACCT>"
EMAIL = "<EMAIL>"
NAME = "<NAME>"
ADDR = "<ADDR>"

BD_MOBILE = re.compile(r"^(?:\+?880|0)1[3-9]\d{8}$")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# a digit run, allowing single space / hyphen / dot separators between digits, optional leading +
_DIGIT_RUN = re.compile(r"(?<![\d])\+?\d(?:[ \-.]?\d)*(?![\d])")
NID_LENGTHS = frozenset({10, 13, 17})
MIN_ACCOUNT_DIGITS = 7  # shorter numbers (amounts, OTP-length codes, package ids) are kept

_SPOKEN_DIGITS: dict[str, str] = {
    # Bangla
    "শূন্য": "0", "শুন্য": "0", "জিরো": "0", "এক": "1", "ওয়ান": "1", "দুই": "2", "টু": "2",
    "তিন": "3", "থ্রি": "3", "চার": "4", "ফোর": "4", "পাঁচ": "5", "ফাইভ": "5",
    "ছয়": "6", "সিক্স": "6", "সাত": "7", "সেভেন": "7", "আট": "8", "এইট": "8",
    "নয়": "9", "নাইন": "9",  # ছয়, নয় (keys are normalised below, so nukta form is moot)
    # Banglish / English
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip
_SPOKEN_DIGITS = {normalize(k): v for k, v in _SPOKEN_DIGITS.items()}
_REPEAT = {normalize(k): n for k, n in {"ডাবল": 2, "double": 2, "ট্রিপল": 3, "triple": 3}.items()}


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    tag: str


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    spans: list[Span] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """Tag counts: safe to log (no content)."""
        out: dict[str, int] = {}
        for s in self.spans:
            out[s.tag] = out.get(s.tag, 0) + 1
        return out


class NERBackend(Protocol):
    """Finds person names and addresses. Returns (start, end, tag) with tag NAME or ADDR."""

    def find(self, text: str) -> list[Span]: ...


class NullNER:
    """Default: regex-only redaction. A real Bangla NER plugs in during P2 (docs/02 §2.6)."""

    def find(self, text: str) -> list[Span]:
        return []


def classify_digits(digits: str) -> str | None:
    """Tag for a run of ASCII digits (optionally with leading +), or None to keep it."""
    plain = digits.lstrip("+")
    if BD_MOBILE.match(digits) or BD_MOBILE.match(plain):
        return PHONE
    if len(plain) in NID_LENGTHS:
        return NID
    if len(plain) >= MIN_ACCOUNT_DIGITS:
        return ACCT
    return None


def _digit_spans(text: str) -> list[Span]:
    ascii_text = to_ascii_digits(text)  # length-preserving, so offsets stay valid
    spans = []
    for m in _DIGIT_RUN.finditer(ascii_text):
        compact = re.sub(r"[ \-.]", "", m.group())
        tag = classify_digits(compact)
        if tag:
            spans.append(Span(m.start(), m.end(), tag))
    return spans


def _spoken_digit_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    run_start = run_end = -1
    run_digits = ""
    pending_repeat = 1

    def flush() -> None:
        nonlocal run_start, run_digits
        if run_digits:
            tag = classify_digits(run_digits)
            if tag:
                spans.append(Span(run_start, run_end, tag))
        run_start, run_digits = -1, ""

    for tok in tokenize(text):
        key = normalize(tok.text)
        if key in _REPEAT:
            if run_start < 0:
                run_start = tok.start
            run_end = tok.end
            pending_repeat = _REPEAT[key]
            continue
        digit = _SPOKEN_DIGITS.get(key)
        if digit is None and key.isdigit():
            digit = key  # mixed: "জিরো 1 সাত"
        if digit is None:
            pending_repeat = 1
            flush()
            continue
        if run_start < 0:
            run_start = tok.start
        run_end = tok.end
        run_digits += digit * pending_repeat
        pending_repeat = 1
    flush()
    return spans


def _merge(spans: list[Span]) -> list[Span]:
    """Drop overlaps, keeping the earliest-starting, then longest span."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out


def redact(text: str, ner: NERBackend | None = None) -> RedactionResult:
    """Replace PII in ``text`` with tags."""
    if not text:
        return RedactionResult(text="")
    spans = [Span(m.start(), m.end(), EMAIL) for m in _EMAIL.finditer(text)]
    spans += _digit_spans(text)
    spans += _spoken_digit_spans(text)
    spans += (ner or NullNER()).find(text)
    merged = _merge(spans)
    parts: list[str] = []
    pos = 0
    for s in merged:
        parts.append(text[pos : s.start])
        parts.append(s.tag)
        pos = s.end
    parts.append(text[pos:])
    return RedactionResult(text="".join(parts), spans=merged)


def contains_pii(text: str) -> bool:
    """True if any regex-detectable PII is present (used by the log scrubber and lints)."""
    return bool(_EMAIL.search(text) or _digit_spans(text) or _spoken_digit_spans(text))
