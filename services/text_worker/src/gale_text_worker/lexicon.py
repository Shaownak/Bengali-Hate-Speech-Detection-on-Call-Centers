"""Lexicon matching on normalised, redacted text (docs/02 §4).

Exact match (multi-word terms included), edit distance ≤ 1 for terms of ≥ 5 letters, and a
Bangla-aware phonetic key. Placeholder tags (``<GALI_1>``) only ever match exactly, so
``<GALI_1>`` never fuzzy-matches ``<GALI_11>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from galecore.bangla import TAG, Token, normalize, phonetic_key, tokenize
from galecore.config import LexiconEntry

Match = Literal["exact", "fuzzy", "phonetic"]
MATCH_SCORE: dict[Match, float] = {"exact": 0.95, "fuzzy": 0.85, "phonetic": 0.8}
FUZZY_MIN_LEN = 5


@dataclass(frozen=True, slots=True)
class LexiconHit:
    entry: LexiconEntry
    match: Match
    surface: str  # the matched text as it appears (already normalised and redacted)
    char_start: int
    char_end: int
    token_start: int
    token_end: int  # exclusive

    @property
    def score(self) -> float:
        return MATCH_SCORE[self.match]


def _edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diffs += 1
        if diffs > 1:
            return False
        if len(a) == len(b):
            i += 1
        j += 1
    return diffs + (len(b) - j) <= 1


class LexiconMatcher:
    def __init__(self, entries: list[LexiconEntry]):
        self.exact: dict[tuple[str, ...], LexiconEntry] = {}
        self.phonetic: dict[str, LexiconEntry] = {}
        self.fuzzy: list[tuple[str, LexiconEntry]] = []
        for e in entries:
            for form in {e.term, *e.variants, *e.romanized}:
                toks = tuple(t.text for t in tokenize(normalize(form)))
                if not toks:
                    continue
                self.exact.setdefault(toks, e)
                if len(toks) == 1 and not TAG.fullmatch(toks[0]):
                    self.phonetic.setdefault(phonetic_key(toks[0]), e)
                    if len(toks[0]) >= FUZZY_MIN_LEN:
                        self.fuzzy.append((toks[0], e))
        self.max_ngram = max((len(k) for k in self.exact), default=1)

    def match(self, text: str) -> list[LexiconHit]:
        """Hits in ``text`` (which must already be normalised), longest exact match first."""
        toks = tokenize(text)
        hits: list[LexiconHit] = []
        i = 0
        while i < len(toks):
            hit = self._exact_at(text, toks, i) or self._approx_at(text, toks, i)
            if hit is None:
                i += 1
                continue
            hits.append(hit)
            i = hit.token_end
        return hits

    def _exact_at(self, text: str, toks: list[Token], i: int) -> LexiconHit | None:
        for n in range(min(self.max_ngram, len(toks) - i), 0, -1):
            key = tuple(t.text for t in toks[i : i + n])
            if key in self.exact:
                a, b = toks[i].start, toks[i + n - 1].end
                return LexiconHit(self.exact[key], "exact", text[a:b], a, b, i, i + n)
        return None

    def _approx_at(self, text: str, toks: list[Token], i: int) -> LexiconHit | None:
        tok = toks[i]
        w = tok.text
        if TAG.fullmatch(w):
            return None
        if len(w) >= FUZZY_MIN_LEN:
            for form, e in self.fuzzy:
                if _edit_distance_le1(w, form):
                    return LexiconHit(e, "fuzzy", w, tok.start, tok.end, i, i + 1)
        ph = self.phonetic.get(phonetic_key(w))
        if ph is not None:
            return LexiconHit(ph, "phonetic", w, tok.start, tok.end, i, i + 1)
        return None
