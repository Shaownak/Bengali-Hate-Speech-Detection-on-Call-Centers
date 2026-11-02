"""Register detector, rules v0 (docs/03 §6.1).

Levels: ``apni`` (আপনি + -েন/-ুন verbs) < ``tumi`` (তুমি + -ো/-ছ verbs) < ``tui`` (তুই + bare or
-িস imperatives). Pronouns decide the level; verb morphology alone is weak evidence and only
nudges the score, it never fires a signal on its own.

Per (call, speaker) state:
- the **agent** moving down a level fires (আপনি → তুমি, তুমি → তুই, আপনি → তুই);
- any speaker moving to তুই fires when the utterance has negative context;
- a **customer** moving to তুমি fires only with negative context (docs/02 §3.4 labels a
  disrespectful switch; a neutral তুমি from a customer is habit/dialect);
- a level used from the speaker's first register-bearing utterance is their neutral baseline
  and never fires (dialect suppression).
The "tiny model" on rule features + text-clf scores comes with in-domain data (P3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from galecore.bangla import normalize, tokenize
from galecore.models.base import RegisterDetector
from galecore.models.stubs import stable_id
from galecore.schemas import AsrSegment, Category, Channel, Evidence, Signal, SignalSource

Level = Literal["apni", "tumi", "tui"]
RANK: dict[Level, int] = {"apni": 0, "tumi": 1, "tui": 2}
VERSION = "register-rules-0.1"

PRONOUNS: dict[Level, frozenset[str]] = {
    "apni": frozenset({"আপনি", "আপনার", "আপনাকে", "আপনারা", "আপনাদের", "apni", "apnar", "apnake"}),
    "tumi": frozenset({"তুমি", "তোমার", "তোমাকে", "তোমরা", "তোমাদের", "tumi", "tomar", "tomake"}),
    "tui": frozenset({"তুই", "তোর", "তোকে", "তোরা", "tui", "tor", "toke"}),
}
VERBS: dict[Level, re.Pattern[str]] = {
    "apni": re.compile(r"(?:েন|ুন)$"),
    "tumi": re.compile(r"(?:ো|ছ|েছ|তেছ)$"),
    "tui": re.compile(r"(?:িস|িস না)$"),
}


@dataclass(frozen=True, slots=True)
class RegisterObservation:
    level: Level
    char_start: int  # of the pronoun that decided it, in the normalised text
    pronoun: str
    verb_agrees: bool


def observe_level(text: str) -> RegisterObservation | None:
    """The lowest-register pronoun in ``text`` (normalised), plus whether verbs agree."""
    toks = tokenize(normalize(text))
    found: RegisterObservation | None = None
    for t in toks:
        for level in ("tui", "tumi", "apni"):
            if t.text in PRONOUNS[level] and (found is None or RANK[level] > RANK[found.level]):
                found = RegisterObservation(level, t.start, t.text, False)
    if found is None:
        return None
    agrees = any(VERBS[found.level].search(t.text) for t in toks)
    return RegisterObservation(found.level, found.char_start, found.pronoun, agrees)


def word_index_at(segment: AsrSegment, char_pos: int) -> int | None:
    """Index into ``segment.words`` of the word covering ``char_pos`` in ``text_redacted``
    (the ASR step guarantees ``text_redacted == " ".join(w.word for w in words)``)."""
    pos = 0
    for i, w in enumerate(segment.words):
        if pos <= char_pos < pos + len(w.word):
            return i
        pos += len(w.word) + 1
    return None


class RuleRegisterDetector(RegisterDetector):
    def __init__(self) -> None:
        self._baseline: dict[tuple[str, Channel], Level] = {}
        self._last: dict[tuple[str, Channel], Level] = {}

    def observe(self, segment: AsrSegment, negative_context: bool = False) -> list[Signal]:
        obs = observe_level(segment.text_redacted)
        if obs is None:
            return []
        key = (segment.call_id, segment.channel)
        base = self._baseline.setdefault(key, obs.level)
        prev = self._last.get(key, base)
        self._last[key] = obs.level
        if RANK[obs.level] <= RANK[base] or RANK[obs.level] <= RANK[prev]:
            return []  # neutral baseline, or no move down since the last utterance
        agent = segment.channel == Channel.AGENT
        if not (agent or negative_context):
            return []
        start, end = segment.ts_start, segment.ts_end
        wi = word_index_at(segment, obs.char_start)
        if wi is not None:
            start, end = segment.words[wi].start, segment.words[wi].end
        score = (0.9 if agent else 0.8) + (0.05 if obs.verb_agrees else 0.0)
        return [
            Signal(
                call_id=segment.call_id,
                channel=segment.channel,
                signal_id=stable_id(segment.segment_id, "register", obs.char_start),
                source=SignalSource.REGISTER,
                category=Category.DISRESPECT_REGISTER,
                score=round(score, 3),
                ts_start=start,
                ts_end=end,
                evidence=Evidence(
                    segment_id=segment.segment_id,
                    text_span=obs.pronoun,
                    word_ids=[] if wi is None else [wi],
                    features={
                        "from_level": float(RANK[prev]),
                        "to_level": float(RANK[obs.level]),
                        "verb_agrees": float(obs.verb_agrees),
                        "negative_context": float(negative_context),
                    },
                ),
                model_version=VERSION,
            )
        ]
