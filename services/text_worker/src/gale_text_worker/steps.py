"""text-worker logic: one redacted ASR segment → lexicon, text-classifier and register signals.

Shared by the post-call pipeline (``gale_postcall``) and, from P4, the streaming text-worker.
Input text is already redacted (D-01) and is normalised again here before matching (CLAUDE.md).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from gale_text_worker.lexicon import LexiconMatcher
from gale_text_worker.register import word_index_at
from galecore.bangla import normalize
from galecore.config import ThresholdsConfig
from galecore.models.base import RegisterDetector, TextClassifier, TextContext
from galecore.models.stubs import stable_id
from galecore.schemas import AsrSegment, Category, Channel, Evidence, Signal, SignalSource

LEXICON_VERSION = "lexicon-matcher-0.1"
ABUSIVE = frozenset(
    {
        Category.SLANG_PROFANITY,
        Category.INSULT,
        Category.THREAT,
        Category.SEXUAL_HARASSMENT,
        Category.HATE_IDENTITY,
        Category.DISMISSIVE_RUDE,
        Category.SARCASM_MOCKERY,
    }
)
NEGATIVE_CONTEXT_SCORE = 0.6  # text-clf score that counts as negative context for register


@dataclass
class TextWorker:
    matcher: LexiconMatcher
    classifier: TextClassifier
    register: RegisterDetector
    thresholds: ThresholdsConfig
    context_turns: int = 2  # previous utterances of each speaker (docs/03 §6)
    _history: deque[tuple[Channel, str]] = field(default_factory=lambda: deque(maxlen=4))

    def process(self, seg: AsrSegment) -> list[Signal]:
        text = normalize(seg.text_redacted)
        signals = self._lexicon(seg, text)
        ctx = TextContext(current=text, speaker=seg.channel, previous=list(self._history))
        clf = self.classifier.classify(ctx)
        self._history.append((seg.channel, text))
        for cat, score in clf.scores.items():
            if score < self.thresholds.categories[cat].log or cat == Category.DISRESPECT_REGISTER:
                continue  # register comes from the register detector, not the classifier alone
            signals.append(
                Signal(
                    call_id=seg.call_id,
                    channel=seg.channel,
                    signal_id=stable_id(seg.segment_id, "text_clf", cat),
                    source=SignalSource.TEXT_CLF,
                    category=cat,
                    score=round(score, 4),
                    ts_start=seg.ts_start,
                    ts_end=seg.ts_end,
                    evidence=Evidence(
                        segment_id=seg.segment_id,
                        features={
                            "severity": float(max(1, clf.severity)),
                            "aggression": clf.aggression,
                        },
                    ),
                    model_version=clf.model_version,
                )
            )
        negative = any(s.source == SignalSource.LEXICON for s in signals) or any(
            clf.scores.get(c, 0.0) >= NEGATIVE_CONTEXT_SCORE for c in ABUSIVE
        )
        signals += self.register.observe(seg, negative_context=negative)
        return signals

    def _lexicon(self, seg: AsrSegment, text: str) -> list[Signal]:
        return lexicon_signals(self.matcher, seg, text)


def lexicon_signals(matcher: LexiconMatcher, seg: AsrSegment, text: str) -> list[Signal]:
    """Lexicon hits in one (normalised) segment as signals with word-precise timing."""
    out = []
    for hit in matcher.match(text):
        first = word_index_at(seg, hit.char_start)
        last = word_index_at(seg, max(hit.char_start, hit.char_end - 1))
        if first is not None and last is not None:
            t0, t1 = seg.words[first].start, seg.words[last].end
            ids = list(range(first, last + 1))
        else:
            t0, t1, ids = seg.ts_start, seg.ts_end, []
        out.append(
            Signal(
                call_id=seg.call_id,
                channel=seg.channel,
                signal_id=stable_id(seg.segment_id, "lexicon", hit.char_start),
                source=SignalSource.LEXICON,
                category=hit.entry.category,
                score=hit.score,
                ts_start=t0,
                ts_end=t1,
                evidence=Evidence(
                    text_span=hit.surface,
                    char_start=hit.char_start,
                    char_end=hit.char_end,
                    word_ids=ids,
                    segment_id=seg.segment_id,
                    term_id=hit.entry.id,
                    features={
                        "severity": float(hit.entry.severity),
                        "match_exact": float(hit.match == "exact"),
                        "context_sensitive": float(hit.entry.context_sensitive),
                    },
                ),
                model_version=LEXICON_VERSION,
            )
        )
    return out
