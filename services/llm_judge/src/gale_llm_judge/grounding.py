"""Post-validation of LLM output in code, not in the model (docs/08 §2).

- every ``evidence`` must be a substring of the referenced utterance (or of some utterance in
  the window when no id is given), else the event is dropped (``llm_ungrounded``). The match is
  exact up to Unicode form (``galecore.bangla.find_canonical``): model tokenizers NFC-normalise,
  which decomposes ড় / য়, while transcripts use the precomposed form. The signal keeps the
  transcript's own span;
- ``speaker`` must match that utterance's channel;
- ``shouting`` / ``harsh_tone`` are accepted only when acoustic signals for that speaker in the
  window support them: the LLM cannot hear audio.
Accepted events become ``Signal(source=llm_window)`` for fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter

from galecore.bangla import find_canonical
from galecore.models.base import WindowInput
from galecore.models.stubs import stable_id
from galecore.schemas import (
    AsrSegment,
    Category,
    Evidence,
    JudgedEvent,
    Signal,
    SignalSource,
    WindowJudgement,
)

REJECTED = Counter(
    "gale_llm_rejected_events_total", "LLM events dropped by post-validation", ["reason"]
)
ACOUSTIC_ONLY = {Category.SHOUTING, Category.HARSH_TONE}
ACOUSTIC_SOURCES = {SignalSource.DSP_SHOUT, SignalSource.SER}
ACOUSTIC_SUPPORT = 0.5


@dataclass(frozen=True, slots=True)
class Grounded:
    signals: list[Signal]
    rejected: dict[str, int]


def ground(
    window: WindowInput, judgement: WindowJudgement, ids: dict[str, str], model_version: str
) -> Grounded:
    by_uid: dict[str, AsrSegment] = {
        ids[s.segment_id]: s for s in window.segments if s.segment_id in ids
    }
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1
        REJECTED.labels(reason).inc()

    out = []
    for ev in judgement.events:
        found = _find(ev, by_uid, window.segments)
        if found is None:
            reject("llm_ungrounded")
            continue
        seg, (start, end) = found
        if seg.channel.value != ev.speaker.value:
            reject("speaker_mismatch")
            continue
        if ev.category in ACOUSTIC_ONLY and not any(
            s.source in ACOUSTIC_SOURCES
            and s.channel == seg.channel
            and s.score >= ACOUSTIC_SUPPORT
            and s.ts_start <= seg.ts_end
            and seg.ts_start <= s.ts_end
            for s in window.signals
        ):
            reject("no_acoustic_support")
            continue
        out.append(
            Signal(
                call_id=window.call_id,
                channel=seg.channel,
                signal_id=stable_id(seg.segment_id, "llm", ev.category, start),
                source=SignalSource.LLM_WINDOW,
                category=ev.category,
                score=ev.confidence,
                ts_start=seg.ts_start,
                ts_end=seg.ts_end,
                evidence=Evidence(
                    text_span=seg.text_redacted[start:end],
                    char_start=start,
                    char_end=end,
                    segment_id=seg.segment_id,
                    features={"severity": float(ev.severity)},
                ),
                model_version=model_version,
            )
        )
    return Grounded(out, rejected)


def _find(
    ev: JudgedEvent, by_uid: dict[str, AsrSegment], segments: list[AsrSegment]
) -> tuple[AsrSegment, tuple[int, int]] | None:
    if ev.utterance_id is not None:
        seg = by_uid.get(ev.utterance_id)
        candidates = [seg] if seg is not None else []
    else:
        candidates = segments
    for s in candidates:
        span = find_canonical(s.text_redacted, ev.evidence)
        if span is not None:
            return s, span
    return None
