"""asr-worker logic: VAD regions → utterance chunks → ASR → normalise + redact → AsrSegment.

D-01: raw recogniser text (``RawSegment``) exists only inside this module's call frames. It is
normalised and redacted here; only ``AsrSegment.text_redacted`` leaves the worker.
Invariant used downstream: ``text_redacted == " ".join(w.word for w in words)``.
"""

from __future__ import annotations

from galecore.bangla import normalize
from galecore.models.base import ASRBackend, AudioChunk, RawSegment
from galecore.models.stubs import stable_id
from galecore.redact import redact
from galecore.schemas import AsrSegment, Word

MAX_CHUNK_S = 30.0
PAD_S = 0.2
JOIN_GAP_S = 0.8


def utterance_chunks(
    regions: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Join VAD regions closer than ``JOIN_GAP_S`` into utterances (≤ 30 s), padded a little."""
    out: list[tuple[float, float]] = []
    for a, b in regions:
        if out and a - out[-1][1] < JOIN_GAP_S and b - out[-1][0] <= MAX_CHUNK_S:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(max(0.0, a - PAD_S), min(duration_s, b + PAD_S)) for a, b in out]


def to_asr_segment(raw: RawSegment, call_id: str) -> AsrSegment:
    """Normalise and redact one raw segment. PII spanning several words collapses into one tag
    word whose timing covers them all."""
    norm = [normalize(w.word) for w in raw.words]
    joined = " ".join(norm)
    result = redact(joined)
    offsets, pos = [], 0
    for n in norm:
        offsets.append((pos, pos + len(n)))
        pos += len(n) + 1
    words: list[Word] = []
    i = 0
    spans = sorted(result.spans, key=lambda s: s.start)
    while i < len(raw.words):
        a, b = offsets[i]
        span = next((s for s in spans if s.start < b and a < s.end), None)
        if span is None:
            words.append(
                Word(
                    word=norm[i],
                    start=raw.words[i].start,
                    end=raw.words[i].end,
                    conf=raw.words[i].conf,
                )
            )
            i += 1
            continue
        j = i
        while j + 1 < len(raw.words) and offsets[j + 1][0] < span.end:
            j += 1
        # the tag plus any non-PII remainder of the first/last word (e.g. punctuation)
        head = norm[i][: max(0, span.start - a)]
        tail = norm[j][max(0, span.end - offsets[j][0]) :]
        words.append(
            Word(
                word=f"{head}{span.tag}{tail}",
                start=raw.words[i].start,
                end=raw.words[j].end,
                conf=min(w.conf for w in raw.words[i : j + 1]),
            )
        )
        i = j + 1
    text = " ".join(w.word for w in words)
    return AsrSegment(
        call_id=call_id,
        channel=raw.channel,
        segment_id=stable_id(call_id, raw.channel, round(raw.ts_start, 3)),
        ts_start=raw.ts_start,
        ts_end=raw.ts_end,
        text_redacted=text,
        words=words,
        is_final=raw.is_final,
        avg_conf=raw.avg_conf,
        model_version=raw.model_version,
    )


def transcribe(asr: ASRBackend, chunk: AudioChunk, call_id: str) -> list[AsrSegment]:
    return [to_asr_segment(r, call_id) for r in asr.transcribe(chunk)]
