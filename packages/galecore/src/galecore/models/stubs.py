"""Deterministic stub models (``GALE_MODEL_MODE=stub``).

No weights, no downloads, same output for the same input. Good enough for the whole system
to run end-to-end on a laptop and for integration tests. Placeholder slang appears as
``<GALI_n>`` tags (CLAUDE.md hard rule 2).
"""

from __future__ import annotations

import re
import uuid

import numpy as np

from galecore.audio import rms_dbfs
from galecore.bangla import normalize, tokenize
from galecore.models.base import (
    ASRBackend,
    AudioChunk,
    AudioToxModel,
    KWSModel,
    LLMJudge,
    RawSegment,
    RawWord,
    RegisterDetector,
    SERModel,
    ShoutDetector,
    TextClassifier,
    TextClfOutput,
    TextContext,
    WindowInput,
)
from galecore.schemas import (
    AgentScore,
    AsrSegment,
    CallScorecard,
    Category,
    Channel,
    CustomerScore,
    Escalation,
    Evidence,
    ProvisionalDecision,
    Signal,
    SignalSource,
    WindowJudgement,
)

STUB_VERSION = "stub-0.1"
_NS = uuid.UUID("6a1e0a1e-0000-4000-8000-000000000a1e")
_GALI = re.compile(r"<GALI_\d+>")
SPEECH_DBFS = -45.0
SHOUT_DB_ABOVE_BASELINE = 10.0


def stable_id(*parts: object) -> str:
    return str(uuid.uuid5(_NS, "|".join(map(str, parts))))


def _windows(chunk: AudioChunk, win_s: float, hop_s: float) -> list[tuple[float, float, float]]:
    """(ts_start, ts_end, dBFS) over sliding windows."""
    sr = chunk.sample_rate
    win, hop = int(win_s * sr), int(hop_s * sr)
    x = chunk.samples
    out = []
    for start in range(0, max(1, len(x) - win + 1), hop):
        seg = x[start : start + win]
        out.append(
            (chunk.offset_s + start / sr, chunk.offset_s + (start + len(seg)) / sr, rms_dbfs(seg))
        )
    return out


def _loudness_signals(
    chunk: AudioChunk, source: SignalSource, category: Category, score: float
) -> list[Signal]:
    wins = _windows(chunk, 1.5, 0.5)
    speech = [db for _, _, db in wins if db > SPEECH_DBFS]
    if not speech:
        return []
    baseline = float(np.median(speech))
    signals = []
    for ts0, ts1, db in wins:
        if db >= baseline + SHOUT_DB_ABOVE_BASELINE:
            signals.append(
                Signal(
                    call_id=chunk.call_id,
                    channel=chunk.channel,
                    signal_id=stable_id(chunk.call_id, chunk.channel, source, ts0),
                    source=source,
                    category=category,
                    score=score,
                    ts_start=round(ts0, 3),
                    ts_end=round(ts1, 3),
                    evidence=Evidence(features={"loudness_z_db": round(db - baseline, 2)}),
                    model_version=STUB_VERSION,
                )
            )
    return signals


class StubASR(ASRBackend):
    """One segment per voiced region (energy-based); text is a numbered placeholder."""

    name = "stub"

    def transcribe(self, chunk: AudioChunk) -> list[RawSegment]:
        frames = _windows(chunk, 0.2, 0.2)
        regions: list[tuple[float, float]] = []
        for ts0, ts1, db in frames:
            if db <= SPEECH_DBFS:
                continue
            if regions and ts0 - regions[-1][1] < 0.6:
                regions[-1] = (regions[-1][0], ts1)
            else:
                regions.append((ts0, ts1))
        segments = []
        for i, (ts0, ts1) in enumerate(regions):
            text = f"স্টাব বাক্য {i + 1}"
            words = [
                RawWord(w, ts0 + j * (ts1 - ts0) / 3, ts0 + (j + 1) * (ts1 - ts0) / 3, 0.9)
                for j, w in enumerate(text.split())
            ]
            segments.append(
                RawSegment(
                    channel=chunk.channel,
                    ts_start=round(ts0, 3),
                    ts_end=round(ts1, 3),
                    text=text,
                    words=words,
                    is_final=True,
                    avg_conf=0.9,
                    model_version=STUB_VERSION,
                )
            )
        return segments


class StubShout(ShoutDetector):
    def detect(self, chunk: AudioChunk) -> list[Signal]:
        return _loudness_signals(chunk, SignalSource.DSP_SHOUT, Category.SHOUTING, 0.9)


class StubSER(SERModel):
    def detect(self, chunk: AudioChunk) -> list[Signal]:
        return _loudness_signals(chunk, SignalSource.SER, Category.HARSH_TONE, 0.7)


class StubAudioTox(AudioToxModel):
    def detect(self, chunk: AudioChunk) -> list[Signal]:
        return []


class StubKWS(KWSModel):
    def detect(self, chunk: AudioChunk) -> list[Signal]:
        return []


class StubTextClassifier(TextClassifier):
    """Placeholder slang ⇒ high slang score; তুই/তোর ⇒ moderate disrespect score."""

    def classify(self, ctx: TextContext) -> TextClfOutput:
        text = normalize(ctx.current)
        scores = dict.fromkeys(Category, 0.02)
        severity = 0
        if _GALI.search(text):
            scores[Category.SLANG_PROFANITY] = 0.95
            scores[Category.INSULT] = 0.4
            severity = 2
        toks = {t.text for t in tokenize(text)}
        if toks & {"তুই", "তোর", "তোকে"}:
            scores[Category.DISRESPECT_REGISTER] = 0.6
            severity = max(severity, 1)
        return TextClfOutput(scores=scores, severity=severity, model_version=STUB_VERSION)


_APNI = {"আপনি", "আপনার", "আপনাকে"}
_TUMI = {"তুমি", "তোমার", "তোমাকে"}
_TUI = {"তুই", "তোর", "তোকে"}
_LEVEL = {"apni": 0, "tumi": 1, "tui": 2}


class StubRegister(RegisterDetector):
    """Pronoun-only register tracking. Emits when an agent moves down a level, or any
    speaker drops to তুই after using আপনি (the real rules add verb morphology, docs/03 §6.1)."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, Channel], str] = {}

    def observe(self, segment: AsrSegment) -> list[Signal]:
        toks = {t.text for t in tokenize(normalize(segment.text_redacted))}
        level = (
            "tui" if toks & _TUI else "tumi" if toks & _TUMI else "apni" if toks & _APNI else None
        )
        if level is None:
            return []
        key = (segment.call_id, segment.channel)
        prev = self._state.get(key)
        self._state[key] = level
        if prev is None or _LEVEL[level] <= _LEVEL[prev]:
            return []
        if segment.channel != Channel.AGENT and level != "tui":
            return []
        return [
            Signal(
                call_id=segment.call_id,
                channel=segment.channel,
                signal_id=stable_id(segment.segment_id, "register"),
                source=SignalSource.REGISTER,
                category=Category.DISRESPECT_REGISTER,
                score=0.9,
                ts_start=segment.ts_start,
                ts_end=segment.ts_end,
                evidence=Evidence(
                    segment_id=segment.segment_id,
                    features={"from_level": _LEVEL[prev], "to_level": _LEVEL[level]},
                ),
                model_version=STUB_VERSION,
            )
        ]


class StubLLMJudge(LLMJudge):
    """Confirms every provisional event, adds none; neutral scorecard."""

    def judge_window(self, window: WindowInput) -> WindowJudgement:
        return WindowJudgement(
            events=[],
            provisional_decisions=[
                ProvisionalDecision(event_id=e.event_id, decision="confirm")
                for e in window.provisional
            ],
        )

    def judge_call(self, call_id: str, segments: list[AsrSegment]) -> CallScorecard:
        return CallScorecard(
            agent=AgentScore(
                professionalism=4,
                empathy=3,
                de_escalation=3,
                respectful_register_maintained=True,
                violations=[],
            ),
            customer=CustomerScore(hostility=1, abuse_toward_agent=False, main_issue_bn="স্টাব"),
            escalation=Escalation(
                escalated=False, who_escalated_first="none", turning_point_utterance_id=None
            ),
            summary_bn="স্টাব সারাংশ",
            summary_en="stub summary",
            coaching_points_bn=[],
        )
