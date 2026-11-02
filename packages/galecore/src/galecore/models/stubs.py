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
    VADModel,
    WindowInput,
)
from galecore.models.oracle import FixtureOracle, OracleEvent
from galecore.schemas import (
    AgentScore,
    AsrSegment,
    CallScorecard,
    Category,
    Channel,
    CustomerScore,
    Escalation,
    EventStatus,
    Evidence,
    FusedEvent,
    ProvisionalDecision,
    Signal,
    SignalSource,
    Speaker,
    WindowJudgement,
)

STUB_VERSION = "stub-0.1"
_NS = uuid.UUID("6a1e0a1e-0000-4000-8000-000000000a1e")
_GALI = re.compile(r"<GALI_\d+(?:_V)?>")
_TAG = re.compile(r"<[A-Z][A-Z0-9_]*>")
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
    """Deterministic ASR.

    With a fixture oracle: the script's own words and timings for words whose midpoint falls in
    the chunk (a perfect recogniser). Without one: one placeholder segment per voiced region.
    """

    name = "stub"

    def __init__(self, oracle: FixtureOracle | None = None) -> None:
        self.oracle = oracle

    def transcribe(self, chunk: AudioChunk) -> list[RawSegment]:
        if self.oracle is not None:
            return self._from_script(chunk, self.oracle)
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

    @staticmethod
    def _from_script(chunk: AudioChunk, oracle: FixtureOracle) -> list[RawSegment]:
        t0, t1 = chunk.offset_s, chunk.offset_s + chunk.duration_s
        out = []
        for turn in oracle.turns:
            if turn.channel != chunk.channel or turn.end < t0 or turn.start > t1:
                continue
            words = [w for w in turn.words if t0 <= (w.start + w.end) / 2 < t1]
            if not words:
                continue
            out.append(
                RawSegment(
                    channel=chunk.channel,
                    ts_start=words[0].start,
                    ts_end=words[-1].end,
                    text=" ".join(w.token for w in words),
                    words=[RawWord(w.token, w.start, w.end, 0.95) for w in words],
                    is_final=True,
                    avg_conf=0.95,
                    model_version=f"{STUB_VERSION}+script",
                )
            )
        return out


class StubVAD(VADModel):
    """Energy VAD on 30 ms frames. The threshold adapts per channel (noise floor + 12 dB and
    peak level - 22 dB), so far-end crosstalk at about -32 dB stays below it."""

    def __init__(self, frame_s: float = 0.03, min_gap_s: float = 0.3, min_len_s: float = 0.15):
        self.frame_s, self.min_gap_s, self.min_len_s = frame_s, min_gap_s, min_len_s

    def speech_regions(self, chunk: AudioChunk) -> list[tuple[float, float]]:
        n = int(self.frame_s * chunk.sample_rate)
        x = chunk.samples[: len(chunk.samples) // n * n].reshape(-1, n)
        if not len(x):
            return []
        db = 10 * np.log10(np.mean(x.astype(np.float64) ** 2, axis=1) + 1e-12)
        thr = max(float(np.percentile(db, 10)) + 12, float(np.percentile(db, 95)) - 22, -60.0)
        regions: list[tuple[float, float]] = []
        for k in np.flatnonzero(db > thr):
            a, b = k * self.frame_s, (k + 1) * self.frame_s
            if regions and a - regions[-1][1] <= self.min_gap_s:
                regions[-1] = (regions[-1][0], b)
            else:
                regions.append((a, b))
        off = chunk.offset_s
        return [
            (round(off + a, 3), round(off + b, 3)) for a, b in regions if b - a >= self.min_len_s
        ]


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
    """Placeholder tags score high on their fake-lexicon category; তুই/তোর gives a moderate
    disrespect score; with a fixture oracle, annotated turn-level categories score 0.9."""

    def __init__(
        self,
        tag_categories: dict[str, tuple[Category, int]] | None = None,
        annotations: dict[str, list[tuple[Category, int]]] | None = None,
    ) -> None:
        self.tags = tag_categories or {}
        self.annotations = {normalize(k): v for k, v in (annotations or {}).items()}

    def classify(self, ctx: TextContext) -> TextClfOutput:
        text = normalize(ctx.current)
        scores = dict.fromkeys(Category, 0.02)
        severity = 0
        for tag in _TAG.findall(text):
            if tag in self.tags:
                cat, sev = self.tags[tag]
            elif _GALI.fullmatch(tag):
                cat, sev = Category.SLANG_PROFANITY, 2
            else:
                continue
            scores[cat] = max(scores[cat], 0.9)
            severity = max(severity, sev)
        toks = {t.text for t in tokenize(text)}
        if toks & {"তুই", "তোর", "তোকে"}:
            scores[Category.DISRESPECT_REGISTER] = max(scores[Category.DISRESPECT_REGISTER], 0.6)
            severity = max(severity, 1)
        for key, cats in self.annotations.items():
            if len(text) >= 8 and (text in key or key in text):
                for cat, sev in cats:
                    scores[cat] = max(scores[cat], 0.9)
                    severity = max(severity, sev)
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

    def observe(self, segment: AsrSegment, negative_context: bool = False) -> list[Signal]:
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
    """Confirms provisional events and adds none. With a fixture oracle it dismisses events that
    the ground truth marks as not alerting (reported speech, self-directed), as a well-tuned
    judge would. The call scorecard is derived from the fused events."""

    def __init__(self, oracle: FixtureOracle | None = None) -> None:
        self.oracle = oracle

    def _context(self, e: FusedEvent) -> OracleEvent | None:
        if self.oracle is None:
            return None
        mid = (e.ts_start + e.ts_end) / 2
        for g in self.oracle.events:
            if (
                not g.alert
                and g.speaker == e.speaker
                and g.category == e.category
                and g.start - 2.0 <= mid <= g.end + 2.0
            ):
                return g
        return None

    def judge_window(self, window: WindowInput) -> WindowJudgement:
        decisions = []
        for e in window.provisional:
            g = self._context(e)
            if g is None:
                decisions.append(ProvisionalDecision(event_id=e.event_id, decision="confirm"))
            else:
                flags = ", ".join(f.value for f in g.context_flags) or "context"
                decisions.append(
                    ProvisionalDecision(
                        event_id=e.event_id, decision="dismiss", reason_bn=f"প্রসঙ্গ: {flags}"
                    )
                )
        return WindowJudgement(events=[], provisional_decisions=decisions)

    def judge_call(
        self, call_id: str, segments: list[AsrSegment], events: list[FusedEvent]
    ) -> CallScorecard:
        live = [e for e in events if e.status != EventStatus.DISMISSED_BY_LLM]
        agent = [e for e in live if e.speaker == Speaker.AGENT]
        cust = [e for e in live if e.speaker == Speaker.CUSTOMER]
        first = min(live, key=lambda e: e.ts_start, default=None)
        abusive = {
            Category.SLANG_PROFANITY,
            Category.INSULT,
            Category.THREAT,
            Category.SEXUAL_HARASSMENT,
            Category.HATE_IDENTITY,
        }
        agent_cats = sorted({e.category.value for e in agent})
        return CallScorecard(
            agent=AgentScore(
                professionalism=max(1, 5 - sum(e.severity for e in agent)),
                empathy=4 if not agent else 2,
                de_escalation=(4 if cust else 3) if not agent else 2,
                respectful_register_maintained=not any(
                    e.category == Category.DISRESPECT_REGISTER for e in agent
                ),
                violations=agent_cats,
            ),
            customer=CustomerScore(
                hostility=min(5, 1 + sum(e.severity for e in cust)),
                abuse_toward_agent=any(e.category in abusive for e in cust),
                main_issue_bn="স্টাব: গ্রাহকের অভিযোগ",
            ),
            escalation=Escalation(
                escalated=bool(live),
                who_escalated_first=first.speaker.value if first else "none",
                turning_point_utterance_id=None,
            ),
            summary_bn=f"স্টাব সারাংশ: এজেন্টের {len(agent)}টি ও গ্রাহকের {len(cust)}টি ঘটনা।",
            summary_en=f"Stub summary: {len(agent)} agent and {len(cust)} customer events.",
            coaching_points_bn=[f"{c}: আচরণ উন্নত করুন" for c in agent_cats][:5],
        )
