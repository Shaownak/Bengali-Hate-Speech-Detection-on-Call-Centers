"""Post-call pipeline (docs/01, Prompt 3): stereo WAV → events, scorecard and timelines.

    WAV → channel split → 16 kHz → VAD → ASR (+ normalise/redact, D-01)
        → audio steps (DSP shouting, SER, audio-tox, KWS, VAD overtalk)
        → text steps (lexicon, text classifier, register)
        → fusion (provisional) → LLM judge (window, grounded) → fusion (final) → LLM call scorecard

Every stage reuses the worker logic the streaming services run from P4. Stage latencies go to
``gale_postcall_stage_seconds`` and into the result.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from prometheus_client import Histogram

from gale_asr_worker.steps import transcribe, utterance_chunks
from gale_audio_worker.overtalk import overtalk_signals
from gale_audio_worker.shout import suppress_line_artefacts
from gale_fusion_engine.fusion import FusionEngine
from gale_fusion_engine.rules import parse_rules
from gale_llm_judge.client import JudgeUnavailableError
from gale_llm_judge.grounding import ground
from gale_llm_judge.windows import utterance_ids, windows_for_events
from gale_postcall.models import PipelineModels, build_models
from gale_text_worker.lexicon import LexiconMatcher
from gale_text_worker.steps import TextWorker
from galecore.audio import MODEL_SR, resample, split_channels
from galecore.config import Settings, get_settings, load_lexicon, load_rules, load_thresholds
from galecore.logging import get_logger
from galecore.models import FixtureOracle
from galecore.models.base import AudioChunk
from galecore.schemas import (
    AsrSegment,
    CallScorecard,
    Channel,
    FusedEvent,
    ProvisionalDecision,
    Signal,
)

log = get_logger(__name__)
STAGE = Histogram("gale_postcall_stage_seconds", "post-call pipeline stage latency", ["stage"])
Layout = Literal["stereo_agent_left", "stereo_agent_right", "mono"]


@dataclass
class PostCallResult:
    call_id: str
    source_file: str
    duration_s: float
    sample_rate_in: int
    layout: Layout
    segments: list[AsrSegment]
    signals: list[Signal]
    events: list[FusedEvent]
    scorecard: CallScorecard | None
    llm_status: Literal["ok", "offline"]
    timelines: dict[str, Any]
    stages_s: dict[str, float]
    models: dict[str, str]
    notes: list[str] = field(default_factory=list)
    audio_16k: dict[Channel, np.ndarray] = field(default_factory=dict, repr=False)


@contextmanager
def _stage(name: str, stages: dict[str, float]) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        stages[name] = round(stages.get(name, 0.0) + dt, 4)
        STAGE.labels(name).observe(dt)


def run_postcall(
    wav: Path,
    settings: Settings | None = None,
    *,
    call_id: str | None = None,
    layout: Layout = "stereo_agent_left",
    models: PipelineModels | None = None,
    queue: str | None = None,
) -> PostCallResult:
    s = settings or get_settings()
    call_id = call_id or str(uuid.uuid4())  # D-18: GAL-E mints its own UUID call ids
    oracle = FixtureOracle.for_audio(wav) if s.model_mode == "stub" else None
    pm = models or build_models(s, oracle)
    thresholds = load_thresholds(s.config_dir)
    engine = FusionEngine(parse_rules(load_rules(s.config_dir)), thresholds, mode="postcall")
    stages: dict[str, float] = {}
    notes: list[str] = []

    with _stage("load", stages):
        raw, sr = sf.read(wav, dtype="float32", always_2d=True)
        if raw.shape[1] == 1:
            layout = "mono"
        chans = split_channels(raw if raw.shape[1] > 1 else raw[:, 0], layout)
        audio = {ch: resample(x, int(sr), MODEL_SR) for ch, x in chans.items()}
        duration = len(raw) / sr
    if Channel.MIXED in audio:
        notes.append(
            "mono recording: no diarization yet, segments are not attributed and never alert (D-15)"
        )

    with _stage("vad", stages):
        regions = {ch: pm.vad.speech_regions(AudioChunk(call_id, ch, x)) for ch, x in audio.items()}

    segments: list[AsrSegment] = []
    with _stage("asr", stages):
        for ch, x in audio.items():
            for a, b in utterance_chunks(regions[ch], duration):
                i0, i1 = int(a * MODEL_SR), int(b * MODEL_SR)
                segments += transcribe(
                    pm.asr, AudioChunk(call_id, ch, x[i0:i1], offset_s=a), call_id
                )
    segments.sort(key=lambda g: (g.ts_start, g.channel.value))

    signals: list[Signal] = []
    with _stage("audio", stages):
        shout = {ch: pm.shout.detect(AudioChunk(call_id, ch, x)) for ch, x in audio.items()}
        if Channel.AGENT in shout and Channel.CUSTOMER in shout:
            shout[Channel.AGENT], shout[Channel.CUSTOMER] = suppress_line_artefacts(
                shout[Channel.AGENT], shout[Channel.CUSTOMER]
            )
        for ch, x in audio.items():
            chunk = AudioChunk(call_id, ch, x)
            signals += (
                shout[ch] + pm.ser.detect(chunk) + pm.audio_tox.detect(chunk) + pm.kws.detect(chunk)
            )
        if Channel.MIXED not in audio:
            signals += overtalk_signals(call_id, regions, audio, MODEL_SR)

    with _stage("text", stages):
        worker = TextWorker(
            LexiconMatcher(load_lexicon(s).entries), pm.text_clf, pm.register, thresholds
        )
        for seg in segments:
            signals += worker.process(seg)

    with _stage("fusion", stages):
        provisional = engine.provisional(call_id, signals)

    decisions: list[ProvisionalDecision] = []
    llm_signals: list[Signal] = []
    llm_status: Literal["ok", "offline"] = "ok"
    with _stage("llm_window", stages):
        for w in windows_for_events(call_id, segments, signals, provisional, queue):
            try:
                j = pm.llm.judge_window(w.input)
            except JudgeUnavailableError:
                llm_status = "offline"
                notes.append("LLM judge offline: events stay provisional")
                break
            decisions += j.provisional_decisions
            g = ground(
                w.input, j, utterance_ids(w.input.segments), getattr(pm.llm, "version", "llm")
            )
            llm_signals += g.signals

    with _stage("fusion", stages):
        judged = engine.apply_judgements(provisional, decisions, signals)
        events = engine.finalize(judged + engine.llm_events(call_id, llm_signals, judged))
    signals += llm_signals

    scorecard: CallScorecard | None = None
    with _stage("llm_call", stages):
        if llm_status == "ok":
            try:
                scorecard = pm.llm.judge_call(call_id, segments, events)
            except JudgeUnavailableError:
                llm_status = "offline"
                notes.append("LLM judge offline: no scorecard")

    timelines = build_timelines(call_id, pm, signals, duration)
    log.info(
        "postcall_done",
        call_id=call_id,
        duration_s=round(duration, 1),
        segments=len(segments),
        signals=len(signals),
        events=len(events),
        llm=llm_status,
        total_ms=round(sum(stages.values()) * 1000),
    )
    return PostCallResult(
        call_id=call_id,
        source_file=wav.name,
        duration_s=round(duration, 3),
        sample_rate_in=int(sr),
        layout=layout,
        segments=segments,
        signals=signals,
        events=events,
        scorecard=scorecard,
        llm_status=llm_status,
        timelines=timelines,
        stages_s=stages,
        models=pm.versions,
        notes=notes,
        audio_16k=audio,
    )


def build_timelines(
    call_id: str, pm: PipelineModels, signals: list[Signal], duration: float
) -> dict[str, Any]:
    """Per speaker: loudness (dBFS) and shout probability per 0.5 s, anger (SER) and a rolling
    30 s escalation curve every 5 s; overtalk markers."""
    out: dict[str, Any] = {
        "loudness": {},
        "shout_p": {},
        "anger_p": {},
        "escalation": {},
        "overtalk": [],
    }
    for ch in (Channel.AGENT, Channel.CUSTOMER):
        tl = pm.shout.timeline.get((call_id, ch), [])
        out["loudness"][ch.value] = [[t, round(db, 1)] for t, db, _, _ in tl]
        out["shout_p"][ch.value] = [[t, round(p, 3)] for t, _, _, p in tl]
        ser = [x for x in signals if x.channel == ch and x.source.value == "ser"]
        out["anger_p"][ch.value] = [[x.ts_start, x.score] for x in ser]
        heat = [(t, p) for t, _, _, p in tl] + [(x.ts_start, x.score) for x in ser]
        out["escalation"][ch.value] = [
            [float(t), round(max((p for u, p in heat if t - 30 <= u <= t), default=0.0), 3)]
            for t in np.arange(0, duration + 5, 5.0)
        ]
    out["overtalk"] = [
        {
            "speaker": x.channel.value,
            "t": x.ts_start,
            "overlap_s": x.evidence.features.get("overlap_s", 0.0),
        }
        for x in signals
        if x.source.value == "vad_overtalk"
    ]
    return out
