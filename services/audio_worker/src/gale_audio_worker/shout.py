"""Shouting detector (docs/03 §3): DSP features as z-scores vs the speaker's rolling baseline.

Baseline: the first 20 s of that speaker's speech, then an EMA over non-shouting speech
windows. Scorer: a LightGBM model from ``$GALE_MODEL_DIR/shout/lgbm.txt`` when one exists
(placeholder until P3 trains it), else a transparent rule on loudness and pitch z-scores.
Guards: no score when VAD says < 40 % speech in the window; ``suppress_line_artefacts`` drops
windows where both channels spike at once.

Real code in both model modes: stub mode just never has a LightGBM file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gale_audio_worker.dsp import FEATURES, HOP_S, Window, windows
from galecore.models.base import AudioChunk, ShoutDetector, VADModel
from galecore.models.stubs import stable_id
from galecore.schemas import Category, Channel, Evidence, Signal, SignalSource

VERSION_RULE = "dsp-shout-rule-0.1"
VERSION_LGBM = "dsp-shout-lgbm"
MIN_SPEECH_RATIO = 0.4
BASELINE_SPEECH_S = 20.0
MIN_BASELINE_WINDOWS = 6
EMA_ALPHA = 0.02
SD_FLOOR = {"rms_db": 2.0, "peak_db": 2.0, "f0_med_st": 1.0, "f0_p90_st": 1.0, "hf_ratio_db": 2.0}
EMIT_SCORE = 0.5


@dataclass
class Baseline:
    windows: list[dict[str, float]] = field(default_factory=list)
    speech_s: float = 0.0
    mean: dict[str, float] | None = None
    sd: dict[str, float] | None = None

    def ready(self) -> bool:
        return self.mean is not None

    def add_initial(self, w: Window) -> None:
        self.windows.append(w.feats)
        self.speech_s += HOP_S * w.speech_ratio
        if self.speech_s >= BASELINE_SPEECH_S:
            self._freeze()

    def _freeze(self) -> None:
        arr = {k: np.array([f[k] for f in self.windows]) for k in FEATURES}
        self.mean = {k: float(np.median(v)) for k, v in arr.items()}
        self.sd = {
            k: max(SD_FLOOR.get(k, 1e-3), float(1.4826 * np.median(np.abs(v - np.median(v)))))
            for k, v in arr.items()
        }

    def finish_early(self) -> None:
        """Short calls: freeze on whatever speech there is (if enough windows)."""
        if self.mean is None and len(self.windows) >= MIN_BASELINE_WINDOWS:
            self._freeze()

    def z(self, feats: dict[str, float]) -> dict[str, float]:
        if self.mean is None or self.sd is None:
            raise RuntimeError("baseline not ready")
        return {k: (feats[k] - self.mean[k]) / self.sd[k] for k in FEATURES}

    def update(self, feats: dict[str, float]) -> None:
        if self.mean is None:
            raise RuntimeError("baseline not ready")
        for k in FEATURES:
            self.mean[k] = (1 - EMA_ALPHA) * self.mean[k] + EMA_ALPHA * feats[k]


def rule_score(z: dict[str, float]) -> float:
    """Loud *and* raised pitch → shouting; loudness alone (handset gain) scores lower."""
    logit = 1.1 * (z["rms_db"] - 3.5) + 0.5 * min(4.0, max(0.0, z["f0_med_st"] - 1.0))
    return 1.0 / (1.0 + math.exp(-logit))


class DSPShoutDetector(ShoutDetector):
    def __init__(self, vad: VADModel, model_dir: Path | None = None) -> None:
        self.vad = vad
        self._baselines: dict[tuple[str, Channel], Baseline] = {}
        self.timeline: dict[tuple[str, Channel], list[tuple[float, float, float, float]]] = {}
        self._lgbm: Any = None
        path = model_dir / "shout" / "lgbm.txt" if model_dir else None
        if path is not None and path.is_file():  # pragma: no cover - needs lightgbm + a model
            import lightgbm

            self._lgbm = lightgbm.Booster(model_file=str(path))

    @property
    def version(self) -> str:
        return VERSION_LGBM if self._lgbm is not None else VERSION_RULE

    def _score(self, z: dict[str, float]) -> float:
        if self._lgbm is not None:  # pragma: no cover
            return float(self._lgbm.predict(np.array([[z[k] for k in FEATURES]]))[0])
        return rule_score(z)

    def detect(self, chunk: AudioChunk) -> list[Signal]:
        """Whole chunk (post-call): baseline from the speaker's first speech, then score."""
        key = (chunk.call_id, chunk.channel)
        base = self._baselines.setdefault(key, Baseline())
        regions = self.vad.speech_regions(chunk)
        wins = windows(chunk.samples, chunk.sample_rate, regions, chunk.offset_s)
        if not base.ready():
            for w in wins:
                if w.speech_ratio >= MIN_SPEECH_RATIO and not base.ready():
                    base.add_initial(w)
            base.finish_early()
        return [
            s for w in wins if (s := self.score_window(chunk.call_id, chunk.channel, w)) is not None
        ]

    def score_window(self, call_id: str, channel: Channel, w: Window) -> Signal | None:
        """Score one window against the speaker's baseline (streaming: windows arrive in order and
        the first 20 s of speech build the baseline before anything is scored)."""
        key = (call_id, channel)
        base = self._baselines.setdefault(key, Baseline())
        if not base.ready() and w.speech_ratio >= MIN_SPEECH_RATIO:
            base.add_initial(w)
        score = 0.0
        z: dict[str, float] = {}
        if base.ready() and w.speech_ratio >= MIN_SPEECH_RATIO:
            z = base.z(w.feats)
            score = self._score(z)
            if score < 0.3:
                base.update(w.feats)
        self.timeline.setdefault(key, []).append(
            (w.t_start, w.feats["rms_db"], z.get("rms_db", 0.0), score)
        )
        if score < EMIT_SCORE:
            return None
        return Signal(
            call_id=call_id, channel=channel,
            signal_id=stable_id(call_id, channel, "dsp_shout", w.t_start),
            source=SignalSource.DSP_SHOUT, category=Category.SHOUTING, score=round(score, 4),
            ts_start=w.t_start, ts_end=w.t_end,
            evidence=Evidence(features={
                "loudness_z": round(z["rms_db"], 3), "f0_z": round(z["f0_med_st"], 3),
                "hf_z": round(z["hf_ratio_db"], 3), "rms_db": round(w.feats["rms_db"], 2),
                "speech_ratio": w.speech_ratio,
            }),
            model_version=self.version,
        )  # fmt: skip


def suppress_line_artefacts(
    a: list[Signal], b: list[Signal], tol_s: float = 0.25
) -> tuple[list[Signal], list[Signal]]:
    """Drop shout windows that spike on both channels at the same time (line artefact)."""

    def clash(s: Signal, others: list[Signal]) -> bool:
        return any(abs(s.ts_start - o.ts_start) <= tol_s for o in others)

    return [s for s in a if not clash(s, b)], [s for s in b if not clash(s, a)]
