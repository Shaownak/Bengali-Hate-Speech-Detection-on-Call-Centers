"""Real SER: emotion2vec+ via FunASR (``GALE_MODEL_MODE=real``).

The pretrained 9-class head is mapped onto our harsh-tone signal (angry + disgusted) until the
fine-tuned head (docs/03 §4) exists; then ``ml/ser/label_map.yaml`` replaces the mapping.
3 s windows with 1 s hop, per channel.
Weights only from ``$GALE_MODEL_DIR/ser/emotion2vec_plus_large``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from galecore.models import RealModelsUnavailableError
from galecore.models.base import AudioChunk, SERModel
from galecore.models.stubs import stable_id
from galecore.schemas import Category, Evidence, Signal, SignalSource

WIN_S, HOP_S = 3.0, 1.0
HARSH_LABELS = ("angry", "disgusted")
EMIT = 0.3


class Emotion2vecSER(SERModel):  # pragma: no cover - needs funasr + weights
    def __init__(self, model_dir: Path) -> None:
        from funasr import AutoModel

        path = model_dir / "ser" / "emotion2vec_plus_large"
        if not path.exists():
            raise RealModelsUnavailableError(
                f"emotion2vec+: {path} is missing; run scripts/download_models.py"
            )
        self.model: Any = AutoModel(model=str(path), disable_update=True)
        self.version = f"emotion2vec_plus_large:{path.name}"

    def detect(self, chunk: AudioChunk) -> list[Signal]:
        sr = chunk.sample_rate
        win, hop = int(WIN_S * sr), int(HOP_S * sr)
        out = []
        for i in range(0, max(1, len(chunk.samples) - win + 1), hop):
            seg = chunk.samples[i : i + win]
            res = self.model.generate(seg, granularity="utterance", extract_embedding=False)[0]
            scores = {
                lab.split("/")[-1].lower(): float(s)
                for lab, s in zip(res["labels"], res["scores"], strict=False)
            }
            p = sum(scores.get(lab, 0.0) for lab in HARSH_LABELS)
            if p < EMIT:
                continue
            t0 = chunk.offset_s + i / sr
            out.append(
                Signal(
                    call_id=chunk.call_id,
                    channel=chunk.channel,
                    signal_id=stable_id(chunk.call_id, chunk.channel, "ser", round(t0, 2)),
                    source=SignalSource.SER,
                    category=Category.HARSH_TONE,
                    score=round(min(1.0, p), 4),
                    ts_start=round(t0, 3),
                    ts_end=round(t0 + len(seg) / sr, 3),
                    evidence=Evidence(features={k: round(v, 4) for k, v in scores.items()}),
                    model_version=self.version,
                )
            )
        return out
