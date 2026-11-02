"""Channel split, native + 16 kHz copies, quality filters (docs/02 §2.2–2.3) → manifest rows.

Quality filters drop a channel when > 80 % is silence, > 5 % of samples clip, or it is < 20 s long.
Dropped channels stay in the manifest with ``dropped`` set, so counts are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from galecore.audio import MODEL_SR, resample, split_channels
from galecore.models.base import AudioChunk
from galecore.models.stubs import StubVAD
from galecore.schemas import Channel
from ml.data.ingest import Recording
from ml.data.manifest import ChannelRow


@dataclass(frozen=True)
class QualityRules:
    max_silence: float = 0.8
    max_clipping: float = 0.05
    min_duration_s: float = 20.0
    clip_level: float = 0.99


def quality(x: np.ndarray, sr: int) -> dict[str, float]:
    regions = StubVAD().speech_regions(
        AudioChunk("q", Channel.MIXED, x.astype(np.float32), sample_rate=sr)
    )
    dur = len(x) / sr
    speech = sum(b - a for a, b in regions)
    return {
        "duration_s": round(dur, 3),
        "silence_ratio": round(1 - speech / dur, 4) if dur else 1.0,
        "clipping_ratio": round(float(np.mean(np.abs(x) >= QualityRules.clip_level)), 5)
        if len(x)
        else 0.0,
        "rms_dbfs": round(float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12)), 2),
    }


def verdict(q: dict[str, float], rules: QualityRules) -> str | None:
    if q["duration_s"] < rules.min_duration_s:
        return "too_short"
    if q["clipping_ratio"] > rules.max_clipping:
        return "clipping"
    if q["silence_ratio"] > rules.max_silence:
        return "silence"
    return None


def process(
    rec: Recording, out_dir: Path, rules: QualityRules | None = None, synthetic: bool = False
) -> list[ChannelRow]:
    """Split one recording into channels, write the copies, return manifest rows."""
    rules = rules or QualityRules()
    audio, sr = sf.read(rec.local_path, dtype="float32", always_2d=True)
    info = sf.info(str(rec.local_path))
    layout = rec.meta.get("layout") or ("mono" if audio.shape[1] == 1 else "stereo_agent_left")
    chans = split_channels(audio if audio.shape[1] > 1 else audio[:, 0], layout)
    rows = []
    for ch, x in chans.items():
        q = quality(x, int(sr))
        reason = verdict(q, rules)
        base = out_dir / rec.call_id
        native = base / f"{ch.value}_{int(sr)}.wav"
        model = base / f"{ch.value}_16k.wav"
        if reason is None:
            base.mkdir(parents=True, exist_ok=True)
            sf.write(native, x, int(sr), subtype="PCM_16")
            sf.write(model, resample(x, int(sr), MODEL_SR), MODEL_SR, subtype="PCM_16")
        rows.append(
            ChannelRow(
                call_id=rec.call_id,
                channel=ch.value,
                source=rec.source,
                source_ref=rec.source_ref,  # type: ignore[arg-type]
                audio_path=str(native) if reason is None else None,
                audio_16k_path=str(model) if reason is None else None,
                duration=q["duration_s"],
                sample_rate=int(sr),
                codec=info.subtype.lower(),
                layout=layout,
                queue=rec.meta.get("queue"),
                date=rec.meta.get("date"),
                direction=rec.meta.get("direction"),
                agent_id=rec.meta.get("agent_id"),
                customer_hash=rec.meta.get("customer_hash"),
                dialect=rec.meta.get("dialect"),
                quality=q,
                dropped=reason,
                synthetic=synthetic,
            )
        )
    kept = {r.channel for r in rows if r.dropped is None}
    if {"agent", "customer"} <= kept:  # stereo review copy for T2/T3 (agent left, customer right)
        stereo = np.stack(
            [
                resample(chans[Channel.AGENT], int(sr), MODEL_SR),
                resample(chans[Channel.CUSTOMER], int(sr), MODEL_SR),
            ],
            axis=1,
        )
        sf.write(stereo_review_path(out_dir, rec.call_id), stereo, MODEL_SR, subtype="PCM_16")
    return rows


def stereo_review_path(out_dir: Path, call_id: str) -> Path:
    return out_dir / call_id / "stereo_16k.wav"
