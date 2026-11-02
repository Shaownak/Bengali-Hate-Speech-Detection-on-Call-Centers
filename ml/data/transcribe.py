"""Batch auto-transcription → redacted seed transcripts for T1 correction (docs/02 §2.4–2.6).

Resumable: one output file per (call, channel) under ``<out>/<call_id>/<channel>.jsonl``, written
atomically; finished files are skipped on re-run. Uses the same ASR step as the services
(VAD → utterance chunks → ASR → normalise + redact), so raw text never reaches disk (D-01).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from gale_asr_worker.steps import transcribe as asr_step
from gale_asr_worker.steps import utterance_chunks
from gale_postcall.models import build_models
from galecore.config import Settings
from galecore.models import FixtureOracle
from galecore.models.base import AudioChunk
from galecore.schemas import AsrSegment, Channel
from ml.data.manifest import ChannelRow

OracleLookup = Callable[[ChannelRow], FixtureOracle | None]


@dataclass
class TranscribeStats:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    segments: int = 0


def output_path(out: Path, row: ChannelRow) -> Path:
    return out / row.call_id / f"{row.channel}.jsonl"


def transcribe_rows(
    rows: list[ChannelRow],
    out: Path,
    settings: Settings,
    oracle_for: OracleLookup | None = None,
    limit: int | None = None,
) -> TranscribeStats:
    stats = TranscribeStats()
    models_cache: dict[str | None, object] = {}
    for row in rows:
        if row.dropped or not row.audio_16k_path:
            continue
        dest = output_path(out, row)
        if dest.exists():
            stats.skipped += 1
            continue
        if limit is not None and stats.done >= limit:
            break
        oracle = oracle_for(row) if oracle_for and settings.model_mode == "stub" else None
        key = oracle.call_id if oracle else None
        if key not in models_cache:
            models_cache[key] = build_models(settings, oracle)
        pm = models_cache[key]
        try:
            x, sr = sf.read(row.audio_16k_path, dtype="float32")
            ch = Channel(row.channel)
            regions = pm.vad.speech_regions(AudioChunk(row.call_id, ch, x, sample_rate=sr))  # type: ignore[attr-defined]
            segs: list[AsrSegment] = []
            for a, b in utterance_chunks(regions, len(x) / sr):
                chunk = AudioChunk(
                    row.call_id, ch, x[int(a * sr) : int(b * sr)], offset_s=a, sample_rate=sr
                )
                segs += asr_step(pm.asr, chunk, row.call_id)  # type: ignore[attr-defined]
        except (OSError, RuntimeError, ValueError):
            stats.failed += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_text("".join(s.model_dump_json() + "\n" for s in segs), encoding="utf-8")
        tmp.replace(dest)
        stats.done += 1
        stats.segments += len(segs)
    return stats


def load_transcript(out: Path, row: ChannelRow) -> list[AsrSegment]:
    p = output_path(out, row)
    if not p.exists():
        return []
    return [
        AsrSegment.model_validate(json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line
    ]
