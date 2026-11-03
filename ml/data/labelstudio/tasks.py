"""Build Label Studio tasks with pre-annotations from the pipeline (docs/02 §3.1).

- T1 (per channel): the redacted ASR seed segments as regions with their text.
- T2 (per call): post-call pipeline events as regions on the right channel, with severity.
- T3 (per call): no pre-annotation (ratings are a human judgement).
``double`` marks tasks for double annotation (20 % by default, deterministic per call id).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from galecore.labelstudio import event_regions
from galecore.schemas import AsrSegment, FusedEvent
from ml.data.manifest import ChannelRow

AudioUrl = Callable[[str], str]


def local_files_url(prefix: str = "/data/local-files/?d=") -> AudioUrl:
    """Label Studio local-files storage URL for a path under its document root."""
    return lambda path: prefix + path.replace("\\", "/")


def double_annotate(call_id: str, fraction: float = 0.2) -> bool:
    return int(hashlib.sha256(call_id.encode()).hexdigest()[:8], 16) / 2**32 < fraction


def t1_task(
    row: ChannelRow, segments: list[AsrSegment], url: AudioUrl, model_version: str
) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    for i, s in enumerate(segments):
        rid = f"s{i}"
        value = {"start": s.ts_start, "end": s.ts_end}
        result.append(
            {
                "id": rid,
                "from_name": "segment",
                "to_name": "audio",
                "type": "labels",
                "value": {**value, "labels": ["speech"]},
            }
        )
        result.append(
            {
                "id": rid,
                "from_name": "transcription",
                "to_name": "audio",
                "type": "textarea",
                "value": {**value, "text": [s.text_redacted]},
            }
        )
    return {
        "data": {
            "audio": url(row.audio_16k_path or ""),
            "call_id": row.call_id,
            "channel": row.channel,
            "seed_text": "\n".join(s.text_redacted for s in segments),
        },
        "predictions": [{"model_version": model_version, "result": result}] if result else [],
        "meta": {"double": double_annotate(row.call_id)},
    }


def t2_task(
    call_id: str, stereo_audio: str, events: list[FusedEvent], url: AudioUrl, model_version: str
) -> dict[str, Any]:
    result = event_regions(events)
    return {
        "data": {"audio": url(stereo_audio), "call_id": call_id},
        "predictions": [{"model_version": model_version, "result": result}] if result else [],
        "meta": {"double": double_annotate(call_id)},
    }


def t3_task(call_id: str, stereo_audio: str, url: AudioUrl) -> dict[str, Any]:
    return {
        "data": {"audio": url(stereo_audio), "call_id": call_id},
        "meta": {"double": double_annotate(call_id)},
    }
