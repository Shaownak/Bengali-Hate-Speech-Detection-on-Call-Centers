"""Evaluation sets (docs/06 §1) in one shape, whatever their source.

- ``fixtures``: the synthetic calls in ``tests/fixtures/calls`` (CI). Gold events from
  ``expected_events.json``, gold transcripts and word timings from ``script.json``; slice metadata
  is assigned deterministically so slicing code is exercised.
- manifest splits (``dev``, ``golden``, ``negatives``, …): a data-engine manifest plus the converted
  annotations (``ml.data convert`` → ``fusion.jsonl`` / ``asr.jsonl``, optional ``t3.jsonl``).
  ``golden`` is only loaded when ``golden.lock`` still matches (CLAUDE.md rule 3: evaluation only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.eval.metrics import Span

SLICE_KEYS = (
    "dialect",
    "agent_gender",
    "customer_gender",
    "queue",
    "codec",
    "duration",
    "time_of_day",
)
NO_ALERT_FLAGS = {"reported_speech", "self_directed", "banter", "quoted_policy"}


@dataclass
class GoldSegment:
    channel: str
    start: float
    end: float
    text: str
    words: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class EvalCall:
    call_id: str
    audio: Path  # stereo, agent left
    events: list[Span]
    segments: list[GoldSegment] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # slice keys
    t3: dict[str, Any] | None = None  # QA-lead call ratings when annotated
    oracle_dir: Path | None = None  # synthetic fixture dir (stub models key off it)


@dataclass
class EvalSet:
    name: str
    calls: list[EvalCall]

    @property
    def gold(self) -> list[Span]:
        return [e for c in self.calls for e in c.events]


def duration_bucket(seconds: float) -> str:
    return (
        "<2m"
        if seconds < 120
        else "2-5m"
        if seconds < 300
        else "5-10m"
        if seconds < 600
        else ">10m"
    )


def time_of_day(iso: str | None) -> str:
    if not iso or len(iso) < 13:
        return "unknown"
    h = int(iso[11:13])
    return "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"


def load_fixtures(calls_dir: Path) -> EvalSet:
    from tools.fixtures_gen.__main__ import ensure_wav

    calls = []
    dialects, queues, genders = (
        ["dhaka", "chattogram", "sylhet"],
        ["billing", "network"],
        ["female", "male"],
    )
    for i, d in enumerate(sorted(p for p in calls_dir.iterdir() if (p / "script.json").is_file())):
        script = json.loads((d / "script.json").read_text(encoding="utf-8"))
        exp = json.loads((d / "expected_events.json").read_text(encoding="utf-8"))
        events = [
            Span(
                script["call_id"],
                e["speaker"],
                e["category"],
                e["t_start"],
                e["t_end"],
                1.0,
                e["severity"],
                e["alert"],
            )
            for e in exp["events"]
        ]
        segs = [
            GoldSegment(
                t["speaker"],
                t["t_start"],
                t["t_end"],
                t["text"],
                [(w["w"], w["t_start"], w["t_end"]) for w in t["words"]],
            )
            for t in script["turns"]
        ]
        meta = {
            "dialect": dialects[i % 3],
            "queue": queues[i % 2],
            "agent_gender": genders[i % 2],
            "customer_gender": genders[(i + 1) % 2],
            "codec": "pcm_s16le",
            "duration": duration_bucket(script["duration_s"]),
            "time_of_day": ["morning", "evening"][i % 2],
        }
        calls.append(EvalCall(script["call_id"], ensure_wav(d), events, segs, meta, None, d))
    return EvalSet("fixtures", calls)


def load_manifest_split(
    manifest: Path, gold_dir: Path, split: str, audio_dir: Path, lock: Path | None = None
) -> EvalSet:
    """A split of an in-house manifest with converted annotations (see ``ml.data``)."""
    from ml.data.manifest import read_manifest
    from ml.data.process import stereo_review_path
    from ml.data.splits import check_golden

    rows = read_manifest(manifest)
    if split == "golden":
        problems = check_golden(rows, lock)
        if problems:
            raise RuntimeError("golden set failed its integrity check: " + "; ".join(problems))
    rows = [r for r in rows if r.split == split and not r.dropped]
    by_call: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = by_call.setdefault(
            r.call_id,
            {
                "dialect": r.dialect or "unknown",
                "queue": r.queue or "unknown",
                "codec": r.codec,
                "duration": duration_bucket(r.duration),
                "time_of_day": time_of_day(r.date),
            },
        )
        m.setdefault("agent_gender", "unknown")

    def jsonl(name: str) -> list[dict[str, Any]]:
        p = gold_dir / name
        return (
            [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
            if p.exists()
            else []
        )

    gold_events = {g["call_id"]: g["events"] for g in jsonl("fusion.jsonl")}
    t3 = {g["call_id"]: g for g in jsonl("t3.jsonl")}
    segs: dict[str, list[GoldSegment]] = {}
    for a in jsonl("asr.jsonl"):
        cid = Path(a["audio"]).parent.name
        ch = Path(a["audio"]).stem.split("_")[0]
        segs.setdefault(cid, []).append(GoldSegment(ch, a["start"], a["end"], a["text"]))
    calls = []
    for cid, meta in sorted(by_call.items()):
        events = [
            Span(
                cid,
                e["speaker"],
                e["category"],
                e["start"],
                e["end"],
                1.0,
                e.get("severity", 1),
                not (set(e.get("flags", [])) & NO_ALERT_FLAGS),
            )
            for e in gold_events.get(cid, [])
        ]
        calls.append(
            EvalCall(
                cid,
                stereo_review_path(audio_dir, cid),
                events,
                segs.get(cid, []),
                meta,
                t3.get(cid),
            )
        )
    return EvalSet(split, calls)
