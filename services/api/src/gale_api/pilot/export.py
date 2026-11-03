"""Nightly feedback export for the retraining pipelines (docs/02 §6 step 1; ADR-020).

    python -m gale_api export-feedback [--since 2026-11-01] [--until …] [--out FILE]

One JSONL row per reviewed event (latest verdict) and per QA-found missed event, with the
**redacted** text of that speaker's overlapping segments, two turns of context and the contributing
signal scores. ``ml.retrain`` reads it to fine-tune the text classifier (``text`` → ``label``) and
to refit fusion calibration (``signals`` → ``label``). Golden calls are dropped (CLAUDE.md rule 3).
The file goes to the object store under ``datasets/feedback/`` — it holds redacted text only and is
never logged.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session

from gale_api.db import Call, EventRow, MissedEvent, ReviewRow, Segment, SignalRow
from gale_api.pilot.report import FP, TP


@dataclass
class ExportStats:
    rows: int = 0
    positives: int = 0
    negatives: int = 0
    golden_dropped: int = 0


def _text(segs: list[Segment], speaker: str, t0: float, t1: float) -> tuple[str, list[str]]:
    mine = [
        g for g in segs if g.channel == speaker and g.ts_start < t1 + 0.5 and t0 - 0.5 < g.ts_end
    ]
    before = [g for g in segs if g.ts_end <= (mine[0].ts_start if mine else t0)][-2:]
    return " ".join(g.text_redacted for g in mine), [
        f"[{g.channel.upper()}] {g.text_redacted}" for g in before
    ]


def export_feedback(
    db: Session, since: datetime, until: datetime, golden: set[str]
) -> tuple[list[dict[str, Any]], ExportStats]:
    stats = ExportStats()
    latest: dict[str, ReviewRow] = {
        r.event_id: r
        for r in db.scalars(
            select(ReviewRow)
            .where(
                ReviewRow.kind == "verdict",
                ReviewRow.created_at >= since,
                ReviewRow.created_at < until,
            )
            .ext(distinct_on(ReviewRow.event_id))
            .order_by(ReviewRow.event_id, ReviewRow.created_at.desc())
        )
    }
    events = {e.id: e for e in db.scalars(select(EventRow).where(EventRow.id.in_(list(latest))))}
    missed = list(
        db.scalars(
            select(MissedEvent).where(
                MissedEvent.created_at >= since, MissedEvent.created_at < until
            )
        )
    )
    call_ids = {e.call_id for e in events.values()} | {m.call_id for m in missed}
    calls = {c.id: c for c in db.scalars(select(Call).where(Call.id.in_(call_ids)))}
    gold = {cid for cid, c in calls.items() if str(cid) in golden or c.external_call_id in golden}
    segs: dict[uuid.UUID, list[Segment]] = defaultdict(list)
    for g in db.scalars(
        select(Segment).where(Segment.call_id.in_(call_ids - gold)).order_by(Segment.ts_start)
    ):
        segs[g.call_id].append(g)
    sig_ids = {s for e in events.values() for s in e.contributing_signal_ids}
    sigs = {
        s.id: s
        for s in db.scalars(
            select(SignalRow).where(
                SignalRow.call_id.in_(call_ids - gold), SignalRow.id.in_(sig_ids)
            )
        )
    }
    rows: list[dict[str, Any]] = []
    for eid, rv in sorted(latest.items()):
        e = events.get(eid)
        if e is None or rv.verdict not in TP | FP:
            continue
        if e.call_id in gold:
            stats.golden_dropped += 1
            continue
        positive = rv.verdict in TP
        text, ctx = _text(segs[e.call_id], e.speaker, e.ts_start, e.ts_end)
        rows.append(
            {
                "uid": f"review:{eid}",
                "kind": "review",
                "call_id": str(e.call_id),
                "event_id": eid,
                "speaker": e.speaker,
                "ts_start": e.ts_start,
                "ts_end": e.ts_end,
                "category": rv.corrected_category or e.category,
                "predicted_category": e.category,
                "severity": rv.corrected_severity or e.severity,
                "label": int(positive),
                "verdict": rv.verdict,
                "text": text,
                "context": ctx,
                "signals": [
                    {"source": s.source, "category": s.category, "score": s.score}
                    for s in (sigs.get(i) for i in e.contributing_signal_ids)
                    if s is not None
                ],
            }
        )
        stats.positives += positive
        stats.negatives += not positive
    for me in missed:
        if me.call_id in gold:
            stats.golden_dropped += 1
            continue
        text, ctx = _text(segs[me.call_id], me.speaker, me.ts_start, me.ts_end)
        rows.append(
            {
                "uid": f"missed:{me.id}",
                "kind": "missed",
                "call_id": str(me.call_id),
                "event_id": None,
                "speaker": me.speaker,
                "ts_start": me.ts_start,
                "ts_end": me.ts_end,
                "category": me.category,
                "predicted_category": None,
                "severity": me.severity,
                "label": 1,
                "verdict": "missed",
                "text": text,
                "context": ctx,
                "signals": [],
            }
        )
        stats.positives += 1
    stats.rows = len(rows)
    return rows, stats


def to_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
