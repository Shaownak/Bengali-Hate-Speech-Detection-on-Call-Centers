"""Writes from bus messages into Postgres (idempotent upserts) and row ↔ schema conversion."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from gale_api.db import (
    Agent,
    Call,
    CallScore,
    EventRow,
    Segment,
    SignalRow,
    ensure_partition,
    now,
    partition_exists,
)
from galecore.bangla import normalize
from galecore.schemas import (
    AsrSegment,
    CallLifecycle,
    CallScorecard,
    Channel,
    Evidence,
    FusedEvent,
    PostcallResult,
    Signal,
    Word,
)


def as_uuid(call_id: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(call_id)
    except ValueError:
        return None  # D-18: every GAL-E call id is a UUID; anything else is not persisted


class Partitions:
    """Remembers which monthly partitions exist so each is created once per process."""

    def __init__(self) -> None:
        self.done: set[str] = set()

    def ensure(self, db: Session, table: str, when: datetime) -> None:
        key = f"{table}:{when.astimezone(UTC):%Y%m}"
        if key in self.done:
            return
        # own committed transaction: a rollback of the caller's write must not undo the DDL
        bind = db.get_bind()
        engine = bind if isinstance(bind, Engine) else bind.engine
        try:
            with engine.begin() as conn:
                ensure_partition(conn, table, when)
        except DBAPIError:  # another replica created it concurrently
            with engine.connect() as conn:
                if not partition_exists(conn, table, when):
                    raise
        self.done.add(key)


def agent_for_code(db: Session, code: str | None) -> uuid.UUID | None:
    if not code:
        return None
    a = db.scalar(select(Agent.id).where(Agent.employee_code == code))
    if a is not None:
        return a
    stmt = (
        insert(Agent)
        .values(id=uuid.uuid4(), employee_code=code, active=True)
        .on_conflict_do_nothing(index_elements=["employee_code"])
    )
    db.execute(stmt)
    return db.scalar(select(Agent.id).where(Agent.employee_code == code))


def ensure_call(db: Session, call_id: uuid.UUID, *, mode: str = "realtime") -> Call:
    call = db.get(Call, call_id)
    if call is None:
        db.execute(
            insert(Call)
            .values(id=call_id, started_at=now(), mode=mode, status="active", created_at=now())
            .on_conflict_do_nothing(index_elements=["id"])
        )
        call = db.get(Call, call_id)
    if call is None:  # pragma: no cover - the insert above guarantees the row
        raise RuntimeError(f"call {call_id} vanished")
    return call


def apply_lifecycle(db: Session, ev: CallLifecycle) -> None:
    cid = as_uuid(ev.call_id)
    if cid is None:
        return
    call = ensure_call(db, cid, mode=ev.mode or "realtime")
    if ev.event == "started":
        if ev.wall_clock is not None:
            call.started_at = datetime.fromtimestamp(ev.wall_clock, UTC)
        call.external_call_id = ev.external_ref or call.external_call_id
        call.queue = ev.queue or call.queue
        call.direction = ev.direction or call.direction
        call.channel_layout = ev.channel_layout or call.channel_layout
        call.mode = ev.mode or call.mode
        call.agent_id = agent_for_code(db, ev.agent_id) or call.agent_id
        call.status = "active"
    elif ev.event in ("ended", "hangup"):
        call.status = "ended"
        call.duration_s = ev.ts
        if ev.wall_clock is not None:
            call.ended_at = datetime.fromtimestamp(ev.wall_clock, UTC)
        else:
            call.ended_at = datetime.fromtimestamp(call.started_at.timestamp() + ev.ts, UTC)


def upsert_segment(db: Session, parts: Partitions, seg: AsrSegment) -> None:
    cid = as_uuid(seg.call_id)
    if cid is None or not seg.is_final:
        return  # partial hypotheses are live-only; only finals persist
    call = ensure_call(db, cid)
    parts.ensure(db, "segments", call.started_at)
    values: dict[str, Any] = {
        "id": seg.segment_id,
        "call_started_at": call.started_at,
        "call_id": cid,
        "channel": seg.channel.value,
        "ts_start": seg.ts_start,
        "ts_end": seg.ts_end,
        "text_redacted": seg.text_redacted,  # redacted upstream (ADR-006); raw never persists
        "text_norm": normalize(seg.text_redacted),
        "words_jsonb": [w.model_dump(mode="json") for w in seg.words],
        "avg_conf": seg.avg_conf,
        "model_version": seg.model_version,
    }
    stmt = insert(Segment).values(**values)
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["id", "call_started_at"],
            set_={k: stmt.excluded[k] for k in values if k not in ("id", "call_started_at")},
        )
    )


def upsert_signal(db: Session, parts: Partitions, sig: Signal) -> None:
    cid = as_uuid(sig.call_id)
    if cid is None:
        return
    call = ensure_call(db, cid)
    parts.ensure(db, "signals", call.started_at)
    db.execute(
        insert(SignalRow)
        .values(
            id=sig.signal_id,
            call_started_at=call.started_at,
            call_id=cid,
            channel=sig.channel.value,
            source=sig.source.value,
            category=sig.category.value,
            score=sig.score,
            ts_start=sig.ts_start,
            ts_end=sig.ts_end,
            evidence_jsonb=sig.evidence.model_dump(mode="json"),
            model_version=sig.model_version,
            shadow=sig.shadow,
        )
        .on_conflict_do_nothing(index_elements=["id", "call_started_at"])
    )


def upsert_event(db: Session, ev: FusedEvent) -> None:
    cid = as_uuid(ev.call_id)
    if cid is None:
        return
    ensure_call(db, cid)
    values: dict[str, Any] = {
        "id": ev.event_id,
        "call_id": cid,
        "speaker": ev.speaker.value,
        "category": ev.category.value,
        "severity": ev.severity,
        "probability": ev.probability,
        "ts_start": ev.ts_start,
        "ts_end": ev.ts_end,
        "contributing_signal_ids": ev.contributing_signals,
        "explanation_bn": ev.explanation_bn,
        "explanation_en": ev.explanation_en,
        "status": ev.status.value,
        "alert": ev.alert,
        "shadow": ev.shadow,
        "updated_at": now(),
    }
    stmt = insert(EventRow).values(**values, created_at=now())
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={k: stmt.excluded[k] for k in values if k != "id"},
        )
    )


def escalation_curve(db: Session, call_id: uuid.UUID, bin_s: float = 60.0) -> list[list[float]]:
    """``[[bin_start_s, agent_events, customer_events], …]`` over confirmed/provisional events."""
    rows = db.execute(
        select(EventRow.ts_start, EventRow.speaker).where(
            EventRow.call_id == call_id, EventRow.status != "dismissed_by_llm"
        )
    ).all()
    bins: dict[int, list[float]] = {}
    for ts, speaker in rows:
        b = bins.setdefault(int(ts // bin_s), [0.0, 0.0])
        b[0 if speaker == "agent" else 1] += 1
    return [[k * bin_s, *v] for k, v in sorted(bins.items())]


def upsert_scorecard(db: Session, call_id: str, card: CallScorecard, version: str) -> None:
    cid = as_uuid(call_id)
    if cid is None:
        return
    ensure_call(db, cid)
    values: dict[str, Any] = {
        "call_id": cid,
        "agent_score": card.agent.professionalism,
        "customer_hostility": card.customer.hostility,
        "escalation_curve_jsonb": escalation_curve(db, cid),
        "scorecard_jsonb": card.model_dump(mode="json"),
        "model_version": version,
    }
    stmt = insert(CallScore).values(**values)
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["call_id"], set_={k: stmt.excluded[k] for k in values if k != "call_id"}
        )
    )


def apply_postcall(db: Session, parts: Partitions, res: PostcallResult) -> None:
    cid = as_uuid(res.call_id)
    if cid is None:
        return
    call = ensure_call(db, cid, mode="postcall")
    call.mode = "postcall"
    call.status = "ended"
    call.duration_s = res.duration_s
    call.ended_at = datetime.fromtimestamp(call.started_at.timestamp() + res.duration_s, UTC)
    call.external_call_id = res.external_ref or call.external_call_id
    call.queue = res.queue or call.queue
    call.channel_layout = res.channel_layout
    call.recording_uri = res.recording_uri or call.recording_uri
    for seg in res.segments:
        upsert_segment(db, parts, seg)
    for sig in res.signals:
        upsert_signal(db, parts, sig)


# --- rows → schemas ----------------------------------------------------------------------------


def segment_of(r: Segment) -> AsrSegment:
    return AsrSegment(
        call_id=str(r.call_id),
        channel=Channel(r.channel),
        segment_id=r.id,
        ts_start=r.ts_start,
        ts_end=r.ts_end,
        text_redacted=r.text_redacted,
        words=[Word.model_validate(w) for w in r.words_jsonb],
        is_final=True,
        avg_conf=r.avg_conf,
        model_version=r.model_version,
    )


def signal_of(r: SignalRow) -> Signal:
    return Signal(
        call_id=str(r.call_id),
        channel=Channel(r.channel),
        signal_id=r.id,
        source=r.source,
        category=r.category,
        score=r.score,
        ts_start=r.ts_start,
        ts_end=r.ts_end,
        evidence=Evidence.model_validate(r.evidence_jsonb),
        model_version=r.model_version,
        shadow=r.shadow,
    )


def event_of(r: EventRow) -> FusedEvent:
    return FusedEvent(
        event_id=r.id,
        call_id=str(r.call_id),
        speaker=r.speaker,
        category=r.category,
        severity=r.severity,
        probability=r.probability,
        ts_start=r.ts_start,
        ts_end=r.ts_end,
        contributing_signals=list(r.contributing_signal_ids),
        explanation_bn=r.explanation_bn,
        explanation_en=r.explanation_en,
        status=r.status,
        alert=r.alert,
        shadow=r.shadow,
    )
