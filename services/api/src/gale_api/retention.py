"""Retention and DSAR deletion (docs/05 §7). Both are irreversible, so:

- retention reads `configs/retention.yaml`; it only *reports* unless the policy is marked
  `approved: true` (DPO sign-off) **and** the caller passes `execute=True` (ADR-018);
- DSAR deletion defaults to a dry run and needs a reason; selectors are logged only as hashes.

Every run writes an `audit_log` row with the counts. Golden-set manifests live in the data
engine, not in this database, so nothing here can touch them (CLAUDE.md hard rule 3).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import ColumnElement, delete, func, or_, select, text
from sqlalchemy.orm import Session

from gale_api.db import (
    PARTITIONED,
    AuditLog,
    Call,
    CallScore,
    EventRow,
    ReviewRow,
    Segment,
    SignalRow,
)
from galecore.objectstore import ObjectStore


class RetentionPolicy(BaseModel):
    version: int
    approved: bool = False
    raw_audio_days: int = Field(ge=1)
    transcripts_months: int = Field(ge=1)
    events_months: int = Field(ge=1)
    audit_log_months: int = Field(ge=1)

    @model_validator(mode="after")
    def _order(self) -> RetentionPolicy:
        if self.transcripts_months > self.events_months:
            raise ValueError("transcripts must not outlive the events that cite them")
        if self.audit_log_months < self.events_months:
            raise ValueError("the audit trail must outlive the data it describes")
        return self


def load_policy(config_dir: Path) -> RetentionPolicy:
    return RetentionPolicy.model_validate(
        yaml.safe_load((config_dir / "retention.yaml").read_text(encoding="utf-8"))
    )


def months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + (d.month - 1) - months, 12)
    return date(y, m + 1, 1)


def audio_keys(call: Call) -> list[str]:
    keys = [f"audio/{call.id}/agent.wav", f"audio/{call.id}/customer.wav"]
    keys += [f"reports/{call.id}/report.json", f"reports/{call.id}/report.html"]
    if call.recording_uri:
        keys.append(call.recording_uri)
    return keys


@dataclass
class Report:
    executed: bool
    audio_calls: int = 0
    objects_deleted: int = 0
    partitions: list[str] = field(default_factory=list)
    calls: int = 0
    events: int = 0
    audit_rows: int = 0


def _old_partitions(db: Session, cutoff: date) -> list[str]:
    """Monthly partitions that end on or before the cutoff month (whole months only)."""
    out = []
    for table in PARTITIONED:
        rows = db.execute(
            text(
                "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
                "JOIN pg_class p ON p.oid = i.inhparent WHERE p.relname = :t"
            ),
            {"t": table},
        ).scalars()
        for name in rows:
            stamp = name.rsplit("_p", 1)[-1]
            if len(stamp) == 6 and stamp.isdigit():
                start = date(int(stamp[:4]), int(stamp[4:]), 1)
                if months_before(start, -1) <= cutoff:  # partition end ≤ cutoff
                    out.append(name)
    return sorted(out)


def run_retention(
    db: Session,
    store: ObjectStore,
    policy: RetentionPolicy,
    *,
    execute: bool,
    now: datetime | None = None,
) -> Report:
    now = now or datetime.now(UTC)
    if execute and not policy.approved:
        raise PermissionError("retention policy not approved by the DPO (configs/retention.yaml)")
    audio_cutoff = now - timedelta(days=policy.raw_audio_days)
    transcripts_cutoff = months_before(now.date(), policy.transcripts_months)
    events_cutoff = datetime.combine(
        months_before(now.date(), policy.events_months), datetime.min.time(), UTC
    )
    audit_cutoff = datetime.combine(
        months_before(now.date(), policy.audit_log_months), datetime.min.time(), UTC
    )
    rep = Report(executed=execute)

    audio_calls = db.scalars(
        select(Call).where(func.coalesce(Call.ended_at, Call.started_at) < audio_cutoff)
    ).all()
    rep.audio_calls = len(audio_calls)
    rep.partitions = _old_partitions(db, transcripts_cutoff)
    old_calls = select(Call.id).where(Call.started_at < events_cutoff)
    rep.calls = db.scalar(select(func.count()).select_from(old_calls.subquery())) or 0
    rep.events = (
        db.scalar(select(func.count(EventRow.id)).where(EventRow.call_id.in_(old_calls))) or 0
    )
    rep.audit_rows = (
        db.scalar(select(func.count(AuditLog.id)).where(AuditLog.ts < audit_cutoff)) or 0
    )

    if execute:
        for call in audio_calls:
            for key in audio_keys(call):
                rep.objects_deleted += store.delete(key)
        for name in rep.partitions:
            db.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        _delete_calls(db, old_calls)
        db.execute(delete(AuditLog).where(AuditLog.ts < audit_cutoff))
    db.add(
        AuditLog(
            action="retention.purge" if execute else "retention.dry_run",
            object_type="system",
            details_jsonb={**asdict(rep), "policy_version": policy.version},
        )
    )
    return rep


def _delete_calls(db: Session, call_ids: Any) -> None:
    event_ids = select(EventRow.id).where(EventRow.call_id.in_(call_ids))
    db.execute(delete(ReviewRow).where(ReviewRow.event_id.in_(event_ids)))
    db.execute(delete(EventRow).where(EventRow.call_id.in_(call_ids)))
    db.execute(delete(CallScore).where(CallScore.call_id.in_(call_ids)))
    db.execute(delete(Segment).where(Segment.call_id.in_(call_ids)))
    db.execute(delete(SignalRow).where(SignalRow.call_id.in_(call_ids)))
    db.execute(delete(Call).where(Call.id.in_(call_ids)))


# --- DSAR ---------------------------------------------------------------------------------------


class DsarIn(BaseModel):
    """Erase every record of a data subject's calls (PDPO 2025). At least one selector."""

    call_ids: list[uuid.UUID] = Field(default_factory=list, max_length=1000)
    external_call_id: str | None = Field(default=None, max_length=255)
    customer_hash: str | None = Field(default=None, max_length=128)
    reason: str = Field(min_length=3, max_length=500)
    dry_run: bool = True

    @model_validator(mode="after")
    def _selector(self) -> DsarIn:
        if not (self.call_ids or self.external_call_id or self.customer_hash):
            raise ValueError("give call_ids, external_call_id or customer_hash")
        return self


class DsarOut(BaseModel):
    dry_run: bool
    calls: list[uuid.UUID]
    events: int
    segments: int
    objects_deleted: int


def _digest(v: str) -> str:
    return hashlib.sha256(v.encode()).hexdigest()[:16]


def run_dsar(db: Session, store: ObjectStore, req: DsarIn, user_id: uuid.UUID | None) -> DsarOut:
    conds: list[ColumnElement[bool]] = []
    if req.call_ids:
        conds.append(Call.id.in_(req.call_ids))
    if req.external_call_id:
        conds.append(Call.external_call_id == req.external_call_id)
    if req.customer_hash:
        conds.append(Call.customer_hash == req.customer_hash)
    calls = db.scalars(select(Call).where(or_(*conds))).all()
    ids = [c.id for c in calls]
    events = db.scalar(select(func.count(EventRow.id)).where(EventRow.call_id.in_(ids))) or 0
    segments = db.scalar(select(func.count()).where(Segment.call_id.in_(ids))) or 0
    deleted = 0
    if not req.dry_run:
        for call in calls:
            for key in audio_keys(call):
                deleted += store.delete(key)
        _delete_calls(db, ids)
    db.add(
        AuditLog(
            user_id=user_id,
            action="dsar.dry_run" if req.dry_run else "dsar.delete",
            object_type="data_subject",
            # selectors are personal data themselves: only hashes go to the audit trail
            details_jsonb={
                "reason": req.reason,
                "calls": len(ids),
                "events": events,
                "segments": segments,
                "objects_deleted": deleted,
                "selector": {
                    "call_ids": len(req.call_ids),
                    "external_call_id": _digest(req.external_call_id)
                    if req.external_call_id
                    else None,
                    "customer_hash": _digest(req.customer_hash) if req.customer_hash else None,
                },
            },
        )
    )
    return DsarOut(
        dry_run=req.dry_run, calls=ids, events=events, segments=segments, objects_deleted=deleted
    )
