"""Retention and DSAR deletion: dry runs change nothing, execution needs approval, audit rows."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text

from gale_api.db import AuditLog, Call, EventRow, Segment, make_sessionmaker, session_scope
from gale_api.retention import RetentionPolicy, run_retention

from .conftest import World

# fixture calls start at epoch 1_790_000_000 = 2026-09-21
POLICY = dict(
    version=1, raw_audio_days=30, transcripts_months=6, events_months=24, audit_log_months=36
)


def _put_audio(w: World) -> None:
    for cid in (w.call_a, w.call_b):
        w.app.state.store.put(f"audio/{cid}/agent.wav", b"x")


def _count(w: World, model: type, **where: object) -> int:
    with make_sessionmaker(w.engine)() as s:
        q = select(func.count()).select_from(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return s.scalar(q) or 0


def test_policy_validation() -> None:
    with pytest.raises(ValidationError, match="outlive"):
        RetentionPolicy(**{**POLICY, "transcripts_months": 30})
    with pytest.raises(ValidationError, match="audit trail"):
        RetentionPolicy(**{**POLICY, "audit_log_months": 12})


def test_dry_run_reports_without_deleting_and_execute_needs_approval(world: World) -> None:
    _put_audio(world)
    store = world.app.state.store
    sessions = make_sessionmaker(world.engine)
    now = datetime(2027, 6, 1, tzinfo=UTC)
    with session_scope(sessions) as db:
        rep = run_retention(db, store, RetentionPolicy(**POLICY), execute=False, now=now)
    assert (rep.executed, rep.audio_calls, rep.calls) == (False, 2, 0)
    assert rep.partitions == ["segments_p202609", "signals_p202609"]
    assert store.exists(f"audio/{world.call_a}/agent.wav")
    assert _count(world, Segment) == 4
    with session_scope(sessions) as db, pytest.raises(PermissionError, match="DPO"):
        run_retention(db, store, RetentionPolicy(**POLICY), execute=True, now=now)

    approved = RetentionPolicy(**POLICY, approved=True)
    with session_scope(sessions) as db:
        rep = run_retention(db, store, approved, execute=True, now=now)
    assert rep.objects_deleted == 2
    assert not store.exists(f"audio/{world.call_a}/agent.wav")
    assert _count(world, Segment) == 0  # whole monthly partitions dropped
    assert _count(world, Call) == 2  # events/calls are kept for 24 months
    with sessions() as s:
        assert s.execute(text("SELECT to_regclass('segments_p202609')")).scalar() is None
    actions = [a for (a,) in sessions().execute(select(AuditLog.action).order_by(AuditLog.id))]
    assert actions == ["retention.dry_run", "retention.purge"]

    later = datetime(2029, 1, 1, tzinfo=UTC)
    with session_scope(sessions) as db:
        rep = run_retention(db, store, approved, execute=True, now=later)
    assert (rep.calls, rep.events) == (2, 4)
    assert _count(world, Call) == 0
    assert _count(world, EventRow) == 0


def test_dsar_dry_run_then_erase(world: World) -> None:
    _put_audio(world)
    h = world.h("dev-admin")
    body = {"call_ids": [world.call_a], "reason": "DSAR #42", "dry_run": True}
    r = world.client.post("/api/v1/dsar", json=body, headers=h)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "dry_run": True,
        "calls": [world.call_a],
        "events": 2,
        "segments": 2,
        "objects_deleted": 0,
    }
    assert _count(world, Call, id=uuid.UUID(world.call_a)) == 1
    r = world.client.post("/api/v1/dsar", json={**body, "dry_run": False}, headers=h)
    assert r.json()["objects_deleted"] == 1
    assert _count(world, Call, id=uuid.UUID(world.call_a)) == 0
    assert _count(world, Segment, call_id=uuid.UUID(world.call_a)) == 0
    assert _count(world, Call, id=uuid.UUID(world.call_b)) == 1  # nobody else's data
    assert not world.app.state.store.exists(f"audio/{world.call_a}/agent.wav")
    assert world.app.state.store.exists(f"audio/{world.call_b}/agent.wav")
    bad = world.client.post("/api/v1/dsar", json={"reason": "no selector"}, headers=h)
    assert bad.status_code == 422
    world.client.post(
        "/api/v1/dsar", json={"external_call_id": "crm-777", "reason": "DSAR #43"}, headers=h
    )
    with make_sessionmaker(world.engine)() as s:
        rows = s.scalars(select(AuditLog).where(AuditLog.action.like("dsar.%"))).all()
    assert [r.action for r in rows] == ["dsar.dry_run", "dsar.delete", "dsar.dry_run"]
    sel = rows[-1].details_jsonb["selector"]
    assert sel["external_call_id"] != "crm-777"  # hashed, never the raw identifier
    assert len(sel["external_call_id"]) == 16
