"""Bus → Postgres persister: lifecycle, finals only, idempotent upserts, post-call results."""

from __future__ import annotations

import uuid

from fakeredis import FakeAsyncRedis
from sqlalchemy import func, select

from gale_api.db import (
    Agent,
    Call,
    CallScore,
    EventRow,
    Segment,
    SignalRow,
    make_engine,
    make_sessionmaker,
)
from gale_api.persist import Persister
from galecore.bus import (
    ASR_SEGMENTS,
    CALL_LIFECYCLE,
    CALL_SCORES,
    EVENTS_FUSED,
    POSTCALL_RESULTS,
    SIGNALS,
    BusClient,
)
from galecore.schemas import (
    CallLifecycle,
    CallScorecard,
    CallScoreMessage,
    EventStatus,
    PostcallResult,
)

from .conftest import SCORECARD, _ev, _seg, _sig


async def _drain(p: Persister) -> None:
    await p.setup()
    for _ in range(20):
        if not await p.poll():
            return


async def test_persister_writes_everything_idempotently(db_url: str) -> None:
    engine = make_engine(db_url)
    sessions = make_sessionmaker(engine)
    bus = BusClient(FakeAsyncRedis())
    cid = str(uuid.uuid4())
    await bus.publish(
        CALL_LIFECYCLE,
        CallLifecycle(
            call_id=cid,
            event="started",
            ts=0,
            agent_id="AG-NEW",
            queue="billing",
            wall_clock=1_790_000_000.0,
            external_ref="pbx-9",
        ),
    )
    final = _seg(cid, 1, "agent", 1.0, "<GALI_1> কথা")  # type: ignore[arg-type]
    partial = final.model_copy(update={"segment_id": f"{cid}:p", "is_final": False})
    for m in (partial, final, final):  # a redelivered final must not duplicate
        await bus.publish(ASR_SEGMENTS, m)
    await bus.publish(SIGNALS, _sig(cid, "s1", "lexicon", "insult", "agent", 1.0, 0.9))  # type: ignore[arg-type]
    ev = _ev(cid, "e1", "agent", "insult", 1.0, alert=False, sev=2)
    await bus.publish(EVENTS_FUSED, ev.model_copy(update={"status": EventStatus.PROVISIONAL}))
    await bus.publish(EVENTS_FUSED, ev.model_copy(update={"alert": True}))  # later state wins
    await bus.publish(
        CALL_SCORES,
        CallScoreMessage(
            call_id=cid, scorecard=CallScorecard.model_validate(SCORECARD), model_version="m"
        ),
    )
    await bus.publish(CALL_LIFECYCLE, CallLifecycle(call_id=cid, event="ended", ts=42.0))
    await bus.publish(SIGNALS, _sig("not-a-uuid", "s9", "lexicon", "insult", "agent", 1.0, 0.9))  # type: ignore[arg-type]

    p = Persister(bus, sessions, min_idle_ms=0)
    await _drain(p)
    with sessions() as s:
        call = s.get(Call, uuid.UUID(cid))
        assert call is not None
        assert (call.status, call.duration_s, call.queue, call.external_call_id) == (
            "ended",
            42.0,
            "billing",
            "pbx-9",
        )
        assert call.started_at.timestamp() == 1_790_000_000.0
        agent = s.scalar(select(Agent).where(Agent.employee_code == "AG-NEW"))
        assert agent is not None
        assert call.agent_id == agent.id  # unknown agents are registered (no team yet)
        assert s.scalar(select(func.count()).select_from(Segment)) == 1
        seg = s.scalars(select(Segment)).one()
        assert seg.text_norm  # normalised copy for search
        assert s.scalar(select(func.count()).select_from(SignalRow)) == 1
        row = s.scalars(select(EventRow)).one()
        assert (row.status, row.alert) == ("confirmed", True)
        assert s.get(CallScore, uuid.UUID(cid)).agent_score == 2  # type: ignore[union-attr]
    engine.dispose()


async def test_persister_stores_postcall_results(db_url: str) -> None:
    engine = make_engine(db_url)
    sessions = make_sessionmaker(engine)
    bus = BusClient(FakeAsyncRedis())
    cid = str(uuid.uuid4())
    res = PostcallResult(
        call_id=cid,
        duration_s=12.5,
        queue="support",
        recording_uri=f"recordings/{cid}.wav",
        segments=[_seg(cid, 1, "customer", 0.5, "হ্যালো")],  # type: ignore[arg-type]
        signals=[_sig(cid, "s1", "dsp_shout", "shouting", "customer", 0.5, 0.8)],  # type: ignore[arg-type]
    )
    await bus.publish(POSTCALL_RESULTS, res)
    await bus.publish(EVENTS_FUSED, _ev(cid, "e1", "customer", "shouting", 0.5, alert=True, sev=2))
    await _drain(Persister(bus, sessions, min_idle_ms=0))
    with sessions() as s:
        call = s.get(Call, uuid.UUID(cid))
        assert call is not None
        assert (call.mode, call.status, call.duration_s) == ("postcall", "ended", 12.5)
        assert call.recording_uri == f"recordings/{cid}.wav"
        assert s.scalar(select(func.count()).select_from(Segment)) == 1
        assert s.scalar(select(func.count()).select_from(EventRow)) == 1
    engine.dispose()
