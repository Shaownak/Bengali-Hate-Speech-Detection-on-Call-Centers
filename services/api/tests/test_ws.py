"""WebSockets: auth handshake, role/scope filtering, 250 ms batches with event coalescing."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fakeredis import FakeRedis
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from gale_api.db import AuditLog, make_sessionmaker
from galecore.bus import ASR_SEGMENTS, CALL_LIFECYCLE, EVENTS_FUSED, encode
from galecore.schemas import CallLifecycle, EventStatus, GaleModel

from .conftest import World, _ev, _seg


def _pub(w: World, *msgs: tuple[str, GaleModel]) -> None:
    r = FakeRedis(server=w.server)
    pipe = r.pipeline()
    for stream, m in msgs:
        pipe.xadd(stream, encode(m))  # type: ignore[arg-type]
    pipe.execute()


def _closed_with(w: World, path: str, first: Any) -> int:
    with w.client.websocket_connect(path) as ws:
        ws.send_json(first)
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    return e.value.code


def test_ws_auth_and_scope_close_codes(world: World) -> None:
    w = world
    assert _closed_with(w, "/ws/live", {"type": "hello"}) == 4401
    assert _closed_with(w, "/ws/live", {"type": "auth", "token": "garbage"}) == 4401
    assert _closed_with(w, "/ws/live", {"type": "auth", "token": w.token("dev-agent")}) == 4403
    assert _closed_with(w, "/ws/live", {"type": "auth", "token": w.token("dev-auditor")}) == 4403
    tok = w.token("dev-supervisor-ctg")
    assert _closed_with(w, f"/ws/calls/{w.call_a}", {"type": "auth", "token": tok}) == 4404
    assert _closed_with(w, "/ws/calls/nope", {"type": "auth", "token": tok}) == 4404


def _batch(ws: Any) -> list[dict[str, Any]]:
    msg = ws.receive_json()
    assert msg["type"] == "batch"
    return msg["items"]  # type: ignore[no-any-return]


def test_ws_live_filters_by_team_and_coalesces(world: World) -> None:
    w = world
    c_dhaka, c_ctg = str(uuid.uuid4()), str(uuid.uuid4())
    with w.client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "auth", "token": w.token("dev-supervisor")})
        snap = ws.receive_json()
        assert snap["type"] == "snapshot"
        assert snap["calls"] == []  # fixture calls have ended
        ev = _ev(c_dhaka, "e1", "agent", "insult", 1.0, alert=False, sev=2)
        _pub(
            w,
            (
                CALL_LIFECYCLE,
                CallLifecycle(call_id=c_dhaka, event="started", ts=0, agent_id="AG-002"),
            ),
            (
                CALL_LIFECYCLE,
                CallLifecycle(call_id=c_ctg, event="started", ts=0, agent_id="AG-003"),
            ),
            (EVENTS_FUSED, ev.model_copy(update={"status": EventStatus.PROVISIONAL})),
            (EVENTS_FUSED, ev.model_copy(update={"alert": True})),
            (EVENTS_FUSED, _ev(c_ctg, "e1", "agent", "insult", 1.0, alert=True, sev=2)),
        )
        items: list[dict[str, Any]] = []
        while len(items) < 2:
            items += _batch(ws)
    assert {i["data"]["call_id"] for i in items} == {c_dhaka}  # the other team's call is filtered
    events = [i for i in items if i["stream"] == EVENTS_FUSED]
    assert len(events) == 1  # two updates of one event in one batch → latest state only
    assert events[0]["data"]["alert"] is True


def test_ws_call_streams_one_call_and_audits(world: World) -> None:
    w = world
    with w.client.websocket_connect(f"/ws/calls/{w.call_a}") as ws:
        ws.send_json({"type": "auth", "token": w.token("dev-agent")})
        snap = ws.receive_json()
        assert snap["type"] == "snapshot"
        assert len(snap["segments"]) == 2
        assert {e["event_id"] for e in snap["events"]} == {w.ev_agent_a, w.ev_customer_a}
        partial = _seg(w.call_a, 9, "agent", 9.0, "…").model_copy(update={"is_final": False})  # type: ignore[arg-type]
        _pub(w, (ASR_SEGMENTS, _seg(w.call_b, 9, "agent", 9.0, "x")), (ASR_SEGMENTS, partial))  # type: ignore[arg-type]
        items = _batch(ws)
    assert [(i["stream"], i["data"]["call_id"]) for i in items] == [(ASR_SEGMENTS, w.call_a)]
    with make_sessionmaker(w.engine)() as s:
        rows = s.scalars(select(AuditLog).where(AuditLog.action == "transcript.live")).all()
        assert [r.object_id for r in rows] == [w.call_a]
