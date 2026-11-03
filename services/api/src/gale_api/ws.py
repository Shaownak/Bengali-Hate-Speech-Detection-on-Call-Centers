"""WebSockets (docs/04 §3): ``/ws/live`` (wallboard) and ``/ws/calls/{id}`` (one call, live).

One ``LiveHub`` per API process tails the bus with plain XREAD (fan-out, no consumer group).
Each connection gets the messages its role and teams allow. Updates go out batched every
250 ms (docs/04 §4); within a batch only the latest state of each event is kept.

Auth: browsers can't set headers on a WebSocket, and a token in the URL would end up in access
logs. So the first client message must be ``{"type": "auth", "token": "<JWT>"}`` within 5 s.
Close codes: 4401 unauthenticated, 4403 forbidden, 4404 unknown or out-of-scope call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from gale_api.audit import write_audit
from gale_api.auth import AuthError, Principal, principal_for
from gale_api.db import Agent, Call, EventRow, Segment
from gale_api.scope import call_scope, event_scope
from galecore.bus import (
    ALERTS,
    ASR_SEGMENTS,
    CALL_LIFECYCLE,
    CALL_SCORES,
    EVENTS_FUSED,
    SIGNALS,
    BusClient,
    decode,
)
from galecore.logging import get_logger
from galecore.metrics import WS_CONNECTIONS
from galecore.schemas import (
    AsrSegment,
    CallLifecycle,
    CallScoreMessage,
    FusedEvent,
    GaleModel,
    Signal,
)
from galecore.streaming import until_ready

log = get_logger(__name__)

STREAMS: dict[str, type[GaleModel]] = {
    CALL_LIFECYCLE: CallLifecycle,
    EVENTS_FUSED: FusedEvent,
    ALERTS: FusedEvent,
    ASR_SEGMENTS: AsrSegment,
    SIGNALS: Signal,
    CALL_SCORES: CallScoreMessage,
}
LIVE_STREAMS = frozenset({CALL_LIFECYCLE, EVENTS_FUSED, ALERTS, CALL_SCORES})
CALL_STREAMS = frozenset({CALL_LIFECYCLE, EVENTS_FUSED, ALERTS, ASR_SEGMENTS, SIGNALS})
AUTH_TIMEOUT_S = 5.0
BATCH_S = 0.25


@dataclass(eq=False)  # identity-hashed: one per connection
class Subscriber:
    principal: Principal
    streams: frozenset[str]
    call_id: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)
    event_pos: dict[str, int] = field(default_factory=dict)

    def push(self, stream: str, msg: GaleModel) -> None:
        item = {"stream": stream, "data": msg.model_dump(mode="json")}
        if stream == EVENTS_FUSED and isinstance(msg, FusedEvent):
            pos = self.event_pos.get(msg.event_id)
            if pos is not None:  # coalesce: latest state of the event in this batch
                self.items[pos] = item
                return
            self.event_pos[msg.event_id] = len(self.items)
        self.items.append(item)

    def take(self) -> list[dict[str, Any]]:
        out, self.items, self.event_pos = self.items, [], {}
        return out


class LiveHub:
    def __init__(self, bus: BusClient, sessions: sessionmaker[Session]) -> None:
        self.bus, self.sessions = bus, sessions
        self.subs: set[Subscriber] = set()
        self.last_ids: dict[str, str] = {}
        self._teams: dict[str, tuple[uuid.UUID | None, float]] = {}

    # --- team resolution (supervisor scoping) ---------------------------------------------------

    def _lookup_team(self, call_id: str, agent_code: str | None) -> uuid.UUID | None:
        with self.sessions() as s:
            if agent_code:
                t = s.scalar(select(Agent.team_id).where(Agent.employee_code == agent_code))
                if t is not None:
                    return t
            try:
                cid = uuid.UUID(call_id)
            except ValueError:
                return None
            return s.scalar(
                select(Agent.team_id).join(Call, Call.agent_id == Agent.id).where(Call.id == cid)
            )

    async def team_of(self, call_id: str, agent_code: str | None = None) -> uuid.UUID | None:
        hit = self._teams.get(call_id)
        if hit is not None and (hit[0] is not None or time.monotonic() - hit[1] < 2.0):
            return hit[0]
        team = await asyncio.to_thread(self._lookup_team, call_id, agent_code)
        self._teams[call_id] = (team, time.monotonic())
        return team

    async def allowed(self, sub: Subscriber, call_id: str) -> bool:
        p = sub.principal
        if sub.call_id is not None:
            return call_id == sub.call_id  # scope was checked when the socket opened
        if p.sees_all_calls:
            return True
        team = await self.team_of(call_id)
        return team is not None and team in p.team_ids

    # --- bus tail ------------------------------------------------------------------------------

    async def start_positions(self) -> None:
        for stream in STREAMS:
            info = cast("list[tuple[bytes, Any]]", await self.bus.redis.xrevrange(stream, count=1))
            self.last_ids[stream] = info[0][0].decode() if info else "0-0"

    async def dispatch(self, stream: str, msg: GaleModel) -> None:
        call_id = getattr(msg, "call_id", "")
        if isinstance(msg, CallLifecycle) and msg.event == "started":
            await self.team_of(call_id, msg.agent_id)
        hidden = (isinstance(msg, FusedEvent) and msg.shadow) or (
            isinstance(msg, Signal) and msg.shadow
        )
        for sub in list(self.subs):
            if hidden and not sub.principal.sees_shadow:
                continue  # shadow mode (ADR-020): not for supervisors / agents
            if stream in sub.streams and await self.allowed(sub, call_id):
                sub.push(stream, msg)

    async def poll(self) -> int:
        rows = cast(
            "list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] | None",
            await self.bus.redis.xread(dict(self.last_ids), count=500),  # type: ignore[arg-type]
        )
        n = 0
        for raw_stream, entries in rows or []:
            stream = raw_stream.decode()
            for msg_id, fields in entries:
                self.last_ids[stream] = msg_id.decode()
                if not fields or b"data" not in fields:
                    continue
                await self.dispatch(stream, decode(fields, STREAMS[stream]))
                n += 1
        return n

    async def run(self, stop: asyncio.Event) -> None:
        if not await until_ready(stop, self.start_positions, "live_hub"):
            return
        while not stop.is_set():
            try:
                n = await self.poll()
            except Exception as e:  # keep tailing; a bad message must not kill live views
                log.warning("live_hub_poll_failed", error=type(e).__name__)
                n = 0
            if not n:
                await asyncio.sleep(0.05)


# --- endpoints -----------------------------------------------------------------------------------

router = APIRouter()


async def _auth(ws: WebSocket) -> Principal | None:
    state = ws.app.state
    try:
        first = await asyncio.wait_for(ws.receive_json(), AUTH_TIMEOUT_S)
        if not isinstance(first, dict) or first.get("type") != "auth":
            raise AuthError("first message must be auth")
        claims = state.verifier.verify(str(first.get("token", "")))

        def load() -> Principal:
            with state.sessions() as s, s.begin():
                return principal_for(s, claims, state.settings.oidc_role_claim)

        return await asyncio.to_thread(load)
    except (AuthError, TimeoutError, ValueError, json.JSONDecodeError):
        await ws.close(code=4401)
        return None


async def _pump(ws: WebSocket, hub: LiveHub, sub: Subscriber, snapshot: dict[str, Any]) -> None:
    await ws.send_json({"type": "snapshot", **snapshot})
    hub.subs.add(sub)
    endpoint = "call" if sub.call_id else "live"
    WS_CONNECTIONS.labels(endpoint).inc()

    async def drain_client() -> None:  # detect disconnects; clients send nothing else
        while True:
            await ws.receive_text()

    reader = asyncio.create_task(drain_client())
    seq = 0
    try:
        while not reader.done():
            await asyncio.sleep(BATCH_S)
            items = sub.take()
            if items:
                seq += 1
                await ws.send_json({"type": "batch", "seq": seq, "items": items})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.subs.discard(sub)
        WS_CONNECTIONS.labels(endpoint).dec()
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await reader


def _live_snapshot(sessions: sessionmaker[Session], p: Principal) -> dict[str, Any]:
    with sessions() as s:
        calls = s.scalars(
            select(Call)
            .where(Call.status == "active", call_scope(p))
            .order_by(Call.started_at.desc())
            .limit(2000)
        ).all()
        ids = [c.id for c in calls]
        alerts = s.scalars(
            select(EventRow)
            .where(EventRow.call_id.in_(ids), EventRow.alert, event_scope(p))
            .limit(5000)
        ).all()
        return {
            "calls": [
                {
                    "call_id": str(c.id),
                    "started_at": c.started_at.isoformat(),
                    "queue": c.queue,
                    "agent_id": str(c.agent_id) if c.agent_id else None,
                    "mode": c.mode,
                }
                for c in calls
            ],
            "alerts": [_event_json(e) for e in alerts],
        }


def _event_json(e: EventRow) -> dict[str, Any]:
    from gale_api.repo import event_of

    return event_of(e).model_dump(mode="json")


def _call_snapshot(
    sessions: sessionmaker[Session], p: Principal, call_id: str
) -> dict[str, Any] | None:
    try:
        cid = uuid.UUID(call_id)
    except ValueError:
        return None
    with sessions() as s:
        call = s.scalar(select(Call).where(Call.id == cid, call_scope(p)))
        if call is None:
            return None
        segs = s.scalars(
            select(Segment).where(Segment.call_id == cid).order_by(Segment.ts_start)
        ).all()
        events = s.scalars(select(EventRow).where(EventRow.call_id == cid, event_scope(p))).all()
        return {
            "call_id": call_id,
            "status": call.status,
            "segments": [
                {
                    "segment_id": g.id,
                    "channel": g.channel,
                    "ts_start": g.ts_start,
                    "ts_end": g.ts_end,
                    "text_redacted": g.text_redacted,
                }
                for g in segs
            ],
            "events": [_event_json(e) for e in events],
        }


async def _open(
    ws: WebSocket,
    perm: str,
    make: Callable[[Principal], dict[str, Any] | None],
) -> tuple[Principal, dict[str, Any]] | None:
    await ws.accept()
    p = await _auth(ws)
    if p is None:
        return None
    if not p.can(perm):
        await ws.close(code=4403)
        return None
    snap = await asyncio.to_thread(make, p)
    if snap is None:
        await ws.close(code=4404)
        return None
    return p, snap


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    state = ws.app.state
    opened = await _open(ws, "live.read", lambda p: _live_snapshot(state.sessions, p))
    if opened is None:
        return
    p, snap = opened
    await _pump(ws, state.hub, Subscriber(p, LIVE_STREAMS), snap)


@router.websocket("/ws/calls/{call_id}")
async def ws_call(ws: WebSocket, call_id: str) -> None:
    state = ws.app.state
    opened = await _open(ws, "live_call.read", lambda p: _call_snapshot(state.sessions, p, call_id))
    if opened is None:
        return
    p, snap = opened
    ip = ws.client.host if ws.client else None
    await asyncio.to_thread(
        write_audit, state.sessions, p, "transcript.live", "call", call_id, ip, {}
    )
    await _pump(ws, state.hub, Subscriber(p, CALL_STREAMS, call_id=call_id), snap)
