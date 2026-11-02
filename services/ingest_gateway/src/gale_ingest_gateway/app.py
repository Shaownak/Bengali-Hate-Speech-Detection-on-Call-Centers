"""FastAPI app for ingest_gateway (P0 transports + P4 streaming).

POST   /v1/rtp/sessions             register RTP streams (SSRC or source addr) → {call_id, rtp_port}
DELETE /v1/rtp/sessions/{call_id}   end the call, flush, return stats
GET    /v1/sessions/{call_id}       live stats
WS     /v1/stream/{call_id}         {"event":"start",...} · binary s16le PCM · {"event":"stop"}
WS     /v1/fork                     FreeSWITCH mod_audio_fork: metadata JSON, then binary L16

Every session publishes ``call.lifecycle`` + ``audio.frames.{shard}`` (``publisher.BusPublisher``)
and its audio is kept in the object store at hang-up. With ``GALE_INGEST_DROP_DIR`` set, recordings
dropped in that folder become ``postcall.jobs`` (file mode).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from gale_ingest_gateway import SERVICE_NAME
from gale_ingest_gateway.gateway import (
    CallEvents,
    GatewayError,
    GatewaySettings,
    Registry,
    RtpSessionRequest,
    WsStart,
    start_rtp,
)
from gale_ingest_gateway.publisher import BusPublisher
from gale_ingest_gateway.sessions import FrameSink, Session, SessionStats
from galecore.bus import POSTCALL_JOBS, BusClient
from galecore.config import Settings, get_settings
from galecore.logging import get_logger
from galecore.objectstore import LocalObjectStore
from galecore.schemas import PostcallJob
from galecore.service import create_app

log = get_logger(__name__)


def _call[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    try:
        return fn(*args, **kwargs)
    except GatewayError as e:
        raise HTTPException(e.status, e.detail) from e


def _claim_dropped(drop: Path) -> list[PostcallJob]:
    """Move each complete ``<name>.wav`` (+ ``.json``) to ``processed/``; describe it as a job."""
    done = drop.parent / "processed"
    jobs = []
    for wav in sorted(drop.glob("*.wav")) if drop.is_dir() else []:
        side = wav.with_suffix(".json")
        meta = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
        call_id = str(uuid.uuid4())  # D-18
        done.mkdir(parents=True, exist_ok=True)
        dest = done / f"{call_id}.wav"
        wav.replace(dest)
        if side.is_file():
            side.replace(dest.with_suffix(".json"))
        ref = meta.get("external_ref") or (
            f"fixture:{Path(meta['script']).parent}" if meta.get("script") else None
        )
        jobs.append(
            PostcallJob(call_id=call_id, uri=str(dest), external_ref=ref, queue=meta.get("queue"))
        )
    return jobs


async def watch_drop_dir(bus: BusClient, drop: Path, stop: asyncio.Event, poll_s: float) -> None:
    """File mode: ``<name>.wav`` (+ optional ``<name>.json``) → ``postcall.jobs``; files move to
    ``processed/`` so each is queued once. The simulator writes the WAV last (atomic rename)."""
    while not stop.is_set():
        for job in await asyncio.to_thread(_claim_dropped, drop):
            await bus.publish(POSTCALL_JOBS, job, idem_key=job.call_id)
            log.info("postcall_job_queued", call_id=job.call_id)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_s)


async def _pump(ws: WebSocket, registry: Registry, session: Session, start: WsStart) -> None:
    carry = b""
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                carry = registry.on_ws_bytes(session, start, msg["bytes"], carry)
            elif msg.get("text") is not None:
                stats = registry.close(session.call_id)
                await ws.send_json({"event": "stats", **stats.model_dump(mode="json")})
                await ws.close()
                return
    except WebSocketDisconnect:
        pass
    if not session.closed:
        registry.close(session.call_id)


def build_app(
    settings: Settings | None = None,
    gateway: GatewaySettings | None = None,
    sink: FrameSink | None = None,
    events: CallEvents | None = None,
) -> FastAPI:
    s = settings or get_settings()
    gs = gateway or GatewaySettings()
    bus = BusClient.from_settings(s)
    publisher = BusPublisher(
        bus, LocalObjectStore(s.object_store_dir) if s.object_store == "local" else None
    )
    registry = Registry(gs, sink or publisher.frame, events if events is not None else publisher)
    stop = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        transport = await start_rtp(registry)
        tasks = [asyncio.create_task(publisher.run(stop))]
        if gs.drop_dir:
            tasks.append(
                asyncio.create_task(watch_drop_dir(bus, Path(gs.drop_dir), stop, gs.drop_poll_s))
            )
        if gs.ari_url:  # pragma: no cover - needs an Asterisk
            from gale_ingest_gateway.ari import AriApp

            ari = AriApp(
                registry,
                gs.ari_url,
                gs.ari_user,
                gs.ari_password,
                app=gs.ari_app,
                rtp_host=gs.ari_rtp_host,
            )
            tasks.append(asyncio.create_task(ari.run(stop)))
        try:
            yield
        finally:
            stop.set()
            transport.close()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = create_app(SERVICE_NAME, ready_checks={"bus": bus.ping}, settings=s, lifespan=lifespan)
    app.state.registry = registry
    app.state.publisher = publisher

    @app.post("/v1/rtp/sessions", status_code=201)
    async def open_rtp(req: RtpSessionRequest) -> dict[str, object]:
        _call(registry.open_rtp, req)
        return {"call_id": req.call_id, "rtp_port": registry.rtp_port}

    @app.delete("/v1/rtp/sessions/{call_id}")
    async def close_rtp(call_id: str) -> SessionStats:
        return _call(registry.close, call_id)

    @app.get("/v1/sessions/{call_id}")
    async def session_stats(call_id: str) -> SessionStats:
        return _call(registry.get, call_id).stats()

    @app.websocket("/v1/stream/{call_id}")
    async def stream(ws: WebSocket, call_id: str) -> None:
        await ws.accept()
        try:
            start = WsStart.model_validate_json(await ws.receive_text())
            session = registry.open_ws(call_id, start)
        except (ValidationError, GatewayError) as e:
            await ws.send_json(
                {"event": "error", "detail": getattr(e, "detail", "invalid start message")}
            )
            await ws.close(code=1008)
            return
        await _pump(ws, registry, session, start)

    @app.websocket("/v1/fork")
    async def fork(ws: WebSocket) -> None:
        """FreeSWITCH ``uuid_audio_fork <uuid> start ws://gw/v1/fork stereo 16k <metadata>``: the
        first text frame is our metadata JSON (call id, external ref, which side is the agent)."""
        await ws.accept()
        try:
            meta = json.loads(await ws.receive_text())
            start = WsStart(
                event="start",
                sample_rate=int(meta.get("sample_rate", 16000)),
                channels=int(meta.get("channels", 2)),
                layout="stereo_agent_right"
                if meta.get("agent_channel") == "right"
                else ("mono" if int(meta.get("channels", 2)) == 1 else "stereo_agent_left"),
                external_ref=meta.get("external_ref") or meta.get("fs_uuid"),
                agent_id=meta.get("agent_id"),
                queue=meta.get("queue"),
            )
            call_id = str(meta.get("call_id") or uuid.uuid4())
            session = registry.open_ws(call_id, start)
        except (ValidationError, GatewayError, ValueError, TypeError) as e:
            await ws.send_json(
                {"event": "error", "detail": getattr(e, "detail", "invalid metadata")}
            )
            await ws.close(code=1008)
            return
        await ws.send_json({"event": "accepted", "call_id": call_id})
        await _pump(ws, registry, session, start)

    return app
