"""FastAPI app for ingest_gateway: health/metrics plus the P0 stub transports (Prompt 2).

POST   /v1/rtp/sessions             register a call's RTP SSRCs → {call_id, rtp_port}
DELETE /v1/rtp/sessions/{call_id}   end the call, flush, return stats
GET    /v1/sessions/{call_id}       live stats
WS     /v1/stream/{call_id}         {"event":"start",...} · binary s16le PCM · {"event":"stop"}
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from gale_ingest_gateway import SERVICE_NAME
from gale_ingest_gateway.gateway import (
    GatewayError,
    GatewaySettings,
    Registry,
    RtpSessionRequest,
    WsStart,
    start_rtp,
)
from gale_ingest_gateway.sessions import CountingSink, FrameSink, SessionStats
from galecore.bus import BusClient
from galecore.config import Settings, get_settings
from galecore.service import create_app


def _call[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    try:
        return fn(*args, **kwargs)
    except GatewayError as e:
        raise HTTPException(e.status, e.detail) from e


def build_app(
    settings: Settings | None = None,
    gateway: GatewaySettings | None = None,
    sink: FrameSink | None = None,
) -> FastAPI:
    s = settings or get_settings()
    bus = BusClient.from_settings(s)
    registry = Registry(gateway or GatewaySettings(), sink or CountingSink())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        transport = await start_rtp(registry)
        try:
            yield
        finally:
            transport.close()

    app = create_app(SERVICE_NAME, ready_checks={"bus": bus.ping}, settings=s, lifespan=lifespan)
    app.state.registry = registry

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
        carry = b""
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    carry = registry.on_ws_bytes(session, start, msg["bytes"], carry)
                elif msg.get("text") is not None:
                    stats = registry.close(call_id)
                    await ws.send_json({"event": "stats", **stats.model_dump(mode="json")})
                    await ws.close()
                    return
        except WebSocketDisconnect:
            pass
        if not session.closed:
            registry.close(call_id)

    return app
