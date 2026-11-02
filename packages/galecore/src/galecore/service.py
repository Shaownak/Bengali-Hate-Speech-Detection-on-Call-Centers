"""Shared service scaffolding: FastAPI app with health, readiness and Prometheus metrics."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from galecore.config import Settings, get_settings
from galecore.logging import configure_logging

ReadyCheck = Callable[[], Awaitable[bool]]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]

# Default local ports (docs/04 §3 API on 8000; workers expose health/metrics only)
SERVICE_PORTS: dict[str, int] = {
    "api": 8000,
    "ingest_gateway": 8001,
    "audio_worker": 8002,
    "asr_worker": 8003,
    "text_worker": 8004,
    "llm_judge": 8005,
    "fusion_engine": 8006,
}


def create_app(
    name: str,
    *,
    version: str = "0.1.0",
    ready_checks: dict[str, ReadyCheck] | None = None,
    settings: Settings | None = None,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    s = settings or get_settings()
    configure_logging(name, s.log_level, s.log_json)
    app = FastAPI(title=f"gal-e {name}", version=version, lifespan=lifespan)
    checks = ready_checks or {}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": name, "version": version, "model_mode": s.model_mode}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        results = {k: await fn() for k, fn in checks.items()}
        ok = all(results.values())
        return JSONResponse(
            {"status": "ready" if ok else "not_ready", "checks": results},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def run(name: str, app: FastAPI, settings: Settings | None = None) -> None:  # pragma: no cover
    import os

    import uvicorn

    s = settings or get_settings()
    port = s.port if "GALE_PORT" in os.environ else SERVICE_PORTS[name]
    uvicorn.run(app, host=s.host, port=port, log_config=None, access_log=False)
