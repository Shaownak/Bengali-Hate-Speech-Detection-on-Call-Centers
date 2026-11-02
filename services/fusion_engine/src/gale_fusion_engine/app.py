"""FastAPI app for fusion_engine: health/readiness/metrics, plus the streaming consumer (P4)
running in the lifespan (``FusionStream``; off with ``GALE_WORKERS=false``)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gale_fusion_engine import SERVICE_NAME
from gale_fusion_engine.stream import FusionStream
from galecore.bus import BusClient
from galecore.config import Settings, get_settings, shard_of
from galecore.service import create_app


def build_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    bus = BusClient.from_settings(s)
    stop = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = (
            asyncio.create_task(FusionStream(bus, s, shard=shard_of(s)).run(stop))
            if s.workers
            else None
        )
        try:
            yield
        finally:
            stop.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    return create_app(SERVICE_NAME, ready_checks={"bus": bus.ping}, settings=s, lifespan=lifespan)
