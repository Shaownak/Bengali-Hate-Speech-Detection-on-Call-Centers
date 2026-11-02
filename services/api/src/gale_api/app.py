"""GAL-E API (docs/04 §3): REST + WebSockets over Postgres, plus the bus → DB persister.

Background tasks in the lifespan: ``Persister`` (consumer group, when ``GALE_WORKERS``) and
``LiveHub`` (bus tail feeding the WebSockets). ``GALE_DB_AUTO_MIGRATE`` runs Alembic at start
(dev stack); in prod migrations run as a separate job.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.engine import Engine

from gale_api import SERVICE_NAME
from gale_api.audit import AuditMiddleware
from gale_api.auth import TokenVerifier
from gale_api.db import head_revision, make_engine, make_sessionmaker, migrate, schema_revision
from gale_api.persist import Persister
from gale_api.routes_admin import router as admin_router
from gale_api.routes_analytics import router as analytics_router
from gale_api.routes_calls import router as calls_router
from gale_api.routes_dev import router as dev_router
from gale_api.routes_me import router as me_router
from gale_api.ws import LiveHub
from gale_api.ws import router as ws_router
from galecore.bus import BusClient
from galecore.config import Settings, get_settings
from galecore.objectstore import ObjectStore, store_from_settings
from galecore.service import create_app


def build_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    bus: BusClient | None = None,
    store: ObjectStore | None = None,
) -> FastAPI:
    s = settings or get_settings()
    bus = bus or BusClient.from_settings(s)
    engine = engine or make_engine(s.db_url)
    sessions = make_sessionmaker(engine)
    hub = LiveHub(bus, sessions)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if s.db_auto_migrate:
            await asyncio.to_thread(migrate, s.db_url)
        stop = asyncio.Event()
        tasks = [asyncio.create_task(hub.run(stop))]
        if s.workers:
            tasks.append(asyncio.create_task(Persister(bus, sessions).run(stop)))
        try:
            yield
        finally:
            stop.set()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(t, 5)

    async def db_ok() -> bool:
        def ping() -> bool:
            with engine.connect() as c:
                return c.execute(text("SELECT 1")).scalar() == 1

        try:
            return await asyncio.to_thread(ping)
        except Exception:
            return False

    head = head_revision()

    async def schema_ok() -> bool:
        """Ready only once `alembic upgrade head` has run (the chart's migration Job)."""
        try:
            return await asyncio.to_thread(schema_revision, engine) == head
        except Exception:
            return False

    app = create_app(
        SERVICE_NAME,
        ready_checks={"bus": bus.ping, "db": db_ok, "schema": schema_ok},
        settings=s,
        lifespan=lifespan,
    )
    app.state.settings = s
    app.state.engine = engine
    app.state.sessions = sessions
    app.state.bus = bus
    app.state.store = store or store_from_settings(s)
    app.state.verifier = TokenVerifier(s)
    app.state.hub = hub

    app.add_middleware(AuditMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type"],
    )
    for r in (calls_router, analytics_router, admin_router, me_router, ws_router):
        app.include_router(r)
    if s.auth_mode == "dev":
        app.include_router(dev_router)
    return app
