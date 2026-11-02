"""FastAPI app for ingest_gateway: health, readiness and metrics (P0 skeleton)."""

from __future__ import annotations

from fastapi import FastAPI

from gale_ingest_gateway import SERVICE_NAME
from galecore.bus import BusClient
from galecore.config import Settings, get_settings
from galecore.service import create_app


def build_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    bus = BusClient.from_settings(s)
    return create_app(SERVICE_NAME, ready_checks={"bus": bus.ping}, settings=s)
