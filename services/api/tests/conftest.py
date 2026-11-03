"""API test fixtures: a real Postgres (embedded pgserver, or ``GALE_TEST_PG_URL``), one migrated
template database cloned per test, the dev seed, and two synthetic calls in different teams.

No real call data: transcripts are placeholder text (CLAUDE.md hard rule 2).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from gale_api import repo
from gale_api.app import build_app
from gale_api.auth import Role, issue_dev_token
from gale_api.db import make_engine, make_sessionmaker, migrate, session_scope
from gale_api.seed import USERS, seed_dev
from galecore.bus import BusClient
from galecore.config import REPO_ROOT, Settings
from galecore.schemas import (
    AsrSegment,
    CallLifecycle,
    CallScorecard,
    Channel,
    Evidence,
    FusedEvent,
    Signal,
)

PGDATA = REPO_ROOT / ".gale" / "tmp" / "pgtest"


@pytest.fixture(scope="session")
def pg_admin_url() -> Iterator[str]:
    url = os.environ.get("GALE_TEST_PG_URL")  # e.g. postgresql+psycopg://u:p@host:5432/postgres
    if url:
        yield url
        return
    import pgserver

    srv = pgserver.get_server(str(PGDATA), cleanup_mode="stop")
    yield srv.get_uri().replace("postgresql://", "postgresql+psycopg://")


def _admin(url: str) -> Engine:
    return create_engine(url, isolation_level="AUTOCOMMIT")


def _with_db(url: str, name: str) -> str:
    # parsed, not split on "/": on Linux pgserver's URL is a unix socket (``?host=/run/user/…``)
    return make_url(url).set(database=name).render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def template_db(pg_admin_url: str) -> Iterator[str]:
    name = f"gale_tpl_{os.getpid()}"
    eng = _admin(pg_admin_url)
    with eng.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        c.execute(text(f'CREATE DATABASE "{name}"'))
    url = _with_db(pg_admin_url, name)
    migrate(url)
    e = make_engine(url)
    with session_scope(make_sessionmaker(e)) as s:
        seed_dev(s, Settings(env="test"))
    e.dispose()
    yield name
    with eng.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    eng.dispose()


@pytest.fixture
def db_url(pg_admin_url: str, template_db: str) -> Iterator[str]:
    name = f"gale_t_{uuid.uuid4().hex[:10]}"
    eng = _admin(pg_admin_url)
    with eng.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"'))
    yield _with_db(pg_admin_url, name)
    with eng.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    eng.dispose()


@dataclass
class World:
    app: FastAPI
    client: TestClient
    settings: Settings
    engine: Engine
    server: FakeServer
    call_a: str  # AG-001, Team Dhaka
    call_b: str  # AG-003, Team Chattogram
    ev_agent_a: str  # agent alert on call A
    ev_customer_a: str  # customer event on call A
    ev_agent_b: str

    def token(self, sub: str) -> str:
        return issue_dev_token(self.settings, sub, Role(USERS[sub][0]))

    def h(self, sub: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(sub)}"}

    def bus(self) -> BusClient:
        return BusClient(FakeAsyncRedis(server=self.server))


def _seg(call_id: str, n: int, ch: Channel, t: float, txt: str) -> AsrSegment:
    return AsrSegment(
        call_id=call_id,
        channel=ch,
        segment_id=f"{call_id}:seg{n}",
        ts_start=t,
        ts_end=t + 2.0,
        text_redacted=txt,
        is_final=True,
        avg_conf=0.9,
        model_version="stub",
    )


def _sig(call_id: str, sid: str, src: str, cat: str, ch: Channel, t: float, s: float) -> Signal:
    return Signal(
        call_id=call_id,
        channel=ch,
        signal_id=f"{call_id}:{sid}",
        source=src,  # type: ignore[arg-type]
        category=cat,  # type: ignore[arg-type]
        score=s,
        ts_start=t,
        ts_end=t + 1.0,
        evidence=Evidence(text_span="<GALI_1>" if src == "lexicon" else None),
        model_version="stub",
    )


def _ev(
    call_id: str, eid: str, speaker: str, cat: str, t: float, alert: bool, sev: int
) -> FusedEvent:
    return FusedEvent(
        event_id=f"{call_id}:{eid}",
        call_id=call_id,
        speaker=speaker,  # type: ignore[arg-type]
        category=cat,  # type: ignore[arg-type]
        severity=sev,  # type: ignore[arg-type]
        probability=0.9,
        ts_start=t,
        ts_end=t + 1.0,
        contributing_signals=[f"{call_id}:s1", f"{call_id}:s2"],
        explanation_bn="প্রমাণ",
        status="confirmed",  # type: ignore[arg-type]
        alert=alert,
    )


SCORECARD: dict[str, Any] = {
    "agent": {
        "professionalism": 2,
        "empathy": 3,
        "de_escalation": 2,
        "respectful_register_maintained": False,
        "violations": ["insult"],
    },
    "customer": {"hostility": 4, "abuse_toward_agent": True, "main_issue_bn": "বিল"},
    "escalation": {
        "escalated": True,
        "who_escalated_first": "customer",
        "turning_point_utterance_id": None,
    },
    "summary_bn": "সারাংশ",
    "summary_en": "summary",
    "coaching_points_bn": ["শান্ত থাকুন"],
}


def load_call(engine: Engine, call_id: str, agent: str, queue: str, word: str) -> tuple[str, str]:
    parts = repo.Partitions()
    with session_scope(make_sessionmaker(engine)) as s:
        repo.apply_lifecycle(
            s,
            CallLifecycle(
                call_id=call_id,
                event="started",
                ts=0.0,
                agent_id=agent,
                queue=queue,
                mode="realtime",
                wall_clock=1_790_000_000.0,
            ),
        )
        repo.upsert_segment(s, parts, _seg(call_id, 1, Channel.CUSTOMER, 1.0, "আমার বিল ঠিক নেই"))
        repo.upsert_segment(s, parts, _seg(call_id, 2, Channel.AGENT, 4.0, f"তুমি {word} <GALI_1>"))
        repo.upsert_signal(
            s, parts, _sig(call_id, "s1", "lexicon", "slang_profanity", Channel.AGENT, 4.5, 0.95)
        )
        repo.upsert_signal(
            s, parts, _sig(call_id, "s2", "dsp_shout", "shouting", Channel.AGENT, 4.0, 0.7)
        )
        repo.upsert_signal(
            s, parts, _sig(call_id, "s3", "ser", "harsh_tone", Channel.CUSTOMER, 1.0, 0.6)
        )
        ea = _ev(call_id, "e1", "agent", "slang_profanity", 4.5, alert=True, sev=3)
        ec = _ev(call_id, "e2", "customer", "harsh_tone", 1.0, alert=False, sev=2)
        repo.upsert_event(s, ea)
        repo.upsert_event(s, ec)
        repo.upsert_scorecard(s, call_id, CallScorecard.model_validate(SCORECARD), "stub")
        repo.apply_lifecycle(s, CallLifecycle(call_id=call_id, event="ended", ts=30.0))
    return ea.event_id, ec.event_id


def make_settings(db_url: str, tmp: Path, **kw: Any) -> Settings:
    return Settings(
        env="test",
        bus_backend="memory",
        auth_mode="dev",
        workers=False,
        db_url=db_url,
        object_store_dir=tmp / "objects",
        object_url_secret="test-secret",
        log_level="WARNING",
        **kw,
    )


@pytest.fixture
def world(db_url: str, tmp_path: Path) -> Iterator[World]:
    settings = make_settings(db_url, tmp_path)
    engine = make_engine(db_url)
    server = FakeServer()
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    ea, ec = load_call(engine, a, "AG-001", "billing", "বাজে")
    eb, _ = load_call(engine, b, "AG-003", "support", "খারাপ")
    app = build_app(settings, engine=engine, bus=BusClient(FakeAsyncRedis(server=server)))
    with TestClient(app) as client:
        yield World(app, client, settings, engine, server, a, b, ea, ec, eb)
    engine.dispose()
