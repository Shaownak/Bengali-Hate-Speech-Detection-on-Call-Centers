"""PostgreSQL schema (docs/01 §4) as SQLAlchemy 2 models, engine/session helpers and migrations.

Deviations from docs/01 §4, recorded in ADR-016:
- segment/signal/event ids are the pipeline's string ids (stable, idempotent upserts), not uuids;
- ``segments`` and ``signals`` are range-partitioned by month on ``call_started_at`` (the call's
  start, so a re-delivered message always lands in the same partition);
  their PK is ``(id, call_started_at)``;
- ``segments.text_norm`` holds ``bangla.normalize(text_redacted)`` for trigram search;
- ``users.agent_id`` links an agent login to its ``agents`` row; ``reviews.kind`` separates QA
  verdicts from agent disputes; ``events.acked_*`` record acknowledgements;
- ``model_versions.research_only`` (ADR-012) and ``model_promotions`` (two-person approval).

Pilot additions (migration 0002, ADR-020): ``model_versions.shadow`` (a registered candidate not yet
promoted), ``pilot_reports`` (weekly report), ``pilot_feedback`` (supervisor usefulness rating),
``missed_events`` (false negatives found by QA), ``al_samples`` (items sent to annotation).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
PARTITIONED = ("segments", "signals")

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


TS = DateTime(timezone=True)


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200))
    supervisor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", use_alter=True), nullable=True
    )


class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[uuid.UUID] = _uuid_pk()
    employee_code: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str | None] = mapped_column(String(200))
    team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = _uuid_pk()
    oidc_sub: Mapped[str] = mapped_column(String(255), unique=True)
    email: Mapped[str | None] = mapped_column(String(320))
    role: Mapped[str] = mapped_column(String(32))
    team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    agent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id"))
    created_at: Mapped[datetime] = mapped_column(TS, default=now)


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (Index("ix_calls_agent_started", "agent_id", "started_at"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    external_call_id: Mapped[str | None] = mapped_column(String(255), index=True)
    started_at: Mapped[datetime] = mapped_column(TS, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(TS)
    duration_s: Mapped[float | None] = mapped_column(Float)
    queue: Mapped[str | None] = mapped_column(String(100), index=True)
    direction: Mapped[str | None] = mapped_column(String(16))
    agent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id"))
    customer_hash: Mapped[str | None] = mapped_column(String(128))
    recording_uri: Mapped[str | None] = mapped_column(Text)
    channel_layout: Mapped[str | None] = mapped_column(String(32))
    language_detected: Mapped[str | None] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(16), default="realtime")
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|ended
    created_at: Mapped[datetime] = mapped_column(TS, default=now)


class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = (
        Index("ix_segments_call_id", "call_id"),
        Index(
            "ix_segments_text_norm_trgm",
            "text_norm",
            postgresql_using="gin",
            postgresql_ops={"text_norm": "gin_trgm_ops"},
        ),
        {"postgresql_partition_by": "RANGE (call_started_at)"},
    )
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    call_started_at: Mapped[datetime] = mapped_column(TS, primary_key=True)
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    channel: Mapped[str] = mapped_column(String(16))
    ts_start: Mapped[float] = mapped_column(Float)
    ts_end: Mapped[float] = mapped_column(Float)
    text_redacted: Mapped[str] = mapped_column(Text)
    text_norm: Mapped[str] = mapped_column(Text)
    words_jsonb: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    avg_conf: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128))


class SignalRow(Base):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_call_id", "call_id"),
        {"postgresql_partition_by": "RANGE (call_started_at)"},
    )
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    call_started_at: Mapped[datetime] = mapped_column(TS, primary_key=True)
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    channel: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(32))
    category: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)
    ts_start: Mapped[float] = mapped_column(Float)
    ts_end: Mapped[float] = mapped_column(Float)
    evidence_jsonb: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    model_version: Mapped[str] = mapped_column(String(128))
    shadow: Mapped[bool] = mapped_column(Boolean, default=False)


class EventRow(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_category_created", "category", "created_at"),
        Index("ix_events_alert", "call_id", postgresql_where=text("alert")),
    )
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    speaker: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(32))
    severity: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    ts_start: Mapped[float] = mapped_column(Float)
    ts_end: Mapped[float] = mapped_column(Float)
    contributing_signal_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    explanation_bn: Mapped[str | None] = mapped_column(Text)
    explanation_en: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    alert: Mapped[bool] = mapped_column(Boolean, default=False)
    shadow: Mapped[bool] = mapped_column(Boolean, default=False)
    acked_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    acked_at: Mapped[datetime | None] = mapped_column(TS)
    created_at: Mapped[datetime] = mapped_column(TS, default=now)
    updated_at: Mapped[datetime] = mapped_column(TS, default=now)


class ReviewRow(Base):
    __tablename__ = "reviews"
    id: Mapped[uuid.UUID] = _uuid_pk()
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), index=True)
    reviewer_user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(16), default="verdict")
    verdict: Mapped[str | None] = mapped_column(String(32))
    corrected_category: Mapped[str | None] = mapped_column(String(32))
    corrected_severity: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=now)


class CallScore(Base):
    __tablename__ = "call_scores"
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"), primary_key=True)
    agent_score: Mapped[int | None] = mapped_column(Integer)
    customer_hostility: Mapped[int | None] = mapped_column(Integer)
    escalation_curve_jsonb: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    scorecard_jsonb: Mapped[dict[str, Any]] = mapped_column(JSONB)
    model_version: Mapped[str] = mapped_column(String(128))


class LexiconEntry(Base):
    __tablename__ = "lexicon_entries"
    id: Mapped[uuid.UUID] = _uuid_pk()
    term: Mapped[str] = mapped_column(Text)
    normalized: Mapped[str] = mapped_column(Text, index=True)
    variants: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    romanized: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    category: Mapped[str] = mapped_column(String(32))
    severity: Mapped[int] = mapped_column(Integer)
    context_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    dialect: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[datetime] = mapped_column(TS, default=now)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class RuleSet(Base):
    __tablename__ = "rules"
    __table_args__ = (UniqueConstraint("name", "version"),)
    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(100))
    yaml: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[datetime] = mapped_column(TS, default=now)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("component", "version"),)
    id: Mapped[uuid.UUID] = _uuid_pk()
    component: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(200))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    metrics_jsonb: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    deployed_at: Mapped[datetime | None] = mapped_column(TS)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    research_only: Mapped[bool] = mapped_column(Boolean, default=False)
    shadow: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))


class ModelPromotion(Base):
    __tablename__ = "model_promotions"
    id: Mapped[uuid.UUID] = _uuid_pk()
    component: Mapped[str] = mapped_column(String(32))
    model_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id"))
    requested_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    approved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=now)
    decided_at: Mapped[datetime | None] = mapped_column(TS)


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_ts", "ts"),)
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    action: Mapped[str] = mapped_column(String(64))
    object_type: Mapped[str] = mapped_column(String(32))
    object_id: Mapped[str | None] = mapped_column(String(255))
    ip: Mapped[str | None] = mapped_column(String(64))
    ts: Mapped[datetime] = mapped_column(TS, default=now, server_default=func.now())
    details_jsonb: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class PilotReport(Base):
    __tablename__ = "pilot_reports"
    id: Mapped[uuid.UUID] = _uuid_pk()
    week: Mapped[str] = mapped_column(String(8), unique=True)  # ISO week, e.g. 2026-W44
    stage: Mapped[str] = mapped_column(String(8))
    payload_jsonb: Mapped[dict[str, Any]] = mapped_column(JSONB)  # aggregates only, no text
    html_key: Mapped[str | None] = mapped_column(Text)  # object-store key of the HTML report
    created_at: Mapped[datetime] = mapped_column(TS, default=now)


class PilotFeedback(Base):
    __tablename__ = "pilot_feedback"
    __table_args__ = (
        UniqueConstraint("user_id", "week"),
        CheckConstraint("rating BETWEEN 1 AND 5", name="rating"),
    )
    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    week: Mapped[str] = mapped_column(String(8), index=True)
    rating: Mapped[int] = mapped_column(Integer)  # usefulness 1..5 (docs/06 §6 P-2)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=now)


class MissedEvent(Base):
    """A false negative found by QA: abuse in a call that GAL-E did not flag (docs/02 §6)."""

    __tablename__ = "missed_events"
    id: Mapped[uuid.UUID] = _uuid_pk()
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    reporter_user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    speaker: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(32))
    severity: Mapped[int] = mapped_column(Integer)
    ts_start: Mapped[float] = mapped_column(Float)
    ts_end: Mapped[float] = mapped_column(Float)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=now, index=True)


class AlSample(Base):
    """One item the active-learning sampler sent to annotation; ``ref`` dedupes across weeks."""

    __tablename__ = "al_samples"
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    ref: Mapped[str] = mapped_column(String(300), unique=True)
    week: Mapped[str] = mapped_column(String(8), index=True)
    bucket: Mapped[str] = mapped_column(String(32))
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"))
    event_id: Mapped[str | None] = mapped_column(String(255))
    pushed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(TS, default=now)


# --- engine, sessions, migrations ---------------------------------------------------------------


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as s, s.begin():
        yield s


def alembic_config(url: str) -> Any:
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def migrate(url: str, revision: str = "head") -> None:
    from alembic import command

    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    from alembic import command

    command.downgrade(alembic_config(url), revision)


def head_revision() -> str:
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(alembic_config("postgresql://unused")).get_current_head()
    if head is None:  # pragma: no cover - the repo always ships migrations
        raise RuntimeError("no migrations found")
    return head


def schema_revision(engine: Engine) -> str | None:
    with engine.connect() as c:
        try:
            return c.execute(text("SELECT version_num FROM alembic_version")).scalar()
        except Exception:
            return None


def month_bounds(d: date) -> tuple[date, date]:
    start = d.replace(day=1)
    nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start, nxt


def partition_name(table: str, when: datetime) -> tuple[str, date, date]:
    if table not in PARTITIONED:
        raise ValueError(table)
    start, end = month_bounds(when.astimezone(UTC).date())
    return f"{table}_p{start:%Y%m}", start, end


def partition_exists(conn: Connection, table: str, when: datetime) -> bool:
    name, _, _ = partition_name(table, when)
    return conn.execute(text("SELECT to_regclass(:n)"), {"n": name}).scalar() is not None


def ensure_partition(conn: Connection, table: str, when: datetime) -> str:
    """Create the monthly partition of ``table`` that holds ``when`` (idempotent)."""
    name, start, end = partition_name(table, when)
    conn.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF {table} "
            f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
        )
    )
    return name
