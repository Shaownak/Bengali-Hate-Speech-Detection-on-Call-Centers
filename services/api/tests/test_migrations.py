"""Alembic up → down → up on a real Postgres, schema == ORM, monthly partitions."""

from __future__ import annotations

from datetime import UTC, datetime

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from gale_api.db import Base, downgrade, ensure_partition, migrate, partition_exists


def _diff(url: str) -> list[object]:
    eng = create_engine(url)
    with eng.connect() as c:
        has_trgm = c.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name='pg_trgm'")
        ).first()
        diff = compare_metadata(MigrationContext.configure(c), Base.metadata)
    eng.dispose()
    if not has_trgm:  # embedded dev Postgres lacks contrib: the trigram index is skipped
        diff = [d for d in diff if "ix_segments_text_norm_trgm" not in str(d)]
    return diff


def test_migration_up_down_up(db_url: str) -> None:
    # db_url is a clone of the migrated template
    assert _diff(db_url) == []
    downgrade(db_url)
    eng = create_engine(db_url)
    assert set(inspect(eng).get_table_names()) == {"alembic_version"}
    eng.dispose()
    migrate(db_url)
    assert _diff(db_url) == []


def test_monthly_partitions(db_url: str) -> None:
    eng = create_engine(db_url)
    when = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
    with eng.begin() as c:
        assert not partition_exists(c, "segments", when)
        assert ensure_partition(c, "segments", when) == "segments_p202612"
        assert ensure_partition(c, "segments", when) == "segments_p202612"  # idempotent
        assert partition_exists(c, "segments", when)
        bounds = c.execute(
            text("SELECT pg_get_expr(relpartbound, oid) FROM pg_class WHERE relname = :n"),
            {"n": "segments_p202612"},
        ).scalar()
        assert "2026-12-01" in bounds
        assert "2027-01-01" in bounds
    eng.dispose()
