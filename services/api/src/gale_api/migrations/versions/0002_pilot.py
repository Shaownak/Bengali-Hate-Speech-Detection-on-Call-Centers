"""Pilot & feedback loop (ADR-020): candidate flag on model versions, weekly reports, usefulness
ratings, QA-found missed events, active-learning samples.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)
UUID = pg.UUID(as_uuid=True)


def _fk(col: str, table: str, ref: str) -> sa.Column:  # type: ignore[type-arg]
    return sa.Column(
        col, UUID, sa.ForeignKey(f"{ref}.id", name=f"fk_{table}_{col}_{ref}"), nullable=False
    )


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("shadow", sa.Boolean, server_default=sa.text("false"), nullable=False),
    )
    op.create_table(
        "pilot_reports",
        sa.Column("id", UUID, nullable=False),
        sa.Column("week", sa.String(8), nullable=False),
        sa.Column("stage", sa.String(8), nullable=False),
        sa.Column("payload_jsonb", pg.JSONB, nullable=False),
        sa.Column("html_key", sa.Text),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_pilot_reports"),
        sa.UniqueConstraint("week", name="uq_pilot_reports_week"),
    )
    op.create_table(
        "pilot_feedback",
        sa.Column("id", UUID, nullable=False),
        _fk("user_id", "pilot_feedback", "users"),
        sa.Column("week", sa.String(8), nullable=False),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("comment", sa.Text),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_pilot_feedback"),
        sa.UniqueConstraint("user_id", "week", name="uq_pilot_feedback_user_id"),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_pilot_feedback_rating"),
    )
    op.create_index("ix_pilot_feedback_week", "pilot_feedback", ["week"])
    op.create_table(
        "missed_events",
        sa.Column("id", UUID, nullable=False),
        _fk("call_id", "missed_events", "calls"),
        _fk("reporter_user_id", "missed_events", "users"),
        sa.Column("speaker", sa.String(16), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("severity", sa.Integer, nullable=False),
        sa.Column("ts_start", sa.Float, nullable=False),
        sa.Column("ts_end", sa.Float, nullable=False),
        sa.Column("comment", sa.Text),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_missed_events"),
    )
    op.create_index("ix_missed_events_call_id", "missed_events", ["call_id"])
    op.create_index("ix_missed_events_created_at", "missed_events", ["created_at"])
    op.create_table(
        "al_samples",
        sa.Column("id", sa.BigInteger, sa.Identity(), nullable=False),
        sa.Column("ref", sa.String(300), nullable=False),
        sa.Column("week", sa.String(8), nullable=False),
        sa.Column("bucket", sa.String(32), nullable=False),
        _fk("call_id", "al_samples", "calls"),
        sa.Column("event_id", sa.String(255)),
        sa.Column("pushed", sa.Boolean, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_al_samples"),
        sa.UniqueConstraint("ref", name="uq_al_samples_ref"),
    )
    op.create_index("ix_al_samples_week", "al_samples", ["week"])


def downgrade() -> None:
    for t in ("al_samples", "missed_events", "pilot_feedback", "pilot_reports"):
        op.drop_table(t)
    op.drop_column("model_versions", "shadow")
