"""Initial schema (docs/01 §4, ADR-016).

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)
UUID = pg.UUID(as_uuid=True)


def _id() -> sa.Column:  # type: ignore[type-arg]
    return sa.Column("id", UUID, nullable=False)


def _fk(col: str, table: str, ref: str, *, nullable: bool = True) -> sa.Column:  # type: ignore[type-arg]
    col_type = sa.String(255) if ref == "events" else UUID
    return sa.Column(
        col,
        col_type,
        sa.ForeignKey(f"{ref}.id", name=f"fk_{table}_{col}_{ref}"),
        nullable=nullable,
    )


def has_trgm() -> bool:
    """pg_trgm ships with every production Postgres (contrib), but not with the embedded dev/test
    server (pgserver). Without it the trigram index is skipped; search still works, unindexed."""
    q = sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_trgm'")
    return op.get_bind().execute(q).first() is not None


def upgrade() -> None:
    trgm = has_trgm()
    if trgm:
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.create_table(
        "teams",
        _id(),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("supervisor_user_id", UUID, nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_teams"),
    )
    op.create_table(
        "agents",
        _id(),
        sa.Column("employee_code", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200)),
        _fk("team_id", "agents", "teams"),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_agents"),
        sa.UniqueConstraint("employee_code", name="uq_agents_employee_code"),
    )
    op.create_index("ix_agents_team_id", "agents", ["team_id"])
    op.create_table(
        "users",
        _id(),
        sa.Column("oidc_sub", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320)),
        sa.Column("role", sa.String(32), nullable=False),
        _fk("team_id", "users", "teams"),
        _fk("agent_id", "users", "agents"),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("oidc_sub", name="uq_users_oidc_sub"),
    )
    op.create_foreign_key(
        "fk_teams_supervisor_user_id_users", "teams", "users", ["supervisor_user_id"], ["id"]
    )
    op.create_table(
        "calls",
        sa.Column("id", UUID, nullable=False),
        sa.Column("external_call_id", sa.String(255)),
        sa.Column("started_at", TS, nullable=False),
        sa.Column("ended_at", TS),
        sa.Column("duration_s", sa.Float),
        sa.Column("queue", sa.String(100)),
        sa.Column("direction", sa.String(16)),
        _fk("agent_id", "calls", "agents"),
        sa.Column("customer_hash", sa.String(128)),
        sa.Column("recording_uri", sa.Text),
        sa.Column("channel_layout", sa.String(32)),
        sa.Column("language_detected", sa.String(16)),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_calls"),
    )
    op.create_index("ix_calls_agent_started", "calls", ["agent_id", "started_at"])
    op.create_index("ix_calls_external_call_id", "calls", ["external_call_id"])
    op.create_index("ix_calls_started_at", "calls", ["started_at"])
    op.create_index("ix_calls_queue", "calls", ["queue"])

    # segments / signals: monthly range partitions on call_started_at (created on demand)
    op.create_table(
        "segments",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("call_started_at", TS, nullable=False),
        sa.Column("call_id", UUID, nullable=False),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("ts_start", sa.Float, nullable=False),
        sa.Column("ts_end", sa.Float, nullable=False),
        sa.Column("text_redacted", sa.Text, nullable=False),
        sa.Column("text_norm", sa.Text, nullable=False),
        sa.Column("words_jsonb", pg.JSONB, nullable=False),
        sa.Column("avg_conf", sa.Float, nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.PrimaryKeyConstraint("id", "call_started_at", name="pk_segments"),
        postgresql_partition_by="RANGE (call_started_at)",
    )
    op.create_index("ix_segments_call_id", "segments", ["call_id"])
    if trgm:
        op.create_index(
            "ix_segments_text_norm_trgm",
            "segments",
            ["text_norm"],
            postgresql_using="gin",
            postgresql_ops={"text_norm": "gin_trgm_ops"},
        )
    op.create_table(
        "signals",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("call_started_at", TS, nullable=False),
        sa.Column("call_id", UUID, nullable=False),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("ts_start", sa.Float, nullable=False),
        sa.Column("ts_end", sa.Float, nullable=False),
        sa.Column("evidence_jsonb", pg.JSONB, nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("shadow", sa.Boolean, nullable=False),
        sa.PrimaryKeyConstraint("id", "call_started_at", name="pk_signals"),
        postgresql_partition_by="RANGE (call_started_at)",
    )
    op.create_index("ix_signals_call_id", "signals", ["call_id"])

    op.create_table(
        "events",
        sa.Column("id", sa.String(255), nullable=False),
        _fk("call_id", "events", "calls", nullable=False),
        sa.Column("speaker", sa.String(16), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("severity", sa.Integer, nullable=False),
        sa.Column("probability", sa.Float, nullable=False),
        sa.Column("ts_start", sa.Float, nullable=False),
        sa.Column("ts_end", sa.Float, nullable=False),
        sa.Column("contributing_signal_ids", pg.ARRAY(sa.Text), nullable=False),
        sa.Column("explanation_bn", sa.Text),
        sa.Column("explanation_en", sa.Text),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("alert", sa.Boolean, nullable=False),
        sa.Column("shadow", sa.Boolean, nullable=False),
        _fk("acked_by", "events", "users"),
        sa.Column("acked_at", TS),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
    )
    op.create_index("ix_events_call_id", "events", ["call_id"])
    op.create_index("ix_events_category_created", "events", ["category", "created_at"])
    op.create_index("ix_events_alert", "events", ["call_id"], postgresql_where=sa.text("alert"))

    op.create_table(
        "reviews",
        _id(),
        _fk("event_id", "reviews", "events", nullable=False),
        _fk("reviewer_user_id", "reviews", "users", nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("verdict", sa.String(32)),
        sa.Column("corrected_category", sa.String(32)),
        sa.Column("corrected_severity", sa.Integer),
        sa.Column("comment", sa.Text),
        sa.Column("created_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_reviews"),
    )
    op.create_index("ix_reviews_event_id", "reviews", ["event_id"])
    op.create_table(
        "call_scores",
        _fk("call_id", "call_scores", "calls", nullable=False),
        sa.Column("agent_score", sa.Integer),
        sa.Column("customer_hostility", sa.Integer),
        sa.Column("escalation_curve_jsonb", pg.JSONB, nullable=False),
        sa.Column("scorecard_jsonb", pg.JSONB, nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.PrimaryKeyConstraint("call_id", name="pk_call_scores"),
    )
    op.create_table(
        "lexicon_entries",
        _id(),
        sa.Column("term", sa.Text, nullable=False),
        sa.Column("normalized", sa.Text, nullable=False),
        sa.Column("variants", pg.ARRAY(sa.Text), nullable=False),
        sa.Column("romanized", pg.ARRAY(sa.Text), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("severity", sa.Integer, nullable=False),
        sa.Column("context_sensitive", sa.Boolean, nullable=False),
        sa.Column("dialect", sa.String(64)),
        sa.Column("notes", sa.Text),
        _fk("created_by", "lexicon_entries", "users"),
        sa.Column("updated_at", TS, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_lexicon_entries"),
    )
    op.create_index("ix_lexicon_entries_normalized", "lexicon_entries", ["normalized"])
    op.create_table(
        "rules",
        _id(),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("yaml", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        _fk("updated_by", "rules", "users"),
        sa.Column("updated_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_rules"),
        sa.UniqueConstraint("name", "version", name="uq_rules_name"),
    )
    op.create_table(
        "model_versions",
        _id(),
        sa.Column("component", sa.String(32), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("version", sa.String(200), nullable=False),
        sa.Column("mlflow_run_id", sa.String(64)),
        sa.Column("metrics_jsonb", pg.JSONB, nullable=False),
        sa.Column("deployed_at", TS),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("research_only", sa.Boolean, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
        sa.UniqueConstraint("component", "version", name="uq_model_versions_component"),
    )
    op.create_table(
        "model_promotions",
        _id(),
        sa.Column("component", sa.String(32), nullable=False),
        _fk("model_version_id", "model_promotions", "model_versions", nullable=False),
        _fk("requested_by", "model_promotions", "users", nullable=False),
        _fk("approved_by", "model_promotions", "users"),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("decided_at", TS),
        sa.PrimaryKeyConstraint("id", name="pk_model_promotions"),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, sa.Identity(), nullable=False),
        sa.Column("user_id", UUID),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("object_type", sa.String(32), nullable=False),
        sa.Column("object_id", sa.String(255)),
        sa.Column("ip", sa.String(64)),
        sa.Column("ts", TS, server_default=sa.func.now(), nullable=False),
        sa.Column("details_jsonb", pg.JSONB, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_ts", "audit_log", ["ts"])


def downgrade() -> None:
    for t in (
        "audit_log",
        "model_promotions",
        "model_versions",
        "rules",
        "lexicon_entries",
        "call_scores",
        "reviews",
        "events",
        "signals",  # partitions are dropped with their parent
        "segments",
    ):
        op.drop_table(t)
    op.drop_constraint("fk_teams_supervisor_user_id_users", "teams", type_="foreignkey")
    for t in ("calls", "users", "agents", "teams"):
        op.drop_table(t)
