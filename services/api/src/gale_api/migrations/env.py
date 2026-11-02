"""Alembic environment: URL comes from the programmatic config (``gale_api.db.alembic_config``)."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from gale_api.db import Base

config = context.config
target_metadata = Base.metadata


def run() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("sqlalchemy.url not set; use gale_api.db.migrate()")
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run()
