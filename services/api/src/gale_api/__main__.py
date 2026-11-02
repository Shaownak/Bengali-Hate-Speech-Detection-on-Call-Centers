"""Run the API: ``python -m gale_api`` | ``migrate`` | ``downgrade [rev]`` | ``seed-dev`` |
``retention [--execute]``."""

from __future__ import annotations

import sys

from gale_api import SERVICE_NAME
from galecore.config import get_settings


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    s = get_settings()
    if args and args[0] == "migrate":
        from gale_api.db import migrate

        migrate(s.db_url)
        print("database at head")
    elif args and args[0] == "downgrade":
        from gale_api.db import downgrade

        downgrade(s.db_url, args[1] if len(args) > 1 else "base")
    elif args and args[0] == "retention":
        from gale_api.db import make_engine, make_sessionmaker, session_scope
        from gale_api.retention import load_policy, run_retention
        from galecore.objectstore import store_from_settings

        policy = load_policy(s.config_dir)
        with session_scope(make_sessionmaker(make_engine(s.db_url))) as db:
            rep = run_retention(db, store_from_settings(s), policy, execute="--execute" in args)
        # counts only: no call ids or content (CLAUDE.md rule 1)
        print(rep)
    elif args and args[0] == "seed-dev":
        from gale_api.db import make_engine, make_sessionmaker, migrate, session_scope
        from gale_api.seed import seed_dev

        if s.env not in ("dev", "test"):
            raise SystemExit("seed-dev only in GALE_ENV=dev/test")
        migrate(s.db_url)
        with session_scope(make_sessionmaker(make_engine(s.db_url))) as db:
            print(seed_dev(db, s))
    else:  # pragma: no cover - starts uvicorn
        from gale_api.app import build_app
        from galecore.service import run

        run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
