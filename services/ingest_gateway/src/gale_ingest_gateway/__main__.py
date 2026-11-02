"""Run ingest_gateway: ``python -m gale_ingest_gateway``."""

from __future__ import annotations

from gale_ingest_gateway import SERVICE_NAME
from gale_ingest_gateway.app import build_app
from galecore.service import run


def main() -> None:
    run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
