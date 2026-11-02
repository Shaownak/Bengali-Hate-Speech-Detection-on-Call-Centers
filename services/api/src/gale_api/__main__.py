"""Run api: ``python -m gale_api``."""

from __future__ import annotations

from gale_api import SERVICE_NAME
from gale_api.app import build_app
from galecore.service import run


def main() -> None:
    run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
