"""Run fusion_engine: ``python -m gale_fusion_engine``."""

from __future__ import annotations

from gale_fusion_engine import SERVICE_NAME
from gale_fusion_engine.app import build_app
from galecore.service import run


def main() -> None:
    run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
