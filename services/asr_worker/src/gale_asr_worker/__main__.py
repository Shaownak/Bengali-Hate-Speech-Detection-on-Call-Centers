"""Run asr_worker: ``python -m gale_asr_worker``."""

from __future__ import annotations

from gale_asr_worker import SERVICE_NAME
from gale_asr_worker.app import build_app
from galecore.service import run


def main() -> None:
    run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
