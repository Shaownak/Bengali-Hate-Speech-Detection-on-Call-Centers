"""Run audio_worker: ``python -m gale_audio_worker``."""

from __future__ import annotations

from gale_audio_worker import SERVICE_NAME
from gale_audio_worker.app import build_app
from galecore.service import run


def main() -> None:
    run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
