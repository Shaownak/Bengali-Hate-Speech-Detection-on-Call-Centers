"""Run llm_judge: ``python -m gale_llm_judge``."""

from __future__ import annotations

from gale_llm_judge import SERVICE_NAME
from gale_llm_judge.app import build_app
from galecore.service import run


def main() -> None:
    run(SERVICE_NAME, build_app())


if __name__ == "__main__":
    main()
