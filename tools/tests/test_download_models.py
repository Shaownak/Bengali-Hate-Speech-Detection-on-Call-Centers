import re

import pytest

from scripts.download_models import LICENSES_MD, PINS, licenses_markdown, main


def test_every_model_is_pinned_by_commit_hash() -> None:
    assert len({p.key for p in PINS}) == len(PINS)
    for p in PINS:
        assert re.fullmatch(r"[0-9a-f]{40}", p.revision), p.key  # CLAUDE.md rule 6
        assert p.licence_status in {"commercial_ok", "verify", "research_only"}


def test_committed_licence_table_is_current() -> None:
    assert LICENSES_MD.read_text(encoding="utf-8") == licenses_markdown()


def test_cli_guards(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    assert "banglabert" in capsys.readouterr().out
    assert main(["nope"]) == 2
    assert main(["banglabert"]) == 3  # research_only needs an explicit opt-in
