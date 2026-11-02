from pathlib import Path

import pytest
from pydantic import ValidationError

from galecore.config import (
    REPO_ROOT,
    Settings,
    load_canonical,
    load_lexicon,
    load_models,
    load_rules,
    load_thresholds,
)
from galecore.schemas import Category, Speaker

CONFIGS = REPO_ROOT / "configs"


def test_repo_root_points_at_repo() -> None:
    assert (REPO_ROOT / "CLAUDE.md").exists()


def test_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GALE_MODEL_MODE", "real")
    monkeypatch.setenv("GALE_PORT", "9999")
    s = Settings()
    assert s.model_mode == "real"
    assert s.port == 9999


def test_dev_auth_forbidden_outside_dev() -> None:
    Settings(env="dev", auth_mode="dev")
    Settings(env="test", auth_mode="dev")
    with pytest.raises(ValidationError, match="GALE_AUTH_MODE=dev"):
        Settings(env="staging", auth_mode="dev")


def test_stub_models_forbidden_in_prod() -> None:
    with pytest.raises(ValidationError, match="stub"):
        Settings(env="prod", model_mode="stub")


def test_thresholds_cover_every_category() -> None:
    t = load_thresholds(CONFIGS)
    assert set(t.categories) == set(Category)
    slang = t.categories[Category.SLANG_PROFANITY]
    assert slang.alert_realtime.for_speaker(Speaker.AGENT) <= slang.alert_realtime.customer
    assert t.hysteresis.cooldown_s == 20


def test_thresholds_missing_category_rejected(tmp_path: Path) -> None:
    text = (CONFIGS / "thresholds.yaml").read_text(encoding="utf-8")
    cut = text.split("  interruption_overtalk:")[0]
    (tmp_path / "thresholds.yaml").write_text(cut, encoding="utf-8")
    with pytest.raises(ValidationError, match="interruption_overtalk"):
        load_thresholds(tmp_path)


def test_rules_and_models_load() -> None:
    rules = load_rules(CONFIGS)
    assert {r.name for r in rules.rules} >= {
        "lexicon_fast_path",
        "sustained_shout",
        "register_drop",
        "overtalk",
    }
    models = load_models(CONFIGS)
    assert all(c.backend in ("stub", "rules_v0") for c in models.components.values())


def test_fake_lexicon_has_only_placeholders() -> None:
    lex = load_lexicon(Settings(env="test"))
    assert lex.fake
    assert len(lex.entries) == 20
    assert all(e.term.startswith("<GALI_") for e in lex.entries)


def test_fake_lexicon_refused_in_prod(tmp_path: Path) -> None:
    s = Settings.model_construct(
        env="prod", lexicon_path=None, config_dir=CONFIGS, model_mode="real", auth_mode="oidc"
    )
    with pytest.raises(ValueError, match="fake lexicon"):
        load_lexicon(s)


def test_canonical_map() -> None:
    assert load_canonical(CONFIGS)["রিচার্য"] == "রিচার্জ"
