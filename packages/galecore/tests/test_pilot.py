from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from galecore.config import REPO_ROOT, Settings
from galecore.pilot import PilotConfig, PilotPolicy, bucket, load_pilot, pilot_path


def _policy(**cfg: object) -> PilotPolicy:
    return PilotPolicy(PilotConfig.model_validate(cfg))


def test_repo_config_validates_and_shows_alerts_in_dev() -> None:
    cfg = load_pilot(pilot_path())
    assert cfg.stage in ("P-0", "P-1", "P-2", "P-3")
    assert PilotPolicy(cfg).alerts_visible(None)  # dev/e2e: nothing hidden
    assert abs(sum(cfg.sampler.quotas.values()) - 1.0) < 1e-9
    assert set(cfg.exit_criteria) == {"P-0", "P-1", "P-2", "P-3"}


def test_settings_override_path(tmp_path: Path) -> None:
    f = tmp_path / "pilot.yaml"
    f.write_text(yaml.safe_dump({"stage": "P-2", "alerts": {"default": "hidden"}}), "utf-8")
    pol = PilotPolicy.load(Settings(pilot_config=f))
    assert pol.cfg.stage == "P-2"
    assert not pol.alerts_visible("billing")
    assert pilot_path(Settings()) == REPO_ROOT / "configs" / "pilot.yaml"
    assert load_pilot(tmp_path / "missing.yaml") == PilotConfig()


def test_alert_visibility_by_queue() -> None:
    pol = _policy(alerts={"default": "hidden", "visible_queues": ["billing"]})
    assert pol.alerts_visible("billing")
    assert not pol.alerts_visible("sales")
    assert not pol.alerts_visible(None)
    assert _policy(alerts={"default": "visible"}).alerts_visible("anything")


def test_bucket_is_stable_uniform_and_salted_by_component() -> None:
    ids = [f"call-{i}" for i in range(4000)]
    b = [bucket("text_clf", c) for c in ids]
    assert b == [bucket("text_clf", c) for c in ids]  # deterministic
    assert all(0 <= x < 100 for x in b)
    share = sum(x < 10 for x in b) / len(b)
    assert 0.08 < share < 0.12
    other = [bucket("fusion", c) for c in ids]
    both = sum(x < 10 and y < 10 for x, y in zip(b, other, strict=True)) / len(b)
    assert both < 0.03  # independent, not the same 10 %


def test_route_percent_and_queues() -> None:
    pol = _policy(
        candidates={
            "text_clf": {"version": "v2", "role": "shadow", "percent": 100},
            "fusion": {"version": "c2", "role": "canary", "percent": 30, "queues": ["billing"]},
            "llm_judge": {"version": "l2", "role": "shadow", "percent": 0},
        }
    )
    c = pol.route("text_clf", "x", None)
    assert c is not None
    assert (c.version, c.role) == ("v2", "shadow")
    assert pol.route("llm_judge", "x", None) is None
    assert pol.route("asr", "x", None) is None
    ids = [f"c{i}" for i in range(2000)]
    assert all(pol.route("fusion", i, "sales") is None for i in ids)
    routed = sum(pol.route("fusion", i, "billing") is not None for i in ids) / len(ids)
    assert 0.25 < routed < 0.35


def test_unknown_candidate_component_rejected() -> None:
    with pytest.raises(ValueError, match="unknown components"):
        PilotConfig.model_validate({"candidates": {"asr": {"version": "x", "role": "canary"}}})
    with pytest.raises(ValueError, match="role"):
        PilotConfig.model_validate({"candidates": {"fusion": {"version": "x", "role": "live"}}})
