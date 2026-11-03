"""Retraining pipeline (ADR-020): promotion gate, calibration fitting, feedback views, and an
end-to-end stub run on the synthetic fixtures (pass and fail)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from gale_fusion_engine.calibration import TableCalibrator
from ml.fusion.calibrate import fit, fit_isotonic, fit_platt
from ml.retrain.__main__ import main, overlay_model_dir, prepare, textclf_view
from ml.retrain.gate import MetricSpec, compare, load_specs, to_markdown


def _rep(metrics: dict[str, Any], set_id: str = "s1", split: str = "golden") -> dict[str, Any]:
    return {"metrics": metrics, "headline": metrics, "set_id": set_id, "split": split}


SPECS = {
    "fusion": [
        MetricSpec("events.f1", "higher", 0.01),
        MetricSpec("ece", "lower", 0.01),
        MetricSpec("grounding", "higher", 0.01),
    ],
    "system": [MetricSpec("false_alerts_per_100_calls", "lower", 1.0)],
}


def test_gate_tolerance_and_direction() -> None:
    base = {
        "fusion": _rep({"events": {"f1": 0.80}, "ece": 0.05}),
        "system": _rep({"false_alerts_per_100_calls": 2.0}),
    }
    ok = {
        "fusion": _rep({"events": {"f1": 0.795}, "ece": 0.059}),
        "system": _rep({"false_alerts_per_100_calls": 2.9}),
    }
    g = compare(SPECS, base, ok)
    assert g.passed, g.regressions
    assert {c.metric: c.status for c in g.checks}["grounding"] == "n/a"
    bad = {
        "fusion": _rep({"events": {"f1": 0.785}, "ece": 0.04}),
        "system": _rep({"false_alerts_per_100_calls": 3.5}),
    }
    g = compare(SPECS, base, bad)
    assert not g.passed
    assert {r["metric"] for r in g.regressions} == {"events.f1", "false_alerts_per_100_calls"}
    assert "FAILED" in to_markdown("fusion", "v2", g)


def test_gate_missing_metric_and_mismatched_sets() -> None:
    base = {
        "fusion": _rep({"events": {"f1": 0.8}, "ece": 0.05}),
        "system": _rep({"false_alerts_per_100_calls": 1}),
    }
    no_ece = {
        "fusion": _rep({"events": {"f1": 0.9}}),
        "system": _rep({"false_alerts_per_100_calls": 1}),
    }
    g = compare(SPECS, base, no_ece)
    assert [r["status"] for r in g.regressions] == ["missing"]
    other = {k: {**v, "set_id": "s2"} for k, v in base.items()}
    g = compare(SPECS, base, other)
    assert not g.passed
    assert "different evaluation sets" in g.problems[0]
    g = compare(SPECS, base, {"fusion": base["fusion"]})
    assert "missing system report for candidate" in g.problems


def test_repo_gate_config() -> None:
    for comp in ("text_clf", "fusion", "llm_judge"):
        specs = load_specs(comp)
        assert "system" in specs
        assert all(m.tolerance > 0 for ms in specs.values() for m in ms)
    assert load_specs("fusion")["fusion"][0] == MetricSpec("events.f1", "higher", 0.01)
    with pytest.raises(KeyError):
        load_specs("asr")


def test_platt_and_isotonic() -> None:
    rng = random.Random(3)
    scores = [rng.random() for _ in range(600)]
    labels = [int(rng.random() < s * s) for s in scores]  # true P(y|s) = s²
    a, _ = fit_platt(scores, labels)
    assert a > 0
    x, y = fit_isotonic(scores, labels)
    assert x == sorted(x)
    assert y == sorted(y)
    assert len(x) < len(scores)
    cal = fit(
        [
            {
                "kind": "review",
                "label": lab,
                "signals": [{"source": "ser", "category": "harsh_tone", "score": s}],
            }
            for s, lab in zip(scores, labels, strict=True)
        ]
        + [{"kind": "missed", "label": 1, "signals": []}],
        "v",
    )
    assert cal["calibrators"]["ser/harsh_tone"]["kind"] in ("isotonic", "platt")
    assert set(cal["counts"]) == {"ser/harsh_tone", "ser/*"}
    tab = TableCalibrator.from_dict(cal)
    assert tab.version == "v"
    few = fit(
        [
            {
                "kind": "review",
                "label": 1,
                "signals": [{"source": "kws", "category": "insult", "score": 0.9}],
            }
        ],
        "v",
    )
    assert few["calibrators"] == {}  # too few samples: identity


def _fb(uid: str, call: str, label: int, score: float = 0.9, cat: str = "insult") -> dict[str, Any]:
    return {
        "uid": uid,
        "kind": "review",
        "call_id": call,
        "label": label,
        "category": cat,
        "predicted_category": cat,
        "severity": 2,
        "text": "তুমি <GALI_1>",
        "context": ["[CUSTOMER] বিল"],
        "signals": [{"source": "lexicon", "category": cat, "score": score}],
    }


def test_prepare_drops_golden_and_dedupes(tmp_path: Path) -> None:
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    f1.write_text(
        json.dumps(_fb("r1", "c1", 1)) + "\n" + json.dumps(_fb("r2", "gold", 1)) + "\n", "utf-8"
    )
    f2.write_text(json.dumps(_fb("r1", "c1", 0)) + "\n", "utf-8")
    st = prepare([f1, f2], tmp_path / "w", {"gold"})
    assert (st["rows"], st["golden_dropped"], st["golden_overlap"], st["positives"]) == (1, 1, 0, 0)
    view = textclf_view([_fb("r1", "c1", 1), _fb("r3", "c2", 0)])
    assert view[0]["labels"]["insult"] == 1
    assert view[0]["labels"]["threat"] is None
    assert view[1]["labels"]["insult"] == 0
    assert view[1]["severity"] == 0


def test_overlay_model_dir(tmp_path: Path) -> None:
    base = tmp_path / "models"
    (base / "fusion").mkdir(parents=True)
    (base / "fusion" / "calibration.json").write_text("{}", "utf-8")
    (base / "asr").mkdir()
    art = tmp_path / "cand"
    art.mkdir()
    (art / "calibration.json").write_text('{"version": "new"}', "utf-8")
    d = overlay_model_dir(base, "fusion", art, tmp_path / "overlay")
    assert json.loads((d / "fusion" / "calibration.json").read_text("utf-8")) == {"version": "new"}
    assert (d / "asr").exists()


def _run(
    tmp: Path, component: str, rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> int:
    fb = tmp / "feedback.jsonl"
    fb.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    monkeypatch.setenv("GALE_ENV", "test")
    monkeypatch.setenv("GALE_MODEL_DIR", str(tmp / "models"))
    from galecore import config

    config.get_settings.cache_clear()
    try:
        return main(
            [
                "run",
                "--component",
                component,
                "--work",
                str(tmp / "work"),
                "--feedback",
                str(fb),
                "--no-golden-manifest",
                "--version",
                f"{component}-test",
                "--split",
                "fixtures",
                "--limit",
                "4",
            ]
        )
    finally:
        config.get_settings.cache_clear()


def test_end_to_end_stub_retrain_passes_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_fb(f"r{i}", f"c{i}", i % 2) for i in range(10)]  # too few to fit: identity calibration
    assert _run(tmp_path, "fusion", rows, monkeypatch) == 0
    cand = json.loads((tmp_path / "work" / "candidate.json").read_text("utf-8"))
    assert cand["gate"]["passed"] is True
    assert cand["gate"]["split"] == "fixtures"
    assert cand["training"]["golden_overlap"] == 0
    assert {"fusion", "system"} <= set(cand["golden"])
    from gale_api.pilot.register import Candidate

    Candidate.model_validate(cand)  # what register-model accepts
    from galecore.config import Settings
    from ml.retrain.__main__ import fetch_feedback, publish

    dest = publish("fusion", tmp_path / "work", "fusion-test", tmp_path / "models")
    assert (dest / "calibration.json").exists()
    assert dest == tmp_path / "models" / "candidates" / "fusion" / "fusion-test"
    s = Settings(object_store_dir=tmp_path / "objects")
    from datetime import UTC, datetime, timedelta

    from galecore.objectstore import store_from_settings

    d = datetime.now(UTC).date() - timedelta(days=2)
    store_from_settings(s).put(f"datasets/feedback/{d}_{d + timedelta(days=1)}.jsonl", b"{}\n")
    files = fetch_feedback(3, tmp_path / "fb", s)
    assert [f.name for f in files] == [f"{d}_{d + timedelta(days=1)}.jsonl"]


def test_end_to_end_bad_calibration_fails_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # every lexicon hit reviewed as a false positive → calibrated lexicon scores collapse
    rows = [_fb(f"r{i}", f"c{i}", 0, score=0.6 + 0.4 * (i % 10) / 10) for i in range(60)]
    rows += [_fb(f"p{i}", f"d{i}", 1, score=0.05) for i in range(30)]
    assert _run(tmp_path, "fusion", rows, monkeypatch) == 1
    cand = json.loads((tmp_path / "work" / "candidate.json").read_text("utf-8"))
    assert cand["gate"]["passed"] is False
    from ml.retrain.__main__ import publish

    with pytest.raises(SystemExit, match="gate failed"):
        publish("fusion", tmp_path / "work", "fusion-test", tmp_path / "models")
    assert cand["gate"]["regressions"]  # the collapsed lexicon scores hurt calibration / events


def test_stub_trainer_for_text_clf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_fb(f"r{i}", f"c{i}", 1) for i in range(4)]
    assert _run(tmp_path, "text_clf", rows, monkeypatch) == 0
    card = json.loads((tmp_path / "work" / "candidate" / "model_card.json").read_text("utf-8"))
    assert card["stub"] is True


def test_cli_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GALE_ENV", "staging")
    from galecore import config

    config.get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="dev/test only"):
            main(
                [
                    "prepare",
                    "--component",
                    "fusion",
                    "--work",
                    str(tmp_path),
                    "--no-golden-manifest",
                ]
            )
        with pytest.raises(SystemExit, match="golden manifest"):
            main(
                [
                    "prepare",
                    "--component",
                    "fusion",
                    "--work",
                    str(tmp_path),
                    "--golden-manifest",
                    str(tmp_path / "x"),
                ]
            )
        with pytest.raises(SystemExit, match="smoke runs"):
            main(["evaluate", "--component", "fusion", "--work", str(tmp_path), "--limit", "2"])
        with pytest.raises(SystemExit, match="--version"):
            main(["train", "--component", "fusion", "--work", str(tmp_path)])
    finally:
        config.get_settings.cache_clear()
