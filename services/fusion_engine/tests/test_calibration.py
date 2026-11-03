from __future__ import annotations

import json
from pathlib import Path

import pytest

from gale_fusion_engine.calibration import (
    Isotonic,
    Platt,
    TableCalibrator,
    calibration_path,
    load_calibrator,
)
from gale_fusion_engine.fusion import identity
from galecore.schemas import Category, Channel, Signal, SignalSource


def _sig(source: SignalSource, cat: Category, score: float) -> Signal:
    return Signal(
        call_id="c",
        channel=Channel.AGENT,
        signal_id="s",
        source=source,
        category=cat,
        score=score,
        ts_start=0,
        ts_end=1,
        model_version="t",
    )


def test_platt_is_a_sigmoid_and_stable_for_large_inputs() -> None:
    f = Platt(a=10, b=-5)
    assert f(0.5) == pytest.approx(0.5)
    assert f(1.0) > 0.99
    assert f(0.0) < 0.01
    assert 0 <= Platt(1000, 0)(-5) <= 1
    assert 0 <= Platt(1000, 0)(5) <= 1


def test_isotonic_interpolates_and_clamps() -> None:
    f = Isotonic((0.2, 0.6), (0.1, 0.9))
    assert f(0.0) == 0.1
    assert f(1.0) == 0.9
    assert f(0.4) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="increasing"):
        Isotonic((0.6, 0.2), (0.1, 0.9))


def test_table_lookup_order(tmp_path: Path) -> None:
    data = {
        "version": "calib-x",
        "calibrators": {
            "text_clf/insult": {"kind": "platt", "a": 0, "b": 0},  # always 0.5
            "text_clf/*": {"kind": "isotonic", "x": [0, 1], "y": [0, 0.5]},
        },
    }
    path = calibration_path(tmp_path, "calib-x")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), "utf-8")
    cal = load_calibrator(path)
    assert isinstance(cal, TableCalibrator)
    assert cal.version == "calib-x"
    assert cal(_sig(SignalSource.TEXT_CLF, Category.INSULT, 0.9)) == pytest.approx(0.5)
    assert cal(_sig(SignalSource.TEXT_CLF, Category.THREAT, 0.8)) == pytest.approx(0.4)
    assert cal(_sig(SignalSource.LEXICON, Category.INSULT, 0.7)) == 0.7  # identity
    with pytest.raises(ValueError, match="unknown calibrator"):
        TableCalibrator.from_dict({"calibrators": {"a/b": {"kind": "beta"}}})


def test_missing_file_is_identity(tmp_path: Path) -> None:
    assert load_calibrator(None) is identity
    assert load_calibrator(calibration_path(tmp_path)) is identity
    assert calibration_path(tmp_path) == tmp_path / "fusion" / "calibration.json"
