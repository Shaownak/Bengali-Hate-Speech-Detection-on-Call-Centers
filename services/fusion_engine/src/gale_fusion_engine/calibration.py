"""Per-(source, category) score calibration for fusion (docs/03 §8.1, D-13).

The file is written by ``ml.fusion.calibrate`` (retraining, ADR-020)::

    {"version": "calib-2026-12",
     "calibrators": {"text_clf/insult": {"kind": "platt", "a": 4.1, "b": -2.3},
                     "lexicon/*": {"kind": "isotonic", "x": [0.1, 0.9], "y": [0.05, 0.97]}}}

Lookup: ``source/category``, then ``source/*``, then the raw score (identity). Platt is
``sigmoid(a·score + b)``; isotonic is a step-wise linear interpolation over increasing ``x``.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gale_fusion_engine.fusion import Calibrator, identity
from galecore.schemas import Signal


@dataclass(frozen=True)
class Platt:
    a: float
    b: float

    def __call__(self, p: float) -> float:
        z = self.a * p + self.b
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


@dataclass(frozen=True)
class Isotonic:
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.x or len(self.x) != len(self.y) or list(self.x) != sorted(self.x):
            raise ValueError("isotonic calibrator needs equal-length x/y with increasing x")

    def __call__(self, p: float) -> float:
        i = bisect.bisect_left(self.x, p)
        if i == 0:
            return self.y[0]
        if i >= len(self.x):
            return self.y[-1]
        x0, x1, y0, y1 = self.x[i - 1], self.x[i], self.y[i - 1], self.y[i]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (p - x0) / (x1 - x0)


def _one(spec: dict[str, Any]) -> Platt | Isotonic:
    if spec["kind"] == "platt":
        return Platt(float(spec["a"]), float(spec["b"]))
    if spec["kind"] == "isotonic":
        return Isotonic(tuple(map(float, spec["x"])), tuple(map(float, spec["y"])))
    raise ValueError(f"unknown calibrator kind {spec['kind']!r}")


class TableCalibrator:
    def __init__(self, table: dict[str, Platt | Isotonic], version: str = "") -> None:
        self.table, self.version = table, version

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableCalibrator:
        cals = {k: _one(v) for k, v in (data.get("calibrators") or {}).items()}
        return cls(cals, str(data.get("version", "")))

    def __call__(self, s: Signal) -> float:
        f = self.table.get(f"{s.source.value}/{s.category.value}") or self.table.get(
            f"{s.source.value}/*"
        )
        return s.score if f is None else min(1.0, max(0.0, f(s.score)))


def load_calibrator(path: Path | None) -> Calibrator:
    """The calibrator in ``path``, or the identity when there is none yet (v0, stub mode)."""
    if path is None or not path.exists():
        return identity
    return TableCalibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))


def calibration_path(model_dir: Path, version: str | None = None) -> Path:
    """Stable: ``<model_dir>/fusion/calibration.json``; candidate:
    ``<model_dir>/candidates/fusion/<version>/calibration.json``."""
    if version is None:
        return model_dir / "fusion" / "calibration.json"
    return model_dir / "candidates" / "fusion" / version / "calibration.json"
