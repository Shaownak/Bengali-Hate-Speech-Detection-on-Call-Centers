"""Fit fusion calibration per (signal source, category) from reviewed feedback (docs/03 §8.1, D-13).

Every contributing signal of a reviewed event is one sample: its raw score, labelled with the
event's verdict (true positive = 1). Per pair: isotonic regression when there are ≥ 200 positives
and ≥ 200 negatives, Platt scaling when both classes have ≥ ``min_class`` samples, otherwise no
calibrator (fusion keeps the raw score). A pooled ``source/*`` calibrator covers sparse pairs.
Output: ``calibration.json`` as read by ``gale_fusion_engine.calibration``.

    python -m ml.fusion.calibrate --feedback train.jsonl --out DIR --version calib-2026-12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ISOTONIC_MIN = 200  # D-13
MIN_CLASS = 20


def fit_platt(
    scores: Sequence[float], labels: Sequence[int], iters: int = 100
) -> tuple[float, float]:
    """Platt (1999) with smoothed targets; Newton's method on (a, b) for sigmoid(a·s + b)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    hi, lo = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
    t = [hi if y else lo for y in labels]
    a, b = 1.0, 0.0
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for s, ti in zip(scores, t, strict=True):
            z = a * s + b
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            d, w = p - ti, max(p * (1 - p), 1e-12)
            g_a += d * s
            g_b += d
            h_aa += w * s * s
            h_ab += w * s
            h_bb += w
        h_aa += 1e-9
        h_bb += 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-18:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return round(a, 6), round(b, 6)


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators; returns increasing breakpoints (x = block mean score, y = rate)."""
    pts = sorted(zip(scores, labels, strict=True))
    blocks: list[list[float]] = []  # [sum_x, sum_y, n]
    for s, y in pts:
        blocks.append([s, float(y), 1.0])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            x2, y2, n2 = blocks.pop()
            blocks[-1][0] += x2
            blocks[-1][1] += y2
            blocks[-1][2] += n2
    xs = [round(b[0] / b[2], 6) for b in blocks]
    ys = [round(b[1] / b[2], 6) for b in blocks]
    # equal x after rounding: keep the last (monotone anyway)
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys, strict=True):
        if out_x and x <= out_x[-1]:
            out_y[-1] = y
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def samples(rows: Iterable[dict[str, Any]]) -> dict[str, list[tuple[float, int]]]:
    out: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "review":
            continue
        for s in r.get("signals") or []:
            key = f"{s['source']}/{s['category']}"
            out[key].append((float(s["score"]), int(r["label"])))
            out[f"{s['source']}/*"].append((float(s["score"]), int(r["label"])))
    return out


def fit(rows: Iterable[dict[str, Any]], version: str, min_class: int = MIN_CLASS) -> dict[str, Any]:
    cals: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for key, xs in sorted(samples(rows).items()):
        scores, labels = [s for s, _ in xs], [y for _, y in xs]
        pos = sum(labels)
        neg = len(labels) - pos
        counts[key] = {"pos": pos, "neg": neg}
        if pos >= ISOTONIC_MIN and neg >= ISOTONIC_MIN:
            x, y = fit_isotonic(scores, labels)
            cals[key] = {"kind": "isotonic", "x": x, "y": y}
        elif pos >= min_class and neg >= min_class:
            a, b = fit_platt(scores, labels)
            cals[key] = {"kind": "platt", "a": a, "b": b}
    return {
        "version": version,
        "method": {"isotonic_min": ISOTONIC_MIN, "min_class": min_class},
        "calibrators": cals,
        "counts": counts,
    }


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.fusion.calibrate", description=__doc__)
    ap.add_argument("--feedback", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--min-class", type=int, default=MIN_CLASS)
    a = ap.parse_args(argv)
    cal = fit(read_jsonl(a.feedback), a.version, a.min_class)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"{len(cal['calibrators'])} calibrators → {a.out / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
