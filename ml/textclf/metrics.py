"""Metrics for the multi-task text classifier (numpy only), and the docs/09 §8 reference check."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml.textclf.data import HEADS, Example

# docs/09 §8: published numbers to sanity-check our pipeline (> 5 points below → debug first)
REFERENCE = {
    "banglamultihate": ("micro-F1 over hate types (BLP-2025 1A/1B)", 0.72),
    "violens": ("macro-F1 over non / passive / direct violence", 0.71),
}
TOLERANCE = 0.05


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def per_head_f1(
    probs: np.ndarray, examples: Sequence[Example], thr: float = 0.5
) -> dict[str, float]:
    """F1 per head over annotated rows only (masked)."""
    y = np.array([e.target for e in examples]) >= 0.5
    m = np.array([e.mask for e in examples]) > 0
    p = probs >= thr
    out = {}
    for j, head in enumerate(HEADS):
        sel = m[:, j]
        if sel.sum() == 0 or y[sel, j].sum() == 0:
            continue
        tp = int((p[sel, j] & y[sel, j]).sum())
        fp = int((p[sel, j] & ~y[sel, j]).sum())
        fn = int((~p[sel, j] & y[sel, j]).sum())
        out[head] = round(f1(tp, fp, fn), 4)
    return out


def source_metric(
    source: str, probs: np.ndarray, examples: Sequence[Example], thr: float = 0.5
) -> float | None:
    """The metric of REFERENCE[source], derived from our heads."""
    idx = {h: i for i, h in enumerate(HEADS)}
    y = np.array([e.target for e in examples]) >= 0.5
    p = probs >= thr
    if source == "banglamultihate":  # micro-F1 over the mapped hate-type labels
        cols = [idx[h] for h in ("slang_profanity", "insult", "hate_identity")]
        tp = int((p[:, cols] & y[:, cols]).sum())
        fp = int((p[:, cols] & ~y[:, cols]).sum())
        fn = int((~p[:, cols] & y[:, cols]).sum())
        return round(f1(tp, fp, fn), 4)
    if source == "violens":  # 3-class macro-F1: threat → direct, insult → passive, else none

        def cls(row_p: np.ndarray) -> int:
            return 2 if row_p[idx["threat"]] else 1 if row_p[idx["insult"]] else 0

        pred = [cls(r) for r in p]
        gold = [cls(r) for r in y]
        scores = []
        for c in (0, 1, 2):
            tp = sum(a == c and b == c for a, b in zip(pred, gold, strict=True))
            fp = sum(a == c and b != c for a, b in zip(pred, gold, strict=True))
            fn = sum(a != c and b == c for a, b in zip(pred, gold, strict=True))
            scores.append(f1(tp, fp, fn))
        return round(float(np.mean(scores)), 4)
    return None


def macro(scores: dict[str, float]) -> float:
    return round(float(np.mean(list(scores.values()))), 4) if scores else 0.0
