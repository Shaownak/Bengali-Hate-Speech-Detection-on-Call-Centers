"""Evaluation metrics (docs/06 §2). numpy only, deterministic.

Text metrics use the shared normaliser (``galecore.bangla.normalize``) for references and
hypotheses alike (docs/02 §2.5): the same normaliser for training, inference and scoring.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from galecore.bangla import normalize, tokenize

BANGLA = re.compile(r"[ঀ-৿]")
FOREIGN_SCRIPT = re.compile(r"[A-Za-zऀ-ॿ؀-ۿ]")  # Latin, Devanagari, Arabic
TAG = re.compile(r"<[A-Z][A-Z0-9_]*>")


# --- ASR ----------------------------------------------------------------------------------------


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _words(text: str) -> list[str]:
    return [t.text for t in tokenize(normalize(text))]


def _chars(text: str) -> list[str]:
    return list("".join(_words(text)))  # spaces and punctuation don't count


def error_rate(refs: Sequence[str], hyps: Sequence[str], unit: str = "word") -> float:
    split = _words if unit == "word" else _chars
    errs = sum(edit_distance(split(r), split(h)) for r, h in zip(refs, hyps, strict=True))
    return errs / max(1, sum(len(split(r)) for r in refs))


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return error_rate(refs, hyps, "word")


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return error_rate(refs, hyps, "char")


def slang_recall(
    refs: Sequence[str], hyps: Sequence[str], match: Callable[[str], list[str]]
) -> float | None:
    """Share of lexicon terms in the reference that the hypothesis also contains (docs/03 §1.1).
    ``match`` maps normalised text to the lexicon entry ids found in it."""
    found = total = 0
    for r, h in zip(refs, hyps, strict=True):
        want = match(normalize(r))
        got = list(match(normalize(h)))
        total += len(want)
        for term in want:
            if term in got:
                got.remove(term)
                found += 1
    return found / total if total else None


def script_drift(hyps: Sequence[str]) -> float:
    """Share of hypothesis letters in Latin / Devanagari / Arabic script (tags excluded)."""
    text = TAG.sub("", "".join(hyps))
    bn, foreign = len(BANGLA.findall(text)), len(FOREIGN_SCRIPT.findall(text))
    return foreign / (bn + foreign) if bn + foreign else 0.0


# --- detection ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    call_id: str
    speaker: str
    category: str
    start: float
    end: float
    score: float = 1.0
    severity: int = 1
    alert: bool = True


def iou(a: Span, b: Span) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def docs06_match(p: Span, g: Span, iou_min: float = 0.3, centre_tol: float = 2.0) -> bool:
    """docs/06 §2: same speaker and category, IoU ≥ 0.3 or predicted centre within gold ± 2 s."""
    if (p.call_id, p.speaker, p.category) != (g.call_id, g.speaker, g.category):
        return False
    centre = (p.start + p.end) / 2
    return iou(p, g) >= iou_min or g.start - centre_tol <= centre <= g.end + centre_tol


def overlap_match(p: Span, g: Span, min_frac: float = 0.5) -> bool:
    """Shouting F1 rule (docs/06 §2): ≥ 50 % of the gold span covered."""
    if (p.call_id, p.speaker, p.category) != (g.call_id, g.speaker, g.category):
        return False
    inter = max(0.0, min(p.end, g.end) - max(p.start, g.start))
    return inter >= min_frac * max(1e-9, g.end - g.start)


@dataclass
class Matched:
    tp: list[tuple[Span, Span]]
    fp: list[Span]
    fn: list[Span]


def match_events(
    pred: Sequence[Span], gold: Sequence[Span], rule: Callable[[Span, Span], bool] = docs06_match
) -> Matched:
    """One-to-one greedy matching in time order (best IoU first among candidates)."""
    free = list(pred)
    tp, fn = [], []
    for g in sorted(gold, key=lambda s: (s.call_id, s.start)):
        cands = [p for p in free if rule(p, g)]
        if not cands:
            fn.append(g)
            continue
        best = max(cands, key=lambda p: iou(p, g))
        free.remove(best)
        tp.append((best, g))
    return Matched(tp, free, fn)


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def prf_by(m: Matched, key: Callable[[Span], str]) -> dict[str, dict[str, float | int]]:
    keys = sorted({key(s) for s in [*m.fp, *m.fn, *(g for _, g in m.tp)]})
    return {
        k: prf(
            sum(key(g) == k for _, g in m.tp),
            sum(key(p) == k for p in m.fp),
            sum(key(g) == k for g in m.fn),
        )
        for k in keys
    }


# --- probabilistic ------------------------------------------------------------------------------


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """ROC AUC via the Mann–Whitney U statistic (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    ranks = _rank(list(scores))
    r_pos = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return round((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def _rank(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = np.array(_rank(list(a))), np.array(_rank(list(b)))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def calibration(probs: Sequence[float], correct: Sequence[int], bins: int = 10) -> dict[str, Any]:
    """Expected calibration error and the reliability-diagram bins."""
    p, y = np.asarray(probs, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in itertools.pairwise(edges):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if not sel.any():
            continue
        conf, acc = float(p[sel].mean()), float(y[sel].mean())
        ece += sel.sum() / len(p) * abs(conf - acc)
        rows.append(
            {
                "lo": round(float(lo), 2),
                "hi": round(float(hi), 2),
                "n": int(sel.sum()),
                "confidence": round(conf, 4),
                "accuracy": round(acc, 4),
            }
        )
    return {"ece": round(float(ece), 4) if len(p) else None, "bins": rows}


def uar(pred: Sequence[str], gold: Sequence[str]) -> float | None:
    """Unweighted average recall over the gold classes present."""
    classes = sorted(set(gold))
    if not classes:
        return None
    recalls = [
        sum(p == g == c for p, g in zip(pred, gold, strict=True)) / sum(g == c for g in gold)
        for c in classes
    ]
    return round(float(np.mean(recalls)), 4)


def latency_summary(values_s: Sequence[float]) -> dict[str, float | int | None]:
    if not values_s:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    v = np.asarray(values_s, dtype=float)
    return {
        "n": len(v),
        "p50": round(float(np.percentile(v, 50)), 4),
        "p95": round(float(np.percentile(v, 95)), 4),
        "p99": round(float(np.percentile(v, 99)), 4),
        "max": round(float(v.max()), 4),
    }


def bootstrap_ci(
    units: Sequence[Any],
    stat: Callable[[Sequence[Any]], float | None],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile CI of ``stat`` over resampled calls (D-13: every golden metric gets a 95 % CI)."""
    if not units:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [units[i] for i in rng.integers(0, len(units), len(units))]
        v = stat(sample)
        if v is not None and not math.isnan(v):
            vals.append(v)
    if not vals:
        return None
    return (
        round(float(np.percentile(vals, 100 * alpha / 2)), 4),
        round(float(np.percentile(vals, 100 * (1 - alpha / 2))), 4),
    )
