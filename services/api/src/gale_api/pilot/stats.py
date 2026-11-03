"""Small, dependency-free statistics for the pilot report."""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval of a proportion k/n (None when n = 0)."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def rate(k: int, n: int) -> float | None:
    return None if n <= 0 else round(k / n, 4)


def per_100(k: int, n: int) -> float | None:
    return None if n <= 0 else round(100 * k / n, 2)


def psi(expected: Sequence[float], actual: Sequence[float], eps: float = 1e-4) -> float | None:
    """Population stability index between two histograms (counts over the same bins).

    < 0.1 stable · 0.1–0.2 moderate shift · > 0.2 significant shift (the usual reading).
    """
    e_tot, a_tot = sum(expected), sum(actual)
    if e_tot <= 0 or a_tot <= 0 or len(expected) != len(actual):
        return None
    out = 0.0
    for e, a in zip(expected, actual, strict=True):
        pe, pa = max(e / e_tot, eps), max(a / a_tot, eps)
        out += (pa - pe) * math.log(pa / pe)
    return round(out, 4)


def median(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
