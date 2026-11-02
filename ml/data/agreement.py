"""Inter-annotator agreement and adjudication export (docs/02 §3.3 quality gates).

- Krippendorff's α (nominal) per category: units are (call, channel, 1 s bin); each annotator who
  did the call codes 1 if one of their spans of that category covers the bin centre, else 0.
  Gates: α ≥ 0.70 per event category, ≥ 0.80 for shouting.
- Span agreement: spans of two annotators matched one-to-one (same channel and category, greedy
  by IoU); reports mean IoU of matches and match F1 at IoU ≥ 0.3 (the docs/06 rule).
- Adjudication export for the QA lead: unmatched spans and severity disagreements (times and
  categories only, no transcript text).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from ml.data.labelstudio.convert import T2Span

GATES = {"default": 0.70, "shouting": 0.80}
IOU_MATCH = 0.3


def krippendorff_alpha_nominal(units: Iterable[list[Hashable]]) -> float | None:
    """α for nominal data. Each unit lists the values given by the coders who coded it."""
    coinc: Counter[tuple[Hashable, Hashable]] = Counter()
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            coinc[(vals[a], vals[b])] += 1 / (m - 1)
    n_c: Counter[Hashable] = Counter()
    for (c, _), v in coinc.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return None
    d_o = sum(v for (c, k), v in coinc.items() if c != k) / n
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n * (n - 1))
    return 1.0 if d_e == 0 else round(1 - d_o / d_e, 4)


def iou(a: T2Span, b: T2Span) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def match_spans(
    a: list[T2Span], b: list[T2Span]
) -> tuple[list[tuple[T2Span, T2Span, float]], list[T2Span], list[T2Span]]:
    pairs = sorted(
        (
            (iou(x, y), i, j)
            for i, x in enumerate(a)
            for j, y in enumerate(b)
            if x.channel == y.channel and x.category == y.category
        ),
        reverse=True,
    )
    used_a, used_b, matches = set(), set(), []
    for score, i, j in pairs:
        if score < IOU_MATCH or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((a[i], b[j], score))
    return (
        matches,
        [x for i, x in enumerate(a) if i not in used_a],
        [y for j, y in enumerate(b) if j not in used_b],
    )


@dataclass
class AgreementReport:
    alpha: dict[str, float | None]
    gates: dict[str, bool | None]
    span_mean_iou: float | None
    span_f1: float | None
    double_annotated_calls: int
    adjudication: list[dict[str, Any]]


def agreement(
    spans: list[T2Span],
    annotators_by_call: dict[str, set[str]],
    durations: dict[str, float],
    categories: list[str],
    bin_s: float = 1.0,
) -> AgreementReport:
    by_call: dict[str, list[T2Span]] = defaultdict(list)
    for s in spans:
        by_call[s.call_id].append(s)
    double = {c: sorted(a) for c, a in annotators_by_call.items() if len(a) >= 2}
    alpha: dict[str, float | None] = {}
    for cat in categories:
        units: list[list[Hashable]] = []
        for cid, anns in double.items():
            n_bins = max(1, math.ceil(durations.get(cid, 0) / bin_s))
            for ch in ("agent", "customer"):
                cover = {
                    a: [
                        (s.start, s.end)
                        for s in by_call[cid]
                        if s.annotator == a and s.channel == ch and s.category == cat
                    ]
                    for a in anns
                }
                for k in range(n_bins):
                    t = (k + 0.5) * bin_s
                    units.append([int(any(x <= t <= y for x, y in cover[a])) for a in anns])
        alpha[cat] = krippendorff_alpha_nominal(units)
    gates = {
        c: None if v is None else v >= GATES.get(c, GATES["default"]) for c, v in alpha.items()
    }

    ious, tp, fp_fn, adjud = [], 0, 0, []
    for cid, anns in double.items():
        for a, b in itertools.combinations(anns, 2):
            sa = [s for s in by_call[cid] if s.annotator == a]
            sb = [s for s in by_call[cid] if s.annotator == b]
            matches, only_a, only_b = match_spans(sa, sb)
            ious += [m[2] for m in matches]
            tp += len(matches)
            fp_fn += len(only_a) + len(only_b)
            for x, y, score in matches:
                if x.severity != y.severity:
                    adjud.append(
                        {
                            "call_id": cid,
                            "channel": x.channel,
                            "category": x.category,
                            "start": round(min(x.start, y.start), 2),
                            "end": round(max(x.end, y.end), 2),
                            "issue": "severity",
                            "annotators": {a: x.severity, b: y.severity},
                            "iou": round(score, 3),
                        }
                    )
            for who, other, rest in ((a, b, only_a), (b, a, only_b)):
                for s in rest:
                    adjud.append(
                        {
                            "call_id": cid,
                            "channel": s.channel,
                            "category": s.category,
                            "start": round(s.start, 2),
                            "end": round(s.end, 2),
                            "issue": "missing",
                            "annotators": {who: "labelled", other: "not labelled"},
                        }
                    )
    f1 = 2 * tp / (2 * tp + fp_fn) if (tp or fp_fn) else None
    return AgreementReport(
        alpha,
        gates,
        round(sum(ious) / len(ious), 4) if ious else None,
        None if f1 is None else round(f1, 4),
        len(double),
        sorted(adjud, key=lambda r: (r["call_id"], r["start"])),
    )
