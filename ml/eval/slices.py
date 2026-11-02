"""Fairness & robustness slices (docs/06 §3).

Per slice value: event precision/recall/F1 and the **false-alarm rate** (false alerts per 100
calls). A slice is flagged when its rate exceeds 1.5× the overall rate, given enough calls for
the comparison to mean something (``min_calls``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ml.eval.metrics import Matched, prf
from ml.eval.sets import SLICE_KEYS, EvalSet

FLAG_RATIO = 1.5


def slice_report(es: EvalSet, m: Matched, min_calls: int = 2) -> dict[str, Any]:
    call_meta = {c.call_id: c.meta for c in es.calls}
    n_calls = len(es.calls)
    overall_fa = 100 * len(m.fp) / n_calls if n_calls else 0.0
    out: dict[str, Any] = {
        "overall_false_alerts_per_100_calls": round(overall_fa, 3),
        "slices": {},
        "flags": [],
    }
    for key in SLICE_KEYS:
        groups: dict[str, list[str]] = defaultdict(list)
        for cid, meta in call_meta.items():
            groups[str(meta.get(key, "unknown"))].append(cid)
        rows = {}
        for value, cids in sorted(groups.items()):
            ids = set(cids)
            tp = sum(g.call_id in ids for _, g in m.tp)
            fp = sum(p.call_id in ids for p in m.fp)
            fn = sum(g.call_id in ids for g in m.fn)
            fa = 100 * fp / len(cids)
            rows[value] = {
                "calls": len(cids),
                **prf(tp, fp, fn),
                "false_alerts_per_100_calls": round(fa, 3),
            }
            ratio = fa / overall_fa if overall_fa else (float("inf") if fa else 0.0)
            if len(cids) >= min_calls and fa > 0 and ratio > FLAG_RATIO:
                out["flags"].append(
                    {
                        "slice": key,
                        "value": value,
                        "false_alerts_per_100_calls": round(fa, 3),
                        "ratio_to_overall": round(ratio, 2) if overall_fa else None,
                    }
                )
        out["slices"][key] = rows
    return out
