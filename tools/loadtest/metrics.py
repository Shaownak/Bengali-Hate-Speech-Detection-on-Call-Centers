"""Prometheus text → per-stage latency percentiles over a load-test window (histogram deltas)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from prometheus_client.parser import text_string_to_metric_families

Key = tuple[str, str]  # (service, stage)


@dataclass
class Snapshot:
    # (service, stage) → {le: cumulative count}
    buckets: dict[Key, dict[float, float]] = field(default_factory=lambda: defaultdict(dict))
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    gauges_max: dict[str, float] = field(default_factory=lambda: defaultdict(float))


COUNTERS = (
    "gale_ingest_frames_dropped",
    "gale_bus_dead_lettered",
    "gale_bus_failed",
    "gale_alerts",
    "gale_events",
)


def parse(text: str, into: Snapshot | None = None) -> Snapshot:
    snap = into or Snapshot()
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name == "gale_stage_latency_seconds_bucket":
                key = (s.labels["service"], s.labels["stage"])
                le = float("inf") if s.labels["le"] == "+Inf" else float(s.labels["le"])
                snap.buckets[key][le] = snap.buckets[key].get(le, 0.0) + s.value
            elif fam.name in COUNTERS and s.name.endswith("_total"):
                snap.counters[fam.name] += s.value
            elif s.name == "gale_bus_lag_seconds":
                snap.gauges_max["bus_lag_s"] = max(snap.gauges_max["bus_lag_s"], s.value)
    return snap


def quantile(buckets: dict[float, float], q: float) -> float | None:
    """Prometheus-style histogram_quantile on cumulative buckets (linear within a bucket)."""
    items = sorted(buckets.items())
    if not items or items[-1][1] <= 0:
        return None
    total = items[-1][1]
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in items:
        if c >= rank:
            if le == float("inf"):
                return prev_le
            span = c - prev_c
            return prev_le + (le - prev_le) * ((rank - prev_c) / span if span else 1.0)
        prev_le, prev_c = le, c
    return prev_le


def window(
    before: Snapshot, after: Snapshot, q: float = 0.95
) -> dict[Key, tuple[float | None, int]]:
    """p-quantile and sample count per (service, stage) between two snapshots."""
    out: dict[Key, tuple[float | None, int]] = {}
    for key, b in after.buckets.items():
        a0 = before.buckets.get(key, {})
        delta = {le: c - a0.get(le, 0.0) for le, c in b.items()}
        n = int(max(delta.values(), default=0))
        if n > 0:
            out[key] = (quantile(delta, q), n)
    return out
