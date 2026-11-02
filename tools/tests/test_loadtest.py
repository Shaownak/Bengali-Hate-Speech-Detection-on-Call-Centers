from tools.loadtest.__main__ import LevelResult, render_report
from tools.loadtest.metrics import Snapshot, parse, quantile, window

TEXT = """# HELP gale_stage_latency_seconds x
# TYPE gale_stage_latency_seconds histogram
gale_stage_latency_seconds_bucket{service="fusion_engine",stage="confirmed_event",le="1.0"} {a}
gale_stage_latency_seconds_bucket{service="fusion_engine",stage="confirmed_event",le="2.0"} {b}
gale_stage_latency_seconds_bucket{service="fusion_engine",stage="confirmed_event",le="+Inf"} {b}
gale_stage_latency_seconds_count{service="fusion_engine",stage="confirmed_event"} {b}
gale_stage_latency_seconds_sum{service="fusion_engine",stage="confirmed_event"} 1.0
# HELP gale_ingest_frames_dropped_total x
# TYPE gale_ingest_frames_dropped_total counter
gale_ingest_frames_dropped_total {d}
# HELP gale_bus_lag_seconds x
# TYPE gale_bus_lag_seconds gauge
gale_bus_lag_seconds{stream="signals",group="fusion_engine"} {lag}
"""


def snap(a: int, b: int, d: int, lag: float) -> Snapshot:
    return parse(
        TEXT.replace("{a}", str(a))
        .replace("{b}", str(b))
        .replace("{d}", str(d))
        .replace("{lag}", str(lag))
    )


def test_quantile_and_window_deltas() -> None:
    assert quantile({1.0: 50, 2.0: 100, float("inf"): 100}, 0.5) == 1.0
    assert quantile({1.0: 50, 2.0: 100, float("inf"): 100}, 0.95) == 1.9
    assert quantile({}, 0.95) is None
    before, after = snap(10, 10, 0, 0.0), snap(10, 30, 2, 3.5)
    w = window(before, after)
    p, n = w[("fusion_engine", "confirmed_event")]
    assert n == 20
    assert 1.0 < p <= 2.0  # all 20 new samples landed in (1, 2]
    assert (
        after.counters["gale_ingest_frames_dropped"] - before.counters["gale_ingest_frames_dropped"]
        == 2
    )
    assert after.gauges_max["bus_lag_s"] == 3.5


def test_report_marks_gpu_rows_tbd() -> None:
    r = LevelResult(level_pct=200, calls=8, wall_s=50, call_minutes=20, passed=True)
    r.p95 = {"fusion_engine/confirmed_event": 1.2}
    md = render_report(
        [r], {"generated": "now", "peak": 4, "speed": 1, "model_mode": "stub", "host": "h"}
    )
    assert "| 200% | 8 |" in md
    assert "fusion_engine/confirmed_event | 1.2 | 6.0" in md
    assert "GPU_ASR" in md
    assert "TBD" in md
