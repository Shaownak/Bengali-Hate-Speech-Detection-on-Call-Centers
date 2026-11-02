"""Generate the Grafana dashboards (docs/05 §4: Ops, ML, Business) into the Helm chart.

    python -m tools.gen_dashboards          # writes infra/helm/gal-e/files/dashboards/*.json

The JSON is generated so panels stay consistent; a test fails if the files drift from this code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "infra" / "helm" / "gal-e" / "files" / "dashboards"
DS = {"type": "prometheus", "uid": "${datasource}"}


def panel(
    title: str,
    exprs: list[tuple[str, str]],
    unit: str = "short",
    kind: str = "timeseries",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": kind,
        "title": title,
        "datasource": DS,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "table", "placement": "bottom"}},
        "targets": [
            {"refId": chr(65 + i), "expr": e, "legendFormat": legend, "datasource": DS}
            for i, (e, legend) in enumerate(exprs)
        ],
        **extra,
    }


def stat(title: str, expr: str, unit: str = "short") -> dict[str, Any]:
    return panel(title, [(expr, "")], unit=unit, kind="stat")


def dashboard(uid: str, title: str, rows: list[list[dict[str, Any]]]) -> dict[str, Any]:
    panels, y, pid = [], 0, 1
    for row in rows:
        w = 24 // len(row)
        h = 4 if all(p["type"] == "stat" for p in row) else 8
        for i, p in enumerate(row):
            panels.append({**p, "id": pid, "gridPos": {"x": i * w, "y": y, "w": w, "h": h}})
            pid += 1
        y += h
    return {
        "uid": uid,
        "title": title,
        "tags": ["gal-e"],
        "timezone": "Asia/Dhaka",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "templating": {
            "list": [
                {
                    "name": "datasource",
                    "type": "datasource",
                    "query": "prometheus",
                    "label": "Data source",
                }
            ]
        },
        "panels": panels,
    }


def p95(stage: str) -> str:
    return f'histogram_quantile(0.95, sum by (le) (rate(gale_stage_latency_seconds_bucket{{stage="{stage}"}}[5m])))'


OPS = dashboard(
    "gale-ops",
    "GAL-E · Ops",
    [
        [
            stat("Confirmed alert p95", "gale:confirmed_latency:p95_10m", "s"),
            stat("Alert p95", "gale:alert_latency:p95_10m", "s"),
            stat("Max bus lag", "max(gale_bus_lag_seconds)", "s"),
            stat("Open dashboard sockets", "sum(gale_ws_connections)"),
        ],
        [
            panel(
                "Per-stage latency p95 (docs/01 §5)",
                [
                    (p95(s), s)
                    for s in (
                        "frame_on_bus",
                        "asr_final",
                        "provisional_event",
                        "llm_decision",
                        "confirmed_event",
                        "alert",
                    )
                ],
                "s",
            ),
            panel(
                "Bus lag by group",
                [("max by (stream, group) (gale_bus_lag_seconds)", "{{group}} · {{stream}}")],
                "s",
            ),
        ],
        [
            panel(
                "HTTP 5xx ratio",
                [
                    (
                        'sum by (service) (rate(gale_http_requests_total{code_class="5xx"}[5m])) / sum by (service) (rate(gale_http_requests_total[5m]))',
                        "{{service}}",
                    )
                ],
                "percentunit",
            ),
            panel(
                "Handler failures / dead letters",
                [
                    ("sum by (group) (rate(gale_bus_failed_total[5m]))", "failed · {{group}}"),
                    (
                        "sum by (group) (increase(gale_bus_dead_lettered_total[15m]))",
                        "DLQ · {{group}}",
                    ),
                ],
            ),
            panel(
                "RTP ingest",
                [
                    ("sum(rate(gale_ingest_rtp_packets_total[1m]))", "packets/s"),
                    ("sum(rate(gale_ingest_rtp_lost_total[1m]))", "lost/s"),
                    ("sum(rate(gale_ingest_frames_dropped_total[1m]))", "frames dropped/s"),
                ],
            ),
        ],
        [
            panel(
                "GPU utilisation",
                [("avg by (Hostname, gpu) (DCGM_FI_DEV_GPU_UTIL)", "{{Hostname}} · {{gpu}}")],
                "percent",
            ),
            panel(
                "GPU memory used",
                [
                    (
                        "DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)",
                        "{{Hostname}} · {{gpu}}",
                    )
                ],
                "percentunit",
            ),
            panel(
                "LLM judge latency p95",
                [
                    (
                        "histogram_quantile(0.95, sum by (le, mode) (rate(gale_llm_judge_seconds_bucket[5m])))",
                        "{{mode}}",
                    )
                ],
                "s",
            ),
        ],
    ],
)

ML = dashboard(
    "gale-ml",
    "GAL-E · ML drift & quality",
    [
        [
            stat("Review false-positive ratio (24 h)", "gale:review_fp_ratio:24h", "percentunit"),
            stat("ASR confidence (1 h)", "gale:asr_confidence:avg1h", "percentunit"),
            stat(
                "LLM dismiss share (24 h)",
                'sum(increase(gale_events_total{stage="dismissed"}[24h])) / sum(increase(gale_events_total{stage="provisional"}[24h]))',
                "percentunit",
            ),
        ],
        [
            panel(
                "Signal score median by source (drift)",
                [
                    (
                        "histogram_quantile(0.5, sum by (le, source) (rate(gale_signal_score_bucket[1h])))",
                        "{{source}}",
                    )
                ],
                "percentunit",
            ),
            panel(
                "Signals per minute by source",
                [("sum by (source) (rate(gale_signals_total[5m])) * 60", "{{source}}")],
            ),
        ],
        [
            panel(
                "ASR confidence (hourly avg vs 7-day avg)",
                [
                    ("gale:asr_confidence:avg1h", "1 h"),
                    ("avg_over_time(gale:asr_confidence:avg1h[7d])", "7 d"),
                ],
                "percentunit",
            ),
            panel(
                "Lexicon hits per minute",
                [
                    (
                        'sum by (category) (rate(gale_signals_total{source="lexicon"}[5m])) * 60',
                        "{{category}}",
                    )
                ],
            ),
            panel(
                "Review verdicts per day",
                [
                    (
                        'sum by (verdict) (increase(gale_reviews_total{kind="verdict"}[1d]))',
                        "{{verdict}}",
                    )
                ],
            ),
        ],
        [
            panel(
                "Grounding rejections (LLM)",
                [("sum by (reason) (rate(gale_llm_rejected_events_total[1h]))", "{{reason}}")],
            ),
            panel(
                "Alert acceptance (reviewed correct)",
                [
                    (
                        'sum(increase(gale_reviews_total{kind="verdict",verdict="correct"}[1d])) / sum(increase(gale_reviews_total{kind="verdict"}[1d]))',
                        "accepted",
                    )
                ],
                "percentunit",
            ),
        ],
    ],
)

BUSINESS = dashboard(
    "gale-business",
    "GAL-E · Business",
    [
        [
            stat("Alerts today", "sum(increase(gale_alerts_total[1d]))"),
            stat(
                "Agent-caused alerts today", 'sum(increase(gale_alerts_total{speaker="agent"}[1d]))'
            ),
            stat(
                "Customer-caused alerts today",
                'sum(increase(gale_alerts_total{speaker="customer"}[1d]))',
            ),
        ],
        [
            panel(
                "Alerts per hour by category",
                [("sum by (category) (increase(gale_alerts_total[1h]))", "{{category}}")],
            ),
            panel(
                "Alerts per hour by speaker",
                [("sum by (speaker) (increase(gale_alerts_total[1h]))", "{{speaker}}")],
            ),
        ],
        [
            panel(
                "Confirmed events per day by category (agent)",
                [
                    (
                        'sum by (category) (increase(gale_events_total{stage="confirmed", speaker="agent"}[1d]))',
                        "{{category}}",
                    )
                ],
                kind="barchart",
            ),
            panel(
                "Confirmed events per day by category (customer)",
                [
                    (
                        'sum by (category) (increase(gale_events_total{stage="confirmed", speaker="customer"}[1d]))',
                        "{{category}}",
                    )
                ],
                kind="barchart",
            ),
        ],
    ],
)

DASHBOARDS = {"gale-ops.json": OPS, "gale-ml.json": ML, "gale-business.json": BUSINESS}


def render(d: dict[str, Any]) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in DASHBOARDS.items():
        (OUT / name).write_text(render(d), encoding="utf-8", newline="\n")
        print(f"wrote {(OUT / name).relative_to(ROOT).as_posix()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
