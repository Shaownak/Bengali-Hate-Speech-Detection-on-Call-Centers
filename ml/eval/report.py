"""Evaluation reports: JSON + Markdown + single-file HTML (with a reliability diagram), and an
MLflow run when ``mlflow`` and ``MLFLOW_TRACKING_URI`` are available (docs/03 §9, docs/06)."""

from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.eval.components import Result


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flat(v, key + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def headline(r: Result) -> dict[str, Any]:
    m = r.metrics
    pick = {
        "system": [
            "events",
            "alerts",
            "agent_sev23_alerts",
            "false_alerts_per_100_calls",
            "call_spearman",
            "call_auc_needs_review",
        ],
        "fusion": ["events", "severity_accuracy", "ece", "context_negatives"],
        "asr": ["cer", "wer", "slang_recall", "script_drift", "rtf"],
        "kws": ["recall", "false_alarms_per_hour"],
        "shout": ["events"],
        "ser": ["uar", "angry_auc"],
        "audiotox": ["toxic_auc"],
        "textclf": ["macro_f1", "ece"],
        "llm": ["decision_accuracy", "json_validity", "grounding_rate", "window_latency_s"],
    }.get(r.component, list(m))
    return {k: m.get(k) for k in pick}


def reliability_svg(cal: dict[str, Any] | None, size: int = 260) -> str:
    if not cal or not cal.get("bins"):
        return ""
    pad, w = 30, size - 40
    pts = " ".join(
        f"{pad + b['confidence'] * w:.1f},{pad + w - b['accuracy'] * w:.1f}" for b in cal["bins"]
    )
    dots = "".join(
        f'<circle cx="{pad + b["confidence"] * w:.1f}" cy="{pad + w - b["accuracy"] * w:.1f}" r="{2 + min(6, b["n"] ** 0.5):.1f}" '
        f'fill="#2563eb" opacity="0.7"><title>{b["lo"]}–{b["hi"]}: n={b["n"]}, acc={b["accuracy"]}</title></circle>'
        for b in cal["bins"]
    )
    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" role="img" aria-label="reliability diagram">'
        f'<rect x="{pad}" y="{pad}" width="{w}" height="{w}" fill="none" stroke="#999"/>'
        f'<line x1="{pad}" y1="{pad + w}" x2="{pad + w}" y2="{pad}" stroke="#bbb" stroke-dasharray="4"/>'
        f'<polyline points="{pts}" fill="none" stroke="#2563eb"/>{dots}'
        f'<text x="{pad}" y="{size - 6}" font-size="10">confidence →</text>'
        f'<text x="4" y="{pad - 8}" font-size="10">accuracy ↑ (ECE {cal.get("ece")})</text></svg>'
    )


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation: `{payload['component']}` on `{payload['split']}`",
        "",
        f"{payload['created']} · models: `{payload['models']}` · calls: {payload['calls']}",
        "",
    ]
    if payload["split"] == "fixtures":
        lines += [
            "> Synthetic fixtures in stub mode: these numbers test the harness, not model quality.",
            "",
        ]
    lines += ["## Headline", "", "| metric | value | 95 % CI |", "|---|---|---|"]
    for k, v in payload["headline"].items():
        ci = payload.get("ci", {}).get(k)
        lines.append(f"| {k} | `{json.dumps(v, ensure_ascii=False)}` | {ci or ''} |")
    if payload.get("slices"):
        s = payload["slices"]
        lines += [
            "",
            "## Fairness & robustness slices",
            "",
            f"Overall false alerts per 100 calls: **{s['overall_false_alerts_per_100_calls']}**. "
            "Flag = slice above 1.5× that rate (docs/06 §3).",
            "",
        ]
        lines += [
            f"- ⚠ `{f['slice']}={f['value']}`: {f['false_alerts_per_100_calls']} per 100 calls"
            for f in s["flags"]
        ] or ["- no flags"]
        for key, rows in s["slices"].items():
            lines += [
                "",
                f"### {key}",
                "",
                "| value | calls | P | R | F1 | false alerts / 100 calls |",
                "|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {v} | {r['calls']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['false_alerts_per_100_calls']} |"
                for v, r in rows.items()
            ]
    lines += [
        "",
        "## All metrics",
        "",
        "```json",
        json.dumps(payload["metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def to_html(payload: dict[str, Any], cal: dict[str, Any] | None) -> str:
    md = html.escape(to_markdown(payload))
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>GAL-E eval {html.escape(payload['component'])}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px}"
        "pre{white-space:pre-wrap;font-size:13px;background:#f6f6f6;padding:12px;border-radius:6px}</style></head><body>"
        f"<h1>GAL-E evaluation · {html.escape(payload['component'])} · {html.escape(payload['split'])}</h1>"
        f"{reliability_svg(cal)}<pre>{md}</pre></body></html>"
    )


def log_mlflow(payload: dict[str, Any], files: list[Path], tags: dict[str, str]) -> str | None:
    """Log to MLflow if installed and configured; returns the run id."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        return None
    mlflow.set_experiment(f"gale-eval-{payload['component']}")  # pragma: no cover - needs a server
    with mlflow.start_run(tags={"split": payload["split"], **tags}) as run:  # pragma: no cover
        mlflow.log_metrics(_flat(payload["metrics"]))
        for f in files:
            mlflow.log_artifact(str(f))
        return str(run.info.run_id)


def write(
    result: Result, payload_extra: dict[str, Any], out_dir: Path, tags: dict[str, str] | None = None
) -> dict[str, Path]:
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "component": result.component,
        "split": result.split,
        "created": created,
        "headline": headline(result),
        "metrics": result.metrics,
        **payload_extra,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{result.component}_{result.split}_{created}"
    paths = {
        "json": stem.with_suffix(".json"),
        "md": stem.with_suffix(".md"),
        "html": stem.with_suffix(".html"),
    }
    paths["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["md"].write_text(to_markdown(payload), encoding="utf-8")
    paths["html"].write_text(to_html(payload, result.calibration), encoding="utf-8")
    run_id = log_mlflow(payload, list(paths.values()), tags or {})
    if run_id:  # pragma: no cover
        payload["mlflow_run_id"] = run_id
    return paths
