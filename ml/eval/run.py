"""Evaluation runner (docs/06).

    make eval COMPONENT=system SPLIT=fixtures
    python -m ml.eval.run --component asr --split dev --manifest … --gold … --audio …

Splits: ``fixtures`` (synthetic calls, CI) or any split of a data-engine manifest (``dev``,
``golden``, ``negatives``, …). ``golden`` is checked against ``golden.lock`` before it loads.
Reports go to ``reports/`` (JSON, Markdown, HTML) and to MLflow when ``MLFLOW_TRACKING_URI`` is set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from galecore.config import REPO_ROOT, Settings, get_settings
from galecore.logging import configure_logging
from ml.eval import metrics as M
from ml.eval.components import EVALUATORS, Context, Result
from ml.eval.report import write
from ml.eval.sets import EvalSet, load_fixtures, load_manifest_split
from ml.eval.slices import slice_report

DATA = Path(os.environ.get("GALE_DATA_DIR", REPO_ROOT / ".gale")) / "data"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "calls"


def load_set(split: str, manifest: Path, gold: Path, audio: Path, lock: Path) -> EvalSet:
    if split == "fixtures":
        return load_fixtures(FIXTURES)
    return load_manifest_split(manifest, gold, split, audio, lock)


def confidence_intervals(r: Result) -> dict[str, tuple[float, float] | None]:
    if not r.per_call or "tp" not in r.per_call[0]:
        return {}

    def f1(units: Any) -> float | None:
        tp, fp, fn = (sum(u[k] for u in units) for k in ("tp", "fp", "fn"))
        if tp + fp + fn == 0:  # a resample with no events at all: F1 undefined, skip it
            return None
        return float(M.prf(tp, fp, fn)["f1"])

    return {"events": M.bootstrap_ci(r.per_call, f1)}


def evaluate(
    component: str,
    es: EvalSet,
    settings: Settings,
    out_dir: Path,
    tags: dict[str, str] | None = None,
) -> tuple[Result, dict[str, Path]]:
    result = EVALUATORS[component](es, Context(settings))
    extra: dict[str, Any] = {"calls": len(es.calls), "models": settings.model_mode}
    extra["ci"] = {k: list(v) if v else None for k, v in confidence_intervals(result).items()}
    if result.matched is not None and component in ("system", "fusion", "shout"):
        extra["slices"] = slice_report(es, result.matched)
    return result, write(result, extra, out_dir, tags)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.eval.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--component", choices=sorted(EVALUATORS), required=True)
    ap.add_argument("--split", default="fixtures")
    ap.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    ap.add_argument("--gold", type=Path, default=DATA / "training")
    ap.add_argument("--audio", type=Path, default=DATA / "audio")
    ap.add_argument("--lock", type=Path, default=DATA / "manifests" / "golden.lock")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    a = ap.parse_args(argv)
    settings = get_settings()
    configure_logging("eval", "WARNING", settings.log_json)
    es = load_set(a.split, a.manifest, a.gold, a.audio, a.lock)
    result, paths = evaluate(a.component, es, settings, a.out)
    print(f"{a.component} on {a.split} ({len(es.calls)} calls):")
    for k, v in result.metrics.items():
        if not isinstance(v, dict) or k in ("events", "alerts", "agent_sev23_alerts"):
            print(f"  {k}: {v}")
    for p in paths.values():
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
