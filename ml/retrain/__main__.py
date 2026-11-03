"""Monthly retraining pipeline (docs/02 §6 step 3; ADR-020). Each step is one Argo task
(``infra/helm/gal-e/templates/retrain.yaml``); ``run`` chains them locally (``make retrain``).

    python -m ml.retrain prepare  --component fusion --feedback a.jsonl … --out W
    python -m ml.retrain train    --component fusion --work W --version calib-2026-12
    python -m ml.retrain evaluate --component fusion --work W [--split golden]
    python -m ml.retrain gate     --component fusion --work W --version calib-2026-12
    python -m ml.retrain publish  --component fusion --work W --version calib-2026-12
    python -m ml.retrain run      --component fusion --feedback … --work W --version …

``prepare --store-days 30`` reads the nightly exports from the object store instead of files.

- **prepare** merges the nightly feedback exports (``python -m gale_api export-feedback``),
  dedupes, drops golden calls and fails if any remain (CLAUDE.md rule 3), splits train/dev by call.
- **train**: ``fusion`` refits per-(source, category) calibration (CPU, D-13); ``text_clf``
  fine-tunes the production checkpoint on the feedback view (GPU, ``--group ml-train``);
  ``llm_judge`` LoRA arrives with Prompt 9. With ``GALE_MODEL_MODE=stub`` the GPU trainers write a
  stub artefact so the whole workflow can be exercised without a GPU.
- **evaluate** runs ``ml.eval`` for production and candidate on the same set, the candidate through
  a model directory where only this component is replaced.
- **gate** applies ``configs/promotion_gate.yaml`` and writes ``candidate.json`` for
  ``python -m gale_api register-model``; exit code 1 when the gate fails.
- **publish** copies a passed candidate to ``$GALE_MODEL_DIR/candidates/<component>/<version>/``,
  where the workers load shadow/canary candidates from.

Nothing here reads or writes the golden manifests' content beyond call ids, and nothing is
promoted automatically: registration creates a shadow candidate for the two-person approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from galecore.config import REPO_ROOT, Settings, get_settings
from ml.retrain.gate import compare, load_specs, to_markdown

MODEL_SUBDIR = {"text_clf": "textclf", "fusion": "fusion", "llm_judge": "llm"}
EVAL_DIR = REPO_ROOT / ".gale" / "data"


class TrainerUnavailableError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def golden_ids(manifest: Path | None) -> set[str]:
    if manifest is None:
        return set()
    if not manifest.exists():
        raise SystemExit(f"golden manifest {manifest} not found (CLAUDE.md rule 3)")
    return {str(r["call_id"]) for r in _jsonl(manifest) if r.get("split") == "golden"}


def _is_dev(call_id: str, frac: float = 0.2) -> bool:
    return int(hashlib.sha256(call_id.encode()).hexdigest()[:8], 16) / 2**32 < frac


# --- steps --------------------------------------------------------------------------------------


def prepare(feedback: list[Path], work: Path, golden: set[str]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for f in feedback:
        for r in _jsonl(f):
            rows[r["uid"]] = r  # later exports win (a re-review changes the label)
    kept = [r for r in rows.values() if r["call_id"] not in golden]
    overlap = {r["call_id"] for r in kept} & golden
    if overlap:  # pragma: no cover - defensive, the filter above makes this empty
        raise SystemExit("training data overlaps the golden set (CLAUDE.md rule 3)")
    train = [r for r in kept if not _is_dev(r["call_id"])]
    dev = [r for r in kept if _is_dev(r["call_id"])]
    work.mkdir(parents=True, exist_ok=True)
    _write_jsonl(work / "train.jsonl", train)
    _write_jsonl(work / "dev.jsonl", dev)
    stats = {
        "rows": len(kept),
        "train": len(train),
        "dev": len(dev),
        "positives": sum(r["label"] for r in kept),
        "golden_dropped": len(rows) - len(kept),
        "golden_overlap": len(overlap),
        "sources": [f.name for f in feedback],
    }
    (work / "prepare.json").write_text(json.dumps(stats, indent=2), "utf-8")
    return stats


def fetch_feedback(days: int, dest: Path, settings: Settings) -> list[Path]:
    """The nightly ``datasets/feedback/<day>_<next day>.jsonl`` exports of the last ``days`` days
    (missing days are skipped: no reviews that night, or the job failed)."""
    from datetime import UTC, datetime, timedelta

    from galecore.objectstore import store_from_settings

    store = store_from_settings(settings)
    dest.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date()
    out = []
    for i in range(days, 0, -1):
        d = today - timedelta(days=i)
        key = f"datasets/feedback/{d}_{d + timedelta(days=1)}.jsonl"
        if store.exists(key):
            f = dest / key.rsplit("/", 1)[-1]
            f.write_bytes(store.get(key))
            out.append(f)
    return out


def publish(component: str, work: Path, version: str, model_dir: Path) -> Path:
    cand = json.loads((work / "candidate.json").read_text("utf-8"))
    if not cand["gate"]["passed"]:
        raise SystemExit("the gate failed: nothing to publish")
    dest = model_dir / "candidates" / component / version
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(work / "candidate", dest)
    return dest


def textclf_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Feedback rows → ``ml.textclf`` view records (open-data schema, docs/09 §5)."""
    from ml.data.open.schema import LABELS

    out = []
    for r in rows:
        if not r.get("text"):
            continue
        labels: dict[str, int | None] = dict.fromkeys(LABELS)
        if r["label"] and r["category"] in labels:
            labels[r["category"]] = 1
        elif not r["label"] and r.get("predicted_category") in labels:
            labels[r["predicted_category"]] = 0  # an annotated negative for what was predicted
        out.append(
            {
                "uid": r["uid"],
                "text": r["text"],
                "context": r.get("context") or [],
                "labels": labels,
                "severity": r.get("severity") if r["label"] else 0,
                "source": "feedback",
                "weight": 1.0,
            }
        )
    return out


def train(component: str, work: Path, version: str, settings: Settings) -> dict[str, Any]:
    out = work / "candidate"
    out.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(work / "train.jsonl")
    card: dict[str, Any] = {"component": component, "version": version, "rows": len(rows)}
    if component == "fusion":
        from ml.fusion.calibrate import fit

        cal = fit(rows, version)
        (out / "calibration.json").write_text(json.dumps(cal, indent=2), "utf-8")
        card["calibrators"] = len(cal["calibrators"])
    elif settings.model_mode == "stub":
        card["stub"] = True  # GPU trainers: placeholder artefact to exercise the workflow
    elif component == "text_clf":  # pragma: no cover - GPU box
        from ml.textclf.train import main as train_textclf

        view = work / "view"
        view.mkdir(exist_ok=True)
        _write_jsonl(view / "textclf_train.jsonl", textclf_view(rows))
        dev = textclf_view(_jsonl(work / "dev.jsonl"))
        _write_jsonl(view / "textclf_dev.jsonl", dev)
        _write_jsonl(view / "textclf_test.jsonl", dev)
        base = settings.model_dir / MODEL_SUBDIR[component]
        rc = train_textclf(["--data", str(view), "--out", str(out), "--base", str(base)])
        if rc:
            raise SystemExit(rc)
    else:
        raise TrainerUnavailableError(
            f"no real trainer for {component} yet (LLM LoRA recipe: Prompt 9 / docs/03 §7)"
        )
    (out / "model_card.json").write_text(json.dumps(card, indent=2), "utf-8")
    return card


def overlay_model_dir(base: Path, component: str, artefact: Path, dest: Path) -> Path:
    """A model directory equal to ``base`` except this component's folder, which is ``artefact``.
    Other components are symlinked (copied where symlinks aren't allowed)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    sub = MODEL_SUBDIR[component]
    if base.exists():
        for child in base.iterdir():
            if child.name in (sub, "candidates"):
                continue
            try:
                (dest / child.name).symlink_to(child, target_is_directory=child.is_dir())
            except OSError:  # pragma: no cover - Windows without symlink rights
                (shutil.copytree if child.is_dir() else shutil.copy2)(child, dest / child.name)
    shutil.copytree(artefact, dest / sub)
    return dest


def evaluate(
    component: str,
    work: Path,
    split: str,
    settings: Settings,
    eval_paths: dict[str, Path],
    limit: int = 0,
) -> dict[str, Any]:
    from ml.eval.run import evaluate as run_eval
    from ml.eval.run import load_set
    from ml.eval.sets import EvalSet

    p = eval_paths
    es = load_set(split, p["manifest"], p["gold"], p["audio"], p["lock"])
    if limit:  # smoke runs only: the gate needs the whole golden set
        es = EvalSet(es.name, es.calls[:limit])
    specs = load_specs(component)
    cand_dir = overlay_model_dir(
        settings.model_dir, component, work / "candidate", work / "modeldir"
    )
    variants = {
        "baseline": settings,
        "candidate": settings.model_copy(update={"model_dir": cand_dir}),
    }
    if component == "llm_judge":  # the candidate LoRA adapter is served under its version name
        version = json.loads((work / "candidate" / "model_card.json").read_text("utf-8"))["version"]
        variants["candidate"] = variants["candidate"].model_copy(update={"llm_model": version})
    reports: dict[str, dict[str, str]] = {}
    for name, s in variants.items():
        reports[name] = {}
        for ev in specs:
            _, paths = run_eval(ev, es, s, work / "eval" / name, tags={"variant": name})
            reports[name][ev] = str(paths["json"])
    (work / "eval" / "reports.json").write_text(json.dumps(reports, indent=2), "utf-8")
    return reports


def gate(component: str, work: Path, version: str, name: str) -> tuple[bool, dict[str, Any]]:
    reports = json.loads((work / "eval" / "reports.json").read_text("utf-8"))
    load = lambda d: {ev: json.loads(Path(p).read_text("utf-8")) for ev, p in d.items()}  # noqa: E731
    base, cand = load(reports["baseline"]), load(reports["candidate"])
    res = compare(load_specs(component), base, cand)
    (work / "gate.md").write_text(to_markdown(component, version, res), "utf-8")
    prep = json.loads((work / "prepare.json").read_text("utf-8"))
    card = json.loads((work / "candidate" / "model_card.json").read_text("utf-8"))
    candidate = {
        "component": component,
        "name": name,
        "version": version,
        "artifact_uri": os.environ.get(
            "GALE_CANDIDATE_URI", f"models/candidates/{component}/{version}/"
        ),
        "research_only": bool(card.get("research_only", False)),
        "mlflow_run_id": os.environ.get("MLFLOW_RUN_ID"),
        "baseline_version": os.environ.get("GALE_BASELINE_VERSION"),
        "golden": {ev: r["headline"] for ev, r in cand.items()},
        "gate": res.to_dict(),
        "training": {
            "rows": prep["rows"],
            "golden_overlap": prep["golden_overlap"],
            "golden_dropped": prep["golden_dropped"],
            "stub": bool(card.get("stub", False)),
        },
    }
    (work / "candidate.json").write_text(json.dumps(candidate, indent=2), "utf-8")
    return res.passed, candidate


# --- CLI ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.retrain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("step", choices=["prepare", "train", "evaluate", "gate", "publish", "run"])
    ap.add_argument("--component", choices=sorted(MODEL_SUBDIR), required=True)
    ap.add_argument("--work", type=Path, required=True, help="working directory of this run")
    ap.add_argument("--feedback", type=Path, nargs="*", default=[])
    ap.add_argument("--store-days", type=int, default=0, help="read feedback from the object store")
    ap.add_argument("--golden-manifest", type=Path, default=EVAL_DIR / "manifests" / "calls.jsonl")
    ap.add_argument("--no-golden-manifest", action="store_true", help="dev/test only")
    ap.add_argument("--version", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--split", default="golden")
    ap.add_argument("--manifest", type=Path, default=EVAL_DIR / "manifests" / "calls.jsonl")
    ap.add_argument("--gold", type=Path, default=EVAL_DIR / "training")
    ap.add_argument("--audio", type=Path, default=EVAL_DIR / "audio")
    ap.add_argument("--lock", type=Path, default=EVAL_DIR / "manifests" / "golden.lock")
    ap.add_argument("--limit", type=int, default=0, help="evaluate the first N calls (smoke)")
    a = ap.parse_args(argv)
    s = get_settings()
    steps = ["prepare", "train", "evaluate", "gate"] if a.step == "run" else [a.step]
    if a.no_golden_manifest and s.env not in ("dev", "test"):
        raise SystemExit("--no-golden-manifest is for GALE_ENV=dev/test only (CLAUDE.md rule 3)")
    if a.limit and a.split == "golden":
        raise SystemExit("--limit is for smoke runs: the gate compares the whole golden set")
    if ({"train", "gate", "publish"} & set(steps)) and not a.version:
        raise SystemExit("--version is required for train/gate")
    passed = True
    for step in steps:
        if step == "prepare":
            gold = set() if a.no_golden_manifest else golden_ids(a.golden_manifest)
            files = list(a.feedback)
            if a.store_days:
                files += fetch_feedback(a.store_days, a.work / "feedback", s)
            print("prepare:", json.dumps(prepare(files, a.work, gold)))
        elif step == "train":
            print("train:", json.dumps(train(a.component, a.work, a.version, s)))
        elif step == "evaluate":
            paths = {"manifest": a.manifest, "gold": a.gold, "audio": a.audio, "lock": a.lock}
            evaluate(a.component, a.work, a.split, s, paths, a.limit)
            print(f"evaluate: reports in {a.work / 'eval'}")
        elif step == "publish":
            print(f"publish: {publish(a.component, a.work, a.version, s.model_dir)}")
        else:
            passed, cand = gate(a.component, a.work, a.version, a.name or f"{a.component} retrain")
            print(f"gate: {'PASSED' if passed else 'FAILED'} → {a.work / 'gate.md'}")
            if not passed:
                for r in cand["gate"]["regressions"]:
                    print(
                        f"  regressed: {r['eval']}.{r['metric']} {r['baseline']} → {r['candidate']}"
                    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
