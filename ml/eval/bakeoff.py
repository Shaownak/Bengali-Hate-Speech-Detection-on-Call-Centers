"""Bake-off runner (docs/06 §4): evaluate several candidates of one component on the same frozen
split with identical preprocessing, then write the ADR results table.

    python -m ml.eval.bakeoff --component asr --split dev --candidates configs/bakeoff/asr.yaml \
        --adr docs/adr/014-asr-bakeoff.md

Candidates file (YAML)::

    component: asr
    primary: cer            # metric to choose by (dotted path into the metrics)
    lower_is_better: true
    candidates:
      - name: whisper_ct2
        settings: {model_mode: real}
        models: {asr_postcall: {backend: whisper_ct2, version: tugstugi-ct2}}
      - name: indic_conformer
        settings: {model_mode: real}
        models: {asr_postcall: {backend: indic_conformer, version: 600m}}

Each candidate gets its own config dir (copy of ``configs/`` with ``models.yaml`` overridden), so
nothing but the model differs. Results are tagged ``bakeoff=<component>`` in MLflow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from galecore.config import REPO_ROOT, get_settings
from ml.eval.run import DATA, evaluate, load_set


def _get(d: dict[str, Any], dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def run_bakeoff(
    spec: dict[str, Any], split: str, out: Path, set_args: dict[str, Path]
) -> list[dict[str, Any]]:
    es = load_set(split, **set_args)  # loaded once: every candidate sees identical data
    base = get_settings()
    rows = []
    for cand in spec["candidates"]:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / ".gale") as tmp:
            cfg = Path(tmp) / "configs"
            shutil.copytree(base.config_dir, cfg)
            if cand.get("models"):
                models = yaml.safe_load((cfg / "models.yaml").read_text(encoding="utf-8"))
                models["components"].update(cand["models"])
                (cfg / "models.yaml").write_text(
                    yaml.safe_dump(models, sort_keys=False), encoding="utf-8"
                )
            settings = base.model_copy(update={"config_dir": cfg, **cand.get("settings", {})})
            result, paths = evaluate(
                spec["component"],
                es,
                settings,
                out,
                {"bakeoff": spec["component"], "candidate": cand["name"]},
            )
        rows.append(
            {
                "name": cand["name"],
                "primary": _get(result.metrics, spec["primary"]),
                "metrics": result.metrics,
                "report": str(paths["md"]),
            }
        )
    lower = bool(spec.get("lower_is_better"))
    ranked = sorted(
        (r for r in rows if r["primary"] is not None), key=lambda r: r["primary"], reverse=not lower
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return rows


def adr_table(
    spec: dict[str, Any], split: str, rows: list[dict[str, Any]], extra: list[str]
) -> str:
    head = ["candidate", spec["primary"], *extra, "rank", "report"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(rows, key=lambda r: r.get("rank", 99)):
        vals = [
            r["name"],
            r["primary"],
            *(_get(r["metrics"], e) for e in extra),
            r.get("rank", "–"),
            f"`{Path(r['report']).name}`",
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def write_adr(path: Path, spec: dict[str, Any], split: str, table: str) -> None:
    winner = (
        spec["candidates"][0]["name"] if not table else table.splitlines()[2].split("|")[1].strip()
    )
    path.write_text(
        f"""# ADR: {spec["component"]} bake-off on `{split}`

- **Status:** Proposed (pending review of latency/cost constraints)
- **Date:** {datetime.now(UTC).date().isoformat()}
- **Spec refs:** docs/06 §4, docs/03

## Context
Candidates evaluated by `python -m ml.eval.bakeoff` on the frozen `{split}` split, with
identical preprocessing.

## Results
{table}

## Decision
Highest-ranked by `{spec["primary"]}`: **{winner}**, subject to the latency and cost limits
(docs/01 §5) and the robustness slices in each report. Tie-break by slices (docs/06 §4).
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.eval.bakeoff",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--split", default="fixtures")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "bakeoff")
    ap.add_argument("--adr", type=Path)
    ap.add_argument("--extra", nargs="*", default=[], help="more metric paths for the table")
    ap.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    ap.add_argument("--gold", type=Path, default=DATA / "training")
    ap.add_argument("--audio", type=Path, default=DATA / "audio")
    ap.add_argument("--lock", type=Path, default=DATA / "manifests" / "golden.lock")
    a = ap.parse_args(argv)
    spec = yaml.safe_load(a.candidates.read_text(encoding="utf-8"))
    rows = run_bakeoff(
        spec,
        a.split,
        a.out,
        {"manifest": a.manifest, "gold": a.gold, "audio": a.audio, "lock": a.lock},
    )
    table = adr_table(spec, a.split, rows, a.extra)
    print(table)
    (a.out / f"bakeoff_{spec['component']}_{a.split}.md").write_text(table + "\n", encoding="utf-8")
    if a.adr:
        write_adr(a.adr, spec, a.split, table)
        print(f"wrote {a.adr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
