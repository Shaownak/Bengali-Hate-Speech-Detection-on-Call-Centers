"""Score an LLM-judge run on a held-out SFT split (docs/03 §7.4, docs/08 §6).

Loads the base weights + the run's LoRA adapter (or the base alone with ``--zero-shot``, the
docs/03 §7.1 screen), generates greedily with thinking off, and writes
``.gale/reports/llm/<run>_<split>.{json,md}``: JSON validity, event P/R/F1 + per category,
decision accuracy, hallucinated-evidence rate, severity accuracy, clean-window FP rate, a
per-scenario table (incl. the prompt-injection red-team scenario) and p50/p95 latency. Reports
hold metrics only, never window text.

    .gale/venv-llm/Scripts/python -m ml.llm.evaluate --base .gale/models/llm/qwen3_5_4b \
        --run qwen3_5_4b_open_v1 --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from galecore.config import REPO_ROOT
from ml.llm.metrics import dumps, evaluate
from ml.llm.train import DATA, RUNS, load_jsonl, prompt_text

REPORTS = REPO_ROOT / ".gale" / "reports" / "llm"


def markdown(name: str, m: dict[str, Any]) -> str:
    keys = [
        "n",
        "json_validity",
        "event_precision",
        "event_recall",
        "event_f1",
        "event_macro_f1",
        "decision_accuracy",
        "hallucinated_evidence_rate",
        "severity_accuracy",
        "clean_window_fp_rate",
        "latency_p50_s",
        "latency_p95_s",
    ]
    lines = [f"# LLM judge eval: {name}", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {m.get(k)} |" for k in keys]
    lines += ["", "| category | P | R | F1 | support |", "|---|---|---|---|---|"]
    lines += [
        f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |"
        for c, v in m["per_category"].items()
    ]
    lines += ["", "| scenario | n | validity | event F1 | decisions |", "|---|---|---|---|---|"]
    lines += [
        f"| {s} | {v['n']} | {v['json_validity']} | {v['event_f1']} | {v['decision_accuracy']} |"
        for s, v in m["by_scenario"].items()
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - needs a GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_jsonl(args.data / f"sft_window_{args.split}.jsonl", args.n)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    name = f"{Path(args.base).name}_zero_shot" if args.zero_shot else args.run
    if not args.zero_shot:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, RUNS / args.run / "adapter")
    model.eval()
    outs, lat = [], []
    with torch.no_grad():
        for r in recs:
            ids = tok(
                prompt_text(tok, r["messages"]), add_special_tokens=False, return_tensors="pt"
            )
            ids = ids.to(model.device)
            t = time.perf_counter()
            g = model.generate(
                **ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            lat.append(time.perf_counter() - t)
            outs.append(tok.decode(g[0, ids["input_ids"].shape[1] :], skip_special_tokens=True))
    REPORTS.mkdir(parents=True, exist_ok=True)
    # raw outputs (open-data windows only, kept under .gale/): re-score without regenerating
    with (REPORTS / f"{name}_{args.split}_outputs.jsonl").open("w", encoding="utf-8") as f:
        for r, o, t in zip(recs, outs, lat, strict=True):
            f.write(json.dumps({"meta": r.get("meta"), "output": o, "latency_s": t}) + "\n")
    m = evaluate(recs, outs)
    lat = sorted(lat)
    m["latency_p50_s"] = round(lat[len(lat) // 2], 3)
    m["latency_p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3)
    m["split"], m["run"] = args.split, name
    (REPORTS / f"{name}_{args.split}.json").write_text(dumps(m), encoding="utf-8")
    (REPORTS / f"{name}_{args.split}.md").write_text(markdown(name, m), encoding="utf-8")
    brief = {k: v for k, v in m.items() if k not in ("per_category", "by_scenario")}
    print(json.dumps(brief, indent=2))
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", default="")
    ap.add_argument("--zero-shot", action="store_true", help="base model only (docs/03 §7.1)")
    ap.add_argument("--split", default="test", choices=["dev", "test", "redteam"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=600)
    a = ap.parse_args(argv)
    if not a.zero_shot and not a.run:
        ap.error("--run is required unless --zero-shot")
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
