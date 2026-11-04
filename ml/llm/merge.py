"""Merge a judge run's LoRA adapter into its base for serving (docs/03 §7.3: merge → quantise →
vLLM).

Writes ``<out>/`` with bf16 safetensors, the tokenizer and chat template, and ``gale_model.json``
(base + revision, adapter run, research_only from the run's model card). The output is what
``quantize.py`` reads, or what vLLM / ``ml.llm.serve --base <out>`` serve directly. Runs on CPU by
default (a 4B model needs ~10 GB RAM); ``--device cuda`` is faster when the GPU is free.

    .gale/venv-llm/Scripts/python -m ml.llm.merge --base .gale/models/llm/qwen3_5_4b \
        --run qwen3_5_4b_open_v1 --out .gale/models/llm/gale-llm-judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.llm.train import RUNS


def merged_card(run_dir: Path, base: str, out: Path) -> dict[str, Any]:
    """Provenance for the merged weights: it inherits the run's licence gate."""
    card = json.loads((run_dir / "model_card.json").read_text(encoding="utf-8"))
    rev = Path(base) / "REVISION"
    return {
        "name": out.name,
        "component": "llm_judge",
        "base": base,
        "base_revision": card.get("base_revision")
        or (rev.read_text(encoding="utf-8").strip() if rev.is_file() else None),
        "adapter_run": card["name"],
        "best": card.get("best"),
        "research_only": bool(card.get("research_only", True)),
        "quantisation": None,
    }


def merge(base: str, run: str, out: Path, device: str) -> Path:  # pragma: no cover - big weights
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = RUNS / run
    tok = AutoTokenizer.from_pretrained(run_dir / "adapter")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, run_dir / "adapter").merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out)
    (out / "gale_model.json").write_text(
        json.dumps(merged_card(run_dir, base, out), indent=2) + "\n", encoding="utf-8"
    )
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - big weights
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args(argv)
    print(f"merged → {merge(a.base, a.run, a.out, a.device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
