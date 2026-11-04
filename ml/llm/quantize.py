"""Quantise a merged judge for vLLM (docs/03 §7.3: AWQ int4 or FP8), on the Linux GPU box.

Uses llm-compressor (vLLM's own quantisation library; install it in the GPU venv there:
``pip install llmcompressor``):

- ``fp8``: FP8 dynamic (weights FP8, activations per token at runtime). Needs no calibration
  data and fits L4 / H100 class GPUs.
- ``awq``: W4A16 AWQ, calibrated on judge-format windows from the **train** split (never dev,
  test or golden; CLAUDE.md rule 3). It halves memory again, for A10-class GPUs.

The output keeps the tokenizer and ``gale_model.json`` (quantisation recorded) and is served by
vLLM (``infra/helm/gal-e`` ``vllm.model``) with guided JSON decoding.

    python -m ml.llm.quantize --merged .gale/models/llm/gale-llm-judge --scheme fp8 \
        --out .gale/models/llm/gale-llm-judge-fp8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ml.llm.train import DATA, load_jsonl

SCHEMES = {"fp8": "FP8_DYNAMIC", "awq": "W4A16"}


def calibration_texts(n: int, seed: int = 7, data: Path = DATA) -> list[list[dict[str, str]]]:
    """``n`` chat transcripts (system + user + assistant) from the train split only."""
    recs = load_jsonl(data / "sft_window_train.jsonl")
    random.Random(seed).shuffle(recs)
    return [r["messages"] for r in recs[:n]]


def recipe(scheme: str) -> Any:  # pragma: no cover - needs llmcompressor
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    ignore = ["lm_head", "re:.*visual.*", "re:.*mlp.gate$"]
    if scheme == "fp8":
        return QuantizationModifier(targets="Linear", scheme=SCHEMES[scheme], ignore=ignore)
    return [AWQModifier(targets="Linear", scheme=SCHEMES[scheme], ignore=ignore)]


def quantize(merged: Path, scheme: str, out: Path, n_calib: int) -> Path:  # pragma: no cover
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(merged)
    model = AutoModelForCausalLM.from_pretrained(merged, dtype="auto", device_map="auto")
    kw: dict[str, Any] = {"model": model, "recipe": recipe(scheme)}
    if scheme == "awq":
        from datasets import Dataset

        texts = [
            tok.apply_chat_template(m, tokenize=False, enable_thinking=False)
            for m in calibration_texts(n_calib)
        ]
        kw.update(
            dataset=Dataset.from_dict({"text": texts}),
            max_seq_length=2048,
            num_calibration_samples=len(texts),
        )
    oneshot(**kw)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    card = json.loads((merged / "gale_model.json").read_text(encoding="utf-8"))
    card.update(name=out.name, quantisation=SCHEMES[scheme], calibration_windows=n_calib)
    (out / "gale_model.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--scheme", choices=sorted(SCHEMES), default="fp8")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib", type=int, default=256, help="AWQ calibration windows (train)")
    a = ap.parse_args(argv)
    print(f"quantised → {quantize(a.merged, a.scheme, a.out, a.calib)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
