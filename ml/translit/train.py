"""Back-transliteration model: Banglish (Roman) → Bangla script, trained on BanglaTLit (docs/09 R4).

GPU box (full run, e.g. BanglaT5 or ByT5-small from GALE_MODEL_DIR):
    python -m ml.translit.train --base $GALE_MODEL_DIR/encoders/banglat5 --epochs 5
CPU smoke (tiny random byte-level T5, no downloads):
    python -m ml.translit.train --tiny --epochs 3
Inference (used by ``ml.data.open`` when $GALE_MODEL_DIR/translit exists):
    python -m ml.translit.train --infer "ami tomake valobashi"

The pairs come from ``python -m ml.data.open build`` (views/translit_pairs.jsonl).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from galecore.bangla import normalize
from galecore.config import REPO_ROOT

PAIRS = REPO_ROOT / ".gale" / "data" / "open" / "views" / "translit_pairs.jsonl"
OUT = REPO_ROOT / ".gale" / "models" / "translit"


def load_pairs(path: Path) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {"train": [], "dev": [], "test": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r.get("split") or "train"].append((r["roman"], normalize(r["bangla"])))
    if not out["dev"]:  # carve a dev set when the release has none
        random.Random(0).shuffle(out["train"])
        k = max(1, len(out["train"]) // 20)
        out["dev"], out["train"] = out["train"][:k], out["train"][k:]
    return out


def cer(pred: list[str], gold: list[str]) -> float:
    import difflib

    errs = sum(
        sum(
            max(i2 - i1, j2 - j1)
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, g, p).get_opcodes()
            if op != "equal"
        )
        for p, g in zip(pred, gold, strict=True)
    )
    return errs / max(1, sum(len(g) for g in gold))


def tiny_model() -> tuple[Any, Any]:
    from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

    tok = ByT5Tokenizer()
    cfg = T5Config(
        vocab_size=384,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return tok, T5ForConditionalGeneration(cfg)


def generate(model: Any, tok: Any, texts: list[str], device: str, batch: int = 32) -> list[str]:
    import torch

    model.eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                texts[i : i + batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
            out += [normalize(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return out


def train(a: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_pairs(a.pairs)
    if a.tiny:
        tok, model = tiny_model()
    else:  # pragma: no cover - needs weights
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(a.base, local_files_only=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr if not a.tiny else 3e-3)
    tr = data["train"][: a.max_train or None]
    for epoch in range(a.epochs):
        model.train()
        random.Random(a.seed + epoch).shuffle(tr)
        total = 0.0
        for i in range(0, len(tr), a.batch):
            src, tgt = zip(*tr[i : i + a.batch], strict=True)
            enc = tok(
                list(src), return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            lab = tok(
                list(tgt), return_tensors="pt", padding=True, truncation=True, max_length=256
            ).input_ids.to(device)
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print(f"epoch {epoch}: loss {total:.3f}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    dev = data["dev"][:500]
    pred = generate(model, tok, [s for s, _ in dev], device)
    card = {
        "model": "gale-translit",
        "base": "tiny-byt5" if a.tiny else a.base,
        "train_pairs": len(tr),
        "dev_cer": round(cer(pred, [g for _, g in dev]), 4),
        "research_only": True,
    }
    (a.out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.translit.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--base", default="")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--infer", nargs="*", help="transliterate these strings with the model in --out"
    )
    a = ap.parse_args(argv)
    if a.infer is not None:
        from ml.data.open.translit import load_transliterator

        for src, out in zip(a.infer, load_transliterator(a.out.parent)(a.infer), strict=True):
            print(f"{src}\t{out}")
        return 0
    if not a.tiny and not a.base:
        ap.error("--base <seq2seq model dir> (or --tiny for a smoke run)")
    print(json.dumps(train(a)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
