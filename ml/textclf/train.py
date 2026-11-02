"""Open-data-only multi-task text classifier (docs/03 §6 stage 1, docs/09 §7).

Masked multi-label focal BCE (γ = 2) with soft targets for weak mappings, per-row domain weights,
an auxiliary ``aggression`` head and cumulative ordinal severity heads; early stopping on dev
macro-F1; per-source report on the official test splits (``public_aux`` suite, docs/06) against
the docs/09 §8 reference numbers.

GPU box (full run):
    python -m ml.textclf.train --base $GALE_MODEL_DIR/encoders/banglabert_large --epochs 5
CPU smoke (no downloads, tiny random encoder + tokenizer built from the data):
    python -m ml.textclf.train --tiny --max-train 2000 --epochs 1

The saved checkpoint is what ``gale_text_worker.textclf_hf.HFTextClassifier`` loads
(``$GALE_MODEL_DIR/textclf``). ``model_card.json`` carries ``research_only`` (licence gate).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from galecore.config import REPO_ROOT
from ml.textclf.data import HEADS, Example, load_view
from ml.textclf.metrics import REFERENCE, TOLERANCE, macro, per_head_f1, source_metric

DATA = REPO_ROOT / ".gale" / "data" / "open" / "views"
OUT = REPO_ROOT / ".gale" / "models" / "textclf"


def masked_focal_bce(logits: Any, target: Any, mask: Any, weight: Any, gamma: float = 2.0) -> Any:
    """Focal BCE over annotated heads only; soft targets; per-row (domain) weights."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    loss = ((1 - pt) ** gamma) * bce * mask * weight[:, None]
    return loss.sum() / mask.sum().clamp(min=1.0)


def tiny_model(texts: list[str], out: Path) -> tuple[Any, Any]:
    """A 2-layer random BERT and a WordPiece tokenizer trained on ``texts`` (offline smoke runs)."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    out.mkdir(parents=True, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
    wp.train_from_iterator(
        texts,
        vocab_size=2000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    # wrap the trained object: transformers 5 ignores BertTokenizerFast(vocab_file=...)
    specials = {f"{k}_token": f"[{k.upper()}]" for k in ("unk", "pad", "cls", "sep", "mask")}
    tok = PreTrainedTokenizerFast(tokenizer_object=wp._tokenizer, **specials)
    if tok.vocab_size < 100:
        raise RuntimeError(f"tokenizer vocabulary has only {tok.vocab_size} entries")
    cfg = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
        num_labels=len(HEADS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(HEADS)),
        label2id={h: i for i, h in enumerate(HEADS)},
    )
    return tok, BertForSequenceClassification(cfg)


def base_model(base: str) -> tuple[Any, Any]:  # pragma: no cover - needs downloaded weights
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        local_files_only=True,
        num_labels=len(HEADS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(HEADS)),
        label2id={h: i for i, h in enumerate(HEADS)},
    )
    return tok, model


def predict(
    model: Any, tok: Any, examples: list[Example], device: str, batch: int = 64
) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(examples), batch):
            enc = tok(
                [e.text for e in examples[i : i + batch]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            out.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(HEADS)))


def evaluate(model: Any, tok: Any, examples: list[Example], device: str) -> dict[str, Any]:
    probs = predict(model, tok, examples, device)
    heads = per_head_f1(probs, examples)
    report: dict[str, Any] = {
        "n": len(examples),
        "per_head_f1": heads,
        "macro_f1": macro(heads),
        "sources": {},
    }
    by_src: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        by_src[e.source].append(i)
    for src, idx in sorted(by_src.items()):
        sub = [examples[i] for i in idx]
        entry: dict[str, Any] = {"n": len(idx), "per_head_f1": per_head_f1(probs[idx], sub)}
        if src in REFERENCE:
            name, ref = REFERENCE[src]
            got = source_metric(src, probs[idx], sub)
            entry.update(
                {
                    "metric": name,
                    "value": got,
                    "reference": ref,
                    "below_reference": got is not None and got < ref - TOLERANCE,
                }
            )
        report["sources"][src] = entry
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tr = load_view(args.data / "textclf_train.jsonl")
    dev = load_view(args.data / "textclf_dev.jsonl", include_synthetic=False)
    test = load_view(args.data / "textclf_test.jsonl", include_synthetic=False)
    if args.max_train:
        random.shuffle(tr)
        tr, dev, test = (
            tr[: args.max_train],
            dev[: args.max_train // 5],
            test[: args.max_train // 2],
        )
    research_only = any(
        json.loads(line).get("research_only")
        for line in (args.data / "textclf_train.jsonl").open(encoding="utf-8")
    )
    tok, model = tiny_model([e.text for e in tr], args.out) if args.tiny else base_model(args.base)
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr if not args.tiny else 1e-3, weight_decay=0.01
    )
    steps = max(1, len(tr) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.1
    )
    best, best_epoch, history = -1.0, -1, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(tr)
        total = 0.0
        for i in range(0, len(tr) - args.batch + 1 or 1, args.batch):
            b = tr[i : i + args.batch]
            enc = tok(
                [e.text for e in b],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            loss = masked_focal_bce(
                model(**enc).logits,
                torch.tensor([e.target for e in b], device=device),
                torch.tensor([e.mask for e in b], device=device),
                torch.tensor([e.weight for e in b], device=device),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
        dev_rep = evaluate(model, tok, dev, device)
        history.append(
            {"epoch": epoch, "train_loss": round(total, 4), "dev_macro_f1": dev_rep["macro_f1"]}
        )
        print(
            f"epoch {epoch}: loss {total:.3f}, dev macro-F1 {dev_rep['macro_f1']:.3f}", flush=True
        )
        if dev_rep["macro_f1"] > best:  # early stopping on dev macro-F1
            best, best_epoch = dev_rep["macro_f1"], epoch
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
        elif epoch - best_epoch >= args.patience:
            break
    test_rep = evaluate(model, tok, test, device)
    card = {
        "model": "gale-textclf",
        "stage": "open-data-only (docs/03 §6 stage 1)",
        "base": "tiny-random" if args.tiny else args.base,
        "heads": list(HEADS),
        "research_only": research_only,
        "promotable": not research_only,
        "train_rows": len(tr),
        "best_epoch": best_epoch,
        "history": history,
        "dev_macro_f1": best,
        "public_aux_test": test_rep,
        "seconds": round(time.time() - t0, 1),
        "device": device,
    }
    (args.out / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.textclf.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--base", default="")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if not a.tiny and not a.base:
        ap.error("--base <encoder dir> (or --tiny for a smoke run)")
    card = train(a)
    print(
        json.dumps({k: card[k] for k in ("research_only", "dev_macro_f1", "best_epoch", "seconds")})
    )
    for src, e in card["public_aux_test"]["sources"].items():
        if "metric" in e:
            flag = (
                "  ⚠ more than 5 points below reference: debug preprocessing"
                if e["below_reference"]
                else ""
            )
            if a.tiny:  # a random 2-layer encoder is not expected to reach the published numbers
                flag = "  (tiny smoke model: compare only runs with a pretrained encoder)"
            print(f"  {src}: {e['metric']} = {e['value']} (reference ≈ {e['reference']}){flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
