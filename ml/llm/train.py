"""LoRA fine-tuning of the LLM judge on SFT windows (docs/03 §7.3, docs/08 §4).

bf16 LoRA (r 16, α 32, dropout 0.05, all linear layers), lr 1e-4 cosine, loss on assistant tokens
only, packing off, thinking off (the prompt ends in an empty think block, as at serving time). The
loop is plain PyTorch over PEFT so it runs on native Windows (no Unsloth / flash-attn needed):

- batch 1 + gradient accumulation, no padding;
- logits are computed only at assistant positions (``logits_to_keep``), so a 150k–250k vocabulary
  never materialises full-sequence logits: the difference between fitting a 4B model on a 12 GB
  card or not;
- every ``--eval-every`` optimiser steps the adapter generates on ``--eval-n`` dev windows and is
  scored with ``ml.llm.metrics`` (JSON validity, event F1, decisions, grounding); the best adapter
  by event F1 (ties → validity) is kept.

Runs in the GPU venv (``make llm-venv``): torch CUDA + transformers + peft. Model weights come
from a local dir (``scripts/download_models.py``), never fetched at train time when a path is
given. Logs carry metrics and counts only, never window text.

    .gale/venv-llm/Scripts/python -m ml.llm.train --base .gale/models/llm/qwen3_5_4b \
        --run qwen3_5_4b_open_v1
    # smoke: --base .gale/models/llm/qwen3_5_0_8b --max-train 200 --epochs 1 --eval-n 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

from galecore.config import REPO_ROOT

DATA = REPO_ROOT / ".gale" / "data" / "llm"
RUNS = REPO_ROOT / ".gale" / "models" / "llm" / "runs"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def prompt_text(tok: Any, messages: list[dict[str, str]]) -> str:
    """System + user rendered with the generation prompt, thinking off (Qwen3.x template
    takes ``enable_thinking``; other templates ignore the extra kwarg)."""
    return str(
        tok.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    )


def end_of_turn(tok: Any) -> str:
    """The text the template puts after an assistant message (``<|im_end|>`` for Qwen)."""
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "ANSWER"},
    ]
    full = str(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False))
    tail = full.split("ANSWER", 1)[1]
    return tail.rstrip("\n") or (tok.eos_token or "")


def encode(tok: Any, rec: dict[str, Any], eot: str) -> tuple[list[int], int]:
    """(input ids, length of the prompt part): the loss covers the rest (answer + end of turn)."""
    p = tok(prompt_text(tok, rec["messages"]), add_special_tokens=False)["input_ids"]
    a = tok(rec["messages"][2]["content"] + eot, add_special_tokens=False)["input_ids"]
    return p + a, len(p)


def generate(model: Any, tok: Any, recs: list[dict[str, Any]], max_new: int = 600) -> list[str]:
    import torch

    model.eval()
    outs = []
    with torch.no_grad():
        for r in recs:
            ids = tok(
                prompt_text(tok, r["messages"]), add_special_tokens=False, return_tensors="pt"
            )
            ids = ids.to(model.device)
            g = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            outs.append(tok.decode(g[0, ids["input_ids"].shape[1] :], skip_special_tokens=True))
    model.train()
    return outs


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:  # pragma: no cover - needs a GPU
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(args.base)
    kw: dict[str, Any] = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.load_4bit:  # QLoRA fallback (docs/03 §7.3 advises against it for Qwen3.5: ADR-021)
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, device_map={"": 0}, **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 adapter weights, bf16 base under autocast
    return model, tok


def stay_awake() -> None:
    """Windows: keep the PC from idle-sleeping while this process trains (released when it
    exits; no power setting changes)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def train(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - needs a GPU
    import torch

    stay_awake()
    from transformers import get_cosine_schedule_with_warmup

    from ml.llm.metrics import evaluate

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    run = RUNS / args.run
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "train.log").open("a", encoding="utf-8")

    def emit(**kv: Any) -> None:
        line = json.dumps({"t": round(time.time(), 1), **kv})
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    train_recs = load_jsonl(args.data / "sft_window_train.jsonl", args.max_train)
    if args.with_calls:  # call-mode scorecards (ml.llm.sft_call), mixed into the same epoch
        train_recs += load_jsonl(args.data / "sft_call_train.jsonl", args.with_calls)
    dev_recs = load_jsonl(args.data / "sft_window_dev.jsonl")
    eval_recs = random.Random(1).sample(dev_recs, min(args.eval_n, len(dev_recs)))

    model, tok = load_model(args)
    eot = end_of_turn(tok)
    data = [encode(tok, r, eot) for r in train_recs]
    kept = [d for d in data if len(d[0]) <= args.max_len]
    emit(
        event="data",
        train=len(train_recs),
        kept=len(kept),
        too_long=len(data) - len(kept),
        max_tokens=max(len(d[0]) for d in data),
        mean_tokens=round(sum(len(d[0]) for d in data) / len(data), 1),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps = math.ceil(len(kept) * args.epochs / args.accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)
    emit(event="start", optimizer_steps=steps, trainable=sum(p.numel() for p in params))

    best: dict[str, Any] = {"event_f1": -1.0}
    step, micro, t0, tokens, run_loss = 0, 0, time.time(), 0, 0.0

    def do_eval(tag: str) -> None:
        nonlocal best
        t = time.time()
        outs = generate(model, tok, eval_recs, args.max_new)
        m = evaluate(eval_recs, outs)
        brief = {k: v for k, v in m.items() if k not in ("per_category", "by_scenario")}
        emit(event="eval", step=step, tag=tag, seconds=round(time.time() - t, 1), **brief)
        better = (m["event_f1"], m["json_validity"] or 0) > (
            best["event_f1"],
            best.get("json_validity") or 0,
        )
        if better:
            best = {**m, "step": step}
            model.save_pretrained(run / "adapter")
            tok.save_pretrained(run / "adapter")
            (run / "eval_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(kept)))
        rng.shuffle(order)
        for i in order:
            ids, n_prompt = kept[i]
            x = torch.tensor([ids], device=model.device)
            # position t predicts token t+1: keep logits at n_prompt-1 … len-2
            keep = torch.arange(n_prompt - 1, len(ids) - 1, device=model.device)
            target = x[0, n_prompt:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, logits_to_keep=keep).logits[0]
            loss = torch.nn.functional.cross_entropy(logits.float(), target) / args.accum
            loss.backward()
            run_loss += loss.item()
            tokens += len(ids)
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = time.time() - t0
                emit(
                    event="train",
                    epoch=epoch,
                    step=step,
                    of=steps,
                    loss=round(run_loss / args.log_every, 4),
                    lr=sched.get_last_lr()[0],
                    tok_s=round(tokens / dt, 1),
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    eta_min=round(dt / step * (steps - step) / 60, 1),
                )
                run_loss = 0.0
            if args.eval_every and step % args.eval_every == 0:
                do_eval(f"step{step}")
    do_eval("final")
    card = {
        "name": args.run,
        "component": "llm_judge",
        "base": str(args.base),
        "base_revision": args.base_revision,
        "lora": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout},
        "quantised_base": "nf4" if args.load_4bit else None,
        "data": {
            "train": str(args.data / "sft_window_train.jsonl"),
            "train_sha256": hashlib.sha256(
                (args.data / "sft_window_train.jsonl").read_bytes()
            ).hexdigest(),
            "windows": len(kept),
        },
        "epochs": args.epochs,
        "lr": args.lr,
        "best": {k: best.get(k) for k in ("step", "event_f1", "json_validity")},
        # open-data prototype (ADR-012 / D-20): never promotable
        "research_only": True,
        "run_tag": "research",
    }
    (run / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    emit(event="done", best_step=best.get("step"), best_event_f1=best.get("event_f1"))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True, help="local model dir (or HF id for a smoke run)")
    ap.add_argument("--base-revision", default=None, help="pinned commit of the base weights")
    ap.add_argument("--run", required=True, help="run name under .gale/models/llm/runs/")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--with-calls", type=int, default=0, help="add N call-mode scorecards")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=100, help="optimiser steps; 0 = end only")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (nf4) base: ADR-021")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)
    train(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
