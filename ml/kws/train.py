"""First KWS model on BAAD (docs/03 §2b, docs/09 A1): small CNN over 1.5 s log-mel windows,
multi-class over the slang terms + ``__other__`` (BAAD's confusable non-slang words, when present)
+ ``__silence__``. Every clip is passed through telephony simulation each epoch.

GPU box (full run):
    python -m ml.kws.train --manifest .gale/data/open/views/audio_manifest.jsonl --epochs 40
CPU smoke (CI on machines with torch; synthetic tone "words", no real audio):
    python -m ml.kws.train --smoke

Labels are term ids (folder names), never written to logs as text beyond their index.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from galecore.audio import resample
from galecore.config import REPO_ROOT
from ml.kws.features import SR, fix_length, log_mel, telephony

MANIFEST = REPO_ROOT / ".gale" / "data" / "open" / "views" / "audio_manifest.jsonl"
OUT = REPO_ROOT / ".gale" / "models" / "kws"
SILENCE, OTHER = "__silence__", "__other__"


def load_clips(manifest: Path, source: str = "baad") -> list[tuple[np.ndarray, str, str | None]]:
    clips = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != source:
                continue
            x, sr = sf.read(r["path"], dtype="float32", always_2d=True)
            clips.append((resample(x.mean(axis=1), sr, SR), r["label"], r.get("speaker")))
    return clips


def smoke_clips(
    n_terms: int = 3, per_term: int = 24, seed: int = 0
) -> list[tuple[np.ndarray, str, str | None]]:
    """Synthetic "words": each term is a distinct two-tone chirp pattern (no real audio)."""
    rng = np.random.default_rng(seed)
    out = []
    t = np.arange(int(0.6 * SR)) / SR
    for k in range(n_terms):
        f1, f2 = 300 + 250 * k, 900 + 300 * k
        for i in range(per_term):
            x = (
                np.sin(2 * np.pi * (f1 + 80 * t) * t) + 0.6 * np.sin(2 * np.pi * f2 * t)
            ) * np.hanning(len(t))
            out.append(
                ((0.2 * x * rng.uniform(0.5, 1.5)).astype(np.float32), f"term_{k}", f"spk{i % 6}")
            )
    return out


def split_by_speaker(
    clips: list[tuple[np.ndarray, str, str | None]], dev_frac: float = 0.15, seed: int = 0
) -> tuple[list[Any], list[Any]]:
    """Hold out whole speakers for dev, so the model can't learn voices."""
    speakers = sorted({s or "unknown" for _, _, s in clips})
    random.Random(seed).shuffle(speakers)
    dev_spk = set(speakers[: max(1, int(len(speakers) * dev_frac))])
    return [c for c in clips if (c[2] or "unknown") not in dev_spk], [
        c for c in clips if (c[2] or "unknown") in dev_spk
    ]


def build_model(n_classes: int) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, n_classes),
    )


def featurise(
    clips: list[Any], labels: list[str], rng: np.random.Generator, augment: bool
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, lab, _ in clips:
        y = (
            telephony(fix_length(x, rng=rng if augment else None), rng)
            if augment
            else fix_length(x)
        )
        xs.append(log_mel(fix_length(y)))
        ys.append(labels.index(lab))
    for _ in range(max(2, len(clips) // 10)):  # silence / line noise class
        noise = rng.normal(0, 10 ** (rng.uniform(-70, -40) / 20), int(1.5 * SR)).astype(np.float32)
        xs.append(log_mel(noise))
        ys.append(labels.index(SILENCE))
    return np.stack(xs)[:, None], np.array(ys)


def train(a: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    clips = smoke_clips() if a.smoke else load_clips(a.manifest)
    if not clips:
        raise SystemExit(
            f"no BAAD clips in {a.manifest}; place BAAD under raw/baad/ and run the open-data build"
        )
    labels = [*sorted({lab for _, lab, _ in clips}), SILENCE]
    tr, dev = split_by_speaker(clips, seed=a.seed)
    model = build_model(len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    xd, yd = featurise(dev, labels, rng, augment=True)  # dev also sounds like a phone line
    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        x, y = featurise(tr, labels, rng, augment=True)
        perm = rng.permutation(len(y))
        for i in range(0, len(y), a.batch):
            j = perm[i : i + a.batch]
            loss = torch.nn.functional.cross_entropy(
                model(torch.from_numpy(x[j])), torch.from_numpy(y[j])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(torch.from_numpy(xd)).argmax(1).numpy() == yd).mean())
        history.append({"epoch": epoch, "dev_acc": round(acc, 4)})
        if acc > best:
            best = acc
            a.out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), a.out / "kws_cnn.pt")
    card = {
        "model": "gale-kws",
        "arch": "cnn-logmel-1.5s",
        "classes": len(labels),
        "dev_acc": round(best, 4),
        "history": history,
        "train_clips": len(tr),
        "dev_clips": len(dev),
        "smoke": a.smoke,
        "research_only": True,
    }
    (a.out / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (a.out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.kws.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    card = train(a)
    print(json.dumps({k: card[k] for k in ("classes", "dev_acc", "train_clips", "dev_clips")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
