"""Training-code tests. Data/metrics/features need numpy only; the trainer smoke tests run where the
opt-in ``ml-train`` group (torch, transformers) is installed and are skipped elsewhere (CI)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from ml.data.open.schema import LABELS
from ml.kws.features import N_MELS, SR, fix_length, log_mel, mel_filterbank, telephony
from ml.kws.train import smoke_clips, split_by_speaker
from ml.textclf.data import HEADS, format_text, to_example
from ml.textclf.metrics import macro, per_head_f1, source_metric


def rec(
    labels: dict[str, int | None],
    sev: int | None = 2,
    conf: dict[str, float] | None = None,
    src: str = "violens",
) -> dict[str, object]:
    full = dict.fromkeys(LABELS)
    full.update(labels)
    return {
        "uid": "u",
        "text": "বাক্য",
        "context": ["আগের কথা", "তার আগের"],
        "labels": full,
        "label_confidence": conf or {},
        "severity": sev,
        "weight": 0.3,
        "source": src,
    }


def test_example_masks_soft_targets_and_ordinal_severity() -> None:
    e = to_example(
        rec({"insult": 1, "threat": 0, "aggression": 1}, sev=2, conf={"aggression": 0.6})
    )
    t = dict(zip(HEADS, e.target, strict=True))
    m = dict(zip(HEADS, e.mask, strict=True))
    assert (t["insult"], m["insult"]) == (1.0, 1.0)
    assert (t["threat"], m["threat"]) == (0.0, 1.0)  # annotated negative
    assert m["slang_profanity"] == 0.0  # not annotated → no loss
    assert t["aggression"] == 0.6  # weak mapping → soft target
    assert [t["severity_1"], t["severity_2"], t["severity_3"]] == [1.0, 1.0, 0.0]  # P(sev ≥ k)
    assert to_example(rec({}, sev=None)).mask[-3:] == [0.0, 0.0, 0.0]
    assert e.weight == 0.3
    assert format_text("এখন", ["এক", "দুই", "তিন"]) == "[SPK] দুই [SPK] তিন [TARGET] এখন"


def test_metrics() -> None:
    exs = [
        to_example(rec({"insult": 1, "threat": 0})),
        to_example(rec({"insult": 0, "threat": 1})),
        to_example(rec({"insult": 0, "threat": 0})),
    ]
    idx = {h: i for i, h in enumerate(HEADS)}
    probs = np.zeros((3, len(HEADS)))
    probs[0, idx["insult"]] = 0.9
    probs[1, idx["threat"]] = 0.9
    scores = per_head_f1(probs, exs)
    assert scores["insult"] == 1.0
    assert scores["threat"] == 1.0
    assert "slang_profanity" not in scores  # never annotated here
    assert source_metric("violens", probs, exs) == 1.0
    assert (
        source_metric("banglamultihate", probs, exs) == 1.0
    )  # micro over T1 labels only (threat is not one)
    assert source_metric("other", probs, exs) is None
    assert macro({"a": 1.0, "b": 0.5}) == 0.75


def test_kws_features() -> None:
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, 201)
    assert (fb.sum(axis=1) > 0).all()
    x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    assert log_mel(fix_length(x)).shape == (148, N_MELS)
    assert len(fix_length(x[:100])) == int(1.5 * SR)
    y = telephony(x, np.random.default_rng(0))
    assert abs(len(y) - len(x)) <= 2
    assert np.abs(y).max() < 1.0
    clips = smoke_clips(n_terms=2, per_term=12)
    tr, dev = split_by_speaker(clips)
    assert {c[2] for c in tr}.isdisjoint({c[2] for c in dev})


def test_textclf_tiny_training(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ml.textclf.train import train

    data = tmp_path / "views"
    data.mkdir()
    rows = []
    for i in range(60):
        bad = i % 2 == 0
        text = f"তুই একটা <GALI_1> নম্বর {i}" if bad else f"আমার বিল নম্বর {i} এসেছে"
        r = rec({"insult": int(bad), "threat": 0}, sev=2 if bad else 0, src="violens")
        r.update({"uid": f"v:{i}", "text": text, "context": [], "research_only": True})
        rows.append(r)
    for split in ("train", "dev", "test"):
        (data / f"textclf_{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    args = argparse.Namespace(
        data=data,
        out=tmp_path / "m",
        base="",
        tiny=True,
        epochs=8,
        batch=16,
        lr=2e-5,
        patience=8,
        max_train=0,
        seed=1,
        cpu=True,
    )
    card = train(args)
    assert card["research_only"] is True
    assert card["promotable"] is False
    assert card["dev_macro_f1"] > 0.5
    assert (tmp_path / "m" / "config.json").is_file()
    cfg = json.loads((tmp_path / "m" / "config.json").read_text(encoding="utf-8"))
    assert list(cfg["id2label"].values()) == list(HEADS)  # what HFTextClassifier reads


def test_kws_smoke_training(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ml.kws.train import train

    card = train(
        argparse.Namespace(
            manifest=tmp_path / "none.jsonl",
            out=tmp_path / "kws",
            smoke=True,
            epochs=4,
            batch=16,
            lr=3e-3,
            seed=3,
        )
    )
    assert card["classes"] == 4  # 3 synthetic terms + silence
    assert card["dev_acc"] > 0.3
    assert (tmp_path / "kws" / "kws_cnn.pt").is_file()


def test_translit_tiny_training(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ml.translit.train import cer, train

    pairs = tmp_path / "p.jsonl"
    pairs.write_text(
        "".join(
            json.dumps({"split": "train", "roman": f"ami {i}", "bangla": f"আমি {i}"}) + "\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )
    card = train(
        argparse.Namespace(
            pairs=pairs,
            out=tmp_path / "t",
            base="",
            tiny=True,
            epochs=1,
            batch=8,
            lr=3e-4,
            max_train=0,
            seed=1,
        )
    )
    assert card["research_only"] is True
    assert (tmp_path / "t" / "config.json").is_file()
    assert cer(["আমি"], ["আমি"]) == 0.0
    assert cer(["আম"], ["আমি"]) == pytest.approx(1 / 3)
