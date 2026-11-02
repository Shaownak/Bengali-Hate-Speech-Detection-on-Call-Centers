"""Text-classifier training data from the open-data ``textclf_*.jsonl`` views (no torch needed).

Output heads (one sigmoid each, docs/03 §6): the 8 unified labels (``aggression`` is the auxiliary
head, D-06) plus ordinal severity as cumulative binaries ``severity_1..3`` (P(sev ≥ k)), which is
what ``gale_text_worker.textclf_hf.HFTextClassifier`` reads back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ml.data.open.schema import LABELS

SEVERITY_HEADS = ("severity_1", "severity_2", "severity_3")
HEADS = (*LABELS, *SEVERITY_HEADS)


@dataclass(frozen=True)
class Example:
    text: str
    target: list[float]  # soft targets per head
    mask: list[float]  # 1 = annotated (contributes loss), 0 = not annotated
    weight: float
    source: str
    uid: str


def format_text(text: str, context: list[str]) -> str:
    """docs/03 §6 input; open data has no speaker roles, so context turns are tagged [SPK]."""
    prev = " ".join(f"[SPK] {c}" for c in context[-2:])
    return f"{prev} [TARGET] {text}".strip()


def to_example(rec: dict[str, object]) -> Example:
    labels: dict[str, int | None] = rec["labels"]  # type: ignore[assignment]
    conf: dict[str, float] = rec.get("label_confidence") or {}  # type: ignore[assignment]
    target, mask = [], []
    for lab in LABELS:
        v = labels.get(lab)
        mask.append(0.0 if v is None else 1.0)
        # weak positive → soft target (label_confidence); negatives stay 0
        target.append(0.0 if v is None else float(v) * float(conf.get(lab, 1.0)))
    sev = rec.get("severity")
    for k in (1, 2, 3):
        mask.append(0.0 if sev is None else 1.0)
        target.append(0.0 if sev is None else float(int(sev) >= k))  # type: ignore[call-overload]
    return Example(
        format_text(str(rec["text"]), list(rec.get("context") or [])),
        target,
        mask,  # type: ignore[arg-type]
        float(rec.get("weight", 1.0)),
        str(rec.get("source")),
        str(rec.get("uid")),
    )  # type: ignore[arg-type]


def load_view(path: Path, include_synthetic: bool = True) -> list[Example]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("synthetic") and not include_synthetic:
                continue
            out.append(to_example(rec))
    return out
