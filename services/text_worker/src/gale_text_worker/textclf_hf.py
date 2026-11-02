"""Real text classifier: a fine-tuned BanglaBERT-family multi-task checkpoint (transformers).

Expects ``$GALE_MODEL_DIR/textclf/`` with a sequence-classification model whose
``config.id2label`` names our categories (multi-label sigmoid) plus optional ``severity_*``
labels. The P1 open-data track (Prompt 4b) produces it; the base encoder choice is gated by the
licence review (Q-03: ship only ``commercial_ok``). Input format per docs/03 §6:
``[AGENT] … [CUST] … [TARGET] <current utterance>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from galecore.models import RealModelsUnavailableError
from galecore.models.base import TextClassifier, TextClfOutput, TextContext
from galecore.schemas import Category, Channel

_TAG = {Channel.AGENT: "[AGENT]", Channel.CUSTOMER: "[CUST]", Channel.MIXED: "[SPK]"}


def format_input(ctx: TextContext) -> str:
    prev = " ".join(f"{_TAG[ch]} {text}" for ch, text in ctx.previous)
    return f"{prev} [TARGET] {_TAG[ctx.speaker]} {ctx.current}".strip()


class HFTextClassifier(TextClassifier):  # pragma: no cover - needs torch, transformers, weights
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = model_dir / "textclf"
        if not (path / "config.json").is_file():
            raise RealModelsUnavailableError(f"text classifier: no fine-tuned checkpoint in {path}")
        self._torch = torch
        self.tok: Any = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model: Any = (
            AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
            .eval()
            .to(device)
        )
        self.labels: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
        self.device = device
        self.version = f"textclf:{path.name}"

    def classify(self, ctx: TextContext) -> TextClfOutput:
        enc = self.tok(format_input(ctx), return_tensors="pt", truncation=True, max_length=256).to(
            self.device
        )
        with self._torch.no_grad():
            probs = self._torch.sigmoid(self.model(**enc).logits[0]).tolist()
        by_label = {self.labels[i]: p for i, p in enumerate(probs)}
        scores = {c: float(by_label.get(c.value, 0.0)) for c in Category}
        sev = max(
            (
                int(lab.split("_")[1])
                for lab, p in by_label.items()
                if lab.startswith("severity_") and p >= 0.5
            ),
            default=0,
        )
        return TextClfOutput(
            scores=scores,
            severity=sev,
            aggression=float(by_label.get("aggression", 0.0)),
            model_version=self.version,
        )
