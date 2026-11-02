"""Real VAD and ASR backends (``GALE_MODEL_MODE=real``). Weights come only from ``GALE_MODEL_DIR``
(placed there by ``scripts/download_models.py``); nothing is downloaded at runtime.

Heavy packages are imported lazily and are not part of the default environment (D-12):
install the ``real`` extra of gale-asr-worker (or ``nemo`` for IndicConformer) on a GPU host.
Not exercised in CI: there are no weights there (``# pragma: no cover``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from galecore.models import RealModelsUnavailableError
from galecore.models.base import ASRBackend, AudioChunk, RawSegment, RawWord, VADModel


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        raise RealModelsUnavailableError(
            f"{what}: {path} is missing; run scripts/download_models.py"
        )
    return path


class SileroVAD(VADModel):  # pragma: no cover
    """Silero VAD (ONNX) from ``$GALE_MODEL_DIR/vad/silero_vad.onnx``."""

    def __init__(self, model_dir: Path, threshold: float = 0.5) -> None:
        from silero_vad.utils_vad import OnnxWrapper

        self.model = OnnxWrapper(str(_need(model_dir / "vad" / "silero_vad.onnx", "Silero VAD")))
        self.threshold = threshold

    def speech_regions(self, chunk: AudioChunk) -> list[tuple[float, float]]:
        import torch
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(
            torch.from_numpy(chunk.samples),
            self.model,
            threshold=self.threshold,
            sampling_rate=chunk.sample_rate,
            return_seconds=True,
        )
        return [(chunk.offset_s + float(t["start"]), chunk.offset_s + float(t["end"])) for t in ts]


class FasterWhisperASR(ASRBackend):  # pragma: no cover
    """faster-whisper (CTranslate2) with the regional Bangla Whisper-medium converted to CT2
    (``$GALE_MODEL_DIR/asr/whisper_ct2``). Lexicon/domain terms go in as hotwords (docs/03 §1.3)."""

    name = "whisper_ct2"

    def __init__(
        self, model_dir: Path, hotwords: list[str] | None = None, device: str = "auto"
    ) -> None:
        from faster_whisper import WhisperModel

        path = _need(model_dir / "asr" / "whisper_ct2", "faster-whisper model")
        self.model = WhisperModel(
            str(path), device=device, compute_type="int8_float16" if device != "cpu" else "int8"
        )
        self.hotwords = " ".join(hotwords or []) or None
        self.version = f"whisper_ct2:{path.name}"

    def transcribe(self, chunk: AudioChunk) -> list[RawSegment]:
        segs, _ = self.model.transcribe(
            chunk.samples,
            language="bn",
            word_timestamps=True,
            beam_size=5,
            vad_filter=False,
            hotwords=self.hotwords,
            condition_on_previous_text=False,
        )
        out = []
        for s in segs:
            words = [
                RawWord(
                    w.word.strip(),
                    chunk.offset_s + w.start,
                    chunk.offset_s + w.end,
                    float(w.probability),
                )
                for w in (s.words or [])
                if w.word.strip()
            ]
            if not words:
                continue
            out.append(
                RawSegment(
                    chunk.channel,
                    words[0].start,
                    words[-1].end,
                    " ".join(w.word for w in words),
                    words,
                    True,
                    float(np.mean([w.conf for w in words])),
                    self.version,
                )
            )
        return out


class NemoIndicConformerASR(ASRBackend):  # pragma: no cover
    """AI4Bharat IndicConformer (NeMo hybrid CTC/RNN-T), ``$GALE_MODEL_DIR/asr/indic_conformer``.
    Word timings come from the CTC head's frame alignment."""

    name = "indic_conformer"

    def __init__(self, model_dir: Path, language: str = "bn") -> None:
        import nemo.collections.asr as nemo_asr

        path = _need(model_dir / "asr" / "indic_conformer", "IndicConformer model")
        nemo_file = next(path.glob("*.nemo"), None)
        if nemo_file is None:
            raise RealModelsUnavailableError(f"no .nemo checkpoint in {path}")
        self.model: Any = nemo_asr.models.ASRModel.restore_from(str(nemo_file))
        self.model.cur_decoder = "ctc"
        self.language = language
        self.version = f"indic_conformer:{nemo_file.stem}"

    def transcribe(self, chunk: AudioChunk) -> list[RawSegment]:
        hyp = self.model.transcribe(
            [chunk.samples],
            batch_size=1,
            language_id=self.language,
            return_hypotheses=True,
            timestamps=True,
        )[0]
        stamps = (getattr(hyp, "timestamp", None) or {}).get("word", [])
        words = [
            RawWord(
                w["word"], chunk.offset_s + float(w["start"]), chunk.offset_s + float(w["end"]), 0.9
            )
            for w in stamps
        ]
        if not words:
            text = getattr(hyp, "text", "") or ""
            if not text.strip():
                return []
            words = [RawWord(text, chunk.offset_s, chunk.offset_s + chunk.duration_s, 0.5)]
        return [
            RawSegment(
                chunk.channel,
                words[0].start,
                words[-1].end,
                " ".join(w.word for w in words),
                words,
                True,
                0.9,
                self.version,
            )
        ]
