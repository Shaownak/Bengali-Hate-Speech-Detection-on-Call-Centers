"""Model registry: picks stub or real implementations from ``GALE_MODEL_MODE``."""

from __future__ import annotations

from dataclasses import dataclass

from galecore.config import Settings, get_settings
from galecore.models.base import (
    ASRBackend,
    AudioToxModel,
    KWSModel,
    LLMJudge,
    RegisterDetector,
    SERModel,
    ShoutDetector,
    TextClassifier,
)
from galecore.models.stubs import (
    StubASR,
    StubAudioTox,
    StubKWS,
    StubLLMJudge,
    StubRegister,
    StubSER,
    StubShout,
    StubTextClassifier,
)

__all__ = ["ModelSet", "RealModelsUnavailableError", "load_models"]


class RealModelsUnavailableError(RuntimeError):
    """Raised when ``GALE_MODEL_MODE=real`` but real backends are not implemented/installed.

    Real backends (faster-whisper, NeMo, emotion2vec+, BanglaBERT-family, vLLM client) are
    delivered in phase P1, Prompt 3 (docs/PROGRESS.md).
    """


@dataclass(frozen=True, slots=True)
class ModelSet:
    asr: ASRBackend
    shout: ShoutDetector
    ser: SERModel
    audio_tox: AudioToxModel
    kws: KWSModel
    text_clf: TextClassifier
    register: RegisterDetector
    llm: LLMJudge


def load_models(settings: Settings | None = None) -> ModelSet:
    s = settings or get_settings()
    if s.model_mode == "stub":
        return ModelSet(
            asr=StubASR(),
            shout=StubShout(),
            ser=StubSER(),
            audio_tox=StubAudioTox(),
            kws=StubKWS(),
            text_clf=StubTextClassifier(),
            register=StubRegister(),
            llm=StubLLMJudge(),
        )
    raise RealModelsUnavailableError(
        "GALE_MODEL_MODE=real: real model backends arrive in P1 (Prompt 3); use stub for now"
    )
