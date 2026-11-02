"""Model registry: picks stub or real implementations from ``GALE_MODEL_MODE``.

``load_models`` builds the stub set (galecore has no heavy dependencies). Real backends live
in the worker packages that own them and are assembled by ``gale_postcall.models``.
"""

from __future__ import annotations

from dataclasses import dataclass

from galecore.config import Settings, get_settings, load_lexicon
from galecore.models.base import (
    ASRBackend,
    AudioToxModel,
    KWSModel,
    LLMJudge,
    RegisterDetector,
    SERModel,
    ShoutDetector,
    TextClassifier,
    VADModel,
)
from galecore.models.oracle import FixtureOracle
from galecore.models.stubs import (
    StubASR,
    StubAudioTox,
    StubKWS,
    StubLLMJudge,
    StubRegister,
    StubSER,
    StubShout,
    StubTextClassifier,
    StubVAD,
)
from galecore.schemas import Category

__all__ = ["FixtureOracle", "ModelSet", "RealModelsUnavailableError", "load_models"]


class RealModelsUnavailableError(RuntimeError):
    """Raised when ``GALE_MODEL_MODE=real`` but a backend's weights or packages are missing."""


@dataclass(frozen=True, slots=True)
class ModelSet:
    vad: VADModel
    asr: ASRBackend
    shout: ShoutDetector
    ser: SERModel
    audio_tox: AudioToxModel
    kws: KWSModel
    text_clf: TextClassifier
    register: RegisterDetector
    llm: LLMJudge


def load_models(settings: Settings | None = None, oracle: FixtureOracle | None = None) -> ModelSet:
    """Stub model set. ``oracle`` keys the stubs to a synthetic fixture (see ``oracle``)."""
    s = settings or get_settings()
    if s.model_mode != "stub":
        raise RealModelsUnavailableError(
            "galecore.models.load_models only builds stubs; use gale_postcall.models.build_models"
        )
    tags: dict[str, tuple[Category, int]] = {
        term: (e.category, e.severity)
        for e in load_lexicon(s).entries
        for term in [e.term, *e.variants]
        if term.startswith("<")
    }
    return ModelSet(
        vad=StubVAD(),
        asr=StubASR(oracle),
        shout=StubShout(),
        ser=StubSER(),
        audio_tox=StubAudioTox(),
        kws=StubKWS(),
        text_clf=StubTextClassifier(tags, oracle.turn_annotations() if oracle else None),
        register=StubRegister(),
        llm=StubLLMJudge(oracle),
    )
