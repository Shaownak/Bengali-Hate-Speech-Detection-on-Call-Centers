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
    TextClfOutput,
    TextContext,
    VADModel,
    WindowInput,
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
from galecore.schemas import AsrSegment, CallScorecard, Category, FusedEvent, WindowJudgement

__all__ = [
    "FixtureOracle",
    "ModelSet",
    "RealModelsUnavailableError",
    "VersionedLLMJudge",
    "VersionedTextClassifier",
    "load_models",
]


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


class VersionedTextClassifier(TextClassifier):
    """A classifier whose outputs carry another version label: a pilot candidate (ADR-020).

    In stub mode the candidate is the stub under its candidate version, which exercises the
    routing end to end; real candidates load from ``$GALE_MODEL_DIR/candidates/text_clf/<version>``.
    """

    def __init__(self, inner: TextClassifier, version: str) -> None:
        self.inner, self.version = inner, version

    def classify(self, ctx: TextContext) -> TextClfOutput:
        out = self.inner.classify(ctx)
        return TextClfOutput(
            scores=out.scores,
            severity=out.severity,
            aggression=out.aggression,
            model_version=self.version,
        )


class VersionedLLMJudge(LLMJudge):
    """An LLM judge reported under a candidate version (ADR-020); see ``VersionedTextClassifier``.
    Real candidates are a second LoRA adapter served by the same vLLM server."""

    def __init__(self, inner: LLMJudge, version: str) -> None:
        self.inner, self.version = inner, version

    def judge_window(self, window: WindowInput) -> WindowJudgement:
        return self.inner.judge_window(window)

    def judge_call(
        self, call_id: str, segments: list[AsrSegment], events: list[FusedEvent]
    ) -> CallScorecard:
        return self.inner.judge_call(call_id, segments, events)
