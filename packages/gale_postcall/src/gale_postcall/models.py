"""Assemble the pipeline's models for ``GALE_MODEL_MODE`` (stub | real).

Stub: galecore's deterministic stubs (keyed to a fixture oracle when one is given).
Real: backends named in ``configs/models.yaml``; components still marked ``stub`` there (KWS,
audio-tox until P3) keep their stub. DSP shouting and the register rules are real code in both
modes. ASR and the LLM judge must be real in real mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from gale_audio_worker.shout import DSPShoutDetector
from gale_text_worker.register import RuleRegisterDetector
from galecore.config import Settings, load_lexicon
from galecore.config import load_models as load_models_config
from galecore.models import FixtureOracle, ModelSet, RealModelsUnavailableError, load_models
from galecore.models.base import (
    ASRBackend,
    AudioToxModel,
    KWSModel,
    LLMJudge,
    RegisterDetector,
    SERModel,
    TextClassifier,
    VADModel,
)


@dataclass
class PipelineModels:
    vad: VADModel
    asr: ASRBackend
    shout: DSPShoutDetector
    ser: SERModel
    audio_tox: AudioToxModel
    kws: KWSModel
    text_clf: TextClassifier
    register: RegisterDetector
    llm: LLMJudge
    versions: dict[str, str]


def build_models(settings: Settings, oracle: FixtureOracle | None = None) -> PipelineModels:
    stubs = load_models(settings.model_copy(update={"model_mode": "stub"}), oracle)
    if settings.model_mode == "stub":
        return PipelineModels(
            vad=stubs.vad,
            asr=stubs.asr,
            shout=DSPShoutDetector(stubs.vad, settings.model_dir),
            ser=stubs.ser,
            audio_tox=stubs.audio_tox,
            kws=stubs.kws,
            text_clf=stubs.text_clf,
            register=RuleRegisterDetector(),
            llm=stubs.llm,
            versions={"mode": "stub", "oracle": "fixture" if oracle else "none"},
        )
    return _real(settings, stubs)  # pragma: no cover - needs weights / GPU packages


def _real(settings: Settings, stubs: ModelSet) -> PipelineModels:  # pragma: no cover
    from gale_asr_worker.backends import FasterWhisperASR, NemoIndicConformerASR, SileroVAD
    from gale_audio_worker.ser import Emotion2vecSER
    from gale_llm_judge.client import OpenAICompatibleJudge
    from gale_text_worker.textclf_hf import HFTextClassifier

    comps = load_models_config(settings.config_dir).components
    md = settings.model_dir

    def backend(name: str) -> str:
        return comps[name].backend if name in comps else "stub"

    vad = SileroVAD(md)
    asr_name = backend("asr_postcall")
    hotwords = [e.term for e in load_lexicon(settings).entries if not e.term.startswith("<")]
    if asr_name == "whisper_ct2":
        asr: ASRBackend = FasterWhisperASR(md, hotwords)
    elif asr_name == "indic_conformer":
        asr = NemoIndicConformerASR(md)
    else:
        raise RealModelsUnavailableError(
            f"models.yaml asr_postcall backend {asr_name!r} is not a real ASR"
        )
    if backend("llm_judge") != "openai":
        raise RealModelsUnavailableError(
            "models.yaml llm_judge backend must be 'openai' (self-hosted vLLM)"
        )
    s = stubs
    return PipelineModels(
        vad=vad,
        asr=asr,
        shout=DSPShoutDetector(vad, md),
        ser=Emotion2vecSER(md) if backend("ser") == "emotion2vec" else s.ser,
        audio_tox=s.audio_tox,
        kws=s.kws,
        text_clf=HFTextClassifier(md) if backend("text_clf") == "hf" else s.text_clf,
        register=RuleRegisterDetector(),
        llm=OpenAICompatibleJudge(
            settings.llm_base_url,
            settings.llm_model,
            window_timeout_s=settings.llm_window_timeout_s,
            call_timeout_s=settings.llm_call_timeout_s,
        ),
        versions={"mode": "real", **{k: f"{v.backend}:{v.version}" for k, v in comps.items()}},
    )
