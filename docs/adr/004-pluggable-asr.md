# ADR-004: Pluggable ASR backend chosen by bake-off

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/00 §2, docs/03 §1, docs/06 §4

## Context
No Bangla ASR model is proven on 8 kHz Bangladeshi call audio. Candidates (tugstugi Whisper-medium, IndicConformer-600M, Qwen3-ASR-0.6B-bn) trade off accuracy, dialect robustness, streaming fitness and the CTC posteriors KWS needs.

## Decision
`ASRBackend` is an interface with implementations `whisper_ct2`, `indic_conformer`, `qwen3_asr` and `stub`. The active backend per mode (real-time / post-call) is set in `configs/models.yaml`, with no code change. The choice is made by a bake-off on the golden set (CER, WER, slang recall, script drift, RTF, p95 latency) and recorded in its own ADR.

## Consequences
- Positive: we can swap or upgrade ASR as evidence changes. Real-time and post-call can use different backends.
- Negative: we maintain several backends. The shared normaliser must be used for all scoring.
