# ADR-001: Multi-signal fusion over an end-to-end LLM

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/00 §1, docs/01 §1, docs/03 §8

## Context
Harshness has two carriers: lexical (what is said) and paralinguistic (how it is said). Shouting and angry delivery are lost in transcripts. ASR is weakest on profanity. Published fusion results (MADA, MuTox) beat single modalities. An LLM on every utterance is too slow and costly for real-time alerts, and is hard to explain.

## Decision
GAL-E combines independent signals into one decision: DSP shouting, SER, audio toxicity, KWS, lexicon, register rules, a text classifier, and an LLM judge. Each signal emits a calibrated probability. The fusion engine (rules + stacker + hysteresis) is the only component that decides what becomes an event or alert. The LLM is the context layer at the top. It confirms, dismisses or adds context-only events, and is never the first line of detection.

## Consequences
- Positive: fast path independent of LLM latency; explainable alerts (contributing signals); graceful degradation when any one model fails.
- Negative: more models to train, calibrate and operate; fusion needs labelled dev data to tune.
