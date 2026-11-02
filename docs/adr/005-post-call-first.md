# ADR-005: Post-call first, real-time second

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/01 intro, docs/07 phases P1/P4

## Context
Post-call QA delivers value without PBX integration, gives data for tuning, and has relaxed latency. Real-time adds transport, streaming ASR and strict latency budgets.

## Decision
Build the post-call pipeline first (P1) as a library reused by the services. Real-time mode (P4) uses the same models and fusion logic. It only adds streaming transport and stricter alert thresholds (`thr_alert` per category × speaker × mode).

## Consequences
- Positive: early end-to-end demo (M1); one code path to test.
- Negative: worker logic must be written streaming-friendly from the start (stateful per-speaker baselines, incremental windows).
