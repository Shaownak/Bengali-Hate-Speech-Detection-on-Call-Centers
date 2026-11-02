# ADR-003: Self-hosted models only for call data

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/00 §8, docs/05 §1, §7, CLAUDE.md hard rule 1

## Context
Call audio and transcripts are personal data under Bangladesh's PDPO 2025. Confidential data may need to stay in-country, and cross-border transfer is restricted.

## Decision
Every model that touches call data runs on self-hosted infrastructure in Bangladesh. No external or cloud AI APIs are used for call audio or text, even redacted. Weights are loaded only from `GALE_MODEL_DIR` and are pinned by commit hash. ML and data namespaces have no outbound internet in prod.

## Consequences
- Positive: data residency and minimisation; no third-party processor.
- Negative: we procure and operate GPUs. Teacher distillation is limited to self-hosted models.
