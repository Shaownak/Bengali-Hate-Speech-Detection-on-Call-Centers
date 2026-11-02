# ADR-006: Redacted text everywhere downstream of redaction

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/01 §2, §4, docs/05 §7, docs/08 intro, CLAUDE.md hard rule 1

## Context
Transcripts contain phone numbers, NIDs, account numbers, names and addresses. Downstream models, storage and the LLM don't need them.

## Decision
PII is replaced with tags (`<PHONE>`, `<NID>`, `<NAME>`, `<ADDR>`, `<ACCT>`) by `galecore.redact`. All downstream models, persistence (`segments.text_redacted`) and the LLM judge see only redacted text. Un-redacted text never persists and is never logged. Where redaction runs relative to the bus is decided in PROGRESS D-01 / Q-01.

## Consequences
- Positive: smaller privacy blast radius; LLM never sees raw PII.
- Negative: redaction errors (missed names) leak through, so recall of the redactor must be tested. Over-redaction can hide evidence.
