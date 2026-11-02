# ADR-007: Human-in-the-loop, advisory alerts only

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/02 §6, docs/04 §2.4, §2.6, docs/06 §6, CLAUDE.md hard rule 4

## Context
Model errors (dialect, loud speakers, banter) can harm agents if acted on automatically. Supervisor trust and agent acceptance depend on transparency.

## Decision
Alerts are advisory. The system never takes automatic punitive action. Every alert shows its evidence and is reviewable. Agents can see and dispute their own events. Review verdicts feed the active-learning and retraining loop.

## Consequences
- Positive: errors are caught by people; reviews generate training data; supports coaching-only policy.
- Negative: needs reviewer time, and alert precision must be high enough to keep that workload acceptable.
