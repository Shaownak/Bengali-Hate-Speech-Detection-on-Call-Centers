# ADR-002: Redis Streams as the message bus

- **Status:** Accepted
- **Date:** 2026-11-02
- **Spec refs:** docs/01 §3, docs/05 §2, §8

## Context
Services exchange audio frames, ASR segments, signals and events with low latency. Expected scale is ≤ 2,000 concurrent channels. The team is small, so operational simplicity matters.

## Decision
Use Redis Streams with consumer groups, at-least-once delivery, and idempotency keys (`signal_id` / `event_id`). All access goes through `galecore.bus.BusClient` so the bus can be swapped for Kafka or NATS later without touching services.

## Consequences
- Positive: one simple, low-latency dependency; KEDA can scale on pending entries.
- Negative: memory-bound retention (maxlen sized for ~15 min buffering). Audio payload size must be kept small (see PROGRESS D-02). Persistence settings interact with privacy (PROGRESS D-01 / Q-01).
