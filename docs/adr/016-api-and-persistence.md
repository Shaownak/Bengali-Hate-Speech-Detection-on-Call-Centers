# ADR-016: API, persistence and access control (Prompt 12)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 12), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/01 §4; docs/04 §1, §3, §4; docs/05; ADR-009, ADR-012, D-18

## Context
Prompt 12 builds `services/api`. The spec gives the table list, the endpoint list and the role table. It leaves open:
- who writes the database;
- how ids, partitions and search work in practice;
- how browsers authenticate WebSockets;
- what "two-person approval" means.

## Decision
1. **Persistence through the bus.**
   - An `api_persister` consumer group in the API process writes `call.lifecycle`, final `asr.segments`, `signals`, `events.fused` (latest state wins), `call.scores` and the new `postcall.results` to Postgres.
   - Every write is an idempotent upsert.
   - Partial ASR hypotheses are live-only.
   - The post-call worker now publishes `PostcallResult` (redacted final segments + signals), so uploaded and dropped recordings reach the database the same way as live calls.
2. **Schema deviations from docs/01 §4.**
   - Segment/signal/event ids are the pipeline's string ids (needed for idempotency), not uuids.
   - `segments` and `signals` are range-partitioned by month on `call_started_at` (PK `(id, call_started_at)`). The partition key is the call's start, so a redelivered message lands in the same partition. Partitions are created on demand, in their own transaction.
   - `segments.text_norm` = `bangla.normalize(text_redacted)` carries a GIN trigram index. The index exists only where `pg_trgm` is available; the embedded dev/test Postgres (pgserver) ships without contrib, so there search runs unindexed.
   - Additions:
     - `users.agent_id`: agent login → agent row;
     - `reviews.kind`: verdict | dispute;
     - `events.acked_by/acked_at`;
     - `model_versions.research_only` (ADR-012);
     - a `model_promotions` table;
     - `calls.duration_s`.
3. **AuthN.**
   - OIDC JWTs from Keycloak: RS256/ES256 via the realm JWKS; `aud` = `gale-api`; role from `realm_access.roles`.
   - The token decides the role. The `users` row is provisioned on first sight and holds team/agent links that admins manage.
   - `GALE_AUTH_MODE=dev` signs HS256 tokens at `POST /api/v1/dev/token`. That route is mounted only in dev mode, and the settings guard refuses dev mode outside dev/test.
4. **AuthZ.**
   - A permission table maps docs/04 §1 roles to actions, with row scoping on top:
     - admin/QA see all calls;
     - supervisors see the teams they supervise plus their own team;
     - agents see their own calls;
     - auditors see audit and aggregate reports only.
   - Out-of-scope objects answer **404**, so their existence doesn't leak.
   - A matrix test covers every endpoint × role and fails when a new endpoint lacks a case.
5. **Audit.**
   - Middleware writes `audit_log` rows for transcript and audio reads (including denied attempts and signed-URL downloads) and for every write under `/api/v1`.
   - `/ws/calls/{id}` audits on open.
   - Rows never contain transcript text.
6. **WebSockets.**
   - The first client message is `{"type":"auth","token":…}`, because browsers can't set headers and a token in the URL would end up in access logs.
   - Each API process tails the bus with plain XREAD (fan-out) and sends each socket the messages its scope allows, batched every 250 ms; within a batch only the latest state of an event is kept.
   - Close codes: 4401, 4403, 4404.
7. **Audio.**
   - `audio-url` returns short-lived signed URLs per channel (gateway-retained live audio) or for the uploaded stereo recording.
   - S3/MinIO pre-signs (`S3ObjectStore`). The local store signs with HMAC and the API serves `/api/v1/objects/…`.
8. **Rules and models.**
   - `PUT /rules` validates and stores a new active version; v1 is seeded from `configs/rules.yaml`.
   - `POST /rules/simulate` replays stored calls through the active vs the candidate rules (`gale_fusion_engine.simulate`), reusing stored LLM dismissals per event.
   - Model promotion is two-step: one admin requests, a *different* admin approves or rejects. `research_only` versions are refused.
9. **Ingest.** `POST /ingest/recordings` stores the WAV in the object store, creates the call row (`queued`) and publishes a `postcall.jobs` entry. The post-call worker fetches object-store keys to a temp file.
10. **Agent identity at ingest.**
    - The gateway accepts optional `agent_id` / `queue` (WS start, RTP registration, FreeSWITCH metadata, Asterisk `GALE_AGENT_ID` / `GALE_QUEUE` channel variables) and puts them on `call.lifecycle`.
    - Unknown agent codes are registered without a team until an admin assigns one; until then only admin/QA see those calls.
11. **Types.** `make gen-types` also dumps the API's OpenAPI document (the production surface, without dev routes) and generates `web/src/types/generated/api.ts` with `openapi-typescript`. The web app calls the API through `openapi-fetch` (`web/src/lib/api.ts`).

## Consequences
- Positive:
  - Live and post-call results land in one schema.
  - RBAC, scoping and audit are enforced server-side and tested per endpoint.
  - The web app gets compile-time-checked API calls.
- Negative / follow-ups:
  - Rule and lexicon changes are stored, but running workers load rules/lexicon at start. Hot reload (a `config.changed` message, or workers reading the DB snapshot per D-10) comes with P6.
  - The simulator can't replay per-event LLM decisions that were never stored; it approximates them from the stored event state.
  - The OIDC path is exercised against Keycloak only in staging; tests use the dev issuer.
