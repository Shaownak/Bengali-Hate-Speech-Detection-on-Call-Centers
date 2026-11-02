# ADR-018: Production infrastructure (Prompt 14)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 14), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/05 §1–9; ADR-002, ADR-003, ADR-009, ADR-015, ADR-016

## Context
docs/05 describes the target cluster. Building it surfaced several issues the spec doesn't resolve:
- how per-call stateful workers scale;
- how the irreversible retention deletes are gated;
- which bus and object-store images can be bundled;
- how to validate all of this without a cluster or Docker on the dev machine (ADR-009).

## Decision
1. **Call-affine sharding instead of KEDA for real-time workers.**
   - asr, audio, text, llm and fusion keep per-call state: utterance buffers, rings, register baselines, debounce, fusion state. Several replicas in one consumer group would split a call across pods.
   - These workers therefore run as **StatefulSets**. Replica *i* of *n* (`GALE_SHARD_COUNT`; the ordinal comes from the hostname) owns the calls with `crc32(call_id) % 8 % n == i`.
   - Each replica reads through its own group `<service>-<i>`, skips calls it doesn't own, and subscribes only to its own audio shards.
   - Scaling is a reviewed `helm upgrade` of `shards` (≤ 8; runbook bus-lag-high.md). Resharding replays call state (ADR-015).
   - KEDA scales only the stateless post-call worker on `postcall.jobs` backlog. The lag metric and alert drive the decision to reshard.
   - The API's persister stays a shared group: its writes are idempotent upserts.
2. **One gateway per PBX.** RTP and ARI sessions live in memory, so a gateway runs as a single replica per PBX/ARI application with a `-rtp` UDP LoadBalancer.
3. **Bundled stores.**
   - Bus: Valkey 8 (BSD-3, Redis-protocol compatible) instead of Redis 7.4, which moved to RSAL/SSPL. AOF fsync every second, `noeviction`.
   - Postgres: CloudNativePG, 3 instances, PITR to S3 with `archive_timeout` 900 s (RPO 15 min), `pg_trgm` created at bootstrap.
   - Object store: always an external S3-compatible store (MinIO stopped publishing free images in 2025); `objectStore.mode: local` is for dev and kind only.
4. **Secrets.** External Secrets Operator by default (`ClusterSecretStore gale-vault`). SOPS-encrypted manifests are the fallback. Nothing secret in values or git (a test checks).
5. **Images.** Multi-stage, non-root, read-only root fs, base images pinned by digest (Python 3.12.16-slim, Node 22.23.0-slim, uv 0.13.6). Web uses the Next.js standalone server. CycloneDX SBOMs come from Trivy, so no extra action has to be trusted.
6. **Migrations.** A Helm hook Job (`post-install`, `pre-upgrade`). The API's `/readyz` reports `schema: false` until the database is at the Alembic head, so new pods don't take traffic early.
7. **Retention and DSAR are irreversible, so they are gated.**
   - Retention reads `configs/retention.yaml` and only reports unless the policy says `approved: true` (DPO) **and** the job runs with `--execute` (Helm `retention.execute`).
   - DSAR is an admin endpoint that defaults to a dry run, needs a reason, and stores only hashes of the selectors in the audit log.
   - Transcripts and signals are dropped by whole monthly partition.
8. **Observability.**
   - New metrics in `galecore.metrics`: bus lag/pending per group, events/alerts by category, signal score histograms (drift), ASR confidence, review verdicts, HTTP requests, WebSocket connections.
   - Spans of one call share trace id = the call UUID across services. OTLP export is on only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
   - Rules and dashboards live in the chart. A test checks they reference only defined metrics, and dashboards are generated (`make dashboards`).
9. **Validation without local Docker/helm.**
   - Static tests run everywhere: values, digests, rules ↔ metrics, dashboards, templates.
   - CI runs `helm lint/template` × 4 value sets + kubeconform (CRD catalog), `promtool check rules`, Trivy (image, fs, config), and a **kind smoke deploy** that replays a synthetic call through the cluster and checks the API persisted events.
   - Binaries in CI are pinned and checksum-verified.
10. **Load tests.**
    - `tools/loadtest` drives the simulator over WebSocket at 25/50/100/200 % of peak, scrapes `/metrics`, and computes per-stage p95 from histogram deltas.
    - It writes `capacity-report.md` with the docs/05 §3 formula. GPU rows stay "TBD" until a staging GPU run.
    - Latencies only count at real-time speed.
11. **Ingress.** ingress-nginx was retired upstream in 2026. The chart's Ingress is controller-agnostic: class and annotations come from values, and only cert-manager is assumed.

## Consequences
- Correct per-call state under horizontal scale, at the price of manual (reviewed) resharding for real-time workers.
- Nothing irreversible runs without an explicit, recorded human decision.
- Validated here: static checks and unit/integration tests. Validated in CI only: chart rendering, promtool, image builds, the kind deploy. Not validated at all yet: a real cluster, GPUs, Keycloak, KEDA/CNPG/ESO behaviour.
