# 05 — Infrastructure, Security & Operations

## 1. Environments
| Env | Purpose | Infra |
|---|---|---|
| `dev` | local development | docker-compose (CPU profile with tiny/stub models; GPU profile optional) |
| `staging` | integration, load tests, shadow models | Kubernetes namespace, 1–2 GPU nodes, synthetic + consented data only |
| `prod` | live | Kubernetes in-country (on-prem DC or Bangladesh-hosted cloud), GPU node pool |

All environments **in Bangladesh** for anything touching real call data (ADR-003, PDPO 2025).

## 2. Kubernetes layout
- Namespaces: `gale-ingest`, `gale-ml`, `gale-app`, `gale-data`, `gale-obs`.
- Helm chart `infra/helm/gal-e` with per-service values; images built per service (multi-stage, non-root, distroless/slim, SBOM via Syft, scan via Trivy).
- GPU: NVIDIA GPU Operator; node pools `gpu-l4` (ASR, SER, KWS, text-clf GPU), `gpu-llm` (vLLM). Consider MIG/time-slicing for small models.
- Autoscaling: **KEDA** ScaledObjects on Redis Streams pending-entries per consumer group (`asr.segments` lag, `audio.frames` lag, LLM request queue). HPA on CPU for api/text-worker.
- Stateful: PostgreSQL (CloudNativePG operator, 3 replicas, PITR backups), Redis (Sentinel or Redis Operator, AOF on), MinIO (distributed, SSE-KMS) or existing S3-compatible store, Keycloak.
- Networking: ingress-nginx + cert-manager (TLS); NetworkPolicies default-deny; PBX → ingest-gateway over private network/VPN only; mTLS between services (Linkerd) recommended.

## 3. Capacity planning (method — fill with measured numbers)
```
channels = 2 × peak_concurrent_calls
GPU_ASR  = channels × RTF_asr / streams_per_gpu_at_latency_target
GPU_SER  = channels / windows_per_sec_capacity  (windows every 1 s per active-speech channel)
LLM_rps  = active_calls × (utterances_per_min / 60) × debounce_factor(≈0.3)
GPU_LLM  = LLM_rps / measured_rps_per_gpu_at_p95≤2.5s
```
Run `tools/loadtest` with the simulator at 25/50/100/200% of expected peak; record RTF, p95 per stage, GPU util, cost per 1,000 call-minutes. Post-call mode can run on the same GPUs off-peak (night batch) — priority classes so real-time pre-empts batch.

## 4. Observability
- **Metrics** (Prometheus): per-stage latency histograms, bus lag, events/alerts per minute by category, model inference time, GPU util/mem (DCGM exporter), WS connections, error counters, **drift metrics** (score distributions per signal per day, ASR avg confidence, OOV rate, lexicon hit rate).
- **Tracing** (OpenTelemetry → Tempo/Jaeger): trace id = `call_id`; spans per stage so a single alert can be traced end-to-end.
- **Logs** (Loki): structured JSON; **no transcript text or PII in logs** (lint rule + log scrubber).
- **Dashboards**: Ops (latency, lag, errors), ML (score drift, precision from reviews, alert acceptance rate), Business (incidents per 100 calls, by team/queue).
- **Alerting** (Alertmanager): bus lag > 10 s, p95 alert latency > 8 s, GPU OOM, error rate > 1%, alert volume anomaly (±3σ), review false-positive rate > 30% over 24 h, ASR confidence drop > 10%.

## 5. MLOps
- **DVC** (remote = MinIO bucket `gale-dvc`) for datasets/manifests; **MLflow** (Postgres backend, MinIO artefacts) for experiments + registry.
- Pipelines (Argo Workflows or simple Kubernetes Jobs + Makefile): `prepare-data → train → eval-golden → export → register`.
- **Retraining cadence**: monthly (text-clf, fusion calibration, LLM LoRA), quarterly (ASR, SER, KWS) or on drift alert.
- **Shadow & canary**: shadow for ≥ 1 week on live traffic; canary 10% of calls → 100% after sign-off.
- **Golden set lock**: golden test manifests are immutable and hashed; CI fails if any training manifest overlaps golden call ids or speakers.

## 6. CI/CD (GitHub Actions or GitLab CI)
- On PR: ruff + ruff-format, mypy --strict (packages/services), pytest (unit + contract tests for schemas), ESLint + tsc + Vitest (web), Docker build, Trivy scan, schema→TS types drift check, **PII-in-logs lint**.
- On main: build & push images (tag = git sha), deploy to staging via Helm, run integration suite with simulator, Playwright e2e, small load test.
- Prod deploy: manual approval, Helm upgrade with `--atomic`, automatic rollback on failed readiness.

## 7. Security & privacy controls
- **Data minimisation**: un-redacted text exists only in memory inside asr-worker → text-worker; only redacted text persists or reaches LLM.
- Encryption: TLS 1.2+ everywhere; SSE-KMS on object store; Postgres TDE/disk encryption; secrets via Kubernetes Secrets + external secret store (Vault/SOPS).
- Access: OIDC SSO, RBAC (see GUI spec), least-privilege service accounts, signed short-lived audio URLs, full audit log, quarterly access review.
- Retention jobs (cron): delete raw audio after policy days; purge redacted transcripts/events after policy months; DSAR deletion endpoint (admin).
- **No outbound internet** from ML/data namespaces in prod (egress deny); model weights pulled into an internal registry/MinIO during build.
- Model supply chain: pin model revisions by commit hash; verify checksums; record licences in `ml/MODEL_LICENSES.md`.
- Pen-test before go-live; threat model (STRIDE) in `docs/adr/`.

## 8. Reliability
- SLOs: real-time pipeline availability 99.5% during business hours; confirmed-alert p95 ≤ 6 s; post-call reports within 10 min of hang-up for 95% of calls.
- Degradation modes: if LLM unavailable → fast-path provisional alerts only (UI shows "LLM offline"); if ASR overloaded → shed real-time for low-priority queues, keep post-call; if GPU node lost → KEDA reschedules, bus buffers (Redis maxlen sized for 15 min).
- Backups: Postgres PITR (RPO 15 min), MinIO versioning + replication; DR runbook tested quarterly.

## 9. Runbooks (create in `docs/runbooks/`)
`bus-lag-high.md`, `gpu-oom.md`, `llm-latency.md`, `false-alarm-spike.md`, `model-rollback.md`, `pbx-stream-drop.md`, `dsar-deletion.md`, `key-rotation.md`.
