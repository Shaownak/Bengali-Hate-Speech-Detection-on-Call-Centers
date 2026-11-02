# ADR-019: Threat model (STRIDE)

- **Status:** Accepted (review before go-live with the pen-test findings)
- **Date:** 2026-11-02
- **Spec refs:** docs/05 §7; CLAUDE.md hard rules 1–4; ADR-003, ADR-006, ADR-016, ADR-018

## Assets
1. Call audio (both speakers, may contain personal data).
2. Un-redacted transcript text: exists **only in memory** in asr-worker → text-worker (ADR-006).
3. Redacted transcripts, signals, events, scorecards (Postgres).
4. Agent performance data: sensitive for the employees; alerts are advisory (rule 4).
5. The real lexicon (slurs; loaded from `GALE_LEXICON_PATH`, never in git).
6. Model weights and the golden set (integrity: rule 3).
7. Credentials: bus, DB, object store, URL-signing key, ARI, OIDC.

## Trust boundaries
PBX network ↔ gateway · cluster namespaces (`gale-ingest`, `gale-ml`, `gale-app`, `gale-data`,
`gale-obs`) · browser ↔ ingress ↔ api/web · cluster ↔ secret store / object store / Keycloak.

## STRIDE

| # | Threat | Where | Mitigations (implemented) | Residual / to do |
|---|---|---|---|---|
| S1 | **Spoofing** a PBX: injecting fake RTP/WS audio | gateway | NetworkPolicy `gateway-pbx` (PBX CIDRs only); LB `loadBalancerSourceRanges`; RTP sessions only for registered SSRC / source address | No media auth in RTP. VPN/private link between PBX and cluster required (docs/05 §2); consider SRTP |
| S2 | Spoofing a user/role | api, WS | OIDC JWT (RS/ES256 via JWKS, `aud`, `iss`, `exp` checked); dev issuer refused outside dev/test (settings guard); WS auth as first message, not in the URL | Keycloak hardening (MFA for admin/QA) is the realm owner's job |
| T1 | **Tampering** with events/reviews | api, DB | RBAC per endpoint + row scoping (tests: full matrix); reviews append-only; audit rows for every write | DB superuser access is ops-only (CNPG), reviewed quarterly |
| T2 | Tampering with model/rules config | admin | Two-person promotion; `research_only` never promotable; rules versioned with rollback; weights pinned by hash (rule 6) | Model artefact signing (cosign) not yet |
| T3 | Supply chain (images, deps, actions) | CI | Base images pinned by digest; `uv.lock`/`pnpm-lock` frozen; Trivy image + fs + IaC scans; SBOMs; trivy action pinned by commit | Renovate for controlled bumps; image signing |
| R1 | **Repudiation** of a transcript/audio view or a change | api | `audit_log` for every transcript/audio read (incl. denied), every write, WS live views, DSAR and retention | Ship audit rows to WORM storage for long-term evidence |
| I1 | **Information disclosure**: raw text or PII in logs/traces | all | structlog scrubber; PII-in-logs lint in CI; spans carry only `gale.call_id`; OTel collector drops body-like attributes; Loki has no body extraction | Periodic log sampling review by the DPO |
| I2 | Raw text to an external LLM | llm_judge | Only redacted text reaches the judge (ADR-006); judge is self-hosted vLLM (ADR-003); ML namespace has **no internet egress** | — |
| I3 | Over-broad access to calls | api, web | Supervisors see their teams only, agents their own calls; out-of-scope = 404 (no existence leak); signed audio URLs, TTL 5 min | Quarterly access review (docs/05 §7) |
| I4 | Data at rest | data | SSE-KMS on the bucket; Postgres disk encryption by the platform; secrets via External Secrets/SOPS; nothing secret in values or git (test) | KMS key ownership with the client |
| I5 | Lexicon exposure (real slurs) | admin UI | Terms masked by default; only admin/QA can read; lexicon file outside the repo (rule 2) | — |
| D1 | **Denial of service**: flood of calls/frames | gateway, bus | Bounded gateway queue (drops + alert); stream MAXLEN; priority classes (real time over batch); per-shard isolation | Rate limiting at the ingress for the API |
| D2 | Bus memory exhaustion | Valkey | `maxmemory` + `noeviction` (writes fail loudly rather than silently evicting); lag alerts | Size from the capacity report |
| E1 | **Elevation** via container escape | all pods | non-root UID, read-only root fs, all capabilities dropped, seccomp RuntimeDefault, PSA `restricted` on every namespace, no service-account tokens mounted | vLLM needs a writable root fs (kernel cache): isolated on its own GPU pool |
| E2 | Lateral movement between namespaces | cluster | Default-deny NetworkPolicies; only the listed flows; ML/data have no internet egress | mTLS between services (Linkerd) recommended, not yet enabled |
| E3 | Misuse of alerts against staff | product | Alerts are advisory; no automatic punitive action anywhere in the code (rule 4); every alert reviewable; agents can dispute | Works-council / HR policy is outside the system |

## Privacy-specific (PDPO 2025)
- Retention and DSAR are implemented but **execute only after the DPO approves** the numbers
  (`configs/retention.yaml`, ADR-018). Backups keep data until their own retention.
- Everything that touches call data runs in-country (ADR-003).

## Before go-live
Pen-test (external), DPO sign-off on retention + DSAR procedure, Keycloak realm review, VPN/private
link for the PBX media path.
