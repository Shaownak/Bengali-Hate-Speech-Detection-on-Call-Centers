# 07 — Roadmap, Team & Milestones

Timeline assumes a small team and that call-data access + legal sign-off happen in weeks 1–3. Phases overlap; the critical path is **data annotation → model fine-tuning → pilot**.

## Team (minimum)
| Role | FTE | Notes |
|---|---|---|
| Tech lead / ML engineer | 1 | owns architecture, ASR/KWS/SER |
| ML/NLP engineer | 1 | text-clf, LLM judge, fusion, eval |
| Backend/platform engineer | 1 | services, bus, telephony, k8s |
| Front-end engineer | 0.5–1 | dashboard |
| Annotation lead (from QA) | 1 | guide owner, adjudication |
| Annotators | 6–10 (part-time) | dialect coverage |
| Legal/DPO, Call-centre ops sponsor | part-time | sign-offs |

Claude Code accelerates all engineering work; it cannot replace data access, annotation, GPU procurement, PBX configuration or sign-offs.

## Hardware (starting point)
- Training: 2 × 48 GB GPUs (or 1 × 80 GB) workstation/server.
- Staging: 1–2 × L4/A10 (24 GB).
- Prod (≈ 200–300 concurrent calls, to be validated by load test): 3–4 × L4/A10 for ASR/SER/KWS/text + 1–2 × L40S (48 GB) or equivalent for vLLM. Scale linearly from load-test measurements.

## Phases

| Phase | Weeks | Engineering deliverables [CODE] | Human deliverables [HUMAN] | Exit gate |
|---|---|---|---|---|
| **P0 Foundations** | 1–2 | Monorepo scaffold, CLAUDE.md, galecore (schemas, config, bus, audio & Bangla utils), docker-compose dev stack, CI, call simulator, synthetic fixtures | Legal basis & DPIA started; data-access approval; dataset licence review; hardware ordered | `make dev-up` runs; CI green |
| **P1 Offline MVP (pretrained + open-data models)** | 2–5 | Batch pipeline: WAV in → per-channel ASR (bake-off backends), DSP shouting, emotion2vec+ zero-shot, lexicon, register rules, LLM zero-shot judge, fusion v0 → JSON report + simple HTML report. **Open-data track (`09_OPEN_DATASETS.md`):** ingest + harmonise all open datasets, train BanglaBERT multi-task classifier on open data only, mine slang-lexicon candidates, train first KWS on BAAD, LLM judge few-shot/LoRA on open synthetic windows | Seed lexicon; 50-call pilot annotation to validate guide | Report runs on 100 real calls; first baseline numbers |
| **P2 Data engine** | 3–8 | Ingestion from recorder, manifests, redaction, Label Studio templates + import/export, augmentation (codec sim), eval harness + golden-set lock, DVC/MLflow | T1/T2/T3 annotation at volume; golden & dev sets double-annotated | golden_v1/dev_v1 frozen; α ≥ 0.7 |
| **P3 Model training** | 6–12 | ASR fine-tune (top 2 backends), KWS (CTC search + classifier), shouting LightGBM, SER head, audio-tox head, BanglaBERT multi-task, LLM SFT (Unsloth), calibration + stacker | Review error analyses weekly; adjust guide | Component targets in `06_EVALUATION.md` met or ADR explaining gaps |
| **P4 Streaming services** | 8–13 | ingest-gateway (Asterisk ARI ExternalMedia + snoop, FreeSWITCH fork, file mode), streaming ASR, workers on Redis Streams, real-time fusion, latency instrumentation | PBX team configures media fork in staging | Simulator + staging PBX: p95 confirmed alert ≤ 6 s |
| **P5 API & GUI** | 9–14 | FastAPI (REST/WS/RBAC/audit), Next.js dashboard (all screens in GUI spec), Keycloak integration | UX review with 3 supervisors + 2 QA analysts | Usability sign-off |
| **P6 Production hardening** | 12–16 | Helm, KEDA, observability, retention jobs, security controls, load & chaos tests, runbooks, model registry promotion flow | Pen-test, DPO sign-off | Load test 2× peak passes; security findings closed |
| **P7 Pilot & feedback loop** | 15–22 | Shadow → supervisor pilot tooling, review queue → active-learning sampler → retrain jobs, weekly metrics report | Supervisors review alerts; QA weekly sampling | Pilot exit criteria (06 §6) |

## Milestone demos
- **M1 (end wk 4):** "Upload a call, get a report" — offline MVP with pretrained models.
- **M2 (end wk 12):** Fine-tuned models beat baseline on golden set; post-call mode processes a full day of calls.
- **M3 (end wk 14):** Live wallboard on simulator + staging PBX.
- **M4 (end wk 16):** Production-ready release candidate.
- **M5 (wk 22):** Pilot complete, go/no-go for rollout.

## Top risks & mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Dialects → high ASR error | missed slang, false flags | Ben-10 + in-house dialect data, KWS independent of ASR, dialect-sliced eval |
| Too few positive examples | weak detectors | oversample QA-flagged calls, targeted volunteer recordings, synthetic carriers for KWS |
| False alarms erode trust | supervisors ignore tool | high-precision alert thresholds, provisional vs confirmed, explanations, weekly threshold tuning |
| Agent pushback / privacy concerns | adoption, legal | coaching-only policy, transparency, agent self-view & dispute, DPIA |
| LLM latency/cost | late alerts | fast path independent; debounce; small fine-tuned model; batch for post-call |
| Licence restrictions on datasets/models | legal | licence register, swap to permissive alternatives, in-house data priority |
| Mono recordings in some queues | speaker attribution errors | pyannote fallback, push ops to enable stereo |
