# Claude Code Prompts — build GAL-E phase by phase

How to use:
1. Create an empty git repo, copy this whole spec folder into it (`CLAUDE.md` at repo root, `docs/`, `prompts/`).
2. Open Claude Code in the repo. Paste **one prompt at a time**, in order. Let it finish, run the verification commands yourself, then move on.
3. After each prompt, review `docs/PROGRESS.md`. If something is off, use the **Fix prompt** at the bottom.
4. Prompts marked 🔒 need things only you can provide (data access, GPUs, PBX, credentials). Claude Code will build the tooling and stop where human input is needed.

Tip: start each prompt in a fresh session (or `/clear`) — CLAUDE.md and the docs carry the context.

---

## Prompt 0 — Kickoff & plan check
```
Read CLAUDE.md and every file in docs/. Do not write code yet.
1. Summarise the system in 10 bullet points to confirm understanding.
2. List any contradictions, gaps or risky assumptions you find in the spec, with your proposed resolution.
3. Create docs/PROGRESS.md with: phase checklist (P0–P7 from docs/07_ROADMAP.md broken into tasks), a "Decisions" section, an "Open questions for the human" section, and "Next step".
4. Create docs/adr/000-template.md and ADR-001..007 from docs/01_ARCHITECTURE.md §7 (short).
Stop after this and wait.
```

## Prompt 1 — P0 Monorepo scaffold & shared core
```
Implement Phase P0 foundations exactly per docs/01_ARCHITECTURE.md §6 and CLAUDE.md.
- uv workspace with packages/galecore, services/* (empty FastAPI/worker skeletons with health endpoints), ml/* (empty modules), web/ (Next.js app skeleton with next-intl bn/en, Tailwind, shadcn/ui).
- packages/galecore:
  - schemas.py: AudioFrame, Word, AsrSegment, Evidence, Signal, FusedEvent, CallLifecycle, Review + Category/Channel enums exactly as in docs/01 §3 (category enum must match docs/02 §3.4).
  - config.py (pydantic-settings; env + YAML in configs/), logging.py (structlog JSON, with a scrubber that drops keys named text/transcript/audio/pcm and redacts phone/NID patterns), bus.py (BusClient over Redis Streams: publish, consumer groups, ack, retry, dead-letter, idempotency helper), audio.py (decode µ-law/A-law, resample 8k→16k via soxr, split stereo, frame into 200 ms), bangla.py (normalize(): NFC, nukta/য় variants, ZWJ/ZWNJ cleanup, Bangla↔ASCII digits, whitespace; phonetic_key(); tokenizer), redact.py (BD mobile numbers incl. Bangla digits and spoken-digit sequences, NID 10/13/17 digits, emails, account-like numbers → tags).
  - Thorough unit tests, especially Bangla normalisation and redaction edge cases.
- Model interfaces + stub implementations in galecore/models/ (ASRBackend, ShoutDetector, SERModel, KWSModel, AudioToxModel, TextClassifier, RegisterDetector, LLMJudge) selected by GALE_MODEL_MODE=stub|real.
- configs/: thresholds.yaml, rules.yaml, models.yaml (stub defaults), lexicon/fake_lexicon.yaml with placeholder terms <GALI_1>..<GALI_20>.
- infra/compose/docker-compose.dev.yml: postgres, redis, minio, keycloak (realm import with roles admin, qa_analyst, supervisor, agent, auditor + demo users), all services (stub mode), web.
- Makefile targets from CLAUDE.md; pre-commit; GitHub Actions CI (lint, type, test, web lint/test, docker build, trivy).
- make gen-types: pydantic → JSON Schema → TypeScript (web/src/types/generated).
Verification: make setup && make check && make dev-up, then curl each service /healthz. Update docs/PROGRESS.md.
```

## Prompt 2 — P0 Synthetic fixtures & call simulator
```
Build test assets and the simulator (no real data, no real slurs):
- tools/fixtures_gen/: generate 10 synthetic stereo 8 kHz Bangla "calls" (2–4 min). Use a pluggable TTS interface; implement a stub that synthesises tones/noise with speech-like envelopes plus a JSON script containing the intended transcript, speaker turns, and ground-truth events (slang placeholders, register switch, insult, shouting segments created by +12 dB gain and raised pitch, overtalk). If a local Bangla TTS model directory is configured, use it; otherwise stub.
- Store as tests/fixtures/calls/<name>/{call.wav, script.json, expected_events.json}.
- tools/simulator/: replays a call as (a) RTP streams per channel (G.711 µ-law) to ingest-gateway, (b) WebSocket PCM, (c) file drop for post-call mode; real-time pacing with --speed option.
- Tests that the simulator output decodes back to the original audio within tolerance.
Verification: make simulate CALL=tests/fixtures/calls/demo_01 prints frames received by ingest-gateway (stub). Update PROGRESS.md.
```

## Prompt 3 — P1 Offline post-call pipeline (end-to-end, pretrained/stub models)
```
Implement the post-call pipeline as a library + CLI (`make report FILE=...`), reusing the same worker logic that services will use later:
stereo WAV → channel split → VAD (Silero) → ASR backend → text-worker steps (normalise, redact, lexicon match exact+fuzzy+phonetic, register detector rules per docs/03 §6.1, text classifier) → audio-worker steps (DSP shouting features with per-speaker rolling baseline per docs/03 §3, SER, audio-tox, KWS) → LLM judge (window + call mode, schema-validated, evidence-grounding check per docs/08 §2) → fusion engine v0 (calibration identity, rules.yaml, hysteresis, severity) → outputs: report.json (FusedEvents + CallScorecard + timelines) and a single-file report.html (waveform image, transcript with highlights, timelines, events table).
Real implementations (GALE_MODEL_MODE=real) for: Silero VAD, faster-whisper (CT2) backend, NeMo IndicConformer backend, emotion2vec+ via FunASR, BanglaBERT via transformers, vLLM/OpenAI-compatible client for the judge, DSP shouting with LightGBM placeholder model (rule fallback if no model file). Load weights only from GALE_MODEL_DIR; write scripts/download_models.py that downloads pinned revisions (by commit hash) into that dir and records licences into ml/MODEL_LICENSES.md.
Integration test: run the pipeline in stub mode on all fixtures and assert events match expected_events.json within tolerance (stub models are deterministic and keyed to the fixture scripts).
Verification: make report FILE=tests/fixtures/calls/demo_01/call.wav opens a correct HTML report. Update PROGRESS.md.
```

## Prompt 4 — P2 Data engine 🔒 (needs access to recordings later)
```
Implement ml/data per docs/02_DATA_PLAN.md §2 and §3:
- Ingestion adapters: local folder, S3/MinIO bucket, and a generic "recorder export CSV + WAV" adapter → manifests (JSONL, schema-validated) with pseudonymised customer ids (salted hash; salt from secret).
- Quality filters, channel split, 16 kHz copies, auto-transcription job (batch, resumable), redaction of seed transcripts.
- Label Studio: project XML templates for T1 transcript correction, T2 event spans (taxonomy, severity, target, flags), T3 call QA; scripts to create projects via API, push tasks (with pre-annotations from the pipeline), pull annotations, and convert to training sets for each model (ASR, KWS, shout, SER, audio-tox, textclf, LLM SFT, fusion).
- Agreement metrics (Krippendorff's alpha per category; span IoU agreement) and an adjudication export.
- Augmentation library: codec simulation (µ-law, A-law, AMR-NB, GSM via ffmpeg/sox), packet loss, noise mixing, band-limit, speed perturb; deterministic seeds.
- Split tool: by call AND speaker, stratified by dialect/queue; golden-set lock (hash file) + CI test that fails on overlap.
- DVC init with MinIO remote; MLflow server in docker-compose.
- ml/data/DATASETS.md licence register template pre-filled with datasets from docs/02 §1.1 (licence column "TO VERIFY").
Everything must run on the synthetic fixtures in CI. Write docs/runbooks/annotation-ops.md for the annotation lead.
Stop and list exactly what the human must provide (bucket paths, credentials, Label Studio host).
```

## Prompt 4b — P1 Open-dataset ingestion, harmonisation & open-data baselines 🔒 (licence sign-off; manual downloads for some sets)
```
Implement ml/data/open per docs/09_OPEN_DATASETS.md:
- ml/data/DATASETS.md register (YAML front-matter or datasets.yaml) with every dataset in 09 §1–3: id, name, version/DOI, url, licence text, commercial_ok (true|false|unknown — default unknown), download_allowed, expected checksum. Ingestion refuses commercial_ok != true for runs tagged prod; research runs are tagged research_only and blocked from model promotion.
- One loader per dataset (T1–T15, R1–R4, A1–A4) → unified schema (09 §4) with partial labels (null = not annotated). Loaders read from data/raw/<source>/ if download is not allowed; auto-download only where download_allowed=true. Unit tests with tiny synthetic sample files mimicking each format (no real content).
- Back-transliteration model: training script on BanglaTLit (seq2seq), inference CLI; used for BanTH, TRAC-2 Roman rows; language-ID filter to drop Hindi/English-only rows.
- Normalisation, cleaning, MinHash near-dedupe across sources, BIDWESH→BD-SHS parent split inheritance, official-test-split preservation, conflict resolution, domain re-weighting, ASR-noise injection + optional TTS→codec→ASR round-trip, spoken-style rewriting hook (self-hosted LLM, synthetic-tagged).
- label_map.yaml exactly as 09 §5 with label_confidence for weak mappings.
- Outputs: unified.jsonl, per-model views (textclf, llm_sft open-synthetic windows, lexicon mining), reports/open_data_card.md.
- Lexicon mining: log-odds/PMI candidate extraction with variants → CSV for human review (terms masked in logs).
- Train the open-data-only BanglaBERT multi-task classifier (masked BCE + soft targets + aggression aux head) and report on each public test split in the public_aux suite; compare against 09 §8 reference numbers.
- Train first KWS model on BAAD (smoke on fixtures in CI, full run documented in runbook).
Stop and list: which datasets need manual download, which licences are unknown, and the commands to run the full jobs on the GPU box.
```

## Prompt 5 — P2 Evaluation harness
```
Implement ml/eval per docs/06_EVALUATION.md:
- Metrics: CER/WER with the shared normaliser, slang recall, script-drift rate; KWS recall & false alarms/hour; event matching (speaker + category + IoU ≥ 0.3 or centre ± 2 s); per-category/per-speaker precision/recall/F1; calibration (ECE, reliability plots); call-level Spearman and AUC; latency summaries.
- Slicing by dialect, gender, queue, codec, duration, time-of-day; fairness flags (FPR > 1.5× overall).
- CLI: make eval COMPONENT=<asr|kws|shout|ser|audiotox|textclf|llm|fusion|system> SPLIT=<dev|golden|...> → MLflow run + markdown/HTML report in reports/.
- Bake-off runner: evaluates multiple candidates with identical preprocessing and writes the ADR results table.
Tests with synthetic fixtures. Update PROGRESS.md.
```

## Prompt 6 — P3a ASR fine-tuning 🔒 (GPU + data)
```
Implement ml/asr per docs/03 §1:
- Training scripts for (a) Whisper (HF Seq2Seq, full FT and LoRA options, SpecAugment, domain-prompt training, telephony augmentation on non-call data, weighted data mixing), (b) NeMo IndicConformer hybrid CTC/RNN-T fine-tune incl. cache-aware streaming config, (c) Qwen3-ASR LoRA via its toolkit.
- Export: merge LoRA → CTranslate2 int8_float16 for Whisper; .nemo + ONNX/TensorRT notes for Conformer; vLLM-ready for Qwen3-ASR.
- Contextual biasing: hotword lists from lexicon + domain terms; measure slang false-insertion rate on negatives_v1.
- Configs for a "smoke" run on fixtures (CPU, 10 steps) and a "full" run.
Run only the smoke runs. Write docs/runbooks/asr-training.md with exact commands, expected VRAM/time, and the bake-off procedure.
```

## Prompt 7 — P3b Acoustic models: shouting, SER, audio-tox, KWS 🔒
```
Implement ml/shout, ml/ser, ml/audio_tox, ml/kws per docs/03 §2–5:
- shout: feature extraction (per-speaker baseline z-scores), LightGBM training, isotonic calibration, ONNX/treelite export.
- ser: emotion2vec+ backbone (frozen → partial unfreeze), multi-task head (categorical + arousal/valence), staged training (SUBESCO/BanglaSER 8k-simulated → in-domain), ONNX fp16 export.
- audio_tox: head on shared embeddings; datasets ADIMA-bn, Bangla hate-speech audio, MuTox-bn loaders (paths configurable).
- kws: (a) CTC keyword search over Conformer posteriors with per-term calibrated thresholds; (b) dedicated multi-label KWS classifier with hard negatives and heavy augmentation.
- Dataset loaders for BAAD, SUBESCO etc. read from configurable local paths (no auto-download unless licence flag set).
Smoke-train each on fixtures; wire real implementations into galecore model interfaces. Update PROGRESS.md and runbooks.
```

## Prompt 8 — P3c Text classifier & register detector 🔒
```
Implement ml/textclf per docs/03 §6:
- Start from the open-data classifier and unified dataset built in Prompt 4b (docs/09_OPEN_DATASETS.md); add in-house call spans (stage 3) with the same masked-loss training.
- ASR-noise augmentation (character/word confusion model estimated from ASR outputs vs references; optional TTS→codec→ASR path).
- BanglaBERT multi-task model (multi-label categories + ordinal severity + target), context input format with speaker tags, focal BCE, FGM adversarial option, early stopping on dev macro-F1, ONNX int8 export, CPU latency benchmark.
- Register detector: rule engine (pronouns + verb morphology) + per-speaker state machine + small logistic model; exhaustive unit tests with synthetic Bangla sentences.
Smoke-train on fixtures. Update PROGRESS.md.
```

## Prompt 9 — P3d LLM judge fine-tuning 🔒
```
Implement ml/llm and services/llm_judge per docs/08_LLM_JUDGE_SPEC.md and docs/03 §7:
- SFT dataset builder from annotations (window + call modes), hard-negative mining, optional synthetic/distilled sources tagged in meta; exact system prompt from services/llm_judge/prompts/.
- Unsloth bf16 LoRA training config for Qwen3.5-9B (primary) and Gemma-4 (challenger); loss on assistant tokens; eval callback computing JSON validity + event F1 on dev windows.
- Merge + quantise (AWQ or FP8) scripts; vLLM deployment config with guided JSON decoding using the schemas; prefix caching.
- services/llm_judge: debounced window triggering, timeouts, circuit breaker, post-validation (evidence substring, speaker match, acoustic-only categories gated by signals), metrics.
- Red-team tests: spoken prompt-injection strings in transcripts must not change outputs format/behaviour.
Smoke-run training with a tiny model (e.g. Qwen3.5-0.8B) on fixtures if GPU available; otherwise unit-test the data builder and service with the stub. Update PROGRESS.md.
```

## Prompt 10 — P3e Fusion engine & calibration
```
Implement services/fusion_engine and ml/fusion per docs/03 §8:
- Per-source isotonic calibration (fit on dev, persist), rules engine from configs/rules.yaml (typed, validated, hot-reloadable), per-category logistic stacker, per (call, speaker, category) state with cool-down, merge window, repeat-escalation, provisional→confirmed/dismissed transitions driven by LLM decisions, alert policy by thresholds.yaml (mode-specific: realtime vs postcall).
- Call-level aggregation: agent professionalism, customer hostility, escalation curve, who-escalated-first.
- Rules simulator: replay stored signals for last N calls with a candidate config and output precision/recall deltas (used by the GUI later).
Property-based tests (hypothesis) for hysteresis and idempotency. Update PROGRESS.md.
```

## Prompt 11 — P4 Streaming services & telephony ingest 🔒 (PBX access)
```
Turn the pipeline into streaming services on Redis Streams per docs/01:
- ingest-gateway: (a) Asterisk ARI app that, for each bridged call, creates two snoop channels (spy in/out) and ExternalMedia channels (RTP, slin16 or ulaw), receives RTP, maps to call_id/channel; (b) FreeSWITCH WebSocket audio fork receiver; (c) generic WebSocket PCM endpoint; (d) post-call file mode (watch bucket/folder). Publishes audio.frames + call.lifecycle, writes encrypted audio to MinIO.
- asr-worker streaming: WhisperLiveKit/SimulStreaming-style policy for Whisper backend or NeMo cache-aware streaming for Conformer; partial + final segments.
- audio-worker, text-worker, llm-judge, fusion-engine as consumers with consumer groups, retries, DLQ, idempotency, OpenTelemetry spans (trace id = call_id), Prometheus latency histograms per stage.
- Backpressure & priorities: real-time pre-empts post-call batch.
- Integration test: simulator → full stack (stub models) → events within latency budget (docs/01 §5); chaos test killing a worker mid-call loses no events.
Write docs/runbooks/pbx-integration.md with Asterisk dialplan/ARI and FreeSWITCH config snippets for the PBX team. Stop and list what the PBX team must configure.
```

## Prompt 12 — P5 API
```
Implement services/api per docs/04 §3 and docs/01 §4:
- SQLAlchemy models + Alembic migrations for the full schema (monthly partitions for signals/segments), repositories, Postgres trigram index on normalised redacted text.
- OIDC JWT validation (Keycloak), RBAC per role table in docs/04 §1 (agents see only their own data; supervisors only their teams), audit logging middleware for audio/transcript access.
- All REST endpoints + WebSockets (/ws/live filtered by role/team, /ws/calls/{id}) with batched updates every 250 ms; signed MinIO URLs; rules simulate endpoint calling the fusion simulator; model promote endpoint with two-person approval.
- OpenAPI → TS client generation for web.
- Tests: RBAC matrix tests for every endpoint, WS tests, migration up/down test.
Update PROGRESS.md.
```

## Prompt 13 — P5 Web dashboard
```
Implement web/ per docs/04_GUI_SPEC.md — all screens in §2 with Bangla as default language and English toggle:
/live wallboard, /live/[callId], /calls + /calls/[callId] (stereo wavesurfer with regions, synced transcript, timelines, scorecard, event table with evidence drill-down, PDF/CSV export), /review (keyboard-first), /analytics (agent, team heatmap, queue, fairness panel), /me, /admin/lexicon (masked terms, approval flow), /admin/rules (YAML + form + simulate), /admin/models, /admin/health (Grafana embeds), /admin/audit.
Use the generated API client and types; TanStack Query for REST, a WS store (Zustand) with reconnect/backoff; virtualised lists for 1,000 live calls; accessible severity (icon + text + colour); dark/light themes; Noto Sans Bengali/Hind Siliguri fonts.
Tests: Vitest for components/stores; Playwright e2e against `make dev-up` + simulator (alert appears on wallboard, review flow updates status, agent cannot see others' calls).
Update PROGRESS.md with screenshots paths.
```

## Prompt 14 — P6 Production hardening 🔒 (cluster)
```
Implement docs/05_INFRA_AND_OPS.md:
- Dockerfiles (multi-stage, non-root, pinned), Helm chart infra/helm/gal-e with values for staging/prod, GPU node selectors/tolerations, resource requests, PodDisruptionBudgets, NetworkPolicies (default deny, egress deny for ML/data), KEDA ScaledObjects on Redis Streams lag, CloudNativePG, Redis, MinIO, Keycloak values or references to existing services.
- Observability: Prometheus rules + alerts from docs/05 §4, Grafana dashboards JSON (Ops, ML drift, Business), Loki, OTel collector, DCGM exporter.
- Retention CronJobs, DSAR deletion job, backup configs, secrets via External Secrets/SOPS.
- Load tests (tools/loadtest, k6 or locust) at 25/50/100/200% target peak using the simulator; capacity report template filling the formula in docs/05 §3.
- Runbooks listed in docs/05 §9.
- Security: Trivy, SBOM, STRIDE threat model doc, log-PII lint in CI.
Validate with `helm template` + kubeconform in CI and a kind-cluster smoke deploy in stub mode. Update PROGRESS.md and list what infra/credentials the human must provide.
```

## Prompt 15 — P7 Pilot tooling & feedback loop
```
Implement pilot & continuous-improvement tooling per docs/06 §6 and docs/02 §6:
- Shadow mode flag (alerts computed but hidden from UI; model versions with shadow=true), canary routing by queue/percentage.
- Weekly metrics report job (precision from reviews, alert acceptance, latency, drift, fairness slices) → markdown/HTML + email/Slack webhook (configurable).
- Active-learning sampler (FPs, FNs, low-confidence, cross-signal disagreement, new candidate slang terms) → pushes tasks to Label Studio.
- Retraining pipelines (Argo Workflows or k8s Jobs) for textclf, fusion calibration, LLM LoRA monthly; promotion gate comparing golden-set reports (no metric regresses > 1 pt).
Update PROGRESS.md; produce docs/PILOT_PLAYBOOK.md for the ops team.
```

---

## Utility prompts

### Verify a phase
```
Act as a strict reviewer. Re-read CLAUDE.md and the docs relevant to phase <Px>. Check the implementation against them: missing features, spec deviations, untested paths, privacy-rule violations (PII in logs, raw text persisted, external calls), golden-set leakage, hard-coded thresholds. Run make check and the phase's verification commands. Output a findings list ranked by severity and fix the high-severity ones.
```

### Fix prompt
```
<describe the problem / paste the error>. Find the root cause, fix it with a test that would have caught it, run make check, and record the fix in docs/PROGRESS.md.
```

### Swap a model
```
Add a new <component> implementation "<name>" behind the existing interface, with config in configs/models.yaml, a download entry with pinned revision + licence, and run the bake-off: make eval COMPONENT=<component> SPLIT=dev for current vs new. Write an ADR with the results table. Do not change the default unless the new model wins on the primary metric within latency budget.
```
