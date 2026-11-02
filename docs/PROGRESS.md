# GAL-E — Progress

Maintained by Claude Code. Updated at the end of every phase/prompt.
Legend: `[ ]` todo · `[x]` done · `[~]` in progress · **[HUMAN]** needs people, access or sign-off · 🔒 blocked on a human input.

Last updated: 2026-11-02 (Prompt 0: kickoff and spec review)

---

## Phase checklist

### P0 — Foundations (weeks 1–2) · Prompts 1–2
Exit gate: `make dev-up` runs; CI green.
- [x] Spec review, PROGRESS.md, ADR template and ADR-001..007 (Prompt 0)
- [ ] uv workspace: `packages/galecore`, `services/*` skeletons with `/healthz` `/readyz` `/metrics`, `ml/*` empty modules
- [ ] `galecore.schemas`: Channel, Category, Target enums; AudioFrame, Word, AsrSegment, Evidence, Signal, FusedEvent, CallLifecycle, Review (+ contract tests)
- [ ] `galecore.config` (pydantic-settings, env + `configs/*.yaml`)
- [ ] `galecore.logging` (structlog JSON + PII scrubber) and PII-in-logs lint
- [ ] `galecore.bus` (BusClient over Redis Streams: publish, consumer groups, ack, retry, dead-letter, idempotency)
- [ ] `galecore.audio` (µ-law/A-law decode, soxr 8k→16k, stereo split, 200 ms framing)
- [ ] `galecore.bangla` (`normalize()`, `phonetic_key()`, tokenizer): tests first
- [ ] `galecore.redact` (BD mobile incl. Bangla and spoken digits, NID 10/13/17, email, account numbers): tests first
- [ ] Model interfaces + deterministic stubs (`ASRBackend`, `ShoutDetector`, `SERModel`, `KWSModel`, `AudioToxModel`, `TextClassifier`, `RegisterDetector`, `LLMJudge`), `GALE_MODEL_MODE=stub|real`
- [ ] `configs/`: `thresholds.yaml`, `rules.yaml`, `models.yaml`, `lexicon/fake_lexicon.yaml` (`<GALI_1>`..`<GALI_20>`)
- [ ] Docker-free dev stack (ADR-009): `tools/devstack` supervisor, fakeredis TCP bus, embedded Postgres (`pgserver`), local `ObjectStore`, `GALE_AUTH_MODE=dev` JWT issuer with 5 roles + demo users; `uv.toml`/`.npmrc`/Makefile put caches and temp under `.gale/` on G:
- [ ] Verify `pgserver` on Windows and fakeredis Streams support (XADD/XREADGROUP/XACK/XCLAIM); record fallbacks in ADR-009
- [ ] `infra/compose/docker-compose.dev.yml` (postgres, redis, minio, keycloak realm, services, web): written for later and Linux CI, optional locally
- [ ] `web/` skeleton: Next.js App Router, TS strict, Tailwind, shadcn/ui, next-intl (bn default, en)
- [ ] `.gitattributes` (`* text=auto eol=lf`), Windows-safe Makefile (`SHELL := bash`), and a Windows CI job (ADR-008)
- [ ] Makefile (`setup`, `dev-up`, `dev-down`, `check`, `gen-types`, `simulate`, `report`, `eval`, `e2e`), pre-commit
- [ ] `make gen-types`: pydantic → JSON Schema → `web/src/types/generated` + drift check
- [ ] GitHub Actions CI: ruff, mypy --strict, pytest + coverage ≥ 80%, eslint, tsc, vitest, docker build, Trivy
- [ ] `tools/fixtures_gen`: 10 synthetic stereo 8 kHz calls with `script.json` + `expected_events.json` (placeholders only)
- [ ] `tools/simulator`: RTP (G.711 µ-law), WebSocket PCM, file-drop modes; `--speed`; round-trip decode test
- [ ] **[HUMAN]** Legal basis and DPIA started (PDPO 2025) 🔒
- [ ] **[HUMAN]** Call-data access approval 🔒
- [ ] **[HUMAN]** Dataset and model licence review started 🔒
- [ ] **[HUMAN]** Hardware ordered 🔒

### P1 — Offline MVP, pretrained + open-data models (weeks 2–5) · Prompts 3, 4b
Exit gate: report runs on 100 real calls; first baseline numbers.
- [ ] Post-call pipeline library + CLI (`make report FILE=…`) reusing the worker logic
- [ ] Silero VAD, per-channel ASR backends (`whisper_ct2`, `indic_conformer`, `qwen3_asr`) behind `ASRBackend`
- [ ] Text steps: normalise → redact → lexicon (exact, fuzzy, phonetic, Banglish) → register rules → text classifier
- [ ] Audio steps: DSP shouting features with per-speaker rolling baseline (rule fallback), emotion2vec+ zero-shot SER, audio-tox, KWS
- [ ] Overtalk/interruption detector from two-channel VAD (see Decision D-05)
- [ ] LLM judge client (window + call mode), JSON-schema validation, evidence-grounding post-check, zero-shot prompt
- [ ] Fusion v0: identity calibration, `rules.yaml`, hysteresis, severity
- [ ] `report.json` + single-file `report.html`
- [ ] `scripts/download_models.py` (pinned commit hashes) + `ml/MODEL_LICENSES.md`
- [ ] Stub-mode integration test against all fixtures' `expected_events.json`
- [ ] Open-data track: `ml/data/DATASETS.md` register with `commercial_ok` gate; loaders for T1–T15, R1–R4, A1–A4
- [ ] Back-transliteration model (BanglaTLit) for Banglish sets
- [ ] Harmonise → `unified.jsonl` (masked labels, dedupe/MinHash, split inheritance, re-weighting) + data card
- [ ] BanglaBERT-family multi-task classifier on open data only (see open question Q-03 on licence)
- [ ] Slang-lexicon candidate mining (log-odds/PMI) → review export
- [ ] First KWS model on BAAD
- [ ] LLM judge baseline on open synthetic windows (few-shot / LoRA)
- [ ] **[HUMAN]** Seed lexicon review and approval 🔒
- [ ] **[HUMAN]** 50-call pilot annotation to validate the guide 🔒
- [ ] **[HUMAN]** Manual download of non-redistributable datasets 🔒

### P2 — Data engine (weeks 3–8) · Prompts 4, 5
Exit gate: `golden_v1` / `dev_v1` frozen; α ≥ 0.7.
- [ ] Recorder/S3 ingestion → manifests (`ml/data/manifests/*.jsonl`)
- [ ] Channel split, 8 kHz native + 16 kHz model copies, quality filters
- [ ] Auto-transcription seed; redaction applied before anything persists
- [ ] Speaker-aware splits (by call and by agent/customer), dialect/queue stratification
- [ ] Augmentation: codec sim (µ-law, A-law, AMR-NB, GSM-FR), packet loss, band-limit, noise, speed, shout-safe gain
- [ ] Label Studio templates T1/T2/T3 + import/export ↔ manifests ↔ training views
- [ ] Eval harness (`make eval COMPONENT=… SPLIT=…`), event-matching rule, fairness slices, bootstrap CIs
- [ ] Golden-set lock: hashed immutable manifests + CI overlap check (calls and speakers)
- [ ] DVC (MinIO remote) + MLflow (Postgres + MinIO)
- [ ] **[HUMAN]** T1/T2/T3 annotation at volume; golden + dev double-annotated; annotator wellbeing plan 🔒

### P3 — Model training (weeks 6–12) · Prompts 6–10
Exit gate: component targets in `06_EVALUATION.md` met, or an ADR explaining the gaps.
- [ ] ASR bake-off (CER, WER, slang recall, script drift, RTF, p95) → ADR; fine-tune top 2; contextual biasing + slang false-insertion check
- [ ] KWS: CTC keyword search over Conformer posteriors + dedicated classifier (top ~100 terms)
- [ ] Shouting LightGBM + isotonic calibration + rule guards
- [ ] SER head on emotion2vec+ (2-stage), ONNX fp16
- [ ] Audio-tox head
- [ ] Text classifier 3-stage training (open → ASR-noised → in-domain), ONNX int8 ≤ 40 ms/utt
- [ ] Register detector: rules + per-speaker state + small logistic model
- [ ] LLM judge: zero-shot screen → Unsloth bf16 LoRA (Qwen3.5-9B vs Gemma-4) → merge → AWQ/FP8 → vLLM guided JSON
- [ ] Fusion: per-source calibration, per-category stacker, alert thresholds per category × speaker × mode
- [ ] MLflow registry names + promotion checks
- [ ] **[HUMAN]** GPUs available 🔒; weekly error-analysis review; guide adjustments

### P4 — Streaming services (weeks 8–13) · Prompt 11
Exit gate: simulator + staging PBX: p95 confirmed alert ≤ 6 s.
- [ ] ingest-gateway: Asterisk ARI ExternalMedia + snoop (per direction), FreeSWITCH audio fork, file mode, G.711/AMR-NB decode
- [ ] Streaming ASR (cache-aware Conformer and/or WhisperLiveKit SimulStreaming)
- [ ] audio-worker, asr-worker, text-worker, llm-judge, fusion-engine on Redis Streams with consumer groups
- [ ] Real-time fusion, debounce, LLM circuit breaker ("LLM offline" degradation)
- [ ] Per-stage latency histograms + OTel traces keyed by `call_id`
- [ ] **[HUMAN]** PBX team configures media fork in staging 🔒

### P5 — API & GUI (weeks 9–14) · Prompts 12–13
Exit gate: usability sign-off.
- [ ] FastAPI REST + WS per `04_GUI_SPEC.md` §3, OIDC (Keycloak), RBAC, audit log on every audio/transcript view, signed audio URLs
- [ ] Alembic migrations for the `01_ARCHITECTURE.md` §4 schema (+ D-08 additions), monthly partitions
- [ ] Web: `/live`, `/live/[callId]`, `/calls`, `/calls/[callId]`, `/review`, `/analytics`, `/me`, `/admin/{lexicon,rules,models,health,audit}`
- [ ] bn/en i18n, dark/light, WCAG 2.1 AA, wallboard 1,000 calls at 2 updates/s
- [ ] Vitest, Playwright e2e against simulator stack, visual regression, Lighthouse a11y ≥ 90
- [ ] **[HUMAN]** UX review with 3 supervisors + 2 QA analysts 🔒

### P6 — Production hardening (weeks 12–16) · Prompt 14
Exit gate: load test at 2× peak passes; security findings closed.
- [ ] Helm chart, KEDA ScaledObjects, NetworkPolicies default-deny, egress deny for ML/data
- [ ] Prometheus rules, Grafana dashboards (Ops/ML/Business), Loki, Alertmanager rules
- [ ] Retention jobs, DSAR deletion endpoint
- [ ] Image hardening (non-root, slim), SBOM (Syft), Trivy
- [ ] Load tests (25/50/100/200%), chaos tests (no lost events)
- [ ] Runbooks in `docs/runbooks/` (8 listed in 05 §9); STRIDE threat model
- [ ] Model promotion flow with second approver
- [ ] **[HUMAN]** In-country cluster, DNS/TLS, SSO 🔒; pen-test; DPO sign-off 🔒

### P7 — Pilot & feedback loop (weeks 15–22) · Prompt 15
Exit gate: pilot exit criteria (`06_EVALUATION.md` §6).
- [ ] Shadow mode (alerts hidden), `shadow=true` signals, weekly shadow comparison
- [ ] Supervisor-pilot tooling, review-queue → nightly feedback export
- [ ] Active-learning sampler (FPs, FNs, low confidence, signal disagreement, new slang candidates)
- [ ] Monthly retrain jobs + promotion gate; weekly metrics report
- [ ] **[HUMAN]** Agent/union communication (coaching-only policy) 🔒; supervisors review alerts; QA weekly sampling

---

## Decisions

Accepted (from spec): ADR-001..007 in `docs/adr/`.
Accepted 2026-11-02: ADR-008 native Windows development; ADR-009 Docker-free local dev (venv + fakeredis + embedded Postgres + local object store + dev auth; all caches on G:).

Resolutions from the Prompt 0 review, **accepted 2026-11-02**. Each gets its own ADR when the phase that needs it is implemented. The ⚠ marks the item that was a human decision.

| ID | Decision | Phase |
|---|---|---|
| D-01 ⚠ | Raw (un-redacted) transcript text never goes on the bus. asr-worker calls `galecore.redact` in-process and publishes only redacted text in `AsrSegment`. Redis persistence for `audio.frames.*` is off or on encrypted disk with short `maxlen`. Resolves 01 §1 vs 05 §7 (see Q-01). | P0 |
| D-02 | Audio on the bus is 16-bit PCM bytes, not base64 float32 JSON (`AudioFrame.pcm_s16le`, metadata JSON + binary field). Cuts bus bandwidth by about 2.7×. `sample_rate` stays fixed at 16000. | P0 |
| D-03 | Single canonical `Target` enum: `agent, customer, company, third_party, self, none`. Open-data targets (individual/group/organization/society) map onto it in `label_map.yaml`. | P0 |
| D-04 | One `ContextFlag` enum used by annotation, text, LLM and fusion: `banter, reported_speech, self_directed, partial, quoted_policy, neutral_register, unsure`. | P0 |
| D-05 | Add `Signal.source` values `vad_overtalk` (two-channel VAD overlap counter, the detector for `interruption_overtalk`) and `call_event` (who hung up, holds; feeds `dismissive_rude`). | P0 |
| D-06 | `aggression` (TRAC-2 auxiliary head) is not a `Category`. It travels as a feature in `Signal.evidence.features` and is used by fusion only. | P0 |
| D-07 | Add `shadow: bool = False` to `Signal`/`FusedEvent` and the `signals`/`events` tables (03 §9 needs it; schema lacks it). | P0 |
| D-08 | DB additions: `users.agent_id` (agent self-view), `reviews.kind` (`verdict` / `dispute`), restricted `agents.gender` and `calls.dialect_region` for the fairness panel (pending Q-05). | P5 |
| D-09 | LLM schemas tightened: `provisional_decisions.new_category` uses the Category enum; all `CallScorecard` sub-object fields are `required` (guided decoding otherwise allows empty objects). | P1 |
| D-10 | Lexicon source of truth = Postgres `lexicon_entries`. The encrypted file at `GALE_LEXICON_PATH` is a bootstrap seed imported once. Workers load a versioned snapshot from the DB and hot-reload on change. Tests use `configs/lexicon/fake_lexicon.yaml`. | P0/P1 |
| D-11 | CTC keyword search runs inside asr-worker (where the Conformer posteriors are) and emits `Signal(source="kws")`. The dedicated KWS classifier stays in audio-worker. Avoids shipping posterior tensors over the bus. | P1/P4 |
| D-12 | Dependency isolation: the uv workspace holds light deps. Heavy GPU stacks (NeMo, FunASR, Unsloth) are per-package optional extras declared as uv `conflicts`. vLLM runs as its own container from the official image, outside the workspace lock. | P0 |
| D-13 | Calibration falls back to Platt/beta when a (source, category) pair has < 200 dev positives. Isotonic is used only above that. All golden metrics are reported with bootstrap 95% CIs. | P3 |
| D-14 | The rules "simulate on last N calls" feature (04 §2.8) and any threshold tooling exclude golden-set call ids at the query level. | P5 |
| D-15 | Mono/diarized audio: `Channel.mixed` is allowed only before diarization. Fusion accepts only `agent`/`customer`. Segments that can't be attributed are stored but never alert. | P1 |

---

## Open questions for the human

**2026-11-02 answers:** Q-01 to Q-05 and Q-07 to Q-09 approved with the defaults below. As a result, D-01 to D-15 are accepted. Q-06 was answered **native Windows, no WSL2** (ADR-008).

Licence review (Q-03, Q-04) is still a **[HUMAN]** legal action. The approval covers only the engineering default: swappable encoder, ship `commercial_ok` only.

| ID | Question | Why it matters | My default if you don't object |
|---|---|---|---|
| Q-01 | **Bus privacy posture.** Spec says un-redacted text "exists only in memory", but `AsrSegment.text` goes over Redis Streams, and Redis AOF (05 §2) writes it to disk. Raw audio frames also land in AOF. OK to redact inside asr-worker and run Redis with AOF off for audio streams (or on encrypted volumes)? | Security posture: CLAUDE.md says ask. | D-01 |
| Q-02 | **Golden-set holdout by agent.** 06 §1 says no agent overlap with training. With a limited agent pool, golden agents are excluded from all training data. Accept that, or hold out by call + customer only and report a separate unseen-agent slice? | Changes how much in-house data is usable. | Strict agent holdout as specced |
| Q-03 | **Model licences.** From what I know, `csebuetnlp/banglabert(_large)` is CC BY-NC-SA 4.0 (non-commercial), and SONAR/MuTox are non-commercial too. If confirmed, the primary text encoder can't ship commercially. Alternatives: MuRIL (Apache-2.0), IndicBERTv2 (MIT), XLM-R (MIT). Also check: emotion2vec+, tugstugi Whisper, Qwen3-ASR-bn fine-tune, pyannote community-1, MinIO (AGPL-3.0). | Licence: irreversible, needs legal. | Build behind `TextClassifier` so the encoder is swappable. Bake off BanglaBERT vs MuRIL vs IndicBERTv2, and ship only `commercial_ok` |
| Q-04 | **Dataset licences.** Almost every set in 09 is `VERIFY`. Who owns `ml/data/DATASETS.md` sign-off, and can research-only sets be used in non-promotable research runs? | Gates P1 open-data track. | Research runs allowed and tagged `research_only` (blocks promotion), per 09 |
| Q-05 | **Fairness attributes.** The fairness panel (04 §2.5, 06 §3) needs agent gender and dialect region. Neither is in the schema, and gender is sensitive data under PDPO. May we store `agents.gender` (restricted to admin/QA, aggregate-only in UI)? | Sensitive personal data. | Add column, nullable, restricted, aggregate-only |
| Q-06 | **Windows dev environment.** You're on Windows. `make`, Docker Compose and some audio libs work best under WSL2. Develop in WSL2 (recommended), or keep native Windows + Git Bash (I'd need `make` installed and a Windows-safe Makefile)? | Affects `make setup` / `make dev-up` from Prompt 1. | ~~WSL2~~ → **Answered: native Windows** (ADR-008) |
| Q-07 | **Repo naming.** Spec layout roots at `gal-e/`; this repo is `Bengali-Hate-Speech-Detection-on-Call-Centers` with the spec at the root. OK to scaffold the monorepo directly at this repo root? | Layout only. | Yes, repo root = `gal-e/` |
| Q-08 | **Model ids.** Qwen3.5, Gemma 4, Qwen3-ASR-bn etc. are taken from the research summary. I'll verify each exists and pin it by commit hash at download time (P1). Do you have a preferred hosting mirror (internal registry/MinIO) for weights? | Supply chain (05 §7). | Download to `GALE_MODEL_DIR` from HF in build, mirror to MinIO |
| Q-09 | **PBX target.** Which platform does the call centre use (Asterisk, FreeSWITCH, Genesys/Avaya/Cisco via SIPREC)? Are recordings stereo for all queues? | P4 scope; mono needs diarization path. | Build Asterisk + file mode first |

---

## Spec review notes (Prompt 0)

Smaller inconsistencies, resolved in-line without an ADR:
- LLM latency: 01 §5 "≤ 3 s after segment" (incl. queue) vs 08 "≤ 2.5 s p95" (model). Both kept: 2.5 s is model p95, 3 s is end-to-end.
- Severity: FusedEvent uses 1–3; text-clf/open data use 0–3 where 0 = none. Mapping 0 → no event.
- SER classes {neutral, frustrated, angry, contempt/aggressive, other} ≠ emotion2vec+'s 9 classes. The head is retrained, so the mapping goes into `ml/ser/label_map.yaml`.
- Prompt order: Prompt 4b (P1 open data) comes after Prompt 4 (P2) in the prompt file. Run it right after Prompt 3; it doesn't depend on in-house data.
- P1 exit gate needs 100 real calls before the P2 recorder ingestion exists. P1 will accept manually placed stereo WAVs outside the repo.
- 300-call golden set with < 5% positives gives few agent sev-3 events. Expect wide CIs; oversample harshness into golden as 02 §1.2 suggests (D-13).

---

## Next step

1. Windows toolchain (ADR-008), 2026-11-02:
   - [x] uv 0.13.6, GNU Make 4.4.1, pnpm 12.16.1 installed via winget (user scope). Corepack couldn't write to `C:\Program Files\nodejs`, so pnpm is the standalone build.
   - [x] PATH fixed by the user: `make` now resolves to GNU Make 4.4.1.
   - [x] Docker Desktop **not** installed (user decision, C: has 0.6 GB free). Local dev is Docker-free (ADR-009).
   - Already present: Python 3.12.3, Node 20.12.2, Git Bash 5.2.
2. Run **Prompt 1: P0 monorepo scaffold & shared core**.
