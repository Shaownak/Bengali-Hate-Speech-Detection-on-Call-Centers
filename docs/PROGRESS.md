# GAL-E — Progress

Maintained by Claude Code. Updated at the end of every phase/prompt.
Legend: `[ ]` todo · `[x]` done · `[~]` in progress · **[HUMAN]** needs people, access or sign-off · 🔒 blocked on a human input.

Last updated: 2026-11-02 (Prompt 5: P2 evaluation harness)

---

## Phase checklist

### P0 — Foundations (weeks 1–2) · Prompts 1–2
Exit gate: `make dev-up` runs; CI green. **Met 2026-11-02** (CI run #3 green on `07ce4bd`).
- [x] Spec review, PROGRESS.md, ADR template and ADR-001..007 (Prompt 0)
- [x] uv workspace: `packages/galecore`, 7 `services/*` skeletons with `/healthz` `/readyz` `/metrics`, `ml/*` empty modules (ml is a plain package, not a workspace member: D-16)
- [x] `galecore.schemas`: enums (Channel, Speaker, Category, Target, ContextFlag, SignalSource, EventStatus) + AudioFrame, Word, AsrSegment, Evidence, Signal, FusedEvent, CallLifecycle, Review, WindowJudgement, CallScorecard; tests check Category against docs/02 and docs/08
- [x] `galecore.config` (pydantic-settings `GALE_*`, validated YAML loaders, dev-auth/stub-in-prod guards)
- [x] `galecore.logging` (structlog JSON + PII scrubber) and `tools/lint_pii_logs.py` (AST lint, in `make lint` + pre-commit)
- [x] `galecore.bus` (BusClient: publish, consumer groups, ack, XAUTOCLAIM retry, `.dlq` dead-letter, idempotency keys, Prometheus counters; binary PCM fields per D-02)
- [x] `galecore.audio` (G.711 µ-law/A-law encode+decode bit-exact vs CCITT reference, soxr resample, stereo split, 200 ms framing, dBFS)
- [x] `galecore.bangla` (`normalize()`, `phonetic_key()`, offset tokenizer; placeholder tags keep case)
- [x] `galecore.redact` (BD mobile incl. Bangla/spoken digits and ডাবল/ট্রিপল, NID 10/13/17, email, ≥7-digit accounts, pluggable NER hook)
- [x] Model interfaces + deterministic stubs (`ASRBackend`, `ShoutDetector`, `SERModel`, `KWSModel`, `AudioToxModel`, `TextClassifier`, `RegisterDetector`, `LLMJudge`), `GALE_MODEL_MODE=stub|real`
- [x] `configs/`: `thresholds.yaml`, `rules.yaml`, `models.yaml`, `bangla_canonical.yaml`, `lexicon/fake_lexicon.yaml` (`<GALI_1>`..`<GALI_20>`)
- [x] Docker-free dev stack (ADR-009): `tools/devstack` (up/status/down), fakeredis TCP bus, embedded Postgres via `pg_ctl`; `uv.toml` / `web/pnpm-workspace.yaml` / Makefile put caches and temp under `.gale/`
- [x] Local `ObjectStore` backend (done in Prompt 3) and `GALE_AUTH_MODE=dev` JWT issuer (moved to P5 Prompt 12, first consumer); config guard already enforces dev-only
- [x] Verified `pgserver` on Windows and fakeredis Streams support; findings in ADR-009 (no `pg_trgm`; no DETACHED_PROCESS)
- [x] `infra/compose/docker-compose.dev.yml` (postgres, redis, minio, keycloak realm with 5 roles + demo users, services) + `infra/docker/service.Dockerfile`: written, **not run locally** (no Docker)
- [x] `web/` skeleton: Next.js 15 App Router, TS strict, Tailwind 4, shadcn/ui button, next-intl (bn default, en), TanStack Query, Zustand; Vitest tests check the Category enum and bn/en labels against the backend schema. wavesurfer.js/Recharts are added in P5 with their first screens
- [x] `.gitattributes` (`* text=auto eol=lf`), Windows-safe Makefile (`SHELL := bash`), and a Windows CI job (ADR-008)
- [x] Makefile (`setup`, `dev-up`, `dev-down`, `dev-status`, `check`, `gen-types`, later-phase targets print where they arrive), pre-commit (local hooks only)
- [x] `make gen-types`: pydantic → JSON Schema (`tools/gen_types.py`) → TypeScript (`web/scripts/gen-types.mjs`) + `make gen-types-check` drift check
- [x] GitHub Actions CI: ruff, mypy --strict, pytest + coverage ≥ 80% (Linux + Windows), eslint, tsc, vitest, types drift, docker build (7 images), Trivy. Green since run #3 (fixes: uv-managed Python on runners, trivy-action pinned by commit)
- [x] `tools/fixtures_gen`: 10 synthetic stereo 8 kHz calls (138–167 s) with `script.json` + `expected_events.json` (placeholders only); pluggable TTS (stub + local VITS); `call.wav` rendered on demand (ADR-010)
- [x] `tools/simulator`: RTP (G.711 µ-law/A-law), WebSocket PCM, file-drop modes; `--speed`; round-trip decode tests; `make simulate`
- [x] ingest-gateway P0 stub transports: RTP session registry + UDP receiver, WebSocket PCM, decode → 16 kHz → 200 ms `AudioFrame` → counting sink; `galecore.rtp` (RFC 3550 codec, loss/reorder tracking)
- [ ] **[HUMAN]** Legal basis and DPIA started (PDPO 2025) 🔒
- [ ] **[HUMAN]** Call-data access approval 🔒
- [ ] **[HUMAN]** Dataset and model licence review started 🔒
- [ ] **[HUMAN]** Hardware ordered 🔒

### P1 — Offline MVP, pretrained + open-data models (weeks 2–5) · Prompts 3, 4b
Exit gate: report runs on 100 real calls; first baseline numbers.
- [x] Post-call pipeline library + CLI (`make report FILE=…`) reusing the worker logic (`packages/gale_postcall`, ADR-011)
- [~] Silero VAD, ASR backends `whisper_ct2` + `indic_conformer` behind `ASRBackend` (written, not run: no weights/GPU); `qwen3_asr` pinned only
- [x] Text steps: normalise → redact → lexicon (exact, fuzzy, phonetic, Banglish) → register rules → text classifier (HF backend written, needs the Prompt 4b checkpoint)
- [x] Audio steps: DSP shouting features with per-speaker rolling baseline (rule fallback, LightGBM hook), emotion2vec+ SER backend (not run); audio-tox and KWS stay stubs until P3
- [x] Overtalk/interruption detector from two-channel VAD with crosstalk trimming (D-05, ADR-011)
- [x] LLM judge client (window + call mode), JSON-schema validation, evidence-grounding post-check, circuit breaker, versioned prompts (tested against a mock server, no real vLLM yet)
- [x] Fusion v0: identity calibration, `rules.yaml` v2 (named rules, params in config), merge, hysteresis/cool-down, escalation, severity, alert policy
- [x] `report.json` + single-file `report.html` (waveform SVG, events, scorecard, timelines, highlighted redacted transcript); also stored via the local `ObjectStore`
- [x] `scripts/download_models.py` (pinned commit hashes) + generated `ml/MODEL_LICENSES.md` (drift-tested)
- [x] Stub-mode integration test against all fixtures' `expected_events.json` (docs/06 matching rule): 10/10 exact, no false positives
- [x] Open-data track: `ml/data/datasets.yaml` → `DATASETS.md` register with the `commercial_ok` gate (prod/research run tags); one loader per dataset T1–T15, R1–R4, A1–A4; T1/T6/T8 auto-downloaded and built for real (57,698 rows), the rest await manual placement
- [~] Back-transliteration: seq2seq trainer + inference (tiny smoke run OK) and a rule-based fallback in the build; real training needs BanglaTLit (manual) + GPU
- [x] Harmonise → `unified.jsonl` (`label_map.yaml` per §5, masked labels + soft weak targets, LID filter, cleaning, MinHash dedupe + weighted conflict votes, official-test preservation, BIDWESH parent splits, domain re-weighting, ASR-noise copies, spoken-style rewrite hook) + data card + per-model views
- [~] BanglaBERT-family multi-task classifier on open data only: trainer (masked focal BCE, soft targets, aggression aux, ordinal severity) + public_aux report vs §8; CPU smoke on real data learns (dev macro-F1 0.45, tiny random encoder); full encoder runs on the GPU box
- [x] Slang-lexicon candidate mining (log-odds with Dirichlet prior + PMI, variant grouping) → review CSV (307 candidates from the real build, kept in `.gale/`)
- [~] First KWS model on BAAD: CNN on log-mel with telephony augmentation, speaker-held-out dev; smoke run OK; real run needs BAAD (manual) + GPU
- [~] LLM judge baseline: open-synthetic SFT windows built (41,989, docs/08 §4 format, grounded evidence); LoRA training is P3
- [ ] **[HUMAN]** Seed lexicon review and approval 🔒
- [ ] **[HUMAN]** 50-call pilot annotation to validate the guide 🔒
- [ ] **[HUMAN]** Manual download of non-redistributable datasets 🔒

### P2 — Data engine (weeks 3–8) · Prompts 4, 5
Exit gate: `golden_v1` / `dev_v1` frozen; α ≥ 0.7.
- [x] Folder / S3-MinIO / recorder-CSV ingestion → schema-validated manifests, salted customer hashes, UUID call ids (tested on synthetic calls + fake S3; real recorder format unverified)
- [x] Channel split, 8 kHz native + 16 kHz model copies + stereo review copy, per-channel quality filters
- [x] Resumable auto-transcription job; seed transcripts redacted in-process before they persist
- [x] Speaker-aware splits (connected components over agent/customer), dialect/queue stratification
- [x] Augmentation: µ-law/A-law in-process, AMR-NB/GSM-FR via ffmpeg/sox (not installed here: untested), packet loss, band-limit, noise, speed, shout-safe gain; seeded
- [x] Label Studio T1/T2/T3 templates (from the enums), REST client, tasks with pipeline pre-annotations, export → training sets for ASR, KWS, shout, SER, audio-tox, textclf, LLM SFT, fusion; agreement (α, span IoU) + adjudication CSV (LS itself mocked)
- [x] Eval harness (`make eval COMPONENT=… SPLIT=…`, `make bakeoff`): metrics for all 9 components, docs/06 event matching, calibration + reliability plot, call-level ρ/AUC, fairness slices with 1.5× flags, bootstrap CIs, JSON/MD/HTML reports, optional MLflow (ADR-014)
- [x] Golden-set lock (`golden.lock` hashes) + overlap check (`make golden-check`, tests); converters refuse golden rows
- [~] DVC config with MinIO remote (written by hand, DVC not installed) + MLflow service in docker-compose (not run: no Docker)
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
- [x] ingest-gateway: Asterisk ARI ExternalMedia + snoop (per direction), FreeSWITCH audio fork, file mode, G.711 decode (AMR-NB still open: needs a codec lib) — runbook `docs/runbooks/pbx-integration.md`
- [x] Streaming ASR policy (energy endpointing, partials 1 s, finals after 0.6 s silence) over any `ASRBackend`; real cache-aware Conformer / SimulStreaming backends come with Prompt 6 🔒
- [x] audio-worker, asr-worker, text-worker, llm-judge, fusion-engine on Redis Streams with consumer groups (+ post-call worker with real-time priority)
- [x] Real-time fusion, debounce, LLM circuit breaker ("LLM offline" degradation)
- [x] Per-stage latency histograms + OTel traces keyed by `call_id`
- [ ] **[HUMAN]** PBX team configures media fork in staging 🔒

### P5 — API & GUI (weeks 9–14) · Prompts 12–13
Exit gate: usability sign-off.
- [x] FastAPI REST + WS per `04_GUI_SPEC.md` §3, OIDC (Keycloak), RBAC, audit log on every audio/transcript view, signed audio URLs (Prompt 12, ADR-016; OIDC verified against the dev issuer only)
- [x] Alembic migrations for the `01_ARCHITECTURE.md` §4 schema (+ D-08 additions), monthly partitions
- [x] OpenAPI → TS types (`web/src/types/generated/api.ts`) + typed `openapi-fetch` client
- [x] Web: `/live`, `/live/[callId]`, `/calls`, `/calls/[callId]`, `/review`, `/analytics`, `/me`, `/admin/{lexicon,rules,models,health,audit}` (Prompt 13, ADR-017)
- [x] bn/en i18n (key parity tested), dark/light, wallboard 1,000 calls at 2 updates/s (reducer load test + virtualised grid)
- [ ] WCAG 2.1 AA audit (icon + text + colour, ARIA meters, skip link and reduced motion are in place; no axe/Lighthouse run yet)
- [x] Vitest (27 tests) + Playwright e2e against the simulator stack (5 scenarios)
- [ ] Visual regression + Lighthouse a11y ≥ 90 (reference screenshots exist in `docs/screenshots/`)
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
Accepted 2026-11-02: ADR-008 native Windows development; ADR-009 Docker-free local dev (venv + fakeredis + embedded Postgres + local object store + dev auth; all caches on G:); ADR-010 synthetic fixtures (script is the source of truth, WAV rendered on demand) and the simulator ↔ gateway stub protocol.

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
| D-16 | `ml/` is a plain package at the repo root (so `python -m ml.eval.run` works), not a uv workspace member; heavy ML deps arrive as root dependency groups per component. | P0 |
| D-17 | `normalize()` lower-cases Latin text but keeps placeholder tags (`<PHONE>`, `<GALI_1>`) intact, so tag and lexicon matching survive normalisation. | P0 |
| D-18 | GAL-E mints its own UUID `call_id` (never a PBX channel name or caller-derived id). The log scrubber keeps `call_id` / `trace_id` unredacted only when the value is a valid UUID; any other value, e.g. `PJSIP/017…`, is still scrubbed. From Q-10, approved 2026-11-02. P4 ingest-gateway mints the UUID when ARI/FreeSWITCH report a new call. | P0 |
| D-19 | Fusion `rules.yaml` v2: named rules with parameters in YAML; shouted abuse raises the abusive event's severity; event spans come from word/window-precise signals; cool-down suppresses alerts, not events. Register: customer → তুমি fires only with negative context. Overtalk counts turn starts on crosstalk-trimmed VAD. Ground truth follows the same merge/overtalk conventions. Details in ADR-011. | P1 |
| D-20 | Open-data prototype runs on `--run-tag research` without the licence review (owner decision 2026-11-02, ADR-012): everything trained is `research_only` / not promotable. Raw data, unified set and lexicon candidates stay under `.gale/`. Training deps are an opt-in `ml-train` group; trainer smoke tests skip where torch is absent (CI). | P1 |
| D-21 | Data engine (ADR-013): uuid5 call ids, salted HMAC customer hashes (salt only from a secret), per-channel quality filters, speaker-connected-component splits, golden lock + converter refusal, LS templates generated from enums, annotator text re-redacted on export. | P2 |
| D-22 | Evaluation definitions (ADR-014): gold with context flags must end dismissed/absent; per-slice FPR = false alerts per 100 calls; call-level 'needs QA review' = alert-worthy agent event or T3 professionalism ≤ 2; bootstrap resamples without events are skipped; golden loads only with a matching lock. | P2 |
| D-23 | Streaming services (ADR-015): one `galecore.streaming.Worker` per service; lexicon fast path on ASR partials; latency measured from the word (lexicon) or utterance end (register/classifier); replay-based restart recovery; post-call jobs yield to real-time backlog; provisional alerts can be retracted via `events.fused`. | P4 |
| D-24 | API & persistence (ADR-016): an `api_persister` consumer writes bus messages to Postgres (idempotent upserts); post-call runs publish `postcall.results`. String pipeline ids; monthly partitions on `call_started_at`; trigram index only where `pg_trgm` exists. Role from the token, scoping from the DB (out of scope = 404). WS auth via first message, 250 ms batches. Two-person model promotion; `research_only` never promotable. Gateway carries optional `agent_id`/`queue`. | P5 |
| D-25 | Web dashboard (ADR-017): client-rendered screens over the generated API client; sign-in via Keycloak PKCE (`oidc-client-ts`) or the dev issuer, token in sessionStorage; one `LiveSocket` (first-message auth, backoff) + pure Zustand reducers; wavesurfer for stereo/per-channel audio; CSV in the browser, PDF via print. Small API additions (`/me`, `/directory`, `/reviews/agreement`, `/thresholds`, note/flag reviews, heatmap, more call filters). | P5 |
| D-15 | Mono/diarized audio: `Channel.mixed` is allowed only before diarization. Fusion accepts only `agent`/`customer`. Segments that can't be attributed are stored but never alert. | P1 |

---

## Open questions for the human

**2026-11-02 answers:** Q-01 to Q-05 and Q-07 to Q-09 approved with the defaults below. Q-10 approved later that day (D-18). As a result, D-01 to D-15 are accepted. Q-06 was answered **native Windows, no WSL2** (ADR-008).

Licence review (Q-03, Q-04) is still a **[HUMAN]** legal action. **2026-11-02:** the owner waived it for the prototype (ADR-012, D-20); it is required again before any production training run. The approval covers only the engineering default: swappable encoder, ship `commercial_ok` only.

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
| Q-10 | **Log scrubber vs identifiers.** The PII scrubber redacts any ≥ 7-digit run in every log value. It mangled ISO timestamps (fixed: timestamps are now added after scrubbing) and also mangles ids with long digit runs, e.g. UUID tails (`…-446655440000` → `<ACCT>`), so `call_id` in logs becomes unusable for tracing. Exempting `call_id` / `trace_id` is a privacy-posture change (some PBX channel names embed caller numbers). Proposal: GAL-E always mints its own UUID `call_id` (never a PBX channel name), and those keys are exempt only when the value is a valid UUID. | Security posture: CLAUDE.md says ask. | **Approved 2026-11-02** → D-18 |
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

## Prompt 1 summary (2026-11-02)

**Built:** uv workspace (galecore + 7 services), shared core (schemas, config, logging + scrubber, bus, audio, Bangla normaliser, redaction, model interfaces + stubs), configs, Docker-free devstack, Next.js skeleton, codegen, Makefile, pre-commit, CI, compose/Dockerfile.

**Verified on this Windows machine:** `make check` passes: ruff, format, PII-in-logs lint, mypy --strict, 135 pytest tests at 99.5% coverage, eslint, tsc, 6 vitest tests, types regenerate. `make dev-up` brings up bus, embedded Postgres, 7 services (all `/readyz` ready) and the web app (Bangla page renders). `make dev-down` stops everything cleanly.

**Stubbed:** every model (`GALE_MODEL_MODE=stub`); real backends raise `RealModelsUnavailableError` until P1.

**Not verified:** GitHub Actions (nothing pushed), Docker image build, compose stack and Trivy (no Docker locally).

**Needs a human:** delete the leftover `G:\.pnpm-store` folder (a broken partial pnpm store from this session; the tool wasn't allowed to remove a drive-root folder). Optionally delete `%LOCALAPPDATA%\pnpm-cache` (~40 MB on C:, pnpm metadata from the first install).

**Incident (2026-11-02):** pnpm 12's global virtual store keeps a junction from `<store>/projects/<hash>` to `web/`. Moving the store followed it and moved web config files into the store. Files were restored from there, nothing was lost, and `enableGlobalVirtualStore: false` now prevents it (ADR-009).

## Prompt 2 summary (2026-11-02)

**Built:**
- `tools/fixtures_gen`: scenario DSL, deterministic stub TTS (local VITS optional), ground-truth derivation, renderer, `gen` / `render` / `check` CLI.
- 10 committed fixture scripts covering slang, insult, threat, sexual harassment, register drop (agent and customer), shouting, overtalk, dismissive-rude, context negatives (reported speech, self-directed) and a clean control.
- `tools/simulator`: RTP µ-law/A-law, WebSocket PCM, file drop; `--speed`; CRC integrity report.
- `galecore.rtp`.
- ingest-gateway stub transports (session registry, UDP RTP receiver, WebSocket endpoint, soxr streaming resample to 200 ms 16 kHz `AudioFrame`s, Prometheus counters).
- `make simulate` / `fixtures` / `fixtures-wav`; ADR-010.

**Verified on this Windows machine:**
- `make check` passes: 171 pytest tests at 99.3% coverage, plus mypy --strict, ruff and the web checks.
- `make simulate CALL=tests/fixtures/calls/demo_01 SPEED=20` against `make dev-up`: 8,330 RTP packets per channel, 0 lost, 833 frames per channel, CRC round trip OK. WebSocket mode is also OK.
- The gateway log has counts only, with no audio or text.

**Fixed along the way:** the log scrubber redacted ISO timestamps as `<ACCT>`. The timestamp is now added after scrubbing, with a regression test. Identifier scrubbing is raised as Q-10.

**Stubbed:**
- The TTS (not real Bangla speech).
- The gateway's frame sink (counts only, no bus publish yet: P4).
- No RTP jitter buffer (P4).

**Needs a human:** ~~answer Q-10~~ (approved → D-18, implemented: UUID exemption in `galecore.logging`, simulator mints UUID call ids). Optional: a local Bangla TTS checkpoint at `GALE_TTS_DIR` for speech-realistic fixtures (licence check first; MMS-TTS is CC-BY-NC-4.0).

## Prompt 3 summary (2026-11-02)

**Built:**
- `packages/gale_postcall`: pipeline, report and CLI.
- Worker logic in every service package: ASR step with in-process redaction, DSP shouting, overtalk, lexicon matcher, register rules, text step, LLM windows/grounding/client, fusion v0 (ADR-011).
- Real backends behind the interfaces (lazy imports, `real` extras).
- Fixture oracle for stub mode.
- `galecore.objectstore`.
- `scripts/download_models.py` + `ml/MODEL_LICENSES.md`.
- `make report`.

**Verified on this Windows machine:**
- `make check`: 230 tests at 99.2% coverage, plus ruff, mypy strict (59 files) and the web checks.
- All 10 fixtures match ground truth exactly in stub mode, including two dismissed context negatives (demo_09).
- `make report FILE=tests/fixtures/calls/demo_01/call.wav` writes JSON + HTML in about 1.5 s; the HTML was checked in a browser.
- `make dev-up` / `dev-down` still work.

**Fixed along the way:**
- Crosstalk from a shouting far end opened the near-end VAD and hid an interruption.
- Breaths inside a turn counted as interruptions.
- The logging config captured a stale `sys.stdout`.

**Stubbed / not run:**
- Every real model (no weights, no GPU here), so there are no real accuracy numbers.
- KWS and audio-tox (P3).
- Calibration is the identity; no stacker.

**Needs a human:**
- A GPU host with `GALE_MODEL_DIR` populated by `scripts/download_models.py` (accept the IndicConformer terms on HF), plus a vLLM server for the judge.
- Licence review: emotion2vec+ (`verify`), BanglaBERT (`research_only`) (Q-03).
- 100 real stereo calls for the P1 exit gate.

## Prompt 4b summary (2026-11-02)

**Built:**
- `ml/data/open/`: register and licence gate; 23 loaders driven by `datasets.yaml`; label mapping (`ml/textclf/label_map.yaml` per docs/09 §5); cleaning, language ID, back-transliteration (rule fallback + seq2seq), MinHash dedupe, harmonisation, augmentation; views (text classifier, LLM SFT) and data card; CLI with `status`, `download` and `build`.
- `ml/lexicon/mine.py`.
- Trainers with CPU smoke modes: `ml/textclf/` (multi-task classifier + public_aux report), `ml/translit/train.py`, `ml/kws/` (log-mel CNN, telephony augmentation).
- `docs/runbooks/open_data.md`; ADR-012.

**Ran on this machine with real data:**
- Downloaded T1, T6 and T8: 57,698 rows → 63,435 unified (338 near-duplicates removed, 6,075 ASR-noised copies, 0 test/train leakage).
- 41,989 SFT windows.
- 307 lexicon candidates for review.
- Tiny text classifier trained on CPU: dev macro-F1 0.45, which shows the pipeline learns. Real numbers need a pretrained encoder on the GPU box.

**Fixed along the way:**
- transformers 5 silently ignores `BertTokenizerFast(vocab_file=…)` (5-token vocabulary).
- Placeholder tags were treated as Latin script and transliterated.

**Verified:** `make check`: 248 tests at 99.2% coverage, plus ruff and mypy. The 3 trainer smoke tests run only where `ml-train` is installed; CI skips them.

**Needs a human:**
- Manual downloads for 17 text/translit sets and the 4 audio sets (list in the runbook).
- Licence review before any production use (ADR-012).
- A GPU box for the encoder, transliteration and KWS runs.
- Human review of the lexicon candidates.

## Prompt 4 summary (2026-11-02, on synthetic fixtures)

**Built:**
- `ml/data`: manifests + pseudonymisation; folder / S3 / recorder-CSV ingestion; processing and quality filters; resumable transcription; splits + golden lock; seeded augmentation; agreement + adjudication; CLI `python -m ml.data`.
- `ml/data/labelstudio`: T1–T3 templates, client, tasks with pre-annotations, converters for 8 training sets.
- docs/02 §1.1 speech corpora added to the dataset register.
- `.dvc/config` (MinIO remote); MLflow in compose; `make golden-check`.
- `docs/runbooks/annotation-ops.md`; ADR-013.

**Verified:** the full flow runs on 3 synthetic calls in tests: ingest → split/lock → transcribe → LS tasks → simulated double annotation → training sets → agreement. `make check` passes.

**Not verified (no access here):** real recorder exports, a live Label Studio, real MinIO/S3, ffmpeg AMR/GSM codecs, DVC and MLflow servers.

**Needs a human:**
- The recording location and recorder CSV column names.
- The pseudonymisation salt (a secret).
- S3/MinIO credentials.
- A Label Studio URL + token.
- The golden call list.
- DVC remote credentials.
- Annotators (see the runbook's setup table).

## Prompt 5 summary (2026-11-02)

**Built:** `ml/eval`:
- metrics: CER/WER with the shared normaliser, slang recall, script drift; KWS recall + false alarms/hour; docs/06 event matching; P/R/F1 by category × speaker; ECE + reliability bins; Spearman, AUC, UAR; latency summaries; bootstrap CIs;
- eval sets (fixtures; manifest splits with golden-lock guard);
- 9 component evaluators on the production code path;
- fairness slices;
- reports (JSON/MD/HTML with a reliability SVG) and optional MLflow;
- `run` (`make eval`) and `bakeoff` (`make bakeoff`, writes the ADR table, example `configs/bakeoff/asr.yaml`).

**Verified:**
- `make check`: 276 tests at 99.1% coverage.
- Every component evaluated on the synthetic fixtures. System: event F1 1.0, alert recall 0.7 (cool-down suppresses repeats), 0 false alerts / 100 calls. ASR CER 0 (oracle). Script drift 4.4% (the Banglish call). KWS recall 0 and audio-tox AUC 0.5 (still stubs).

**Found and fixed:** bootstrap resamples with no events were scored as F1 = 0, which widened the CIs.

**Needs a human:** annotated `dev_v1` / `golden_v1` (Prompt 4 flow) before any real numbers; an MLflow server for run tracking.

## Prompt 11 summary (2026-11-02)

**Built:**
- `galecore.streaming`: Worker base (consumer groups, ticks, lifecycle restore, per-call replay), `CallClock`, `gale_stage_latency_seconds`, OTel spans with trace id from the call UUID.
- ingest-gateway: bus publisher (bounded queue, drop counter, call audio retained to the object store), Asterisk ARI app (snoop + ExternalMedia), FreeSWITCH `/v1/fork`, WS `/v1/stream/{call_id}`, watched drop folder → `postcall.jobs`.
- Streaming workers: asr (endpointing, partial/final), audio (shout / SER / audio-tox / KWS / overtalk on ring buffers), text (lexicon on partials, full steps on finals), llm-judge (3 s debounce, scorecard at hang-up → `call.scores`), fusion (realtime re-fuse, alerts).
- `python -m gale_postcall worker` (port 8007) in the dev stack.
- New schemas `LlmDecision`, `CallScoreMessage`, `PostcallJob`; `CallLifecycle.external_ref` / `wall_clock`.

**Verified:**
- `tests/integration/test_streaming.py`: the fixtures give the expected events when streamed; the latency budget is met (provisional ≤ 2 s, confirmed ≤ 6 s p95); concurrent unpaced calls; text + fusion killed mid-call and restarted with nothing lost; post-call worker waits for the real-time backlog.
- Live `make dev-up` + `make simulate CALL=tests/fixtures/calls/demo_01 SPEED=4`: confirmed events, 2 alerts, scorecard and latency metrics. A leftover dropped WAV from Prompt 2 was picked up by the post-call worker as expected.

**Stubbed / limits:** stub models only; ARI tested against a fake ARI (no real PBX here); ASR/audio buffers are in memory, so a crash there loses the utterance in progress; AMR-NB decode not done.

**Needs a human:** the PBX team configures the media fork in staging (runbook checklist) 🔒; p95 on a staging PBX for the P4 exit gate.

## Prompt 12 summary (2026-11-02)

**Built:** `services/api`:
- SQLAlchemy 2 models and Alembic migration 0001 for the full docs/01 §4 schema (+ D-24 additions). Monthly partitions for segments/signals, created on demand. Trigram index on normalised redacted text.
- Bus → DB persister (`api_persister` group); post-call results via the new `postcall.results` stream.
- Auth: OIDC JWT (Keycloak JWKS) or the dev issuer. RBAC permission table + row scoping (supervisor teams, agent's own calls). Audit middleware.
- Every docs/04 §3 endpoint: calls (filters, transcript search), segments, events (with top signals), timelines, signed audio URLs, reviews/disputes, acks, review queue, agent/team/queue analytics, lexicon CRUD (QA proposes, admin activates, soft delete), rules (versioned) + `rules/simulate` (new `gale_fusion_engine.simulate`), models + two-person promotion, audit, recording upload.
- WebSockets `/ws/live` and `/ws/calls/{id}`, role/team-filtered, batched every 250 ms.
- `S3ObjectStore` (MinIO pre-signed URLs).
- Gateway, simulator and Asterisk/FreeSWITCH metadata carry `agent_id`/`queue`.
- Dev seed (`python -m gale_api seed-dev`: 2 teams, 3 agents, one user per role, model versions), run by `make dev-up`.
- `make gen-types` now also emits OpenAPI → `api.ts`; `web/src/lib/api.ts` is the typed client.

**Verified:**
- RBAC matrix: 27 endpoint cases × 7 roles + anonymous, with a test that fails if an endpoint has no case.
- Scoping, reviews, acks, promotion (self-approval refused, research_only refused), rules validation + simulation diff, audit rows, signed-URL download (+ tampered link refused), upload → `postcall.jobs`.
- Persister idempotency; WS close codes, team filtering and coalescing.
- Migration up/down/up with ORM == schema.
- Post-call worker on an object-store key.
- Tests run on a real Postgres (embedded pgserver; `GALE_TEST_PG_URL` for an external one).

**Stubbed / limits:** Keycloak not exercised (dev issuer only); workers don't hot-reload rules/lexicon edits yet (P6); the embedded Postgres has no `pg_trgm`, so dev search is unindexed.

**Needs a human:** a Keycloak realm `gale` with client `gale-api` and the five realm roles, for staging.

## Prompt 13 summary (2026-11-02)

**Built:** `web/` per docs/04 §2, Bangla by default with an English toggle:
- `/live`: KPIs, filters, virtualised call grid with agent/customer risk meters, alert feed with acknowledge / false alarm, optional sound and desktop notification for severity 3, offline banner.
- `/live/[callId]`: two-lane streaming transcript with partials greyed and flagged spans, register-switch markers, loudness/anger/overtalk timelines, event stack, note / flag-for-QA / acknowledge.
- `/calls` (all spec filters incl. transcript search, CSV) and `/calls/[callId]`: stereo waveform with event regions, synced transcript with PII tags, timelines + escalation curve, scorecard, event table with evidence drill-down and review controls, CSV/PDF.
- `/review`: keyboard-first (J/K, 1–4, Space plays span ±3 s), inter-reviewer agreement.
- `/analytics`: agent scorecard vs team median, coaching moments, team heatmap, queues, fairness panel (pending Q-05).
- `/me`: own calls, scorecard, disputes.
- `/admin/lexicon`: masked terms with reveal, proposal → approval, CSV import, 7-day match preview.
- `/admin/rules`: YAML + form view, simulate on last N calls, history with rollback, thresholds table.
- `/admin/models` (two-person promotion), `/admin/health` (API readiness + Grafana embeds), `/admin/audit` (filters, CSV).
- `make e2e` / `make e2e-install` (Chromium under `.gale/ms-playwright`).

**Verified:**
- Vitest: 27 tests covering reducers, socket backoff and refusals, a 1,000-call load test, CSV/YAML/keyboard helpers, accessible components, bn/en parity.
- Playwright against `make dev-up` + simulator: alert appears on the wallboard; call detail; a keyboard review sets `reviewed_false`; an agent can't see another agent's calls (UI and API 404); admin screens in English.
- `next build` passes.

**Screenshots** (synthetic data, written by the e2e run): `docs/screenshots/01-live-wallboard.png`, `02-live-call.png`, `03-call-detail.png`, `04-call-explorer.png`, `05-review-queue.png`, `06-agent-self-view.png`, `07-analytics.png`, `08-team-heatmap.png`, `09-rules-simulate.png`, `10-lexicon.png`, `11-models.png`, `12-audit.png`.

**Stubbed / limits:** Keycloak sign-in not exercised; fairness panel waits on Q-05; simulation shows alert deltas, not P/R; thresholds read-only; no visual-regression or Lighthouse run yet.

**Needs a human:** the UX review with 3 supervisors + 2 QA analysts (P5 exit gate) 🔒; answer Q-05 for the fairness panel.

## Next step

Run **Prompt 6: P3a ASR fine-tuning** on the GPU box (🔒 GPU + data). Prompts 6–10 need a GPU and annotated data. Prompt 11 (P4) is built on the simulator; its exit gate waits on a staging PBX. Prompts 12–13 (P5 API + dashboard) are done; the P5 exit gate is a UX review with supervisors and QA. **Prompt 14 (P6 production hardening)** needs a cluster 🔒; locally, the remaining P5 items are visual regression + a Lighthouse/axe accessibility run.
