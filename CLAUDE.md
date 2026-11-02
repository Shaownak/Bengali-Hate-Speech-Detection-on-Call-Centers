# CLAUDE.md — GAL-E (Gali & Aggression Listening Engine)

## What this project is
A production system that detects Bangla slang/gali, abusive & negative language, disrespectful register (আপনি→তুমি/তুই), harsh tone and shouting in **Bangladeshi call-centre calls**, per speaker (agent vs customer), in **post-call QA mode** and **real-time alert mode**. It is a multi-signal fusion system: DSP + speech-emotion + keyword spotting + ASR + Bangla text classifier + fine-tuned LLM judge → fusion engine → alerts & dashboard.

## Source of truth (read before working on a phase)
- `docs/00_RESEARCH_SUMMARY.md` — why each model/tool was chosen
- `docs/01_ARCHITECTURE.md` — services, message contracts, DB schema, repo layout, latency budget
- `docs/02_DATA_PLAN.md` — datasets, processing, annotation taxonomy (category enum lives here + in schemas)
- `docs/03_MODELS_AND_TRAINING.md` — training recipes per model
- `docs/04_GUI_SPEC.md` — dashboard screens, roles, API surface
- `docs/05_INFRA_AND_OPS.md` — k8s, observability, MLOps, security
- `docs/06_EVALUATION.md` — metrics, targets, test pyramid, pilot
- `docs/07_ROADMAP.md` — phases P0–P7
- `docs/08_LLM_JUDGE_SPEC.md` — LLM prompts and JSON schemas
- `docs/09_OPEN_DATASETS.md` — open Bangla hate-speech/abuse datasets (text, Banglish, audio): catalogue, licence gate, label mapping, harmonisation pipeline, per-model usage
- `docs/PROGRESS.md` — **you maintain this**: what's done, decisions, open questions, next step

If the spec is ambiguous, choose the simplest option consistent with it, record the decision as an ADR in `docs/adr/NNN-title.md`, and continue. If the spec is contradictory or a choice is irreversible (data deletion, licence, security posture), stop and ask.

## Tech stack & conventions
- **Python 3.12**, `uv` for envs/deps (workspace with `packages/*`, `services/*`, `ml/*`). FastAPI, pydantic v2, SQLAlchemy 2 + Alembic, redis-py (asyncio), structlog, OpenTelemetry.
- Lint/format/type: `ruff`, `ruff format`, `mypy --strict` on `packages/` and `services/`. Tests: `pytest` (+ `pytest-asyncio`), coverage ≥ 80% on `packages/` and service logic.
- **Web**: Next.js (App Router), TypeScript strict, Tailwind, shadcn/ui, TanStack Query, Zustand, wavesurfer.js, Recharts, next-intl (bn default, en). pnpm. Vitest + Playwright.
- Shared message/API types: pydantic models in `packages/galecore/schemas.py` are authoritative → generate TS types (`make gen-types`). Never hand-edit generated files.
- Config via env + YAML in `configs/`; validated by pydantic settings. Thresholds/rules are config, never hard-coded.
- Every model sits behind an interface (`ASRBackend`, `SERModel`, `KWSModel`, `TextClassifier`, `LLMJudge`, `ShoutDetector`) with a **`stub` implementation** that returns deterministic fake outputs so the whole system runs on a laptop without GPUs or model downloads (`GALE_MODEL_MODE=stub`). Real implementations load weights from a local model dir (`GALE_MODEL_DIR`), never from the internet at runtime.
- Bangla text: always normalise with `galecore.bangla.normalize()` (NFC + project rules) before matching, training or scoring.
- Commits: small, conventional commits (`feat:`, `fix:`, `chore:`…). Run `make check` before committing.

## Hard rules (non-negotiable)
1. **Privacy**: never log transcript text, audio, or PII. Only redacted text may be persisted or sent to the LLM. No external/cloud AI APIs for call data — all models self-hosted.
2. **No real call data, no real slurs in the repo.** Fixtures are synthetic; slang in tests uses placeholders (`<GALI_1>`…). The real lexicon is loaded from a secure path at runtime (`GALE_LEXICON_PATH`).
3. **Golden set integrity**: never train or tune thresholds on `golden_*` manifests; CI check must enforce no call/speaker overlap.
4. Alerts are advisory; never build automatic punitive actions.
5. Don't weaken tests to make them pass; fix the code or explain in PROGRESS.md.
6. Pin model revisions (HF commit hash) and record licences in `ml/MODEL_LICENSES.md`.

## Common commands (create these in the Makefile during P0)
```
make setup          # uv sync + pnpm install + pre-commit
make dev-up         # docker compose (postgres, redis, minio, keycloak, all services in stub mode, web)
make dev-down
make check          # ruff, mypy, pytest, eslint, tsc, vitest
make gen-types      # pydantic → TS types
make simulate CALL=tests/fixtures/calls/demo_01   # replay a synthetic stereo call as live RTP/WS
make report FILE=path/to/stereo.wav               # offline post-call report (JSON + HTML)
make eval COMPONENT=asr SPLIT=dev                 # eval harness
make e2e            # playwright against dev stack
```

## Definition of done (per task)
Code + tests + types pass `make check`; works in stub mode via `make dev-up`; docs updated (PROGRESS.md, ADR if a decision was made); no TODOs without an issue reference; latency/metrics instrumentation added for new pipeline stages.

## Working style
- Work phase by phase as instructed in the prompt; at the end of each phase, update `docs/PROGRESS.md` and print a short summary: what was built, how to run it, what's stubbed, what needs a human.
- Prefer boring, well-documented libraries. Keep services small. Write the test first for parsers, normalisers, rules and fusion logic.
