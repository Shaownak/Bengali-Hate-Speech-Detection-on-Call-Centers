# GAL-E developer commands (CLAUDE.md). Works on Linux/macOS and on native Windows with
# GNU Make >= 4 + Git Bash (ADR-008). Local dev needs no Docker (ADR-009).

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
GALE := $(ROOT)/.gale

# ADR-009: every cache / temp / model dir lives on the repo drive, never on C:
export UV_CACHE_DIR := $(GALE)/uv-cache
export UV_LINK_MODE := copy
export TMP := $(GALE)/tmp
export TEMP := $(GALE)/tmp
export TMPDIR := $(GALE)/tmp
export HF_HOME := $(GALE)/hf
export GALE_MODEL_DIR := $(GALE)/models
export PRE_COMMIT_HOME := $(GALE)/pre-commit
export PYTHONUTF8 := 1
export NEXT_TELEMETRY_DISABLED := 1

ifeq ($(OS),Windows_NT)
  PY := $(ROOT)/.venv/Scripts/python.exe
else
  PY := $(ROOT)/.venv/bin/python
endif
UV ?= uv
PNPM ?= pnpm
DEV_STACK ?= local
PY_PKGS := -p galecore -p gale_api -p gale_ingest_gateway -p gale_audio_worker \
           -p gale_asr_worker -p gale_text_worker -p gale_llm_judge -p gale_fusion_engine

.PHONY: help setup dev-up dev-down dev-status check check-py check-web lint fmt typecheck test \
        test-web gen-types gen-types-check simulate fixtures fixtures-wav report eval e2e clean-caches

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-16s %s\n",$$1,$$2}'

$(GALE)/tmp:
	@mkdir -p $@

setup: $(GALE)/tmp ## uv sync + pnpm install + pre-commit hooks
	$(UV) sync
	cd web && $(PNPM) install
	$(PY) -m pre_commit install

# --- dev stack ---------------------------------------------------------------------------

dev-up: $(GALE)/tmp ## start bus, db and all services in stub mode (+ web); DEV_STACK=compose for Docker
ifeq ($(DEV_STACK),compose)
	docker compose -f infra/compose/docker-compose.dev.yml up -d --build
else
	$(PY) -m tools.devstack up --web
endif

dev-down: ## stop the dev stack
ifeq ($(DEV_STACK),compose)
	docker compose -f infra/compose/docker-compose.dev.yml down
else
	$(PY) -m tools.devstack down
endif

dev-status: ## show dev stack status
	$(PY) -m tools.devstack status

# --- quality gates -----------------------------------------------------------------------

check: check-py check-web gen-types-check ## ruff, mypy, PII lint, pytest, eslint, tsc, vitest, types drift

check-py: lint typecheck test

lint: ## ruff + format check + PII-in-logs lint
	$(PY) -m ruff check packages services ml tools
	$(PY) -m ruff format --check packages services ml tools
	$(PY) -m tools.lint_pii_logs

fmt: ## auto-format Python
	$(PY) -m ruff check --fix packages services ml tools
	$(PY) -m ruff format packages services ml tools

typecheck: ## mypy --strict on packages and services
	$(PY) -m mypy $(PY_PKGS)

test: $(GALE)/tmp ## pytest with coverage gate (>= 80%)
	$(PY) -m pytest -p no:cacheprovider --cov --cov-report=term

check-web: ## eslint, tsc, vitest
	cd web && $(PNPM) lint && $(PNPM) typecheck && $(PNPM) test

test-web:
	cd web && $(PNPM) test

gen-types: ## pydantic → JSON Schema → TypeScript (web/src/types/generated)
	$(PY) -m tools.gen_types
	cd web && node scripts/gen-types.mjs

gen-types-check: gen-types ## fail if generated types are out of date
	@git diff --exit-code -- web/src/types/generated || \
	  (echo "generated types are stale: run 'make gen-types' and commit" && exit 1)

# --- later phases (targets exist so the CLAUDE.md command list is stable) -----------------

CALL ?= tests/fixtures/calls/demo_01
MODE ?= rtp
SPEED ?= 1
simulate: ## replay a synthetic call: make simulate CALL=tests/fixtures/calls/demo_01 [MODE=rtp|ws|file] [SPEED=10]
	$(PY) -m tools.simulator $(CALL) --mode $(MODE) --speed $(SPEED)

fixtures: ## regenerate synthetic fixture calls (script.json, expected_events.json, call.wav)
	$(PY) -m tools.fixtures_gen gen

fixtures-wav: ## render missing call.wav files from the committed scripts
	$(PY) -m tools.fixtures_gen render

report: ## offline post-call report (Prompt 3)
	@echo "report: arrives in P1 Prompt 3. See docs/PROGRESS.md." && exit 2

eval: ## eval harness (Prompt 5)
	@echo "eval: arrives in P2 Prompt 5. See docs/PROGRESS.md." && exit 2

e2e: ## Playwright against the dev stack (Prompt 13)
	@echo "e2e: arrives in P5 Prompt 13. See docs/PROGRESS.md." && exit 2

clean-caches: ## delete .gale caches (keeps pgdata, models)
	rm -rf $(GALE)/uv-cache $(GALE)/pnpm-cache $(GALE)/tmp/* $(GALE)/pre-commit
