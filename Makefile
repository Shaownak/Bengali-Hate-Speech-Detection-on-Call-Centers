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
PY_PKGS := -p galecore -p gale_postcall -p gale_api -p gale_ingest_gateway -p gale_audio_worker \
           -p gale_asr_worker -p gale_text_worker -p gale_llm_judge -p gale_fusion_engine

.PHONY: help setup dev-up dev-down dev-status check check-py check-web lint fmt typecheck test \
        test-web gen-types gen-types-check simulate fixtures fixtures-wav golden-check report eval bakeoff e2e e2e-install helm-check dashboards loadtest \
        pilot-report al-sample retrain clean-caches

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

golden-check: ## fail if a golden call/speaker appears in another split or golden.lock drifted
	$(PY) -m ml.data golden-check

fixtures: ## regenerate synthetic fixture calls (script.json, expected_events.json, call.wav)
	$(PY) -m tools.fixtures_gen gen

fixtures-wav: ## render missing call.wav files from the committed scripts
	$(PY) -m tools.fixtures_gen render

OPEN ?= 1
report: ## offline post-call report: make report FILE=tests/fixtures/calls/demo_01/call.wav [OPEN=0]
	@test -n "$(FILE)" || (echo "usage: make report FILE=path/to/stereo.wav" && exit 2)
	$(PY) -m gale_postcall report $(FILE) $(if $(filter 1,$(OPEN)),--open,)

COMPONENT ?= system
SPLIT ?= fixtures
eval: ## evaluate a component: make eval COMPONENT=<asr|kws|shout|ser|audiotox|textclf|llm|fusion|system> SPLIT=<fixtures|dev|golden|…>
	$(PY) -m ml.eval.run --component $(COMPONENT) --split $(SPLIT)

bakeoff: ## compare candidates: make bakeoff CANDIDATES=configs/bakeoff/asr.yaml SPLIT=dev [ADR=docs/adr/0NN-x.md]
	$(PY) -m ml.eval.bakeoff --candidates $(CANDIDATES) --split $(SPLIT) $(if $(ADR),--adr $(ADR),)

HELM ?= helm
KUBECONFORM ?= kubeconform
CHART := infra/helm/gal-e
CRD_SCHEMAS := https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json

helm-check: ## helm lint + template every values file + kubeconform (needs helm, kubeconform on PATH)
	$(HELM) lint $(CHART) -f $(CHART)/values-prod.yaml
	@mkdir -p $(GALE)/tmp
	@for v in "" values-staging.yaml values-prod.yaml values-kind.yaml; do \
	  echo "== $${v:-values.yaml}"; \
	  $(HELM) template gale $(CHART) $${v:+-f $(CHART)/$$v} --set global.imageTag=check \
	    > $(GALE)/tmp/rendered.yaml; \
	  $(KUBECONFORM) -strict -summary -ignore-missing-schemas -kubernetes-version 1.31.0 \
	    -schema-location default -schema-location '$(CRD_SCHEMAS)' $(GALE)/tmp/rendered.yaml; \
	done

dashboards: ## regenerate the Grafana dashboards in the Helm chart
	$(PY) -m tools.gen_dashboards

loadtest: ## load test at 25/50/100/200% of PEAK concurrent calls against `make dev-up`
	$(PY) -m tools.loadtest --peak $${PEAK:-4} --levels 25,50,100,200

pilot-report: ## weekly pilot report (ADR-020): make pilot-report [WEEK=2026-W44] [NOTIFY=1] → .gale/reports/pilot
	$(PY) -m gale_api pilot-report $(if $(WEEK),--week $(WEEK)) $(if $(NOTIFY),--notify) --out $(GALE)/reports/pilot

al-sample: ## active-learning sample: make al-sample [WEEK=…] [PUSH=1] → .gale/al (needs the golden manifest)
	$(PY) -m gale_api al-sample $(if $(WEEK),--week $(WEEK)) $(if $(PUSH),--push) --out $(GALE)/al

retrain: ## retrain + gate: make retrain COMPONENT=fusion FEEDBACK=fb.jsonl [SPLIT=golden] [VERSION=…]
	$(PY) -m ml.retrain run --component $(COMPONENT) --work $(GALE)/retrain/$(COMPONENT) \
	  --feedback $(FEEDBACK) --split $${SPLIT:-golden} --version $${VERSION:-$(COMPONENT)-local}

e2e: ## Playwright against the dev stack + simulator (run `make dev-up` first)
	cd web && PLAYWRIGHT_BROWSERS_PATH=$(GALE)/ms-playwright $(PNPM) exec playwright test

e2e-install: ## download Chromium for Playwright into .gale/ (not the user profile)
	cd web && PLAYWRIGHT_BROWSERS_PATH=$(GALE)/ms-playwright $(PNPM) exec playwright install chromium

clean-caches: ## delete .gale caches (keeps pgdata, models)
	rm -rf $(GALE)/uv-cache $(GALE)/pnpm-cache $(GALE)/tmp/* $(GALE)/pre-commit
