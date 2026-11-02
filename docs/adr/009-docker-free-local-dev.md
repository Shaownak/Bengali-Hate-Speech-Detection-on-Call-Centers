# ADR-009: Docker-free local development (venv + local processes)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Shaownak Shahriar (project owner)
- **Spec refs:** CLAUDE.md "Common commands" (`make dev-up`), docs/01 §2, docs/05 §1, ADR-008

## Context
The developer won't install Docker Desktop for now. On 2026-11-02 the C: drive had 0.6 GB free and G: had 86.8 GB. CLAUDE.md defines `make dev-up` as a docker-compose stack (Postgres, Redis, MinIO, Keycloak, services, web). Staging and prod stay Kubernetes and are unaffected.

## Decision
The local dev stack runs from the uv-managed virtualenv (`.venv` in the repo on G:) with no containers. Each infrastructure dependency sits behind an interface whose dev backend is a Python package or a local folder:

| Need | Prod/staging | Local dev (`GALE_ENV=dev`) |
|---|---|---|
| Bus (`BusClient`) | Redis Streams | `fakeredis` TCP server started by `make dev-up` (same redis-py client code); in-process `memory` bus for unit tests and `make report` |
| Database | PostgreSQL 16 | Embedded Postgres from pip (`pgserver`), data dir under `.gale/` so Postgres features (arrays, partitions, `pg_trgm`) still work. SQLite is **not** used. |
| Object store (`ObjectStore`) | MinIO/S3, SSE-KMS | Local folder `.gale/objects/` with signed-URL emulation |
| Auth | Keycloak / OIDC | `GALE_AUTH_MODE=dev`: local JWT issuer with demo users for the 5 roles. Startup refuses it unless `GALE_ENV=dev`. |
| Process supervision | Kubernetes | `make dev-up` / `dev-down` run services + web via a small Python supervisor (`tools/devstack`), PID files in `.gale/` |

**Disk placement:** everything goes on the repo drive, never C:
- `uv.toml` `cache-dir = ".gale/uv-cache"`
- `.npmrc` `store-dir=.gale/pnpm-store`
- Makefile exports `TMP`/`TEMP`, `HF_HOME` and `GALE_MODEL_DIR` under `.gale/`
- `.gale/` is git-ignored.

**Containers stay written but optional.** `infra/compose/docker-compose.dev.yml` and Dockerfiles are still authored and exercised in Linux CI. They become the local path again when Docker is installed (`make dev-up DEV_STACK=compose`).

## Consequences
- Positive: no Docker and no C: usage. `make dev-up` works on this laptop. Code paths are the same, only backends differ by config.
- Negative: fakeredis and embedded Postgres differ from production servers in edge cases (performance, persistence, some commands). The Linux CI compose job is the parity check. Real GPU models still can't run locally (ADR-008).
- Follow-ups: in Prompt 1, verify that `pgserver` runs on Windows and that fakeredis supports XADD/XREADGROUP/XACK/XCLAIM. If either fails, record the replacement here.
