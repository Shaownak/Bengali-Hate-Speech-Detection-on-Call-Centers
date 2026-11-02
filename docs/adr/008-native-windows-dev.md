# ADR-008: Native Windows development environment

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Shaownak Shahriar (project owner)
- **Spec refs:** CLAUDE.md "Common commands", docs/05 §1 (`dev` env), PROGRESS Q-06

## Context
The developer works on Windows 11 and chose to stay native rather than use WSL2. CLAUDE.md requires `make` targets, uv, pnpm and a docker-compose dev stack. Prod and staging are Linux (Kubernetes). Some GPU/audio dependencies (NeMo, vLLM, pyworld) don't build or run reliably on native Windows.

## Decision
- **Host tooling:** GNU Make ≥ 4.x, uv, pnpm (via corepack), Docker Desktop (Linux containers), and Git for Windows. The Makefile sets `SHELL := bash` and uses only POSIX commands available in Git Bash. No `cmd`/PowerShell-specific recipes.
- **Line endings:** `.gitattributes` forces `eol=lf` for all text files, so shell scripts and YAML behave the same in containers and CI.
- **What runs natively:** `galecore`, service logic in stub mode, tests, lint/type checks and the web app. Paths go through `pathlib`; no hard-coded `/`. No POSIX-only stdlib (`fcntl`, `signal.SIGKILL`) without a guard.
- **What runs in containers only:** Postgres, Redis, MinIO and Keycloak, plus all `GALE_MODEL_MODE=real` GPU stacks (NeMo, vLLM, FunASR, Unsloth). These are Linux-only extras (uv environment markers `sys_platform == 'linux'`) and never needed for `make check`.
- **CI:** Linux is the primary job. A `windows-latest` job runs `make check` in stub mode so the Windows workflow doesn't rot.

## Consequences
- Positive: the developer keeps their environment. Stub-mode code stays portable.
- Negative: real-model work (P1 real backends, P3 training) happens in Linux containers or on the GPU server, not on the host. Docker Desktop's file-sharing makes bind-mounted dev containers slower than native Linux. Line-ending and path bugs need the Windows CI job to catch them.
- Follow-ups: the Windows toolchain install is a human step, listed in PROGRESS "Next step".
