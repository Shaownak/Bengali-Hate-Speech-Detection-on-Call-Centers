# One image per service: docker build -f infra/docker/service.Dockerfile --build-arg SERVICE=api .
# Multi-stage, non-root, slim (docs/05 §2). GPU images for real models come in P3/P4.
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS build
ARG SERVICE
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services ./services
RUN uv sync --locked --no-dev --no-default-groups --package gale-${SERVICE//_/-} --no-editable

FROM python:${PYTHON_VERSION}-slim-bookworm
ARG SERVICE
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" GALE_HOST=0.0.0.0 GALE_PORT=8000 SERVICE=${SERVICE}
RUN useradd --system --uid 10001 --no-create-home gale
WORKDIR /app
COPY --from=build --chown=gale:gale /app/.venv /app/.venv
COPY --chown=gale:gale configs /app/configs
ENV GALE_CONFIG_DIR=/app/configs
USER 10001
EXPOSE 8000
CMD ["sh", "-c", "exec python -m gale_${SERVICE}"]
