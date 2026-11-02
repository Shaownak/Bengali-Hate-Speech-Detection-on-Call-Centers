# One image per Python service (docs/05 §2): multi-stage, non-root, slim, base images pinned by
# digest (bump with the Renovate/Dependabot PR that updates both tag and digest).
#
#   docker build -f infra/docker/service.Dockerfile --build-arg SERVICE=api .
#   docker build -f infra/docker/service.Dockerfile --build-arg SERVICE=postcall_worker \
#       --build-arg PACKAGE=gale-postcall --build-arg RUN_CMD="python -m gale_postcall worker" .
#
# Real-model images (GPU) add the ml extras on top of this in P3; weights are never baked in:
# they come from the internal model store (GALE_MODEL_DIR volume), docs/05 §7.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build
ARG SERVICE
ARG PACKAGE=""
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/local/bin/python3
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services ./services
RUN pkg="${PACKAGE:-gale-${SERVICE//_/-}}" \
 && uv sync --locked --no-dev --no-default-groups --package "$pkg" --no-editable

FROM python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
ARG SERVICE
ARG RUN_CMD=""
LABEL org.opencontainers.image.source="https://github.com/Shaownak/Bengali-Hate-Speech-Detection-on-Call-Centers" \
      org.opencontainers.image.title="gale-${SERVICE}"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" \
    GALE_HOST=0.0.0.0 GALE_PORT=8000 GALE_CONFIG_DIR=/app/configs \
    SERVICE=${SERVICE} GALE_CMD=${RUN_CMD}
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin gale
WORKDIR /app
COPY --from=build --chown=root:root /app/.venv /app/.venv
COPY --chown=root:root configs /app/configs
# code and config are root-owned and read-only for the runtime user; the chart adds a /tmp emptyDir
USER 10001
EXPOSE 8000
CMD ["sh", "-c", "exec ${GALE_CMD:-python -m gale_${SERVICE}}"]
