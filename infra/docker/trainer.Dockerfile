# Retraining / evaluation image (ADR-020): the workspace packages the post-call pipeline needs for
# `ml.eval`, plus `ml/` and `configs/`. Runs the steps of `python -m ml.retrain` in the Argo
# workflow (infra/helm/gal-e/templates/retrain.yaml). Weights come from the models PVC, never baked.
#
#   docker build -f infra/docker/trainer.Dockerfile .                               # CPU steps
#   docker build -f infra/docker/trainer.Dockerfile --build-arg TRAIN_GROUPS="--group ml-train" .
#
# The GPU variant (ml-train: torch + transformers) is built on the GPU build host; CI builds the
# CPU variant to scan it.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build
ARG TRAIN_GROUPS=""
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/local/bin/python3
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services ./services
# gale-postcall pulls in the worker packages ml.eval runs; gale-api for the feedback schema checks
# shellcheck disable=SC2086
RUN uv sync --locked --no-dev --no-default-groups --group ml-data ${TRAIN_GROUPS} \
      --package gale-postcall --no-editable \
 && uv pip install --python /app/.venv/bin/python --no-deps ./services/api

FROM python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
LABEL org.opencontainers.image.source="https://github.com/Shaownak/Bengali-Hate-Speech-Detection-on-Call-Centers" \
      org.opencontainers.image.title="gale-trainer"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app \
    GALE_CONFIG_DIR=/app/configs HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin gale
WORKDIR /app
COPY --from=build --chown=root:root /app/.venv /app/.venv
COPY --chown=root:root configs /app/configs
COPY --chown=root:root ml /app/ml
USER 10001
CMD ["python", "-m", "ml.retrain", "--help"]
