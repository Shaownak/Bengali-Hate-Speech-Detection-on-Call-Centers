#!/usr/bin/env bash
# CI-only: replay a synthetic fixture call through the deployed cluster (WebSocket ingest via
# port-forward) and assert the API persisted it with events. Stub models, synthetic audio.
set -euo pipefail
kubectl -n gale-app rollout status deploy/gale-api --timeout=300s
for sts in gale-asr-worker gale-audio-worker gale-text-worker gale-llm-judge gale-fusion-engine; do
  kubectl -n gale-ml rollout status statefulset/$sts --timeout=300s
done
kubectl -n gale-ingest port-forward svc/gale-ingest-gateway 8001:8000 >/tmp/pf-gw.log 2>&1 &
kubectl -n gale-app port-forward svc/gale-api 8000:8000 >/tmp/pf-api.log 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
for _ in $(seq 30); do curl -fs localhost:8000/readyz >/dev/null && curl -fs localhost:8001/healthz >/dev/null && break; sleep 2; done
curl -fs localhost:8000/readyz
CALL="$(uv run python -c 'import uuid; print(uuid.uuid4())')"
uv run python -m tools.simulator tests/fixtures/calls/demo_01 --mode ws --speed 20 \
  --gateway http://127.0.0.1:8001 --spawn never --call-id "$CALL" --agent AG-001 --queue billing
TOKEN="$(curl -fs -X POST localhost:8000/api/v1/dev/token -H 'Content-Type: application/json' \
  -d '{"sub":"ci-admin","role":"admin"}' | uv run python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
for _ in $(seq 60); do
  N="$(curl -fs -H "Authorization: Bearer $TOKEN" "localhost:8000/api/v1/calls/$CALL/events" \
    | uv run python -c 'import json,sys; print(len(json.load(sys.stdin)))' 2>/dev/null || echo 0)"
  [ "${N:-0}" -gt 0 ] && break
  sleep 3
done
echo "events persisted for $CALL: ${N:-0}"
test "${N:-0}" -gt 0
