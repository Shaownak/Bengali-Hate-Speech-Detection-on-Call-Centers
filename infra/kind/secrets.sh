#!/usr/bin/env bash
# CI-only: namespaces and throwaway secrets for the kind smoke deploy (values-kind.yaml).
# Real environments get secrets from External Secrets or SOPS (infra/secrets/README.md).
set -euo pipefail
PW="$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)"
SIGN="$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40)"
BUS_URL="redis://:${PW}@gale-bus.gale-data.svc.cluster.local:6379/0"
DB_URL="postgresql://gale:${PW}@gale-pg-rw.gale-data.svc.cluster.local:5432/gale"
for ns in gale-ingest gale-ml gale-app gale-data gale-obs; do
  kubectl create namespace "$ns"
  kubectl label namespace "$ns" app.kubernetes.io/part-of=gal-e pod-security.kubernetes.io/enforce=restricted
done
for ns in gale-ingest gale-ml gale-app gale-data; do
  kubectl -n "$ns" create secret generic gale-bus \
    --from-literal=GALE_BUS_URL="$BUS_URL" --from-literal=REDIS_PASSWORD="$PW"
done
for ns in gale-ml gale-app; do
  kubectl -n "$ns" create secret generic gale-db --from-literal=GALE_DB_URL="$DB_URL"
done
kubectl -n gale-data create secret generic gale-pg-owner --type=kubernetes.io/basic-auth \
  --from-literal=username=gale --from-literal=password="$PW"
kubectl -n gale-app create secret generic gale-signing --from-literal=GALE_OBJECT_URL_SECRET="$SIGN"
# synthetic fixture scripts for the stub models (values-kind.yaml devFixtures); no audio, no real data
kubectl -n gale-ml create configmap gale-dev-fixtures \
  --from-file=demo_01.script.json=tests/fixtures/calls/demo_01/script.json \
  --from-file=demo_01.expected_events.json=tests/fixtures/calls/demo_01/expected_events.json
