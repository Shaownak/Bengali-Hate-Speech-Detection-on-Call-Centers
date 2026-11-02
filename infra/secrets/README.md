# Secrets for GAL-E (docs/05 §7)

Secrets never live in Helm values or in git in plain text. Two supported ways:

## 1. External Secrets Operator (default, `externalSecrets.enabled: true`)

The chart renders `ExternalSecret` objects against a `ClusterSecretStore` (default name
`gale-vault`: HashiCorp Vault, or your cloud's secret manager, in-country). Entries the store must hold
(remote key → properties):

| Remote key | Properties | Used by |
|---|---|---|
| `gale/bus` | `GALE_BUS_URL` (`redis://:<pw>@gale-bus.gale-data.svc.cluster.local:6379/0`), `REDIS_PASSWORD` | all services, Valkey, KEDA |
| `gale/db` | `GALE_DB_URL` (`postgresql://gale:<pw>@gale-pg-rw.gale-data.svc.cluster.local:5432/gale`), `username` (`gale`), `password` | api, post-call worker, migrations, CloudNativePG owner |
| `gale/objectstore` | `GALE_S3_ACCESS_KEY`, `GALE_S3_SECRET_KEY` | gateway, workers, api |
| `gale/signing` | `GALE_OBJECT_URL_SECRET` (≥ 32 random bytes) | api (signed audio links) |
| `gale/ingest-ari` | `GALE_INGEST_ARI_PASSWORD` | gateway (Asterisk only) |
| `gale/pg-backup` | `ACCESS_KEY_ID`, `ACCESS_SECRET_KEY` | CloudNativePG backups |

Rotation: see `docs/runbooks/key-rotation.md`.

## 2. SOPS (clusters without a secret store)

Encrypt Secret manifests with an age key held by the ops team, commit only the encrypted files,
and apply them before `helm install` (with `externalSecrets.enabled=false`):

```bash
sops --encrypt --age <ops-age-public-key> secrets.yaml > infra/secrets/prod.enc.yaml   # once
sops --decrypt infra/secrets/prod.enc.yaml | kubectl apply -f -                        # each deploy
```

`secrets.example.yaml` shows the shape with placeholder values; `.sops.yaml` sets which fields
are encrypted. Never commit a decrypted file. `.gitignore` blocks `*.dec.yaml` and `secrets.yaml`.
