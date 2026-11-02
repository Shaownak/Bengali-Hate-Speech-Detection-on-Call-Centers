# Runbook: key and credential rotation (quarterly, or immediately on suspected exposure)

All credentials live in the secret store (`infra/secrets/README.md`); External Secrets refreshes
Kubernetes Secrets hourly (`externalSecrets.refreshInterval`). Pods read env at start, so finish
with a rollout restart.

| Secret | Rotate how | Restart |
|---|---|---|
| `gale/bus` (Valkey password) | Set the new password in the store. Valkey reads it at start: restart `sts/gale-bus` first (short bus outage; producers retry), then all GAL-E workloads | all |
| `gale/db` | `ALTER ROLE gale PASSWORD …` via CNPG (`kubectl cnpg psql gale-pg -n gale-data`), then update the store (`password` + `GALE_DB_URL`) | api, post-call worker |
| `gale/objectstore` | Create a new access key in MinIO/S3, update the store, restart, then revoke the old key | gateway, ml, api |
| `gale/signing` | New random 32 bytes. Outstanding audio links (TTL 5 min) stop working: rotate off-peak | api |
| `gale/ingest-ari` | Change it in Asterisk `ari.conf` **and** the store at the same time; the gateway reconnects | gateway |
| `gale/pg-backup` | New S3 key for backups; verify the next `ScheduledBackup` succeeds | (CNPG picks it up) |
| TLS certificates | cert-manager renews automatically; check `kubectl get certificate -A` | — |
| Keycloak client secrets / signing keys | Rotate in Keycloak; the API fetches the JWKS, nothing to restart | — |

```bash
for ns in gale-ingest gale-ml gale-app; do
  kubectl -n $ns rollout restart deploy,statefulset -l app.kubernetes.io/part-of=gal-e
done
```
Record the rotation (date, who, which keys) in the ops log. On suspected exposure, also review
`/admin/audit` for unusual transcript/audio access during the exposure window.
