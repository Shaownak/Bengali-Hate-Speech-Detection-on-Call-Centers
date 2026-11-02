# Runbook: disaster recovery (docs/05 §8; test quarterly)

Targets: Postgres RPO 15 min (continuous WAL archive, `archive_timeout` 900 s), object store via
MinIO versioning + replication. The bus is transient: in-flight live calls are lost on a full
outage, and recordings can be re-processed post-call.

## Restore Postgres to a point in time (CloudNativePG)
1. Pick the target time (UTC) just before the incident.
2. Create a recovery cluster next to the broken one:
   ```yaml
   apiVersion: postgresql.cnpg.io/v1
   kind: Cluster
   metadata: { name: gale-pg-restore, namespace: gale-data }
   spec:
     instances: 3
     bootstrap:
       recovery:
         source: gale-pg
         recoveryTarget: { targetTime: "2026-11-02 10:15:00+00" }
     externalClusters:
       - name: gale-pg
         barmanObjectStore:   # same destinationPath / endpointURL / credentials as values.yaml
           destinationPath: s3://gale-pg-backups/
           endpointURL: https://s3.gale.internal
           s3Credentials:
             accessKeyId: { name: gale-pg-backup, key: ACCESS_KEY_ID }
             secretAccessKey: { name: gale-pg-backup, key: ACCESS_SECRET_KEY }
   ```
3. Point the app at it: update `GALE_DB_URL` (host `gale-pg-restore-rw`) in the secret store,
   then restart the api. The migration hook is a no-op if the schema is already at head.
4. Verify: `/readyz` is green (`schema: true`); the call explorer shows calls up to the target time.

## Quarterly drill (staging)
Restore last night's backup to a scratch cluster, run `pytest services/api/tests/test_migrations.py`
against it (`GALE_TEST_PG_URL`), time the restore, and record RTO / RPO in the ops log.
