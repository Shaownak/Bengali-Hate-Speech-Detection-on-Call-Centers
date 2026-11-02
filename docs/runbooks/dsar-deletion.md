# Runbook: DSAR deletion and retention (PDPO 2025; docs/05 §7)

Both operations are **irreversible**. They are audited: `audit_log` actions `dsar.dry_run`,
`dsar.delete`, `retention.dry_run`, `retention.purge`.

## Data-subject erasure request
1. Log the request (ticket id) and confirm the requester's identity per the DPO's procedure.
2. Find the subject's calls: by CRM / PBX call id (`external_call_id`), the customer hash, or
   explicit call ids from the call explorer.
3. **Dry run** (default). As an admin:
   ```bash
   curl -X POST https://gale-api.../api/v1/dsar -H "Authorization: Bearer $TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"external_call_id":"<crm id>","reason":"DSAR <ticket>","dry_run":true}'
   ```
   Check the count of calls / events / segments with the DPO.
4. **Erase:** repeat with `"dry_run": false`. This deletes audio (live and uploaded), reports,
   redacted segments, signals, events, reviews, scorecards and the call rows.
5. **Object versions:** with bucket versioning on, step 4 leaves delete markers. The bucket
   lifecycle rule expires non-current versions within 7 days; confirm with the storage team if
   the DPO needs proof sooner.
6. **Backups:** PITR backups keep the data until their retention (`postgres.cnpg.backup.retentionPolicy`,
   30 d). Tell the requester the backup horizon, as the DPO policy states.
7. Close the ticket with the `audit_log` row id. The audit row stores only hashes of the selectors.

## Retention job
- `CronJob gale-retention` runs daily as a **dry run** until both hold:
  `configs/retention.yaml` has `approved: true` (DPO sign-off of the numbers), and the Helm value
  `retention.execute=true`.
- Read the dry-run counts: `kubectl -n gale-app logs job/<latest gale-retention job>` or the
  `retention.dry_run` audit rows.
- Transcripts and signals are removed by dropping whole monthly partitions (fast; no bloat).
