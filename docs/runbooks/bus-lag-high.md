# Runbook: bus lag high (`GaleBusLagHigh`, `GaleIngestFramesDropped`, `GaleDeadLetters`)

**Impact:** live alerts arrive late (the 6 s confirmed-alert SLO is at risk); if the gateway's
queue fills, audio frames are dropped and those seconds are lost to real-time analysis. Post-call
can still recover them, because the gateway keeps the call audio.

## 1. Find the slow consumer
Grafana **GAL-E · Ops** → "Bus lag by group". Or:
```bash
kubectl -n gale-data exec sts/gale-bus -- sh -c 'valkey-cli -a "$REDIS_PASSWORD" --no-auth-warning \
  XINFO GROUPS audio.frames.0'        # pending / lag per group (repeat per stream)
```
Groups are named `<service>` or `<service>-<shard>` (sharded real-time workers, ADR-018).

## 2. Usual causes
| Symptom | Cause | Action |
|---|---|---|
| one `<svc>-<n>` group lags, others fine | that shard's pod is slow/crashlooping | `kubectl -n gale-ml describe pod gale-<svc>-<n>`; check GPU OOM (gpu-oom.md) |
| all shards of asr/audio lag | load above capacity | add shards (below) |
| `llm_judge` lags | vLLM slow (llm-latency.md) | alerts stay provisional; fix vLLM first |
| `postcall_worker` lags | batch backlog | expected at peak: real time pre-empts batch; KEDA scales it |
| DLQ growing | a message fails repeatedly | `XRANGE dlq:<stream> - + COUNT 5` (ids only); fix the bug, then replay |

## 3. Add capacity (real-time workers)
Real-time workers are sharded StatefulSets; their replica count is the shard count (max 8):
```bash
helm upgrade gale infra/helm/gal-e -f values-prod.yaml --reuse-values \
  --set services.asr_worker.shards=6 --set services.audio_worker.shards=6 --atomic
```
Resharding restarts the StatefulSet. Text and fusion rebuild per-call state by replay. ASR and audio
lose at most the utterance or window in progress on calls that move shard (ADR-015, ADR-018).
Do it outside peak if you can.

## 4. If lag > 15 min
The bus keeps about 15 min of frames (MAXLEN). Beyond that, real-time results for the oldest calls
are lost. Re-run those calls through post-call mode from the retained audio
(`POST /api/v1/ingest/recordings`), and note the window in the incident log.
