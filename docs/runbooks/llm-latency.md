# Runbook: LLM latency / LLM offline (`GaleAlertLatencyHigh`, `GaleConfirmedLatencySLO`, `GaleLLMOffline`)

**Impact (by design, ADR-015):** if the judge is slow or down, events stay **provisional** and the
UI shows the fast-path badge. Nothing is lost: decisions are retried after the debounce, and the
call scorecard is produced at hang-up when the judge is back.

1. Grafana Ops → "LLM judge latency p95" and "Per-stage latency p95" (`llm_decision`, `confirmed_event`).
2. vLLM health: `kubectl -n gale-ml logs deploy/gale-vllm --tail=100`; `curl gale-vllm:8000/health` from a pod.
3. Circuit breaker: `gale_llm_judge_calls_total{outcome="circuit_open"}` rising means the client
   stopped calling to protect vLLM. It re-probes automatically.
4. **Common causes:** GPU OOM (gpu-oom.md); queue too deep at peak (raise vLLM replicas, or the
   `llm_judge` debounce in config); a new adapter with longer outputs (model-rollback.md).
5. **Never** route call data to an external LLM API as a workaround (CLAUDE.md rule 1, ADR-003).
