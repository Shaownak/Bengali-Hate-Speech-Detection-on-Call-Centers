# Runbook: GPU out of memory (`GaleGpuOOM`, `GaleGpuMemoryNearFull`)

**Impact:** an ASR/SER/KWS shard or vLLM restarts. Its calls lag until it is back (bus-lag-high.md).
LLM OOM leaves alerts provisional (fast path only).

1. Which pod and GPU: `kubectl -n gale-ml get pods -o wide | grep -v Running`; Grafana Ops → "GPU memory used".
2. **Cause checklist**
   - New model version just promoted (larger, or a longer max length)? → model-rollback.md.
   - Two GPU workloads on one card (MIG/time-slicing misconfigured)? Check the `nvidia.com/gpu`
     requests in `kubectl describe node`.
   - vLLM: `--gpu-memory-utilization` too high, or `--max-model-len` raised. Lower it in
     `vllm.extraArgs` and `helm upgrade`.
   - ASR batch/stream count per GPU above what the capacity report measured.
3. **Mitigate:** add shards (spreads calls over more GPUs), or move the noisy workload to its own
   node pool (`gpuPools`).
4. **After:** record peak memory per model in the capacity report (`make loadtest` on staging).
