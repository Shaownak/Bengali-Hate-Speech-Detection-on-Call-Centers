<!-- laptop dev stack (make dev-up, stub models, single instance per worker), 2026-11-02. Measured BEFORE the llm_event split: 'provisional_event' p95 at ≥ 50 % includes LLM-only events (see PROGRESS, Prompt 14 findings). -->
# GAL-E capacity report

Generated 2026-11-02 17:30 · peak = 4 concurrent calls · speed 1.0× · models: **stub** · target: `127.0.0.1`

> Stub-model runs exercise the pipeline (bus, workers, fusion, persistence) only. Fill the GPU
> rows below from a staging run with real models before sizing production (docs/05 §3).

## Results per level

| level | calls | call-min | errors | frames dropped | dead letters | max bus lag (s) | pass |
|---|---|---|---|---|---|---|---|
| 25% | 1 | 2.78 | 0 | 0 | 0 | 0.0 | ✅ |
| 50% | 2 | 5.13 | 0 | 0 | 0 | 0.01 | ❌ |
| 100% | 4 | 10.38 | 0 | 0 | 0 | 0.06 | ❌ |
| 200% | 8 | 20.77 | 0 | 0 | 0 | 0.09 | ❌ |

## p95 latency per stage (s)

| stage | 25% | 50% | 100% | 200% | budget |
|---|---|---|---|---|---|
| api/event_persisted | 2.4 | 8.2 | 7.4 | 7.4 |  |
| asr_worker/asr_final | 1.442 | 1.414 | 1.411 | 1.42 |  |
| audio_worker/dsp_shout | 0.231 | 0.235 | 0.242 | 0.238 |  |
| audio_worker/overtalk | 9.8 | 9.8 | 9.2 | 9.2 |  |
| fusion_engine/alert | 0.95 | 5.6 | 5.3 | 5.3 | 8.0 |
| fusion_engine/confirmed_event | 2.85 | 8.8 | 8.4 | 8.4 | 6.0 |
| fusion_engine/provisional_event | 0.925 | 8.8 | 8.0 | 8.0 | 2.0 |
| ingest_gateway/frame_on_bus | 0.193 | 0.203 | 0.19 | 0.196 |  |
| llm_judge/llm_decision | 2.85 | 8.8 | 8.0 | 8.0 |  |
| text_worker/lexicon_partial | 2.85 | 2.85 | 2.85 | 2.85 |  |
| text_worker/text_signal | 2.9 | 5.65 | 4.7 | 5.35 |  |

## Sizing (docs/05 §3)

```
channels = 2 × peak_concurrent_calls = 2 × 4 = 8
GPU_ASR  = channels × RTF_asr / streams_per_gpu_at_latency_target  = TBD (measure RTF_asr on L4)
GPU_SER  = channels / windows_per_sec_capacity                    = TBD
LLM_rps  = active_calls × (utterances_per_min / 60) × 0.3         = TBD (utterances/min from staging)
GPU_LLM  = LLM_rps / measured_rps_per_gpu_at_p95≤2.5s             = TBD
cost per 1,000 call-minutes                                       = TBD
```

Worker shards (Helm `services.*.shards`): set so the 200 % level passes with bus lag < 10 s.
