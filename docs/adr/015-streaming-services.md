# ADR-015: Streaming services on Redis Streams (Prompt 11)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 11), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/01 §1–3, §5; docs/08 §1; D-01, D-02, D-11, D-18

## Context
Prompt 11 turns the post-call pipeline into streaming services. The spec fixes the streams and the latency budget, but leaves open how workers keep per-call state, recover from crashes, prioritise real time over batch, and how the gateway learns which RTP stream belongs to which speaker.

## Decision
1. **One worker per service**, a `galecore.streaming.Worker` subclass running inside the service's FastAPI lifespan (`GALE_WORKERS=false` turns it off).
   - It reuses `BusClient` consumer groups (at-least-once, XAUTOCLAIM retries, DLQ, idempotency keys) and the exact step code of the post-call pipeline.
   - New streams: `llm.decisions`, `call.scores`, `postcall.jobs`. New messages: `LlmDecision`, `CallScoreMessage`, `PostcallJob`.
   - `CallLifecycle` gains `external_ref` and `wall_clock`.
2. **Gateway.**
   - Publishes `call.lifecycle` (started/ended) and frames through a bounded queue. When the queue is full it drops and counts frames rather than block the receive path.
   - Keeps each call's 16 kHz audio and writes it to the object store at hang-up (SSE-KMS on MinIO in prod).
   - RTP streams are keyed by SSRC (simulator) or by source address (Asterisk ExternalMedia).
   - The ARI app snoops the agent channel (`spy=in` agent, `spy=out` customer). The agent leg is marked in the dialplan (`__GALE_ROLE=agent`).
   - FreeSWITCH `mod_audio_fork` connects to `/v1/fork` with a metadata frame.
   - File mode: a watched folder becomes `postcall.jobs`.
3. **ASR streaming policy.**
   - Energy endpointing on 200 ms frames (12 dB over a running noise floor, at least −45 dBFS so crosstalk stays out), with one frame of pre-roll.
   - A partial segment every 1 s; a final segment after 0.6 s of silence.
   - This endpoint-then-decode policy works for any `ASRBackend`. Whisper SimulStreaming and NeMo cache-aware streaming slot in with the real backends.
4. **Fast path on partials.** The text worker runs the **lexicon only** on partial segments. Lexicon signal ids depend on (segment, char offset), so the final derives the same ids and nothing is published twice. Classifier and register wait for the final segment.
5. **Audio worker.** A 20 s ring buffer per channel. The newest 1.5 s shouting window is scored every 0.5 s against the speaker's baseline, using a new `DSPShoutDetector.score_window`; the baseline needs the first 20 s of that speaker's speech. SER/audio-tox/KWS run on 3 s windows. Overtalk runs on both rings every 2 s, and each onset is published once.
6. **LLM judge.**
   - At most one window request per call every 3 s.
   - Provisional events that arrive meanwhile are judged together.
   - If the judge is offline, events stay provisional and are retried later.
   - At hang-up, the whole-call scorecard.
7. **Fusion.**
   - Re-fuses the call's full signal set (same `FusionEngine`, real-time thresholds) about 100 ms after anything new arrives.
   - Publishes only changed events; alerts go to their own stream.
   - Provisional alerts are retracted by later `events.fused` updates (e.g. `dismissed_by_llm`); the UI must follow `events.fused`, not only `alerts`.
8. **Crash recovery.**
   - Unacked messages are reclaimed by any consumer in the group (XAUTOCLAIM).
   - A worker meeting an unknown call first replays that call's lifecycle; text and fusion also replay its earlier segments / signals / decisions, so register baselines and event state survive a restart.
   - Limit: ASR utterance buffers and audio rings are in memory, so a crash there loses at most the utterance or window in progress. The gateway keeps the call audio, so post-call mode can recover it.
9. **Priorities.** The post-call worker (`python -m gale_postcall worker`, port 8007) takes no job while the ASR/audio groups have more than `backlog_limit` frames pending or lagging.
10. **Latency.**
    - `gale_stage_latency_seconds{service,stage}` = wall clock minus (call start + the evidence's media time).
    - Lexicon hits are measured from the word; utterance-level steps from the end of the utterance, matching the docs/01 §5 rows "ASR final 1.2 s after end of utterance" and "100 ms after segment".
    - OTel spans carry `gale.call_id`; the trace id is derived from the call UUID.

## Consequences
- Positive:
  - Every fixture gives the same events streaming and post-call.
  - The real-time test meets the docs/01 §5 budget (provisional ≤ 2 s, confirmed ≤ 6 s).
  - Crash/restart of text and fusion mid-call loses nothing.
  - The live dev stack replays calls over RTP end to end.
- Negative / not verified:
  - No real Asterisk or FreeSWITCH here; the ARI app is tested against a fake ARI.
  - Latency was measured with stub models on a laptop; real ASR/SER on GPUs will move the numbers.
  - Fusion re-fuses the whole call on each update. That is fine for minutes-long calls; very long calls may need incremental fusion.
