# ADR-010: Synthetic fixture calls and the call simulator

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 2), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** prompts Prompt 2; docs/06 §2 (integration tests with synthetic calls); docs/01 §2 (ingest-gateway); CLAUDE.md hard rules 1–2

## Context
Prompt 2 asks for 10 synthetic stereo 8 kHz calls (2–4 min) with a pluggable TTS, ground truth, and a simulator that replays them to the ingest-gateway as RTP, WebSocket PCM or a file drop. Three things were open:
1. Ten 2–4 minute stereo WAVs are about 50 MB. Committing them bloats the repo for every clone and CI run.
2. The real gateway (Asterisk ARI ExternalMedia, FreeSWITCH fork) arrives in P4. The simulator needs something to talk to now, and something to tell it which RTP stream belongs to which call and channel.
3. With a real TTS the word timings change, so the ground truth can't be written by hand.

## Decision
**Fixtures.**
- `tools/fixtures_gen` composes each scenario (`scenarios.py`) into a timed `script.json` (turns, word spans, voices, render parameters) and derives `expected_events.json` from it.
- Ground truth comes from the script: lexicon placeholders take their category and severity from the fake lexicon; shouted turns get `shouting`; a pronoun register below the speaker's first one gets `disrespect_register`; negative gaps get `interruption_overtalk`. Turn-level `extra` categories are listed explicitly.
- Both JSON files are committed. `call.wav` is git-ignored and rendered on demand. With the stub TTS the render is bit-for-bit reproducible.
- `python -m tools.fixtures_gen check` (run in the test suite) fails if the committed scripts drift from the generator.

**TTS.**
- `StubTTS` makes a speech-like harmonic signal with syllable envelopes. Shouting is +12 dB gain and ×1.35 pitch. Each channel gets −32 dB crosstalk and a −60 dBFS noise floor.
- `LocalVitsTTS` (transformers VITS, local files only) is used when `GALE_TTS_DIR` (or `$GALE_MODEL_DIR/tts`) holds a checkpoint. It renders placeholders with the stub voice because they can't be spoken.
- The default is `--tts auto`. The committed fixtures were made with the stub.

**Simulator ↔ gateway (P0 stub protocol).**
- The simulator registers a call's SSRCs with `POST /v1/rtp/sessions`, streams 20 ms G.711 packets (one UDP socket per channel) to the gateway's single RTP port, and ends the call with `DELETE`. The response carries per-channel stats.
- WebSocket mode sends a JSON `start`, interleaved s16le chunks and a `stop`, and gets the stats back.
- The gateway decodes, resamples to 16 kHz, cuts 200 ms `AudioFrame`s and hands them to a `FrameSink`. In P0 the sink only counts; P4 publishes to `audio.frames`.
- Stats include a CRC-32 of the decoded input PCM. The simulator compares it with its own G.711 round trip, which makes every replay an integrity check.
- File mode drops `<call_id>.wav` plus a `.json` sidecar into `.gale/drop/incoming` atomically (the WAV is renamed in last).

## Consequences
- Positive: the repo stays small; fixtures are reproducible and reviewable as text diffs; `make simulate` works without `make dev-up` (it starts an in-process gateway when none answers).
- Negative / trade-offs:
  - The stub voice is not Bangla speech. Real ASR on these files is meaningless until a local TTS is configured. Prompt 3 stubs key off `script.json`.
  - The gateway decodes RTP in arrival order with no jitter buffer. That's fine on localhost; P4 adds reordering.
- Follow-ups:
  - P4 replaces session registration with ARI / FreeSWITCH call events and the counting sink with a bus publisher.
  - If a VITS checkpoint (e.g. MMS-TTS, CC-BY-NC-4.0) is ever used, pin it and record its licence in `ml/MODEL_LICENSES.md` (CLAUDE.md rule 6).
