# ADR-013: Data engine choices (Prompt 4)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 4), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/02 §1.2, §2, §3; docs/06 §1 (golden set); CLAUDE.md rules 1–3; D-18

## Context
Prompt 4 asks for ingestion, processing, transcription, Label Studio round trips, agreement metrics, augmentation, splits with a golden lock, and DVC/MLflow. It must all run on synthetic fixtures in CI. No recordings, recorder export, Label Studio host, ffmpeg or Docker exist on the build machine.

## Decision
1. **Identities.**
   - `call_id = uuid5(namespace, source|recording_id)`: a GAL-E UUID (D-18) that stays the same if a recording is ingested twice.
   - Customer numbers become `HMAC-SHA256(salt, digits)` at ingest. Leading `+880` is folded to `0`, so formatting doesn't change the hash.
   - The salt comes only from `GALE_PSEUDONYM_SALT[_FILE]`; ingest refuses to run without it.
   - The clear number is never stored or printed.
2. **Quality filters are per channel.** A dropped channel stays in the manifest with a reason. The stereo 16 kHz review copy used for T2/T3 is written only when both channels are kept.
3. **Splits.** Calls connected by a shared agent or customer form one component, and a component goes to one split, stratified by (dialect, queue). Golden pins pull their whole component, and a conflict is an error. With a small agent pool this can starve training; Q-02 was approved as specced.
4. **Golden lock.** `golden.lock` stores sha256 + counts of golden call ids and speaker ids. `check_golden` (CLI, CI) fails on overlap or drift. Every training-set converter refuses golden rows. T1 still transcribes golden calls: they are the ASR test set.
5. **Label Studio.**
   - Templates are generated from the schema enums, so the taxonomy cannot drift.
   - Audio comes through LS local-files storage.
   - T2/T3 use a stereo `splitchannels` view (left agent, right customer).
   - Pre-annotations come from the same post-call pipeline as production.
   - 20 % of tasks are marked for double annotation, chosen deterministically per call.
   - Text that annotators type (corrections, notes) is redacted again on export.
6. **Agreement.** Krippendorff's α (nominal) per category over 1 s bins per channel. Span agreement uses one-to-one IoU matching with the docs/06 threshold (0.3).
7. **Augmentation.** µ-law/A-law run in-process. AMR-NB and GSM-FR go through ffmpeg/sox when installed; their tests skip otherwise. Gain is refused on shouting samples.
8. **Infra.**
   - `.dvc/config` is written by hand (DVC isn't installed here) with a MinIO remote; credentials go only in `config.local`.
   - MLflow is added to `docker-compose.dev.yml` (Postgres backend, MinIO artifacts). Locally it isn't run (ADR-009).
9. **Register.** The docs/02 §1.1 speech corpora (S1–S8) are added to `ml/data/datasets.yaml` with licence "TO VERIFY". They have no loaders until their P3 recipes.

## Consequences
- Positive:
  - The whole flow (ingest → split → transcribe → LS tasks → simulated export → training sets → agreement) is tested end to end on the synthetic calls.
  - Recorder differences become config, not code.
- Negative / not verified:
  - Real recorder formats, a live Label Studio instance (API differences between versions), S3 against real MinIO, ffmpeg codecs, and DVC/MLflow are all untested here.
  - Annotation volumes and agreement numbers need people.
