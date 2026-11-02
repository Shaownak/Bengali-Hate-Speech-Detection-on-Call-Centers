# Runbook: annotation operations (for the annotation lead)

How call audio becomes labelled training data (docs/02 §2–3). Commands run on the in-country
data host as `python -m ml.data …`. Everything lives under `$GALE_DATA_DIR/data`; nothing with
audio, transcripts or customer data goes into git.

## 0. One-time setup (with IT / platform)

| Item | Who provides it | Where it goes |
|---|---|---|
| Recording location: folder path, **or** S3/MinIO bucket + prefix, **or** recorder CSV export + WAV folder | Call-centre IT | `--folder`, `--s3 bucket/prefix --endpoint URL`, or `--recorder-csv … --audio-dir …` |
| Recorder CSV column names (file, recording id, caller number, agent id, queue, start time, direction, channel layout) | Recorder admin | `RecorderColumns` in `ml/data/ingest.py` if they differ from the defaults |
| Which stereo channel is the agent (left by default) | Recorder admin | sidecar / CSV `layout` = `stereo_agent_left` or `stereo_agent_right` |
| Pseudonymisation salt (≥ 16 random bytes, never rotated once used) | Security lead | `GALE_PSEUDONYM_SALT_FILE` → secret file; **never** in git or chat |
| S3/MinIO credentials (read-only on recordings) | IT | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` env vars |
| Label Studio host (self-hosted, in-country) + API token of a service user | Platform team | `GALE_LS_URL`, `GALE_LS_TOKEN`; LS local-files storage pointed at `$GALE_DATA_DIR/data/audio` |
| Golden-set call list (300 calls, chosen with the QA lead) | QA lead | a text file of call ids, used once with `split --golden … --lock` |
| DVC remote credentials (MinIO bucket `gale-dvc`) | IT | `dvc remote modify --local minio access_key_id …` (writes the git-ignored `.dvc/config.local`) |

## 1. Ingest a batch

```bash
python -m ml.data ingest --s3 recordings/2026/11 --endpoint https://minio.internal
```

Customer numbers become salted hashes at ingest; call ids become GAL-E UUIDs (D-18). Channels that
are > 80 % silence, > 5 % clipped or < 20 s long are marked `dropped` (kept in the manifest for audit).

## 2. Splits and the golden set (once, before any annotation is used for training)

```bash
python -m ml.data split --golden golden_calls.txt --lock      # writes golden.lock
python -m ml.data golden-check                                # also run by CI on every manifest
```

Calls that share an agent or a customer always land in the same split. The golden set never moves.
Re-locking needs the QA lead's sign-off. Golden calls are annotated (double) but **never** reach a
training set or threshold tuning (CLAUDE.md rule 3); the converters refuse them.

## 3. Seed transcripts and Label Studio projects

```bash
python -m ml.data transcribe                                  # resumable; re-run after interruptions
python -m ml.data ls-push --task T1 --batch 2026-11-a         # transcript correction, per channel
python -m ml.data ls-push --task T2 --batch 2026-11-a         # event spans, pre-annotated by the pipeline
python -m ml.data ls-push --task T3 --batch 2026-11-a         # call-level QA
```

Pre-annotations are suggestions: annotators must check every region, delete wrong ones, and add what
the pipeline missed. 20 % of tasks (and every golden/dev call) are marked for **double annotation**.

## 4. Quality gates (weekly)

```bash
python -m ml.data ls-pull --project-id <T2 id> --out t2.json
python -m ml.data agreement --t2 t2.json
```

`reports/agreement.json`: Krippendorff's α per category (gate ≥ 0.70; shouting ≥ 0.80), span IoU and
match F1. `reports/adjudication.csv`: every unmatched span and severity disagreement. The QA lead fills
the `decision` column; adjudicated labels go back into Label Studio as the final annotation.
Annotators below the gate are retrained (docs/02 §3.3).

## 5. Training sets

```bash
python -m ml.data ls-pull --project-id <T1 id> --out t1.json
python -m ml.data convert --t1 t1.json --t2 t2.json
```

Writes `training/{asr,kws,shout,ser,audio_tox,textclf,llm_sft,fusion}.jsonl`. Corrected text and
notes are redacted again on the way out (annotators sometimes type a number). Version the outputs with
DVC (`dvc add $GALE_DATA_DIR/data/training && dvc push`).

## 6. Annotator wellbeing (abusive content)

- Sessions of at most 90 minutes on T2, with breaks; rotate T2 with T1/T3.
- Anyone may opt out of a call or a batch without giving a reason.
- Weekly check-in; access to counselling as agreed with HR.
- Never share clips outside Label Studio; no downloads to personal devices.

## Augmentation (for the ML team)

`ml/data/augment_audio.py`: G.711 µ-law / A-law in-process; AMR-NB and GSM-FR need `ffmpeg`
(with libopencore-amrnb / libgsm) or `sox` on the training host; packet loss, band-limit, noise at
5–20 dB SNR, speed 0.9/1.0/1.1, and gain (refused on shouting samples). Every call takes a seeded RNG.
