# 02 — Data Plan & Annotation Guide

> Data, not model choice, will decide whether GAL-E works. Public Bangla data is mostly read speech and social-media text; our target is spontaneous 8 kHz phone speech with dialects and code-mixing. Budget most effort here.

Legend: **[HUMAN]** = cannot be done by Claude Code (needs people, access, or sign-off). **[CODE]** = tooling Claude Code builds.

---

## 1. Data sources

### 1.1 Public data (for pre-training / fine-tuning / auxiliary eval)

> **Open hate-speech / abuse datasets (22 sets: 15 Bangla-script text, 3 Banglish, 4 audio, plus a transliteration corpus) are catalogued in detail — sizes, labels, links, licence status, label mapping, harmonisation and per-model usage — in `09_OPEN_DATASETS.md`.** They are the primary training source for the text classifier, lexicon seed, KWS and LLM-judge baseline until in-house labels exist. The table below is a summary.

| Use | Dataset | Size | License / access* | How we use it |
|---|---|---|---|---|
| ASR | OOD-Speech (Bengali.AI) | ~1,178 h train + 23 h OOD test | check | ASR fine-tune (downsampled to 8 kHz + codec sim) |
| ASR | IndicVoices (AI4Bharat) Bengali | ~600 h | check | ASR fine-tune; spontaneous speech |
| ASR | OpenSLR-53 Large Bengali | ~211 h | open | ASR fine-tune |
| ASR | Common Voice Bengali | 400 h+ | CC0 | ASR fine-tune |
| ASR (dialect) | Ben-10 | 78 h, 10 dialects | check | dialect fine-tune + dialect eval |
| ASR (domain) | Bengali telecom customer-care speech (BUET, 2026) | 10k utts / 26.8 h, synthetic | CC-BY-4.0 | domain vocabulary; never sole source |
| KWS | BAAD | 114 slang words, 5,277 clips | CC BY 4.0 | KWS training + eval |
| Audio abuse | ADIMA (Bengali split) | ~1,200 clips | research | audio-tox head; eval |
| Audio abuse | MuTox (Bengali split) | — | check | audio-tox eval/aux training |
| Audio abuse | Bangla Hate Speech & Non-Hate Audio (2025) | 3,000 clips | check | audio-tox, severity |
| SER | SUBESCO | 7,000 utts, 7 emotions | CC-BY-4.0 | SER head fine-tune |
| SER | BanglaSER, KBES, BanSpEmo | small | check | SER augmentation |
| Text abuse | BLP-2025 Task 1 / BanglaMultiHate | ~50k | shared-task terms | text classifier |
| Text abuse | BD-SHS | 50,200+ | open (GitHub) | text classifier |
| Text abuse (dialect) | BIDWESH | 9,183 × 3 dialects | check | dialect robustness |
| Text abuse | Bangla Multilabel Cyberbully/Sexual-Harassment/Threat | 12,557 | check | threat & sexual-harassment heads |
| Text abuse | Bangla-ToCo | — | check | context-aware toxicity |
| Text abuse (context) | BanglaCAT (2026) | 19.5k train / 2.3k test | check | context-aware: sarcasm, veiled threats, banter → LLM judge |
| Text aggression | TRAC-2 Bengali | 5,971 | check | overt/covert aggression aux head |
| Text violence | Vio-Lens (BLP-2023) | ~6k | check | threats |
| Banglish | BanTH, Code-mixed Chaos | 37.3k / 10.2k | check | code-mix robustness (back-transliterated) |
| …more | see `09_OPEN_DATASETS.md` | | | |

\* **[HUMAN] Licence review is mandatory before commercial use.** Some research datasets prohibit commercial use. Keep a `ml/data/DATASETS.md` register: name, version/DOI, licence, permitted use, owner sign-off.

### 1.2 In-house call data (the most important source) — [HUMAN] access, [CODE] tooling
- Pull **historical stereo recordings + CDR metadata** (queue, duration, agent id, date). Target corpus for phase 1: **3,000–5,000 calls** (≈ 250–400 h) sampled across queues, shifts, agents, regions, complaint vs non-complaint, escalated vs normal.
- **Oversample harshness**: use QA-flagged calls, complaint-coded calls, supervisor escalations, long calls, and calls where existing QA scores were low. Without oversampling, positives are too rare (< 2–5% of calls).
- Store in-country, encrypted (SSE-KMS on MinIO/S3). Access via short-lived signed URLs only.
- **Pseudonymise** `customer_number` → salted hash. Agent ids kept (needed for coaching), access-controlled.

---

## 2. Data processing pipeline [CODE]

1. **Ingest** → manifest (`ml/data/manifests/*.jsonl`: `audio_path, call_id, channel, duration, sample_rate, codec, queue, date, split`).
2. **Channel split & resample**: keep native 8 kHz copy; create 16 kHz model copy (`soxr` HQ).
3. **Quality filters**: drop channel if > 80% silence, clipping ratio > 5%, or duration < 20 s.
4. **Auto-transcribe** with best baseline ASR → seed transcripts for human correction.
5. **Text normalisation** (`galecore.bangla`): NFC; unify ক্ষ/য়/ড়/ঢ় variants; strip ZWJ/ZWNJ inconsistencies; Bangla/English digit normalisation; number-to-words (configurable); common English loanword canonical forms (রিচার্জ, সিম, ওটিপি…). **The same normaliser must be used for training targets, inference and WER scoring.**
6. **PII redaction** (regex for BD mobile numbers `(\+?880|0)1[3-9]\d{8}` incl. Bangla digits and spoken-digit sequences, NID 10/13/17-digit patterns, bKash/Nagad account mentions, emails; NER for person names/addresses) → `<PHONE>`, `<NID>`, `<NAME>`, `<ADDR>`, `<ACCT>` tags.
7. **Splits by call AND by agent/customer** (no speaker leakage). Stratify by dialect region and queue.
8. **Augmentation for audio models**: G.711 µ-law/A-law, AMR-NB, GSM-FR codec simulation; packet loss (1–5%); band-limiting; office/street noise (MUSAN-style) at 5–20 dB SNR; speed perturbation 0.9/1.0/1.1; gain ±10 dB (careful: gain augmentation must not destroy shouting labels — apply only to non-shout samples or apply consistently to the speaker baseline).

---

## 3. Annotation programme

### 3.1 Tooling [CODE]
**Label Studio** (self-hosted, in-country) with three project templates generated from `ml/data/labelstudio/`:
- **T1 Transcript correction** — audio + editable ASR seed text, per channel, with a "contains slang" quick flag.
- **T2 Event spans** — waveform with time-range regions on each channel; labels = taxonomy below; attributes = severity (1–3), target (agent/customer/company/third party), confidence (sure/unsure), free-text note.
- **T3 Call-level QA** — overall agent professionalism (1–5), customer hostility (1–5), was the call escalated by agent behaviour (y/n), de-escalation quality (1–5).

Import/export scripts convert Label Studio JSON ↔ our manifests and ↔ training sets for every model.

### 3.2 Volumes (phase 1 target) [HUMAN]

| Task | Volume | Purpose |
|---|---|---|
| T1 transcript correction | 80–120 h of channel audio (≈ 1,000 calls), dialect-stratified | ASR fine-tune + golden ASR test |
| T2 event spans | 2,000 calls (all positives-oversampled) | all detectors + fusion |
| T3 call QA | same 2,000 calls | call-level model + business KPI alignment |
| Golden test set | **300 calls, never used for training or threshold tuning**, double-annotated | final acceptance |
| Calibration/dev set | 300 calls, double-annotated | thresholds, calibration, stacker |
| Slang clip recording | 50–100 volunteers × lexicon terms read in natural sentences, on phones | KWS |

### 3.3 Annotators [HUMAN]
- 6–10 native Bangla annotators covering major dialect regions (Dhaka, Chattogram, Sylhet, Barishal, Rajshahi/Rangpur, Khulna, Noakhali), plus 1–2 QA leads from the call centre.
- Training: 1-day workshop on this guide, 50-call calibration round, then weekly calibration meetings.
- Wellbeing: abusive content exposure → rotation, opt-out, session limits.
- **Quality gates**: 20% double-annotation; Krippendorff's α ≥ 0.70 for event category, ≥ 0.80 for "shouting"; disagreements adjudicated by QA lead; annotators below gate are retrained.

### 3.4 Annotation guide (summary — full version in Label Studio instructions)

**Rule 0 — Label the speaker, not the call.** Every span belongs to exactly one channel.

| Category | Definition | Include | Exclude |
|---|---|---|---|
| `slang_profanity` | Gali, swear words, vulgar slang, in any dialect or Banglish | Mild-to-severe gali, euphemistic or partially-said gali (mark `partial=true`) | Neutral colloquialisms; quoted reports ("he called me X") → label but set `reported_speech=true` |
| `insult` | Attacks a person's intelligence, worth, family, appearance | "আপনি কি পাগল?" style personal attacks | Complaints about service quality |
| `threat` | Threatens harm, legal or physical, doxxing, "I'll see you outside" | Physical, reputational, legal intimidation | Legitimate "I will file a complaint with BTRC" (label `threat` only if intimidating, else none) |
| `sexual_harassment` | Sexual remarks, propositions, innuendo | Toward agent or customer | — |
| `hate_identity` | Derogatory remarks on religion, gender, region, ethnicity, disability | Regional slurs too | — |
| `disrespect_register` | Speaker switches to তুমি/তুই (or verb forms) toward a stranger in a disrespectful way | Agent→customer especially; mark the switch point | Customer who uses তুমি consistently and neutrally (dialect/habit) — set `neutral_register=true` |
| `harsh_tone` | Angry, aggressive, contemptuous delivery even if words are polite | Raised, clipped, hostile prosody | Merely loud line/handset |
| `shouting` | Voice clearly raised well above the speaker's baseline | — | Loud background, bad gain |
| `sarcasm_mockery` | Mocking, sarcastic put-downs | — | Friendly humour |
| `dismissive_rude` | Rude without profanity: "আমার কিছু করার নাই, রাখেন", hanging up on customer, refusing help rudely | — | Firm but polite policy statements |
| `interruption_overtalk` | Repeatedly cutting the other speaker off (≥ 3 times in 60 s) | Agent behaviour mainly | Single accidental overlap |

**Severity**: 1 = mild/ambiguous (would a QA lead mention it?), 2 = clear violation (coaching required), 3 = severe (immediate supervisor attention: threats, sexual harassment, severe gali by agent).

**Context flags**: `banter` (friendly, both parties fine), `reported_speech`, `self_directed` ("ধুর, আমার কপাল খারাপ"), `partial`, `unsure`.

---

## 4. Slang lexicon [HUMAN content, CODE tooling]
- Seed from BAAD's 114 terms + annotator contributions + T2 spans (auto-extracted candidate terms reviewed weekly).
- Each entry: canonical form, spelling variants, romanised/Banglish forms, dialect, category, default severity, `context_sensitive` (words that are only abusive in some contexts), example sentences.
- Stored in Postgres (`lexicon_entries`) and managed via the admin UI; versioned; **real lexicon is not committed to git** (stored as an encrypted seed file loaded at deploy time). Tests use a fake lexicon.
- Matching: normalised exact match, edit-distance ≤ 1 for terms ≥ 5 chars, phonetic key (Bangla-aware: merge শ/ষ/স, জ/য, ন/ণ, ই/ঈ, উ/ঊ, র/ড়), plus KWS term list derived from the same entries.

---

## 5. Synthetic & weak-supervision data [CODE]
- **TTS slang insertion** for KWS only: generate carrier sentences with slang terms using a Bangla TTS with multiple voices, pass through codec simulation. Label as synthetic; never in test sets.
- **LLM-assisted text augmentation** (redacted transcripts only, self-hosted model): paraphrase positive spans into other dialects/registers; generate hard negatives (polite but firm, complaints without abuse, banter). Human spot-check ≥ 10%.
- **ASR-noise injection for text models**: run TTS→codec→ASR on clean social-media abusive text to produce realistic ASR-style errors; also train on actual ASR outputs of labelled calls.

---

## 6. Active learning loop (post-pilot) [CODE + HUMAN]
1. Supervisors' review verdicts (`reviews` table) flow back nightly.
2. Weekly sampler selects: false positives, false negatives found by QA, low-confidence events, disagreements between signals (e.g. SER says angry, text says neutral), new candidate slang terms.
3. 200–400 items/week to annotators → retrain monthly → shadow-evaluate → promote if golden-set metrics don't regress.

---

## 7. Data governance checklist [HUMAN sign-off]
- [ ] Legal basis confirmed (PDPO 2025) and call-recording notice covers automated analysis
- [ ] Data Protection Impact Assessment done
- [ ] Retention: raw audio N days (policy), redacted transcripts/events M months, training snapshots under DVC with access control
- [ ] Data-subject request process (delete a customer's data incl. from training snapshots on next retrain)
- [ ] Agent/union communication on how scores are used (coaching, not automatic penalties)
- [ ] Dataset licence register approved
