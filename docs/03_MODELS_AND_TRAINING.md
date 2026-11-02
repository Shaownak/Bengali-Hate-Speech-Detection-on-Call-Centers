# 03 — Models & Training Recipes

Every component follows the same lifecycle:
`baseline (pretrained) → fine-tune on public → fine-tune on in-domain → calibrate → export (ONNX/CT2/AWQ) → register in MLflow → shadow → promote`.

All training code lives in `ml/<component>/`, is config-driven (Hydra or plain YAML + pydantic), logs to MLflow, reads data via DVC-tracked manifests, and ends with `python -m ml.eval.run --component <x> --split golden`.

GPU assumptions for training: 1–2 × 48 GB (L40S/A6000) or 1 × 80 GB (A100/H100). Inference: L4/A10 (24 GB) class.

---

## 1. ASR

### 1.1 Bake-off (week 1 of modelling)
Backends: `whisper_ct2` (tugstugi regional Whisper-medium), `indic_conformer` (AI4Bharat 600M), `qwen3_asr` (Qwen3-ASR-0.6B-Bengali). Optionally BuzzASR/bengali (Whisper-large-v3 FT) for post-call.
Eval on: golden 8 kHz call set (human transcripts), Ben-10 test (dialect), OOD-Speech OOD test.
Metrics: CER, WER (after shared normalisation), **slang recall** (lexicon terms present in reference that appear in hypothesis), script-drift %, RTF on L4, streaming p95 latency.

### 1.2 Fine-tuning recipe (apply to top 1–2 backends)
- **Data mix** (per epoch sampling weights): in-house corrected transcripts 40% · IndicVoices-bn + OOD-Speech 30% · Ben-10 10% · Common Voice/OpenSLR 10% · telecom synthetic 5% · slang carrier sentences 5%.
- **Telephony simulation on all non-call data**: 16k→8k→16k resample + random codec (µ-law/A-law/AMR-NB/GSM) + noise + packet loss.
- Whisper: full fine-tune (medium) or LoRA r=32 on encoder+decoder, lr 1e-5 (full) / 1e-4 (LoRA), warmup 500, bf16, SpecAugment, 3–5 epochs, batch ≈ 64 utts eff., max 30 s chunks; **prompt/prefix with domain vocabulary list** during training to enable contextual biasing at inference. Merge LoRA → convert to CTranslate2 (`ct2-transformers-converter --quantization int8_float16`).
- IndicConformer (NeMo): fine-tune hybrid CTC/RNN-T with Bengali tokenizer kept; lr 1e-4–2e-4, Noam/cosine, 50–100k steps, SpecAugment; enable **cache-aware streaming** config for real-time (chunk 160–320 ms, right context small).
- Qwen3-ASR-bn: continue LoRA fine-tune with the official toolkit; serve via vLLM.
- **Acceptance (initial targets, revise after baseline):** golden CER ≤ 15%, WER ≤ 30%, slang recall ≥ 80% (KWS covers the rest), script drift ≈ 0%.

### 1.3 Contextual biasing
- Lexicon terms + domain terms (package names, "রিচার্জ", "ব্যালেন্স"…) passed as hotwords: Whisper `initial_prompt`/hotwords in faster-whisper; NeMo word boosting (GPU-PB / shallow fusion) for Conformer.
- Guard: biasing must not raise false slang insertions — measure **slang false-insertion rate** on clean negative calls.

---

## 2. Keyword spotting (slang from audio)

Two complementary detectors, both produce `Signal(source="kws")`:

**2a. CTC keyword search** (open-vocabulary, every lexicon term):
- Use IndicConformer CTC posteriors (even if ASR text comes from Whisper, run the Conformer encoder's CTC head in parallel — cheap).
- For each term (and variants) build token sequences; run a sliding CTC forced-alignment / token-passing keyword search; score = normalised log-likelihood ratio vs filler model.
- Threshold per term calibrated on dev set.

**2b. Dedicated KWS classifier** (top ~100 high-severity terms):
- Model: small Conformer or wav2vec2/emotion2vec-encoder + attention pooling over 1.0–1.5 s windows; multi-label output (term ids + "other speech" + "silence").
- Data: BAAD clips, in-house volunteer recordings, T2 spans from real calls, TTS carriers; heavy codec/noise augmentation; hard negatives = acoustically similar non-slang words (BAAD's 43 confusables + our own).
- Target: per-term recall ≥ 90% at ≤ 1 false alarm / hour of speech per term-group.

---

## 3. Shouting detector

- Features per 0.5 s hop over 1.5 s window, per channel: short-term LUFS, RMS dBFS, peak & clipping ratio, F0 median/p90/range (pyworld or torchcrepe-tiny), spectral tilt, HF energy ratio (2–4 kHz / 0.3–1 kHz), spectral flatness, speech rate proxy; **each as z-score vs that speaker's rolling baseline** (first 20 s of their speech + EMA).
- Model: LightGBM binary + isotonic calibration. Train on T2 `shouting` spans; negatives sampled from same speakers.
- Rules guard: suppress if VAD says < 40% speech in window, or if both channels spike simultaneously (line artefact).
- Target: F1 ≥ 0.85 on golden.

---

## 4. Speech emotion / harsh tone (SER)

- Backbone: **emotion2vec+ large** (frozen first, then partially unfrozen top 4 layers).
- Head: attention-pooling + MLP, multi-task: (a) categorical {neutral, frustrated, angry, contempt/aggressive, other}, (b) arousal & valence regression (weak labels from SUBESCO mapping + in-domain).
- Training stages: (1) SUBESCO + BanglaSER (acted, 8k-simulated) → (2) in-domain T2 `harsh_tone`/`shouting` spans + neutral spans (real calls). Class-balanced sampling, label smoothing 0.1, mixup on embeddings.
- Windows: 3 s with 1 s hop, aggregate to utterance with max + mean.
- Export: ONNX (fp16). Target: UAR ≥ 0.70 on in-domain golden for {neutral, frustrated, angry}; angry-vs-rest AUC ≥ 0.88.

---

## 5. Audio-toxicity head (cheap end-to-end audio signal)

- Embedding: reuse emotion2vec+/Conformer encoder outputs (already computed) **or** SONAR speech encoder (MuTox approach) if licence permits.
- Head: logistic/MLP → p(toxic). Train on ADIMA-bn + Bangla hate-speech audio + MuTox-bn + in-domain T2 positives.
- Role: tie-breaker and recall booster in fusion, never alerts alone.

---

## 6. Text classifier (BanglaBERT multi-task)

- Init: `csebuetnlp/banglabert_large` (compare with BanglaHateBERT, `banglabert` base for latency).
- Input: redacted, normalised ASR text of current utterance + previous 2 utterances of both speakers, speaker-tagged: `[AGENT] … [CUST] … [TARGET] <current utterance>`.
- Heads (multi-label sigmoid): `slang_profanity, insult, threat, sexual_harassment, hate_identity, dismissive_rude, sarcasm_mockery, disrespect_register` + severity head (ordinal 0–3) + target head (agent/customer/company/third-party/none).
- Training stages:
  1. Open datasets harmonised per `09_OPEN_DATASETS.md` (BanglaMultiHate, BD-SHS, BanglaCAT, BIDWESH, Vio-Lens, TRAC-2, cyberbully/harassment sets, back-transliterated BanTH/Code-mixed Chaos, …) mapped to our taxonomy (`ml/textclf/label_map.yaml`). **Masked multi-label loss** (un-annotated categories give no loss), soft targets for weak mappings, domain re-weighting (personal abuse ↑, political/religious ↓), auxiliary `aggression` head from TRAC-2. Only `commercial_ok` sources in production runs.
  2. **ASR-noised** version of stage-1 data (TTS→codec→ASR or character-level noise model fitted to our ASR confusion statistics).
  3. In-domain T2 spans on real ASR output (not human transcripts!).
- Loss: focal BCE (γ=2) for imbalance; lr 2e-5, 5 epochs, early stop on dev macro-F1; R-Drop or FGM adversarial training (helped BLP-2025 teams).
- Optional ensemble (+ MuRIL / IndicBERTv2) for post-call mode only.
- Export ONNX int8 (dynamic quantisation), target CPU latency ≤ 40 ms/utt.
- Target: macro-F1 ≥ 0.70 on in-domain golden; slang/insult/threat F1 ≥ 0.80.

### 6.1 Register detector (rules + tiny model)
- Pronoun & verb morphology rules: আপনি/আপনার/আপনাকে + -েন verbs (polite) vs তুমি/তোমার + -ো/-েছ verbs vs তুই/তোর + bare imperatives.
- Track per-speaker register state through the call; emit `disrespect_register` when an **agent** moves down a level, or any speaker moves to তুই level with negative context; suppress if the speaker used that level neutrally from the start (dialect).
- Small logistic model on rule features + text-clf scores to decide.

---

## 7. LLM judge

Full prompt/schema in `08_LLM_JUDGE_SPEC.md`.

### 7.1 Model selection
- Primary: **Qwen3.5-9B (instruction-tuned checkpoint) + LoRA**. Challenger: **Gemma-4 (E4B for speed / 26B-A4B for quality) + LoRA**. Optional high-quality post-call: Qwen3.5-27B + LoRA.
- Zero-shot screen first (prompt only) on 200 dev windows; then fine-tune the best two.

### 7.2 SFT data
- Unit = window (≤ 30 s, both channels, speaker-tagged, redacted) + aligned signal summary (shout/SER/KWS/text-clf scores as compact text) → target JSON (schema in 08).
- Sources: (a) T2/T3 human labels converted to target JSON (gold), (b) hard negatives (banter, polite firmness, complaints), (c) synthetic augmentation (dialect paraphrases) — ≤ 30% of mix, (d) **optional teacher distillation**: a stronger self-hosted model (e.g., Qwen3.5-122B-A10B) produces rationales on redacted windows, humans verify a sample; never send call text to external APIs.
- Size target: 8k–15k windows + 1.5k whole-call scorecards for phase 1.

### 7.3 Training (Unsloth)
- bf16 LoRA (Unsloth advises against 4-bit QLoRA on Qwen3.5): r=16–32, α=32, dropout 0.05, target all linear layers, lr 1e-4 cosine, 2–3 epochs, max seq 4k (window) / 16k (call), loss on assistant tokens only, packing off for JSON fidelity.
- VRAM: 9B bf16 LoRA ≈ 22 GB (fits 1 × 48 GB with headroom).
- Thinking/reasoning mode **off** in production (latency); include a short `rationale_bn` field instead.
- Merge → quantise (AWQ int4 or FP8) → serve with vLLM `--guided-decoding-backend xgrammar` (or current vLLM structured-output option) using the JSON schema.

### 7.4 Eval
- Event-level micro/macro-F1 vs gold, JSON validity 100%, schema-consistent severity, hallucinated-evidence rate (evidence span must be a substring of the input) = 0 tolerance, p95 latency ≤ 2.5 s window mode on 1 × L4/A10 at target concurrency.

---

## 8. Fusion engine

1. **Calibration**: each signal source → isotonic regression on dev set → calibrated probability.
2. **Rules layer** (`configs/rules.yaml`), examples:
   - `kws.slang ≥ 0.9 OR lexicon.exact(severity≥2)` → provisional `slang_profanity` (sev from lexicon), alert if speaker=agent.
   - `dsp_shout ≥ 0.8 for ≥ 2 s` → `shouting`.
   - `shouting AND (text_clf.insult|threat|slang ≥ 0.6)` → severity +1.
   - `llm_window.dismissed(category)` within 8 s → set provisional event `dismissed_by_llm` unless kws ≥ 0.97.
   - `register switch agent tumi/tui` → `disrespect_register` sev 2.
3. **Stacker**: per category logistic regression over calibrated signals + context features (speaker role, queue, time into call, prior events) trained on dev; outputs final probability.
4. **Hysteresis**: per (call, speaker, category) cool-down 20 s; merge events < 5 s apart; escalate severity if repeated ≥ 3 times in 2 min.
5. **Alert policy**: real-time alert only if `probability ≥ thr_alert[category][speaker]` (high precision); everything above `thr_log` is stored for post-call QA.
6. Call-level outputs: agent professionalism score, customer hostility score, escalation curve (rolling anger per speaker), "who escalated first" heuristic.

---

## 9. Model registry & promotion
- MLflow registry names: `gale-asr-<backend>`, `gale-kws`, `gale-shout`, `gale-ser`, `gale-audiotox`, `gale-textclf`, `gale-llm-judge`, `gale-fusion`.
- `configs/models.yaml` pins versions per environment. Promotion requires: golden-set report with no metric regressing > 1 pt absolute, latency within budget, signed off by ML lead + QA lead.
- Shadow mode: new version runs in parallel on live traffic, writes signals with `shadow=true`, compared weekly.
