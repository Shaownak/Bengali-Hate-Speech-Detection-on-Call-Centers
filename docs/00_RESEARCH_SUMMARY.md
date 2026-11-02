# 00 — State-of-the-Art Research Summary (as of November 2026)

> Purpose: give the build team (and Claude Code) the evidence behind every model/tool choice in GAL-E.
> Every "default choice" below is a **hypothesis to be confirmed by the bake-off in `06_EVALUATION.md`** on our own 8 kHz call-centre golden set. Public benchmark numbers do not transfer to phone audio.

---

## 1. The core finding that shapes the whole design

Harsh behaviour in a call has two independent carriers:

| Carrier | What it captures | Survives transcription? | Detector family |
|---|---|---|---|
| **Lexical** (what is said) | slang, gali, insults, threats, sexual harassment, disrespectful register (আপনি→তুমি/তুই) | Yes | ASR → text models → LLM |
| **Paralinguistic** (how it is said) | shouting, anger, aggression, contempt, interruptions | **No** — lost in text | DSP features + speech emotion models |

Research confirms both channels matter and are complementary:
- ASR models are weak exactly on profanity — profane words are under-represented in training corpora and often not spoken clearly or completely (ADIMA, ICASSP 2022). → we add **audio-domain keyword spotting** so slang detection does not depend only on the transcript.
- On ADIMA's Bengali split, the best fusion of audio, emotion and text modalities (82.4% accuracy) beat the best single modality (audio, 80.8%) — but naive concatenation did not, so fusion design matters (MADA, 2022). Separate work showed abuse can be detected from acoustic/prosodic features alone, without transcripts.
- Meta's MuTox (audio toxicity, 30 languages incl. Bengali) found an end-to-end audio classifier outperformed ASR→text cascades of similar language coverage. → we keep an **audio-embedding toxicity head** as a third, cheap signal.

**Design consequence:** GAL-E is a *multi-signal fusion system*, not "ASR + LLM". The LLM is the context judge at the top of a tiered stack, not the first line of detection.

---

## 2. Bangla ASR (speech → text)

### 2.1 Candidates

| Model | Type / size | Bengali status | Notes for us |
|---|---|---|---|
| `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium` | Whisper-medium fine-tune | The de-facto Bangla starting point in 2026 papers | Most DL Sprint 4.0 (2026) teams fine-tuned it further. One team reached WER 15.55 on the private test after fine-tuning on ~21k segments built from YouTube audiobooks and dramas. **But on Ben-10 regional dialects its corpus WER is ~0.69** (BengaliAI leaderboard) → dialects are a major risk. |
| `BuzzASR/bengali` | Whisper-large-v3 full fine-tune | Monolingual Bengali (EMNLP 2026 Findings) | Trained on FLEURS + Common Voice 25 → clean read speech; must test on phone audio. |
| `ai4bharat/indic-conformer-600m-multilingual` | 600M hybrid CTC + RNN-T Conformer, 22 Indic languages, MIT | Supports Bengali; required resample to 16 kHz | Best pooled WER in a 2026 16-model Indic benchmark (22.02%). A production team transcribing 2,750 calls/day picked it over Whisper for telephony: more clean segments, **zero foreign-script drift**, runs on CPU (~7.1 s CPU per audio minute). **CTC head gives frame posteriors we can reuse for keyword spotting.** Trained mostly on Indian Bengali → test Bangladeshi accents. |
| `SayedShaun/Qwen3-ASR-0.6B-Bengali` | Qwen3-ASR-0.6B + LoRA (merged) | Community fine-tune; base Qwen3-ASR does **not** natively support Bengali | ~1,170 h training mix (IndicVoices ~600 h, OpenSLR-53, Vaani, Ben-10…). Servable with vLLM (high throughput). Unproven on phone audio. |
| Whisper-large-v3(-turbo) zero-shot | — | Weak | Script drift / very high WER on Bengali in 2026 Indic benchmark → **do not use zero-shot**. |

### 2.2 Telephony-specific evidence
- Phone calls are much harder than read-speech benchmarks; conversational Bengali-English code-mixed baselines sit around 30–37% WER.
- A 2026 BUET paper built a **synthetic Bengali telecom customer-care speech set** (10,000 utterances, 26.82 h, CC-BY-4.0, recharge/OTP/SIM/refund/mobile-banking intents) and fine-tuned the tugstugi Whisper with private telecom recordings. Lessons: (a) domain vocabulary matters, (b) **text normalisation (numbers, abbreviations, spelling variants) dominates residual WER**, (c) report **CER alongside WER** for Bangla, (d) synthetic single-speaker studio audio must be mixed with real call audio.
- Ben-10 (78 h, 10 regional dialects, IJCNLP-AACL 2025): foundation models "struggle heavily" on dialects in zero-shot *and* fine-tuned settings; dialect-specific training helps.

### 2.3 Streaming
- Whisper is built for complete utterances; naive chunking cuts words. **WhisperLiveKit** (MIT) packages SimulStreaming (AlignAtt policy) / LocalAgreement with Silero VAD over a WebSocket, faster-whisper backend. LoRA adapters only work with the native Whisper backend → **merge fine-tuned weights and convert to CTranslate2** instead.
- CTC/RNN-T Conformers (IndicConformer) are naturally better suited to low-latency streaming than encoder-decoder Whisper.

### 2.4 Decision
Run a **3-way bake-off** (fine-tuned tugstugi-Whisper vs fine-tuned IndicConformer vs Qwen3-ASR-bn) on our 8 kHz golden set. Metrics: CER, WER, **slang-word recall**, script-drift rate, real-time factor, p95 latency.
**Prior expectation:** IndicConformer for real-time + KWS posteriors; Whisper-medium fine-tune as the higher-accuracy post-call pass if it wins on CER.

---

## 3. Keyword spotting (slang/gali directly from audio)
- BAAD dataset: 114 Bangla slang words, 5,277 clips (plus 43 confusable non-slang words in the paper), 60 speakers from 20+ districts — the main public seed for audio-level slang KWS.
- Approach (standard, no single SOTA model for Bangla): (1) **CTC keyword search** over IndicConformer posteriors for the lexicon (handles hundreds of terms, open-vocabulary); (2) a small **dedicated KWS classifier** for the top ~100 high-severity terms trained on BAAD + our own clips + TTS/codec augmentation. Fires even when ASR "cleans up" a word.

---

## 4. Paralinguistics: shouting, anger, aggression

### 4.1 Shouting
Pure DSP is reliable and cheap: short-term loudness (LUFS / RMS dBFS), F0 level and range, spectral tilt / high-frequency energy ratio, clipping ratio — **relative to each speaker's own rolling baseline** (absolute loudness depends on handset/gain). A small gradient-boosted classifier over these features, calibrated, is the production standard.

### 4.2 Speech emotion recognition (SER)
- **emotion2vec+ large** (~300M params, 16 kHz, fine-tuned on 42,526 h pseudo-labelled data, 9 classes: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown): the strongest open SER foundation model, designed to be language-agnostic and explicitly aimed at call-centre analytics. Its training data is mostly Chinese/English → **fine-tune a head on Bangla**.
- Bangla SER data: **SUBESCO** (7,000 utterances, 20 actors, 7 emotions, CC-BY-4.0, 48 kHz, acted), **BanglaSER**, KBES (900), BanSpEmo. Recent Bangla SER papers report ~86–93% accuracy on SUBESCO — but that is acted, clean speech; real calls will be far lower. **In-domain labelled call snippets are mandatory.**
- Audio LLMs (Qwen3-Omni etc.) do paralinguistics but have limited speech-language coverage and are expensive; paralinguistic understanding in audio LLMs is still weak per 2026 benchmarks (VoxParadox). → not in the critical path; optional research track.

### 4.3 Audio-domain toxicity
- **MuTox** (Meta): audio toxicity dataset across 30 languages incl. Bengali + SONAR-embedding classifier; beat ASR→text cascades. Use its Bengali split for evaluation and as an auxiliary training signal.
- **ADIMA**: 11,775 clips / 65 h, 10 Indic languages incl. Bengali (~1,200 utterances per language), real ShareChat audio-chatroom speech, κ≈0.88.
- **Bangla Hate Speech & Non-Hate Audio (2025)**: 3,000 clips with severity (mild/moderate/severe), dialect variation.

---

## 5. Bangla text toxicity / abuse classification
- **BanglaBERT** remains the best single encoder for Bangla hate speech (BLP-2025 Task 1 papers: beat mBERT and XLM-R despite being smaller). Leaderboard-level systems are **ensembles** (BanglaBERT + MuRIL + IndicBERTv2) with multi-task heads (type, severity, target) at ~72–73% micro-F1 — fine-grained classes remain hard (class imbalance, sexism/religious hate under-represented).
- **BanglaHateBERT**: BanglaBERT further pre-trained on 1.5 M offensive posts — good init for the abuse head.
- Datasets: BLP-2025 Task 1 / **BanglaMultiHate** (~50k comments, type/severity/target), **BD-SHS** (50,200+ offensive comments, hierarchical labels), **BIDWESH** (9,183 instances translated into Barishal, Noakhali, Chittagong dialects — covers regional swear words and code-switching), Bangla Multilabel Cyberbully/Sexual-Harassment/Threat/Spam (12,557), Bangla-ToCo (context-aware toxic comments).
- **Domain gap:** all are social-media *written* text. Spoken call transcripts differ (disfluency, ASR errors, no punctuation). → **train/fine-tune on ASR outputs**, not clean text (noise augmentation).

---

## 6. LLMs for the context-judge layer
- General open models now have solid Bengali: **Qwen3.5** (open weights Feb 2026; 0.8B–122B; 201 languages) and **Gemma 4** (Apache-2.0, 140+ languages; E2B/E4B/26B-A4B/31B). Bengali-specialised: **TigerLLM** (Llama-3.2-1B / Gemma-2-9B + Bangla-TextBook + Bangla-Instruct), TituLLM.
- Bengali evaluation: **BnMMLU** (ACL 2026 Findings, 134k questions, 24 model variants) — use it only as a sanity filter; our task-specific golden set is what decides.
- Fine-tuning: **Unsloth** supports Qwen3.5 (bf16 LoRA VRAM: 4B ≈10 GB, 9B ≈22 GB, 27B ≈56 GB). Unsloth advises **against QLoRA 4-bit on Qwen3.5** (quantisation differences) → use bf16 LoRA. Gemma-4 E2B/E4B are supported too.
- Serving: **vLLM** with structured/guided JSON output.
- Privacy: call transcripts are personal data → **self-host**; any teacher-model distillation must use redacted or synthetic text only (see §8).

**Decision:** primary judge = **Qwen3.5-9B + LoRA** (bf16 LoRA training ≈ 22 GB → one 48 GB GPU with headroom; serving on 1× L4/A10/L40S with AWQ/FP8 quantisation), challenger = **Gemma-4-E4B / 26B-A4B + LoRA**, chosen by golden-set F1 + latency.

---

## 7. Telephony integration
- Asterisk: **ARI ExternalMedia** gives a bidirectional real-time RTP stream; attach it to **snoop channels (one per direction)** to stream agent and customer separately — the community-recommended path over AudioSocket (better docs/samples).
- FreeSWITCH: `mod_audio_stream` / `mod_audio_fork` (WebSocket fork).
- Enterprise contact-centre platforms (Genesys, Avaya, Cisco): **SIPREC** recording sessions or vendor streaming APIs.
- Post-call: pull stereo recordings from the recorder/S3 + CDR metadata.

---

## 8. Legal / privacy (Bangladesh)
- **Personal Data Protection Ordinance 2025** (Ordinance No. 61 of 2025, gazetted 6 Nov 2025): first comprehensive data-protection law; consent-based; sensitive data has heightened safeguards; cross-border transfer only with equivalent protection; per a Dhaka law-firm brief, *confidential/restricted* personal data must be stored within Bangladesh. Some provisions/enforcement start ~18 months after promulgation (~May 2027).
- **Action:** legal counsel must confirm current status and obligations before pilot. Engineering defaults: in-country hosting, PII redaction before any model that is not strictly necessary, encryption, retention limits, audit logs, no external AI APIs on raw call data.

---

## 9. References
- ADIMA — https://arxiv.org/abs/2202.07991
- MADA multimodal abuse detection — https://arxiv.org/pdf/2204.02263
- Cross-lingual audio abuse few-shot — https://arxiv.org/pdf/2412.01408
- MuTox — https://arxiv.org/abs/2401.05060
- BAAD — https://data.mendeley.com/datasets/w24g8xn23c/3
- Bangla Hate Speech audio (2025) — https://data.mendeley.com/datasets/tgjbpp4z5b
- emotion2vec / emotion2vec+ — https://github.com/ddlBoJack/emotion2vec
- SUBESCO — https://zenodo.org/records/4526477
- Bangla SER (SpectroFusion-ViT) — https://arxiv.org/html/2603.00746v1
- BLP-2025 Task 1 (Retriv) — https://arxiv.org/abs/2511.07304 ; BanglaBERT comparison — https://arxiv.org/abs/2512.02845
- BanglaHateBERT — https://aclweb.org/anthology/2022.restup-1.2.pdf
- BD-SHS — https://arxiv.org/abs/2206.00372
- BIDWESH — https://arxiv.org/abs/2507.16183
- Bangla multilabel cyberbully dataset — https://data.mendeley.com/datasets/sz5558wrd4/3
- DL Sprint 4.0 ASR papers — https://arxiv.org/html/2603.03158v1 ; https://awesomepapers.io/speech-audio/papers/2603.19256 ; https://arxiv.org/abs/2605.08214
- BuzzASR Bengali — https://huggingface.co/BuzzASR/bengali
- Indic ASR 16-model benchmark — https://sthanika.ai/research/what-it-actually-said
- Telephony ASR in production (IndicConformer) — https://zenthos.in/blogs/open-source-speech-to-text-2026
- Qwen3-ASR — https://github.com/QwenLM/Qwen3-ASR ; Bengali fine-tune — https://huggingface.co/SayedShaun/Qwen3-ASR-0.6B-Bengali
- Synthetic Bengali telecom speech (BUET) — https://arxiv.org/html/2608.20346v1
- Ben-10 dialect ASR — https://arxiv.org/abs/2510.23252 ; leaderboard — https://huggingface.co/datasets/bengaliAI/ben10-asr-results
- OOD-Speech — https://arxiv.org/abs/2305.09688
- WhisperLiveKit — https://pypi.org/project/whisperlivekit/
- Asterisk ExternalMedia + snoop — https://community.asterisk.org/t/103197
- Unsloth Qwen3.5 fine-tuning — https://unsloth.ai/docs/models/qwen3.5/fine-tune
- TigerLLM — https://aclanthology.org/2025.acl-short.69.pdf
- BnMMLU — https://aclanthology.org/2026.findings-acl.593/
- Bangladesh PDPO 2025 — https://digitalpolicyalert.org/change/18756 ; https://mahbub-law.com/key-highlights-of-the-personal-data-protection-ordinance-2025-for-businesses
