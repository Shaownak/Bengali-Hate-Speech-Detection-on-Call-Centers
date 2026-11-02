# 09 — Open-Source Bangla Hate-Speech / Abuse Datasets: Catalogue & Usage Plan

> **Goal:** get GAL-E's text classifier, slang lexicon, LLM judge and audio models to a strong baseline **using only open datasets**, before any in-house call data is labelled. In-house data then becomes a *domain adaptation* step, not a from-scratch effort.
>
> ⚠️ **Licence gate [HUMAN]:** "open" ≠ "free for commercial use". Every row has a `licence` field; anything marked *VERIFY* or *research-only* must be checked by legal before it is used to train a model that ships commercially. The ingestion code must read `ml/data/DATASETS.md` and **refuse to include a dataset whose `commercial_ok` flag is not `true`** in production training runs (research runs may include it with a `research_only` tag that blocks promotion).

---

## 1. Text datasets (Bangla script)

| # | Dataset (year) | Size | Labels | Source text | Where | Licence* | GAL-E role |
|---|---|---|---|---|---|---|---|
| T1 | **BanglaMultiHate** / BLP-2025 Task 1 (ACL 2026) | 50,746 comments (shared-task split: 35,522 train / 2,512 dev / test) | hate **type**, **severity** (little-none / mild / severe), **target** | YouTube news comments | `huggingface.co/datasets/aridhasan/BanglaMultiHate`, `github.com/AridHasan/blp25_task1` | VERIFY | **Primary** for type + severity + target heads |
| T2 | **BD-SHS** (LREC 2022) | 50,281 | 3 levels: HS/NH → target (individual/male/female/group) → type (slander, religion, gender, call-to-violence) | FB/YouTube/news comments | `github.com/naurosromim/hate-speech-dataset-for-Bengali-social-media`, Kaggle `naurosromim/bdshs` | VERIFY | Primary binary + type/target |
| T3 | **BanglaCAT** (2026) | 19,453 train / 2,329 test, Bangla + English row-aligned | Toxic/Non-toxic → 11 types: Insult, Threat, Hate Speech, Profanity, Harassment, Identity Attack, Toxic Endorsement, Incitement, **Mockery/Ridicule**, Sexual Harassment, Dehumanization; **context-aware** (sarcasm, veiled threats, consensual banter, quoted speech) | Social media threads | Mendeley DOI 10.17632/8vr3w68c7x | VERIFY | **Key for LLM judge** (context, banter vs abuse) and sarcasm/threat heads |
| T4 | **BIDWESH** (2025) | 9,183 × 3 dialects (Barishal, Noakhali, Chittagong) | hate binary, type, target | BD-SHS translated into dialects | arXiv 2507.16183 (links in paper) | VERIFY | Dialect robustness. **Derived from BD-SHS → must share split with its source rows** |
| T5 | Bangla Multilabel Cyberbully / Sexual Harassment / Threat / Spam (2024) | 12,557 | multi-label: bully, sexual, religious, threat, spam (+ victim gender/profession) | Social media | Mendeley 10.17632/sz5558wrd4.3 | VERIFY | threat + sexual_harassment heads |
| T6 | **Vio-Lens** / BLP-2023 Task 1 | 2,700 train / 1,330 dev / 2,016 test | non-violence / passive violence (derogatory, abusive, slang) / direct violence | YouTube comments on 9 violent incidents | `github.com/blp-workshop/blp_task1` | VERIFY | threat (direct violence), insult (passive) |
| T7 | **TRAC-2 Bengali** (LREC 2020) | 5,971 (3,826 / 957 / 1,188) | Non / **Covertly** / **Overtly aggressive**; Gendered / Non-gendered | YouTube; Bangla + Roman script | TRAC-2 shared task release | VERIFY | **Aggression** signal; covert aggression ≈ sarcasm/dismissive |
| T8 | Bengali Hate Speech Dataset (Karim et al. 2020) | 3,418 (subset release) | Personal, Political, Religious, Geopolitical, Gender-abusive (all hateful) | News, FB, blogs | HF `bn_hate_speech`; UCI #719 | MIT (HF card) / CC-BY-4.0 (UCI) | type head (positives only) |
| T9 | Romim et al. 2020 hate-speech dataset | ~30,000 | hate / not-hate across 7 domains | FB/YouTube | GitHub (paper) | VERIFY | binary pre-training; **dedupe vs T2/HS-BAN** |
| T10 | HS-BAN (2021) | 50,000+ | binary (40% hate) | Social media | arXiv 2112.01902 | VERIFY | binary; likely overlaps T2/T9 → dedupe |
| T11 | Bangla Online Comments (BRAC, 2021) | 44,001 | bully / harassment categories + degree | Online comments | Mendeley 10.17632/9xjx8twk8p | VERIFY | insult / harassment |
| T12 | BanglaHarassNet (2026) | 5,500+ | hate, cyberbullying, sexual harassment, **trolling** (sarcastic/mocking), neutral, with severity | Multi-platform, Bangla/English/Banglish | Mendeley 10.17632/cw6jrvxmhf.1 | VERIFY | sarcasm_mockery, sexual_harassment, severity |
| T13 | Bangla-ToCo (2025) | modest | toxic / non-toxic with conversational context | Social media | DIU / paper | VERIFY | context negatives for LLM |
| T14 | BOISHOMMO (2025) | 2,000+ | multi-label (race, gender, religion, politics…) | Social media | arXiv 2504.08408 | VERIFY | hate_identity |
| T15 | Bengali Hate (Das et al., AACL 2022) | — | hate / offensive / normal; Bangla + Roman Bengali | Social media | `github.com/hate-alert/Bengali_Hate` | VERIFY | offensive vs hate distinction |

## 2. Romanised / code-mixed (Banglish) datasets

| # | Dataset | Size | Labels | Where | Licence* | Use |
|---|---|---|---|---|---|---|
| R1 | **BanTH** (NAACL Findings 2025) | 37,350 (29,879 / 3,736 / 3,735) | hate binary + multi-label: Political, Religious, Gender, Personal Offence, Abusive/Violence, Origin, Body-shaming | arXiv 2410.13281 (links in paper) | VERIFY | Back-transliterate → Bangla script; code-mix robustness |
| R2 | **Code-mixed Chaos** (2025) | 10,234 (5,117 toxic / 5,117 non-toxic) **+ Bangla-translated version** | 9 toxic types: Vulgar, Religious-hostility, **Troll**, **Insult**, Loathe, **Threat**, Race, **Sexual**, Political | Mendeley 10.17632/23dp3t88vk | VERIFY | Use the Bangla version directly; vulgar/insult/threat/sexual/troll heads |
| R3 | MixTox-SA (2025) | 13,035 (toxic / non-toxic / neutral) | Bangla, Hindi, Banglish, English | Mendeley 10.17632/bsdcd7zkyz | **Research/education only** per dataset page | Eval only unless licence clarified; drop Hindi rows |
| R4 | BanglaTLit (2024) | 42.7k pairs + 245.7k romanised pre-training corpus | romanised ↔ Bangla script | ACL/arXiv | VERIFY | Train the **back-transliteration** model used to convert R1/R2/T7 Roman text |

## 3. Audio datasets (spoken abuse / slang)

| # | Dataset | Size | Labels | Where | Licence* | Use |
|---|---|---|---|---|---|---|
| A1 | **BAAD** v3 (Mendeley, 28 Dec 2022; paper: Data in Brief 2023) | 114 slang words, 5,277 clips (paper version adds 43 confusable non-slang words, 6,100 clips total), 60 native speakers (65% male / 35% female), 20+ districts, WAV | isolated word-level slang | Mendeley 10.17632/w24g8xn23c.3 | **CC BY 4.0** (commercial use allowed with attribution; confirm in legal review) | **KWS training** + lexicon seed + KWS eval. Isolated words, not phone-channel → needs codec/noise augmentation and in-sentence recordings |
| A2 | Bangla Hate Speech & Non-Hate Audio (2025) | 3,000 clips | hate / non-hate + severity (mild/moderate/severe), dialects | Mendeley 10.17632/tgjbpp4z5b | VERIFY | audio-tox head; text via ASR → text-clf eval on speech |
| A3 | ADIMA (Bengali split) | ~1,200 utterances (of 11,775 / 65 h total) | abusive / non-abusive, real audio-chatroom speech | ShareChat release (ICASSP 2022) | research — VERIFY | audio-tox head; **realistic** noisy speech eval |
| A4 | MuTox (Bengali split) | part of 30-language set | toxic / non-toxic (+ categories) | Meta (seamless_communication) | VERIFY | audio-tox eval / aux training |

\* Licence column reflects what is visible on dataset pages/papers as of Nov 2026; **all must be re-verified** and recorded in `ml/data/DATASETS.md` with `commercial_ok: true|false|unknown`.

---

## 4. Harmonised schema (`ml/data/open/unified.jsonl`)

```json
{
  "uid": "bdshs:000123",
  "source": "bdshs", "source_split": "train", "source_id": "123",
  "parent_uid": null,                       // e.g. BIDWESH rows point to their BD-SHS parent
  "text_orig": "...", "script_orig": "bangla|roman|mixed",
  "text_bn": "...",                          // Bangla script (normalised with galecore.bangla.normalize)
  "dialect": "standard|barishal|noakhali|chittagong|unknown",
  "context": ["prev comment 1", "..."],      // only for context-aware sets (BanglaCAT, Bangla-ToCo)
  "labels": {                                // GAL-E taxonomy; null = NOT ANNOTATED in this source
    "slang_profanity": 1, "insult": null, "threat": 0, "sexual_harassment": null,
    "hate_identity": 0, "sarcasm_mockery": null, "dismissive_rude": null, "aggression": null
  },
  "toxic_any": 1,                            // always filled where the source has a binary/derivable label
  "severity": 2,                             // 0..3 or null
  "target": "individual|group|organization|society|null",
  "context_flags": ["banter"],               // from BanglaCAT etc.
  "licence_id": "bdshs", "commercial_ok": false,
  "weight": 1.0
}
```

**Partial labels are the norm** (each source annotates different categories). Train with **masked multi-label loss**: categories that are `null` for a row contribute no loss. Never convert "not annotated" into "negative".

## 5. Label mapping → GAL-E taxonomy (`ml/textclf/label_map.yaml`)

| Source label | GAL-E label(s) |
|---|---|
| BanglaMultiHate type `Profane` / `Abusive` | `slang_profanity` / `insult` (verify exact label names in the release) |
| BanglaMultiHate type `Religious Hate`, `Sexism`, `Political Hate` | `hate_identity` (political → `hate_identity` with low weight, see §7) |
| BanglaMultiHate severity little-none / mild / severe | severity 0 / 1–2 / 3 (calibrate on dev) |
| BD-SHS `slander` / `callToViolence` / `religion`, `gender` | `insult` / `threat` / `hate_identity` |
| BanglaCAT Profanity, Insult, Threat, Incitement, Mockery/Ridicule, Sexual Harassment, Identity Attack, Hate Speech, Dehumanization, Harassment | `slang_profanity`, `insult`, `threat`, `threat`, `sarcasm_mockery`, `sexual_harassment`, `hate_identity`, `hate_identity`, `insult`(sev+1), `insult` |
| BanglaCAT context cases (banter, quoted) | `context_flags` + toxic_any per gold |
| Cyberbully set bully / sexual / religious / threat / spam | `insult` / `sexual_harassment` / `hate_identity` / `threat` / drop |
| Vio-Lens passive / direct violence | `insult` (+`slang_profanity` if lexicon hit) / `threat` |
| TRAC-2 OAG / CAG / NAG; GEN | `aggression`=1 (overt → also `insult` weak label) / `aggression`=1 + `sarcasm_mockery` weak / 0; `hate_identity` (gender) |
| Code-mixed Chaos Vulgar / Insult / Threat / Sexual / Troll / Religious, Race / Loathe / Political | `slang_profanity` / `insult` / `threat` / `sexual_harassment` / `sarcasm_mockery` / `hate_identity` / `insult` / `hate_identity`(low weight) |
| BanglaHarassNet trolling / cyberbullying / sexual harassment / hate | `sarcasm_mockery` / `insult` / `sexual_harassment` / `hate_identity` |
| BanTH Personal Offence / Abusive-Violence / Gender, Religious, Origin / Body-shaming / Political | `insult` / `slang_profanity`+`threat`(weak) / `hate_identity` / `insult` / `hate_identity`(low weight) |
| Karim et al. Personal / Gender-abusive / Religious, Geopolitical, Political | `insult` / `hate_identity` / `hate_identity` (positives only; toxic_any=1) |

`aggression` is an auxiliary head (not an alert category) that helps `harsh_tone` / `dismissive_rude` detection from text and is used as a feature in fusion.

Weak mappings get `label_confidence: 0.6` → used as soft targets.

## 6. Processing pipeline [CODE] — `ml/data/open/`

1. **Download** (only datasets with `download_allowed: true` in DATASETS.md; otherwise expect manual placement under `data/raw/<source>/`), record checksum + version/DOI.
2. **Parse** each source with a dedicated loader → unified schema.
3. **Script handling**: Roman/Banglish rows → back-transliterate with a model trained on BanglaTLit (seq2seq, e.g. BanglaT5 fine-tune); keep `text_orig`. Prefer an official Bangla translation where provided (Code-mixed Chaos). Rows with Hindi/English-only content dropped (language-ID filter).
4. **Normalise** with `galecore.bangla.normalize`; strip URLs, @mentions, hashtags, emoji (keep an `emoji_present` feature), repeated characters (collapse ≥3).
5. **De-duplication & leakage control**:
   - exact + near-duplicate (MinHash, Jaccard ≥ 0.85 on character 5-grams) **across all sources**;
   - BIDWESH rows inherit the split of their BD-SHS parent;
   - BD-SHS / HS-BAN / Romim overlap → keep one copy, merge labels;
   - keep each source's official **test** split as test; build our own train/dev from the rest; global dedupe ensures no test text appears in any train split.
6. **Conflict resolution**: if duplicates disagree, majority vote weighted by source quality (inter-annotator agreement reported in paper); unresolved → `null`.
7. **Speech-style adaptation** (closing the social-media → phone-call gap):
   - **ASR-noise injection**: character/word confusion model fitted on our ASR outputs (initially on public ASR test sets, later on call data);
   - **TTS → telephony codec → ASR round-trip** for a 10–20% subset (most realistic noise);
   - **Spoken-style rewriting** (self-hosted LLM, synthetic-tagged): convert comment-style text into a second-person utterance as it might be said on a call, preserving label; human spot-check 10%.
8. **Re-weighting for the call-centre domain**: personal insults, profanity, threats, sexual harassment, mockery, aggression → weight 1.0; political/religious/geopolitical hate (dominant in social media, rare in calls) → 0.3; spam dropped.
9. Output: `unified.jsonl`, per-model training views (text-clf, LLM SFT, lexicon mining), and a **data card** (`reports/open_data_card.md`) with counts per label/source/dialect/script, duplicates removed, licence status.

## 7. How each GAL-E model uses the open data

| Model | Open data used | How |
|---|---|---|
| **Text classifier** (BanglaBERT multi-task) | T1–T15, R1–R2 (commercial_ok only for prod) | Stage 1 train on unified set (masked loss, soft weak labels); Stage 2 ASR-noised copy; Stage 3 in-house call spans. **Phase P1 deliverable: open-data-only classifier.** |
| **Slang lexicon seed** | T1 (Profane/Abusive), T2, T3 (Profanity), R2 (Vulgar), A1 (BAAD 114 words) | Log-odds/PMI term mining on profane vs non-toxic rows → candidate list with variants → **human review** → `lexicon_entries` (masked in UI). |
| **KWS** | A1 BAAD + TTS of reviewed lexicon terms + A2/A3 positives | multi-label KWS training, BAAD confusables as hard negatives. |
| **Audio-toxicity head** | A2, A3, A4 | train/eval; never alerts alone. |
| **LLM judge SFT** | T3 BanglaCAT (context, banter, quoted speech), T1 (type/severity/target), T6, T7, T12, R2 | Convert to **synthetic call windows**: place the (speech-style-rewritten) text as one speaker's utterance inside a templated agent–customer exchange, attach simulated signals, target JSON from mapped labels. Tag `source=open_synthetic`; cap at ≤ 30% of final SFT mix once in-house data exists (100% for the P1 zero-shot→few-shot baseline). |
| **Register detector** | (none directly) | Rules; open text used only to mine তুই/তুমি-with-insult co-occurrence statistics. |
| **Evaluation** | official test splits of T1, T2, T3, T6, T7, R1, A2, A3, A4 | `public_aux` regression suite — reported every release, **never** used for acceptance (see 06). |

## 8. Expected baselines (to sanity-check our implementation)
- BanglaMultiHate / BLP-2025 Task 1: transformer ensembles ≈ 72–73% micro-F1 on 1A/1B (mid-leaderboard systems); BanglaBERT best single model; LoRA-tuned LLMs rival BanglaBERT.
- Vio-Lens: best benchmark ≈ 71% macro-F1 in the dataset paper.
- If our single-dataset reproduction is > 5 points below these, debug preprocessing before going further.

## 9. Known limitations of open data (why in-house labels are still needed)
- Written social-media comments, mostly about politics/religion/celebrities — not customer-service dialogue.
- Little or no **agent-side** behaviour (disrespectful register, dismissive rudeness, interruption) — these categories depend on in-house labels.
- No prosody: shouting and harsh tone cannot be learned from text datasets.
- Dialect coverage limited to BIDWESH's three dialects + Ben-10 audio.
- Label definitions differ across datasets → handled by mapping + masked loss, but residual noise remains.
