# GAL-E — Gali & Aggression Listening Engine
### Implementation spec pack for building with Claude Code

GAL-E detects Bangla slang/gali, abusive and negative language, disrespectful register (আপনি→তুমি/তুই), harsh tone and shouting in Bangladeshi call-centre calls. It scores each speaker (agent vs customer) separately and runs in two modes: post-call QA for every call, and real-time supervisor alerts.

## What's in this folder
| File | What it is |
|---|---|
| `CLAUDE.md` | Project instructions Claude Code reads automatically: stack, conventions, hard rules, commands |
| `docs/00_RESEARCH_SUMMARY.md` | State of the art as of Nov 2026, with sources, and why each model was chosen |
| `docs/01_ARCHITECTURE.md` | Services, data flow, message schemas, DB schema, latency budget, repo layout |
| `docs/02_DATA_PLAN.md` | Public datasets, in-house call data, processing, annotation taxonomy and guide, lexicon, governance |
| `docs/03_MODELS_AND_TRAINING.md` | Training recipes: ASR, keyword spotting, shouting, speech emotion, audio toxicity, BanglaBERT, LLM judge, fusion |
| `docs/04_GUI_SPEC.md` | Dashboard screens, roles and permissions, API surface |
| `docs/05_INFRA_AND_OPS.md` | Kubernetes, GPUs, capacity planning, observability, MLOps, security and privacy |
| `docs/06_EVALUATION.md` | Metrics, initial targets, fairness checks, test pyramid, pilot plan |
| `docs/07_ROADMAP.md` | Phases P0–P7 (~22 weeks), team, hardware, risks |
| `docs/08_LLM_JUDGE_SPEC.md` | LLM prompts, JSON schemas, SFT format, serving and evaluation |
| `docs/09_OPEN_DATASETS.md` | Open Bangla hate-speech/abuse datasets (15 Bangla-script text, 3 Banglish + 1 transliteration corpus, 4 audio): sizes, labels, links, licence gate, label mapping, harmonisation pipeline, how each model uses them |
| `prompts/CLAUDE_CODE_PROMPTS.md` | 17 ordered prompts to paste into Claude Code (incl. 4b: open-dataset ingestion), plus review, fix and model-swap prompts |

## How to use it
1. `git init gal-e && cd gal-e`, then copy this folder's contents in so `CLAUDE.md` sits at the repo root.
2. Open Claude Code in the repo and paste **Prompt 0**. Then continue one prompt at a time.
3. Everything runs in **stub mode** (no GPUs, no model downloads, synthetic calls) until you supply the 🔒 items below.

## What only you (the humans) can provide 🔒
- [ ] Legal sign-off and DPIA under Bangladesh's Personal Data Protection Ordinance 2025. Confirm its current status with counsel.
- [ ] Access to historical stereo call recordings and CDR metadata, kept in-country
- [ ] An annotation team (6–10 native speakers covering the main dialects) and a QA lead to own the guide
- [ ] Licence review of every public dataset and model before commercial use
- [ ] GPUs (training: 2 × 48 GB; inference: L4/A10-class plus one 48 GB card for the LLM, sized by load test)
- [ ] A PBX or contact-centre team to configure media forking (Asterisk ARI ExternalMedia, FreeSWITCH fork, or SIPREC)
- [ ] A Kubernetes cluster (in-country), DNS/TLS, SSO (Keycloak or existing IdP)
- [ ] Agreement with agents and management that alerts are for coaching, not automatic penalties

## Reality check
Claude Code can write essentially all of the software, training and evaluation code. Accuracy, though, comes from **your call data and annotations**. Expect the first fine-tuned version to reach usable precision on clear-cut cases (slang, shouting, threats). Subtle cases (sarcasm, dialect-specific insults) improve through the review → retrain loop during the pilot. Treat the targets in `06_EVALUATION.md` as starting points to agree with your QA lead after the first baseline.
