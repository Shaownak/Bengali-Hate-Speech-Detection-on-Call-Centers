# Runbook: open-data track (Prompt 4b, docs/09)

All paths are under `$GALE_DATA_DIR/data` (default `.gale/data`, git-ignored). Raw data and the
lexicon candidate CSV contain real abusive text: keep them off the repo and out of logs.

## 1. Get the data

```bash
python -m ml.data.open status            # what is in place
python -m ml.data.open download          # T1 BanglaMultiHate, T6 Vio-Lens, T8 Karim et al. (public)
```

Everything else is a **manual download**: get the files from the source and put them under
`$GALE_DATA_DIR/data/raw/<key>/`, matching the `files` patterns in `ml/data/datasets.yaml`. Many
entries are `format: VERIFY` (column names taken from the paper). If a loader can't find a column,
its error lists the columns it saw: fix `columns:` for that dataset in the YAML and re-run.

| id | key | where | notes |
|---|---|---|---|
| T2 | `bdshs` | Kaggle `naurosromim/bdshs` (login) | `train.csv`, `val.csv`, `test.csv` |
| T3 | `banglacat` | Mendeley 10.17632/8vr3w68c7x | train/test files |
| T4 | `bidwesh` | links in arXiv 2507.16183 | needs a column pointing at the BD-SHS parent row |
| T5 | `cyberbully_multilabel` | Mendeley 10.17632/sz5558wrd4.3 | xlsx/csv |
| T7 | `trac2_bn` | TRAC-2 shared-task release (request form) | Bangla + Roman rows |
| T9 | `romim2020` | GitHub (paper) | binary |
| T10 | `hsban` | arXiv 2112.01902 links | binary |
| T11 | `brac_online_comments` | Mendeley 10.17632/9xjx8twk8p | |
| T12 | `banglaharassnet` | Mendeley 10.17632/cw6jrvxmhf.1 | |
| T13 | `bangla_toco` | DIU / paper | with conversational context |
| T14 | `boishommo` | arXiv 2504.08408 links | |
| T15 | `hatealert_bengali` | GitHub `hate-alert/Bengali_Hate` | **tweet ids only**: hydrate texts under the X terms, save as `*hydrated*.csv` |
| R1 | `banth` | arXiv 2410.13281 links | Roman script → back-transliterated |
| R2 | `codemixed_chaos` | Mendeley 10.17632/23dp3t88vk | use the file with the Bangla translation column |
| R3 | `mixtox_sa` | Mendeley 10.17632/bsdcd7zkyz | research/education only → **eval only** |
| R4 | `banglatlit` | ACL Findings EMNLP 2024 links | romanised ↔ Bangla pairs |
| A1 | `baad` | Mendeley 10.17632/w24g8xn23c.3 | `baad/<term>/<speaker>_<n>.wav` |
| A2 | `bn_hate_audio` | Mendeley 10.17632/tgjbpp4z5b | `<label>/` folders |
| A3 | `adima_bn` | ShareChat ADIMA release | Bengali split only |
| A4 | `mutox_bn` | Meta seamless_communication (MuTox) | Bengali split only |

**Licences:** every dataset is `commercial_ok: unknown` except T1 (CC BY-NC-SA 4.0 → `false`) and
R3 (research only → `false`). The owner chose to run the prototype on `--run-tag research` without the
legal review (ADR-012): everything trained is `research_only` and cannot be promoted. Before any model
ships, set `commercial_ok` per dataset after review and rebuild with `--run-tag prod`.

## 2. Build the harmonised set (CPU is fine)

```bash
python -m ml.data.open build --run-tag research --asr-noise 0.15
python -m ml.lexicon.mine                 # → data/open/reports/lexicon_candidates.csv (human review)
```

Outputs: `data/open/unified.jsonl`, `views/textclf_{train,dev,test}.jsonl`,
`views/llm_sft_open_synthetic.jsonl`, `views/audio_manifest.jsonl` (A1–A4),
`views/translit_pairs.jsonl` (R4), `reports/open_data_card.md`, `build_stats.json`. The build exits
non-zero if any test text still has a near-duplicate in train/dev.

Once R4 is in place and the transliteration model is trained (step 3), re-run the build: it then uses
the model instead of the rule fallback for BanTH, TRAC-2 and other Roman rows.

## 3. GPU box

```bash
uv sync --group ml-train --group download        # torch, transformers, huggingface_hub
python scripts/download_models.py banglabert_large muril indicbert_v2 --allow-research-only
# a seq2seq base for transliteration (e.g. BanglaT5 or ByT5-small), placed in $GALE_MODEL_DIR/encoders/<name>

# back-transliteration on BanglaTLit
python -m ml.translit.train --base $GALE_MODEL_DIR/encoders/banglat5 --out $GALE_MODEL_DIR/translit --epochs 5
python -m ml.data.open build --run-tag research   # rebuild with the model

# open-data text classifier (stage 1): one run per encoder for the bake-off (Q-03)
for enc in banglabert_large muril indicbert_v2; do
  python -m ml.textclf.train --base $GALE_MODEL_DIR/encoders/$enc --out $GALE_MODEL_DIR/textclf_$enc --epochs 5 --batch 32
done
# model_card.json → public_aux_test.sources: BanglaMultiHate micro-F1 ≈ 0.72 and Vio-Lens macro-F1 ≈ 0.71 expected
# (docs/09 §8); more than 5 points below → debug preprocessing first. Copy the best to $GALE_MODEL_DIR/textclf.

# first KWS on BAAD (after placing A1 and rebuilding)
python -m ml.kws.train --manifest $GALE_DATA_DIR/data/open/views/audio_manifest.jsonl --epochs 40
```

CPU smoke versions (run in `make check` when the `ml-train` group is installed): `--tiny` for textclf and
translit, `--smoke` for KWS.

## 4. LLM judge baseline

`views/llm_sft_open_synthetic.jsonl` is in the docs/08 §4 chat format (system prompt = the production
`window_system.md`, user message rendered by the production code, target = mapped labels). It is the
few-shot pool and the LoRA training set for the P1 baseline; the Unsloth training recipe is P3 (docs/03 §7).
