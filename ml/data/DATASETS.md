# Open-dataset register

Generated from `ml/data/datasets.yaml` by `python -m ml.data.open register --write-md`; edit the YAML.

- `commercial_ok`: `true` | `false` | `unknown`. **Runs tagged `prod` refuse anything not `true`.** `research` runs may use every dataset; their outputs are tagged `research_only` and blocked from model promotion (docs/09, ADR-012).
- `download`: `yes` = fetched by `python -m ml.data.open download`; `manual` = place the files under `$GALE_DATA_DIR/data/raw/<key>/`.
- `format: VERIFY` = column names come from the paper/README; check them when the files are first placed (the loader names the columns it found if they don't match).

| id | key | dataset | source | licence | commercial_ok | download | files | format |
|---|---|---|---|---|---|---|---|---|
| T1 | `banglamultihate` | BanglaMultiHate / BLP-2025 Task 1 (subtask 1C) | [link](https://github.com/AridHasan/blp25_task1) | CC BY-NC-SA 4.0 (legal code shipped in the repo) | **false** | yes | train: `1C_train.tsv`, dev: `1C_dev.tsv`, test: `1C_test_with_labels.tsv` | tsv-unquoted |
| T2 | `bdshs` | BD-SHS (LREC 2022) | [link](https://www.kaggle.com/datasets/naurosromim/bdshs) | MIT (code repo); data licence VERIFY on Kaggle | **unknown** | manual | train: `train.csv`, dev: `val.csv`, test: `test.csv` | VERIFY |
| T3 | `banglacat` | BanglaCAT (2026) | [10.17632/8vr3w68c7x](https://data.mendeley.com/datasets/8vr3w68c7x) | VERIFY | **unknown** | manual | train: `train.*`, test: `test.*` | VERIFY |
| T4 | `bidwesh` | BIDWESH (2025) — BD-SHS in Barishal / Noakhali / Chittagong dialects | [link](https://arxiv.org/abs/2507.16183) | VERIFY | **unknown** | manual | all: `*.csv` | VERIFY |
| T5 | `cyberbully_multilabel` | Bangla Multilabel Cyberbully / Sexual Harassment / Threat / Spam (2024) | [10.17632/sz5558wrd4.3](https://data.mendeley.com/datasets/sz5558wrd4/3) | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| T6 | `violens` | Vio-Lens / BLP-2023 Task 1 | [link](https://github.com/blp-workshop/blp_task1) | VERIFY (no licence file in the repo) | **unknown** | yes | train: `train.csv`, dev: `dev.csv`, test: `test.csv` | csv |
| T7 | `trac2_bn` | TRAC-2 Bengali (LREC 2020) | [link](https://sites.google.com/view/trac2/shared-task) | VERIFY (shared-task release terms) | **unknown** | manual | train: `*train*.csv`, dev: `*dev*.csv`, test: `*test*.csv` | VERIFY |
| T8 | `karim_bn_hate` | Bengali Hate Speech Dataset (Karim et al. 2020) | [link](https://huggingface.co/datasets/rezacsedu/bn_hate_speech) | MIT (HF card) / CC BY 4.0 (UCI | **unknown** | yes | train: `train.parquet` | parquet |
| T9 | `romim2020` | Romim et al. 2020 Bengali hate speech | [link](https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset) | VERIFY | **unknown** | manual | all: `*.csv` | VERIFY |
| T10 | `hsban` | HS-BAN (2021) | [link](https://arxiv.org/abs/2112.01902) | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| T11 | `brac_online_comments` | Bangla Online Comments (BRAC, 2021) | [10.17632/9xjx8twk8p](https://data.mendeley.com/datasets/9xjx8twk8p) | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| T12 | `banglaharassnet` | BanglaHarassNet (2026) | [10.17632/cw6jrvxmhf.1](https://data.mendeley.com/datasets/cw6jrvxmhf/1) | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| T13 | `bangla_toco` | Bangla-ToCo (2025) | DIU / paper | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| T14 | `boishommo` | BOISHOMMO (2025) | [link](https://arxiv.org/abs/2504.08408) | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| T15 | `hatealert_bengali` | Bengali Hate (Das et al., AACL 2022) | [link](https://github.com/hate-alert/Bengali_Hate) | VERIFY | **unknown** | manual | all: `*hydrated*.csv` | VERIFY |
| R1 | `banth` | BanTH (NAACL Findings 2025) | [link](https://arxiv.org/abs/2410.13281) | VERIFY | **unknown** | manual | train: `*train*.*`, dev: `*dev*.*`, test: `*test*.*` | VERIFY |
| R2 | `codemixed_chaos` | Code-mixed Chaos (2025) — Bangla-translated version | [10.17632/23dp3t88vk](https://data.mendeley.com/datasets/23dp3t88vk) | VERIFY | **unknown** | manual | all: `*.*` | VERIFY |
| R3 | `mixtox_sa` | MixTox-SA (2025) | [10.17632/bsdcd7zkyz](https://data.mendeley.com/datasets/bsdcd7zkyz) | research / education only (dataset page) | **false** | manual | all: `*.*` | VERIFY |
| R4 | `banglatlit` | BanglaTLit (2024) — romanised ↔ Bangla pairs | [link](https://aclanthology.org/2024.findings-emnlp.859/) | VERIFY | **unknown** | manual | train: `*train*.*`, dev: `*dev*.*`, test: `*test*.*` | VERIFY |
| A1 | `baad` | BAAD v3 (Bangla Abusive Audio Dataset) | [10.17632/w24g8xn23c.3](https://data.mendeley.com/datasets/w24g8xn23c/3) | CC BY 4.0 (confirm in legal review) | **unknown** | manual | all: `**/*.wav` | wav-folders |
| A2 | `bn_hate_audio` | Bangla Hate Speech & Non-Hate Audio (2025) | [10.17632/tgjbpp4z5b](https://data.mendeley.com/datasets/tgjbpp4z5b) | VERIFY | **unknown** | manual | all: `**/*.wav` | wav-folders |
| A3 | `adima_bn` | ADIMA (Bengali split) | [link](https://sharechat.com/research/adima) | research — VERIFY | **unknown** | manual | all: `**/*.wav` | wav-folders |
| A4 | `mutox_bn` | MuTox (Bengali split) | [link](https://github.com/facebookresearch/seamless_communication) | VERIFY | **unknown** | manual | all: `**/*.wav` | wav-folders |
| S1 | `ood_speech` | OOD-Speech (Bengali.AI) — ~1,178 h train + 23 h OOD test | [link](https://huggingface.co/datasets/bengaliAI/OOD-Speech) | TO VERIFY | **unknown** | manual | all: `**/*` | VERIFY |
| S2 | `indicvoices_bn` | IndicVoices (AI4Bharat) Bengali — ~600 h spontaneous | [link](https://ai4bharat.iitm.ac.in/indicvoices) | TO VERIFY | **unknown** | manual | all: `**/*` | VERIFY |
| S3 | `openslr53` | OpenSLR-53 Large Bengali ASR — ~211 h | [link](https://www.openslr.org/53/) | TO VERIFY (open per docs/02) | **unknown** | manual | all: `**/*` | VERIFY |
| S4 | `commonvoice_bn` | Common Voice Bengali — 400 h+ | [link](https://commonvoice.mozilla.org/bn/datasets) | TO VERIFY (CC0 per docs/02) | **unknown** | manual | all: `**/*` | VERIFY |
| S5 | `ben10` | Ben-10 — 78 h, 10 dialects | [link](https://huggingface.co/datasets/bengaliAI/ben10) | TO VERIFY | **unknown** | manual | all: `**/*` | VERIFY |
| S6 | `buet_telecom_speech` | Bengali telecom customer-care speech (BUET, 2026) — synthetic, 26.8 h | TO VERIFY | TO VERIFY (CC BY 4.0 per docs/02) | **unknown** | manual | all: `**/*` | VERIFY |
| S7 | `subesco` | SUBESCO — 7,000 utterances, 7 emotions | [link](https://zenodo.org/records/4526477) | TO VERIFY (CC BY 4.0 per docs/02) | **unknown** | manual | all: `**/*.wav` | VERIFY |
| S8 | `banglaser` | BanglaSER / KBES / BanSpEmo — small SER sets | TO VERIFY | TO VERIFY | **unknown** | manual | all: `**/*.wav` | VERIFY |
