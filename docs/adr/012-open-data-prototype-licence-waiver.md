# ADR-012: Open-data track runs as a research prototype without the licence review

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Shaownak Shahriar (project owner): "ignore the license term. the system is prototype work."
- **Spec refs:** docs/09 header (licence gate) and §6–8; prompts Prompt 4b (🔒 licence sign-off); Q-03, Q-04; CLAUDE.md "irreversible (licence) → ask"

## Context
Prompt 4b is marked as blocked on licence sign-off. Almost every dataset in docs/09 is `VERIFY`. One release ships a non-commercial licence (BanglaMultiHate: CC BY-NC-SA 4.0), and one is research-only (MixTox-SA). The owner decided not to wait for a legal review, because GAL-E is prototype work at this stage.

## Decision
- **Licence gate (unchanged).** It is implemented as specified:
  - `ml/data/datasets.yaml` records `commercial_ok: true | false | unknown` (default `unknown`) for all 23 datasets.
  - Runs tagged `prod` refuse anything not `true`.
- **Research tag.** The prototype runs with `--run-tag research`:
  - every dataset may be used;
  - every row and every model card carries `research_only: true` / `promotable: false`.
- **No promotion.** Nothing trained this way may be promoted to production; the model registry check comes with P3 promotion (docs/03 §9).
- **Licence status recorded as found.**
  - T1 `false` (CC BY-NC-SA 4.0 legal code shipped in the repo).
  - R3 `false` (research/education only); also `eval_only`.
  - All others `unknown`.
  - Nothing is marked `true` without a review.
- **Real abusive text stays out of the repo** (CLAUDE.md rule 2). Raw data, `unified.jsonl` and the lexicon candidate CSV live under `.gale/` (git-ignored). Tools print counts only.

## Consequences
- Positive:
  - The open-data track runs now.
  - Switching to production later is a config change plus a rebuild (`commercial_ok: true` per reviewed dataset, `--run-tag prod`), with no code change.
- Negative:
  - Models trained in this phase cannot ship.
  - If a reviewed licence forbids even research use, those rows and any models trained on them must be deleted.
- Follow-ups:
  - Legal review of each dataset (Q-04) and of the encoders (Q-03, `ml/MODEL_LICENSES.md`) before the first production training run.
