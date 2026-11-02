# ADR-014: Evaluation harness choices (Prompt 5)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 5), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/06 §1–4; docs/03 §1.1, §9; D-13; CLAUDE.md rule 3

## Context
docs/06 lists metrics per component but leaves open how some of them apply to a post-call system: what counts as an "alert" match, what the fairness "false-positive rate" is per slice, and how call-level scores are defined before the stacker exists.

## Decision
1. **Same code path.** Every evaluator runs the production pipeline (`gale_postcall` models + worker logic). Eval sets come in one shape: synthetic fixtures (CI) or a data-engine manifest split plus converted annotations.
2. **Gold "alert-worthy".**
   - Gold events with a context flag (reported speech, self-directed, banter, quoted policy) should not alert. The system passes if they are dismissed or absent (`context_negatives`).
   - Fixtures use their `alert` field.
3. **Fusion vs system.**
   - `fusion` scores live events (not dismissed) against alert-worthy gold with the docs/06 matching rule.
   - `system` adds alert-level precision/recall, the docs/06 headline slices (agent severity 2–3 alerts, customer abuse alerts), false alerts per 100 calls, call-level metrics and processing real-time factor.
   - Alerts suppressed by the cool-down count as alert misses on purpose: the number shows what a supervisor would actually see.
4. **Call level (until the stacker).**
   - "Needs QA review" = any alert-worthy agent event, or a T3 professionalism score ≤ 2.
   - The system score is the highest agent event probability plus half the highest overall.
   - Spearman ρ compares the system severity sum with the negated T3 professionalism (or the gold severity sum when no T3 exists).
5. **Fairness.** The per-slice "false-positive rate" is false alerts per 100 calls in the slice, flagged above 1.5× the overall rate when the slice has at least 2 calls. Slices: dialect, agent/customer gender, queue, codec, duration bucket, time of day.
6. **Confidence intervals.** Bootstrap 95 % CIs resample calls (D-13). Resamples with no events are skipped (F1 is undefined there), not scored as 0.
7. **Shouting** uses the docs/06 "≥ 50 % of the gold span covered" rule. SER and audio-tox are scored on 3 s speech windows (AUC, UAR).
8. **Golden.** A `golden` split loads only when `golden.lock` still matches. Evaluating on golden is allowed; tuning on it is not (rule 3), and no tool writes thresholds from eval output.
9. **Outputs.** JSON + Markdown + HTML (reliability diagram) in `reports/` (git-ignored). MLflow logging happens when `mlflow` is installed and `MLFLOW_TRACKING_URI` is set. The bake-off runner copies `configs/` per candidate with `models.yaml` overridden, and writes the ADR results table.

## Consequences
- Positive: one command per component (`make eval`, `make bakeoff`), and CI exercises the whole harness on synthetic calls.
- Negative:
  - Fixture numbers are near-perfect by construction (stub models + oracle); they validate the harness, not the models.
  - KWS and audio-tox stubs score 0 recall / AUC 0.5 until P3.
  - Latency is post-call processing time; real-time alert latency comes with P4 streaming.
