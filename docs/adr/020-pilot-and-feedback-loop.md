# ADR-020: Pilot tooling and the continuous-improvement loop (Prompt 15)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 15), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/06 §6 (pilot plan), docs/02 §6 (active learning), docs/03 §8–9, ADR-007, ADR-016, ADR-018

## Context
The pilot moves from offline replay (P-0) to shadow live (P-1: alerts hidden), a supervisor pilot
(P-2) and rollout (P-3). Each stage has exit criteria. Reviews must flow back into annotation and
monthly retraining, and a retrained model may only be promoted if the golden set does not regress.
The spec leaves several things open:
- where shadow/canary routing lives;
- how the metrics are defined;
- how false negatives are captured;
- which workflow engine runs the retraining;
- what "no metric regresses > 1 pt" means for metrics that are not proportions.

## Decision
1. **One pilot policy file.**
   - `configs/pilot.yaml` holds the stage, alert visibility, candidates, exit criteria and the
     report and sampler settings. Clusters get theirs from the Helm values (`pilot.config` →
     ConfigMap `gale-pilot` → `$GALE_PILOT_CONFIG`). It is a file, not a DB table, because the ML
     workers have no database access (ADR-018).
   - The repo default shows every alert (dev/e2e). Staging and production start in P-1 with
     alerts hidden; a test enforces this.
2. **Shadow mode = `shadow` flags, not separate pipelines.**
   - Fusion publishes events of calls whose queue is not visible with `FusedEvent.shadow = true`.
     Those events are persisted.
   - The API (`scope.event_scope`) hides shadow events from supervisors and agents everywhere:
     lists, counts, analytics, WebSockets, and reviews/acks (404).
   - QA and admins see them, with a badge and a review-queue filter, for P-1 daily sampling.
3. **Candidate routing is deterministic per (component, call).**
   - `sha256(component:call_id)` → [0, 100) < `percent`, optionally limited to `queues`.
   - `shadow`: the candidate runs beside the stable model. Its outputs are published as
     `Signal.shadow = true` with `~<version>` ids, which fusion never fuses. Agreement is counted
     in `gale_shadow_compare_total`: per (segment, category) for text, per decision for the LLM,
     per alert at hang-up for fusion.
   - `canary`: the candidate replaces the stable model for the routed calls.
   - Candidates exist for the three retrained components only: `text_clf`, `fusion` (calibration)
     and `llm_judge`.
   - In stub mode a candidate is the stub under its version label, so routing is tested end to end.
     Real weights load from `$GALE_MODEL_DIR/candidates/<component>/<version>/`.
4. **Metric definitions (weekly report).**
   - An event's verdict is its latest QA/supervisor verdict. `correct` and `wrong_severity` count
     as true positives; `false_positive` and `wrong_category` as false positives.
   - Precision = TP/(TP+FP), with a 95 % Wilson interval.
   - Alert acceptance = accepted reviewed alerts / reviewed alerts.
   - Drift = per-source score-histogram PSI vs the previous week (flag > 0.2), plus ASR confidence
     and events per 100 calls.
   - Fairness slices: team, queue, direction, daypart, duration. A slice is flagged when it has
     ≥ 30 calls and ≥ 10 reviews and its precision is > 0.10 below overall, or its alert rate is
     > 1.5× overall. Gender and dialect slices wait on Q-05.
   - QA coverage = ended calls with a scorecard / ended calls.
   - Latency p95 and shadow agreement come from Prometheus (`GALE_PROMETHEUS_URL`). Without it
     they are "unknown", never "pass".
   - Usefulness = mean weekly supervisor rating (new `pilot_feedback`, 1–5).
   - The report holds aggregates only (no call, agent or event ids). It is stored in
     `pilot_reports` + the object store, and optionally sent by SMTP or a Slack-compatible webhook
     as a short summary with a dashboard link.
5. **False negatives** are captured explicitly: `missed_events` (QA/supervisor "Report a miss" on
   the call page). They feed the sampler, the export and the report.
6. **Active learning.**
   - A weekly sampler fills five buckets with quotas: false positive, false negative, low
     confidence, signal disagreement (acoustic vs lexical, or LLM dismissed a strong lexicon/KWS
     hit), and new terms (tokens frequent in flagged spans, rare elsewhere in the same calls, not
     in the lexicon).
   - Limits: at most `per_call_max` items per call; unused quota flows to other buckets; the
     sample is deterministic per week.
   - Never twice: `al_samples.ref` is unique.
   - **Golden calls are excluded, and the sampler/export refuse to run without the golden
     manifest** outside dev/test (rule 3).
   - Output: Label Studio T2-format tasks with the pipeline's events as pre-annotations. The
     Label Studio client moved to `galecore.labelstudio`.
7. **Retraining.**
   - Argo Workflows: a `WorkflowTemplate` + monthly `CronWorkflow` per component, in `gale-ml`
     (no internet egress).
   - Steps: prepare (nightly feedback exports from the object store, golden dropped) → train (CPU
     calibration for fusion, GPU for text_clf; LLM LoRA disabled until Prompt 9) → evaluate
     (production and candidate on the same golden set; the candidate via a model directory where
     only its component is replaced) → gate → publish (models PVC) → register.
   - The trainer image is `infra/docker/trainer.Dockerfile`.
   - Fusion calibration is implemented for real (`ml.fusion.calibrate`: isotonic at ≥ 200
     positives and negatives, else Platt, else none; D-13). Real-time and post-call fusion now load
     `$GALE_MODEL_DIR/fusion/calibration.json` (identity when absent).
8. **Promotion gate.**
   - `configs/promotion_gate.yaml` lists the gated metrics per component, with direction and
     tolerance. "1 pt" = 0.01 for metrics in [0, 1]; false alerts per 100 calls tolerate 1.0;
     LLM p95 window latency tolerates 0.25 s.
   - Both reports must share split and `set_id` (a new fingerprint of the evaluation set).
   - A metric the candidate lacks fails; one neither side has is "n/a".
   - A passed candidate is **registered as a shadow `model_versions` row**
     (`model_versions.shadow`, migration 0002).
   - Promotion stays the two-person approval. It refuses a version whose recorded gate failed,
     and approval clears `shadow`. Nothing is promoted automatically.

## Consequences
- The pilot can run P-0 → P-3 by editing one values block: no code change, no restart beyond a
  rollout. Every stage's measurable exit criteria appear in the weekly report.
- Supervisors never see shadow-mode output. QA keeps full visibility, so P-1 precision can be
  sampled daily.
- Candidates are compared on live traffic before any promotion, without affecting events.
- Trade-offs:
  - Candidate routing changes need a worker rollout (the config is read at start).
  - The real text-classifier fine-tune and the LLM LoRA trainer still depend on the GPU prompts
    (6–10). Their pipeline steps are wired, but only stub-tested here.
  - Argo, Label Studio, SMTP/webhook and Prometheus are external dependencies the cluster owner
    provides.
