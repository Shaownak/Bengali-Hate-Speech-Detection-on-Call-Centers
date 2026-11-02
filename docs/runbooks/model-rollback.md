# Runbook: model rollback (`GaleAsrConfidenceDrop`, drift, a bad promotion)

Promotion needs two admins (`/admin/models`, ADR-016); rollback uses the same flow, pointed at
the previous version.

1. `/admin/models` → the component → the previously active version → **Request promotion**.
   A second admin approves. `research_only` versions can never be promoted (ADR-012).
2. Deploy the version: real-model pods read `configs/models.yaml` and the weights in the models PVC
   at start. Update `models.yaml` to the rolled-back revision (pinned commit hash, CLAUDE.md rule
   6) in a PR, then `helm upgrade` (restarts the affected StatefulSet shards).
3. Verify on the ML dashboard: ASR confidence, signal score medians and review FP rate return to
   their 7-day baselines within an hour.
4. Write down what broke in the model card / MLflow run, and keep the bad version for the
   post-mortem eval (`make eval` on the dev split; never tune on golden, CLAUDE.md rule 3).
