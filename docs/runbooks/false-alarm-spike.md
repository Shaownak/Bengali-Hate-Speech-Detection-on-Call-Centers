# Runbook: false-alarm spike (`GaleFalsePositiveRateHigh`, `GaleAlertVolumeAnomaly`)

**Impact:** supervisors lose trust in alerts. Alerts are advisory (CLAUDE.md rule 4), so no
automatic action was taken against anyone. Still, act the same day.

1. **Scope:** Grafana **ML** dashboard → review verdicts, alert acceptance, signal score medians.
   Grafana **Business** → which category or speaker drives the spike.
2. **Typical causes**
   - A lexicon entry that matches a common word in a new context. Find it in `/admin/lexicon`
     ("matches in last 7 days") and deactivate or mark it context-sensitive.
   - A rules or threshold change. Compare `/admin/rules` history, and roll back to the previous
     version (one click; the simulator shows the delta first).
   - Model drift after promotion: model-rollback.md.
   - A new queue or product name the ASR mishears (check ASR confidence on the ML dashboard).
3. **Contain:** raise the alert threshold for the affected category in `configs/thresholds.yaml`
   (reviewed PR), or switch the category to log-only. Never delete events: reviewers' verdicts
   feed retraining.
4. **Follow-up:** add the reviewed false positives to the next annotation batch (data engine), and
   add a regression fixture with a placeholder term (`<GALI_n>`).
