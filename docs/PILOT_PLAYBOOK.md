# GAL-E pilot playbook (ops team)

How to run the pilot from shadow mode to rollout, read the weekly report, keep the
annotation → retraining loop going, and promote a retrained model safely.

Background: docs/06 §6 (pilot plan), docs/02 §6 (active learning), ADR-020 (design).

> **GAL-E alerts are advisory.** No alert, score or report may trigger an automatic disciplinary
> action (CLAUDE.md rule 4). Every alert can be reviewed and disputed. Before P-2, agents and their
> representatives must have the coaching-only policy in writing.

---

## 1. Roles

| Who | Does |
|---|---|
| **Pilot lead** (ops) | Owns the stage changes, signs off exit criteria, ticks the manual ones |
| **QA lead / analysts** | Daily sampling in the review queue (incl. shadow events); report misses; Label Studio annotation |
| **Supervisors** (P-2+) | Review every alert of their team; rate usefulness weekly |
| **ML engineer** | Weekly report triage, sampler, retraining runs, candidate rollout |
| **Platform admin** | Helm values, secrets, Argo, Label Studio, alert receivers |
| **DPO** | Approves the retention policy and the fairness attributes (Q-05) |

## 2. One-time setup (platform admin)

1. **Secrets.** Store `gale/pilot` in the secret store with four keys:
   - `GALE_LS_TOKEN`: Label Studio API token;
   - `GALE_SMTP_USER`, `GALE_SMTP_PASSWORD`;
   - `GALE_REPORT_WEBHOOK_URL`: chat webhook.

   Set unused keys to `""`; External Secrets requires all four.
2. **Golden manifest.** The data team creates the ConfigMap `gale-golden-manifest` in `gale-app`
   **and** `gale-ml`, with `calls.jsonl` listing `{"call_id": …, "split": "golden"}` (call ids
   only). Without it the sampler, the feedback export and the retraining **refuse to run**, by
   design (CLAUDE.md rule 3).
3. **Label Studio** (self-hosted, in-country).
   - Create the project **`GAL-E T2 active learning`** with the T2 labelling config
     (`python -m ml.data ls-templates --out …` writes `T2.xml`).
   - Add the recordings bucket as storage, so that `audio_prefix` + `recording_uri` resolves.
   - If Label Studio runs outside the cluster, add its CIDR to `pilot.egressCidrs`.
4. **Notifications.**
   - Set `pilot.config.report.notify.email.to` / `webhook.enabled`.
   - Add the SMTP relay and webhook CIDRs to `pilot.egressCidrs`.
   - Set `dashboard_url`.
5. **Argo Workflows** (controller + CRDs) in the cluster, for `retrain.enabled: true`.
   - The models PVC must be writable by the retrain pods, since `publish` writes `candidates/`.
   - Set `networkPolicies.apiServerCidrs`, so Argo's executor can report step results.
6. **Prometheus**: `pilot.prometheusUrl` must reach Prometheus, so that latency and shadow agreement
   appear in the report.

## 3. Stages

Each stage change is one Helm values change: `pilot.config`. Workers read it at start, so after
`helm upgrade` roll them:

```bash
for ns in gale-ml gale-app; do kubectl -n $ns rollout restart statefulset,deploy -l app.kubernetes.io/part-of=gal-e; done
```

| Stage | `pilot.config` | Who sees alerts | Exit criteria (checked in the weekly report) |
|---|---|---|---|
| **P-0 Offline replay** (2 wk) | `stage: P-0` | nobody (post-call batch: `POST /ingest/recordings`) | precision ≥ 0.80 on ≥ 200 QA-reviewed flags; golden targets met |
| **P-1 Shadow live** (2–3 wk) | `stage: P-1`, `alerts.default: hidden` | QA + admin only (badge **Shadow**) | latency p95 within budget; daily QA precision ≥ 0.80; no stability incidents (manual) |
| **P-2 Supervisor pilot** (4 wk) | `stage: P-2`, `alerts.visible_queues: [<pilot queues>]` | supervisors of the 2–3 pilot teams | alert acceptance ≥ 75 %; usefulness ≥ 4/5; no fairness red flags |
| **P-3 Rollout** | `stage: P-3`, `alerts.default: visible` | everyone in scope; agents get self-view | incidents per 100 calls trending down; QA coverage 100 %; time-to-intervene tracked (manual) |

Shadow-mode events are stored like any other. They are hidden from supervisors and agents in every
screen, count and live feed, and QA reviews them in **Review → Events: Shadow mode only**. Moving a
queue to visible affects new calls only; earlier events stay shadow.

**Rolling back a stage**: set the previous values and roll the workers. Nothing is deleted.

## 4. Weekly rhythm

| When | What | Who |
|---|---|---|
| Daily (P-1) | Review 20+ shadow alerts in the review queue (keyboard: J/K, 1–4) | QA |
| Any time | **Report a miss** on a call page when GAL-E missed abuse (span, speaker, category) | QA, supervisors |
| Nightly 02:15 | `gale-export-feedback`: yesterday's verdicts + misses → `datasets/feedback/…jsonl` | automatic |
| Mon 07:00 | `gale-pilot-report`: last week's report → `/admin/pilot`, email/webhook | automatic |
| Mon 08:00 | `gale-al-sampler`: 300 items (FP 30 %, FN 25 %, low-confidence 20 %, disagreement 15 %, new terms 10 %) → Label Studio | automatic |
| Mon | Report triage (section 5); annotators work through the sample | pilot lead, ML, QA |
| Fri | Supervisors rate the week's usefulness (Analytics page card) | supervisors |
| 1st of month | `gale-retrain-<component>` Argo runs; check gate results (section 6) | ML |

Run any job by hand:

```bash
kubectl -n gale-app create job --from=cronjob/gale-pilot-report pilot-report-manual
```

Locally (dev stack):

```bash
make pilot-report WEEK=2026-W44
```

```bash
make al-sample WEEK=2026-W44
```

## 5. Reading the weekly report (`/admin/pilot`)

- **Exit criteria**: ✅ met · ❌ not met · ❔ unknown (missing data, e.g. Prometheus unreachable or
  too few reviews) · 📝 manual (the pilot lead decides). A stage ends only when every criterion
  is ✅ or signed off.
- **Precision / acceptance** come with 95 % intervals. If fewer than ~100 events were reviewed,
  treat the number as indicative and increase QA sampling.
- **Drift**: PSI > 0.2 on a signal source (⚠) means its score distribution moved since last week.
  - Check for a new queue, campaign or audio path (codec, PBX change) first.
  - If ASR confidence dropped too, see `runbooks/model-rollback.md`.
- **Fairness**: a red flag means a team / queue / time-of-day / call-length slice has precision
  > 10 pts below overall, or > 1.5× the overall alert rate, with at least 30 calls.
  - Investigate before P-2/P-3: lexicon terms specific to a dialect, a noisy queue, a team with
    different scripts.
  - Agent gender and dialect slices are pending Q-05 (DPO).
- **KPIs**: QA coverage, agent incidents and customer abuse per 100 calls, repeat-offender agents
  (count only; names are in the dashboard, for coaching), disputes.
- **Shadow / canary candidates**: shadow signal counts and the agreement with production.

## 6. Retraining and promotion

1. On the 1st, each enabled component's CronWorkflow runs **prepare → train → evaluate → gate →
   publish → register**:
   ```bash
   argo -n gale-ml list
   ```
   ```bash
   argo -n gale-ml get @latest
   ```
2. **Gate.** The candidate and production are evaluated on the golden set in the same run. Any
   gated metric that gets worse by more than its tolerance fails the run:
   - 1 pt = 0.01 for proportions;
   - 1 per 100 calls for false alerts;
   - 0.25 s for LLM latency.

   The gate report is `gate.md` in the run's work volume. A failed run registers nothing; look at
   which metric regressed, and at the training data (`prepare.json`), before retrying.
3. A passed run registers the model as a **shadow candidate**: `/admin/models` shows **shadow** and
   **golden gate ✓**. Nothing changes in production yet.
4. **Shadow it**: add it to `pilot.config.candidates`, then roll the workers.
   ```yaml
   candidates:
     fusion: { version: fusion-20261201, role: shadow, percent: 100 }
   ```
   Run it for ≥ 1 week. The report's candidate section and the `GaleCandidateDisagreement` alert
   show how often it disagrees with production. Check disagreements in the data before going
   further.
5. **Canary** (optional): `role: canary, percent: 10, queues: [<one queue>]`. The routed calls
   use the candidate for real; watch precision and acceptance for those queues.
6. **Promote**: one admin clicks **Promote** in `/admin/models`, a second admin approves (two-person
   rule). Versions with a failed gate or `research_only` cannot be promoted. Then remove the
   candidate entry from `pilot.config` and roll out the new default model (`configs/models.yaml` /
   models PVC).
7. **Roll back**: `runbooks/model-rollback.md`. Promote the previous version (two-person) or
   remove the canary entry.

The LLM LoRA retrain is disabled (`retrain.components.llm_judge.enabled: false`) until its training
recipe lands (Prompt 9). The text classifier fine-tune needs the GPU trainer image
(`--build-arg TRAIN_GROUPS="--group ml-train"`).

## 7. Alerts for the ML on-call

| Alert | Meaning | Action |
|---|---|---|
| `GalePilotJobFailed` | report / sampler / export job failed | `kubectl -n gale-app logs job/<name>`. Common causes: golden manifest ConfigMap missing, Label Studio token or project missing, SMTP/webhook egress |
| `GaleCandidateDisagreement` | a candidate disagrees with production on > 30 % of outputs (≥ 50 in a day) | Sample the disagreements before canary/promotion; consider removing the candidate |
| `GaleFalsePositiveRateHigh` | reviewers mark many alerts false | `runbooks/false-alarm-spike.md` |
| `GaleAsrConfidenceDrop` | ASR confidence fell vs 7-day average | `runbooks/model-rollback.md` |

## 8. Privacy and safety checklist

- Reports, notifications and job logs hold **aggregates only**. Never paste transcript text or call
  ids into chat.
- Label Studio is self-hosted and in-country; sampled tasks carry redacted text and the recording
  path only.
- The candidate-term list (new slang) goes only to Label Studio and the lexicon team, never into git
  (CLAUDE.md rule 2).
- Golden calls never enter sampling, export or training. The jobs refuse to run without the
  manifest.
- Supervisors' usefulness comments are free text: ask them not to name customers or agents.
