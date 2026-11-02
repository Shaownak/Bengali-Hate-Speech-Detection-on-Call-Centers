# 06 — Evaluation, Acceptance Criteria & Pilot

> No system of this kind is "perfect". The goal is **measured, high-precision alerts with known recall**, continuously improved by human review. Every number below is an *initial target*; set final targets with the QA lead after the first baseline report.

## 1. Evaluation sets (immutable, hashed)
| Set | Content | Used for |
|---|---|---|
| `golden_v1` | 300 calls, double-annotated T1+T2+T3, dialect & queue stratified, **no overlap of calls/agents/customers with training** | final acceptance, release gates |
| `dev_v1` | 300 calls, double-annotated | thresholds, calibration, stacker, model selection |
| `dialect_v1` | Ben-10 test + in-house dialect calls | robustness report |
| `public_aux` | ADIMA-bn, MuTox-bn, BLP-2025 test, SUBESCO test split | sanity / regression only |
| `negatives_v1` | 500 calls with no labelled harshness (incl. banter, loud lines, angry-but-polite) | false-alarm rate per hour |

## 2. Metrics per component
| Component | Primary metrics | Initial target (golden) |
|---|---|---|
| ASR | CER, WER (shared normaliser), slang recall, script drift, RTF, p95 latency | CER ≤ 15%, WER ≤ 30%, drift ≈ 0 |
| KWS | per-term recall, false alarms/hour | recall ≥ 0.90, FA ≤ 1/h per group |
| Shouting | F1 (event, ≥ 50% overlap) | ≥ 0.85 |
| SER | UAR {neutral, frustrated, angry}, angry AUC | UAR ≥ 0.70, AUC ≥ 0.88 |
| Text clf | macro-F1, per-category F1 | macro ≥ 0.70; slang/insult/threat ≥ 0.80 |
| LLM judge | event micro/macro-F1, JSON validity, evidence-grounding, p95 latency | validity 100%, grounding 100%, p95 ≤ 2.5 s |
| **Fused system — alerts** | precision, recall, F1 per category × speaker; false alerts per 100 calls; time-to-alert | **agent sev-2/3 alerts: precision ≥ 0.85, recall ≥ 0.75**; customer abuse alerts precision ≥ 0.80; < 1 false alert / 100 calls |
| Fused system — call level | Spearman ρ between system scores and QA-lead T3 scores; AUC for "call needs QA review" | ρ ≥ 0.6, AUC ≥ 0.85 |
| Latency | provisional / confirmed alert p95 | ≤ 2 s / ≤ 6 s |

Event matching rule: predicted event matches a gold span if same speaker, same category (or category-group), temporal IoU ≥ 0.3 or centre within gold span ± 2 s.

## 3. Fairness & robustness report (required for every release)
- Metrics sliced by: dialect region, agent gender, customer gender (if annotated), queue, handset/codec, call duration bucket, time of day.
- Flag any slice where false-positive rate > 1.5× overall → investigate before release.
- Specific risks: dialect words mistaken for slang; regional pronoun habits flagged as disrespect; louder speaking styles flagged as shouting (mitigated by per-speaker baseline).

## 4. Bake-off protocol (model selection)
1. Freeze `dev_v1`/`golden_v1`.
2. For each component run candidates with identical preprocessing; log to MLflow with tag `bakeoff=<component>`.
3. Choose by primary metric subject to latency/cost constraints; tie-break by robustness slices.
4. Write ADR with the table of results.

## 5. Test pyramid for software
- Unit: schemas, normaliser (Bangla edge cases), redaction regexes (Bangla digits, spoken digits), lexicon matcher, register rules, fusion rules, hysteresis.
- Contract: each service consumes/produces valid schema messages (fixtures in `tests/fixtures/messages/`).
- Integration: docker-compose stack + `tools/simulator` replaying **synthetic** fixture calls (TTS-generated Bangla dialogues, including scripted slang placeholders and shouting via gain/pitch) → assert expected events within latency budget.
- E2E UI: Playwright.
- Load: k6/locust at 2× expected peak.
- Chaos: kill a worker / GPU pod during load; assert no lost events (at-least-once + idempotency).

## 6. Pilot plan
| Stage | Duration | Scope | Exit criteria |
|---|---|---|---|
| P-0 Offline replay | 2 weeks | last month's calls, post-call mode | golden targets met; QA lead reviews 200 random flags → precision ≥ 0.8 |
| P-1 Shadow live | 2–3 weeks | 1–2 queues real-time, **alerts hidden** | latency SLOs met; daily precision from QA sampling ≥ 0.8; no stability incidents |
| P-2 Supervisor pilot | 4 weeks | 2–3 teams see alerts; review every alert | alert acceptance ≥ 75%; supervisors rate usefulness ≥ 4/5; no fairness red flags |
| P-3 Rollout | gradual | all queues; agents get self-view | KPIs: incidents per 100 calls trend, QA coverage 100% vs prior sample-based %, time-to-intervene |

Business KPIs to track from day 1: % calls QA-covered, agent incidents per 100 calls, customer-abuse incidents (agent welfare), repeat-offender agents coached, CSAT/complaint correlation.
