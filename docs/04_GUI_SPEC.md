# 04 — GUI Specification (web dashboard)

Stack: **Next.js (App Router) + TypeScript (strict) + Tailwind CSS + shadcn/ui + TanStack Query + Zustand (live state) + wavesurfer.js (waveform, regions, stereo split) + Recharts (charts) + next-intl (bn/en)**.
Auth: OIDC (Keycloak) via `next-auth`; API JWT forwarded. Live data via WebSocket (`/ws/live`, `/ws/calls/{id}`).
Fonts: Noto Sans Bengali / Hind Siliguri for Bangla, Inter for Latin. All UI strings bilingual (default **Bangla**, toggle English). Dark & light themes. WCAG 2.1 AA (colour is never the only cue: severity also has icons + text).

---

## 1. Roles & permissions

| Role | Can see | Can do |
|---|---|---|
| `admin` | everything | users, roles, model/threshold config, lexicon, rules |
| `qa_analyst` | all calls, events, transcripts (redacted), audio | review events, annotate, export reports, edit lexicon (proposal) |
| `supervisor` | own teams' live calls & history | acknowledge alerts, listen-in link (if PBX supports), review, coach notes |
| `agent` | **only own** calls, scorecards, coaching notes | comment/dispute an event |
| `auditor` | audit log, read-only reports | — |

Every view of audio or transcript writes an `audit_log` row.

---

## 2. Screens

### 2.1 Live wallboard (`/live`) — supervisor default
- Grid/list of active calls: agent name, queue, duration, **two risk meters (agent / customer)** = rolling max fused probability over last 60 s, coloured by severity, icon for top category.
- Right panel **alert feed** (newest first): time, agent, speaker, category (Bangla label + icon), severity chip, provisional/confirmed badge, 1-line explanation (Bangla), buttons: *Open call* · *Acknowledge* · *False alarm*.
- Filters: team, queue, severity ≥, speaker (agent/customer/both), category.
- Sound/desktop notification for severity 3 (configurable). Alert de-duplication per call.
- Live KPIs header: active calls, alerts last hour, agent-caused vs customer-caused, median time-to-acknowledge.

### 2.2 Live call view (`/live/[callId]`)
- Streaming transcript in two lanes (agent left, customer right) with partial text greyed, flagged spans highlighted (colour by category, underline style by severity), register-switch markers (আপনি→তুমি ⇣).
- Real-time timelines: loudness (per speaker), anger probability, overtalk markers.
- Event stack with provisional→confirmed transitions animated; LLM explanation expands inline.
- Actions: acknowledge, add supervisor note, "flag for QA".

### 2.3 Call explorer (`/calls`) + Call detail (`/calls/[callId]`)
- Search & filter: date range, agent, team, queue, category, severity, speaker, reviewed status, score ranges, free-text search over redacted transcripts (Postgres `pg_trgm` on normalised text; OpenSearch optional later).
- Detail page:
  - **Stereo waveform** (wavesurfer.js split channels) with event regions; click region → seek & play.
  - Synchronized transcript (word-level highlight during playback), PII shown as tags.
  - Timelines: loudness z-score, anger probability, emotion stack, escalation curve, who-escalated-first marker.
  - **QA scorecard** (from LLM call-mode): professionalism, empathy, de-escalation, compliance flags, summary in Bangla + English.
  - Event table: category, speaker, severity, probability, contributing signals (expandable: KWS score, SER score, text-clf scores, LLM rationale), status, review controls.
  - Export PDF (call report) and CSV.

### 2.4 Review queue (`/review`)
- Queue of events needing human verdict (sorted by severity, then uncertainty). Keyboard-first: `J/K` next/prev, `1` correct, `2` wrong category, `3` false positive, `4` missed severity, `Space` play span ±3 s.
- Corrections write `reviews` rows → active-learning pipeline.
- Inter-reviewer agreement widget for QA leads.

### 2.5 Agent & team analytics (`/analytics`)
- Agent scorecard: events per 100 calls by category (agent-caused only), trend, comparison with team median, top coaching moments (links to spans).
- Team heatmap: hour-of-day × weekday for customer hostility and agent incidents.
- Queue view: which queues/issues drive customer hostility (e.g., recharge failure, billing).
- Fairness panel: event rates by dialect region detected/assumed and by agent gender (aggregated, admin/QA only) to catch model bias.

### 2.6 Agent self-view (`/me`)
- Own calls, own scorecard, coaching notes, ability to dispute an event (creates review item). No visibility of other agents.

### 2.7 Lexicon admin (`/admin/lexicon`)
- CRUD with variants, romanised forms, dialect, category, severity, context-sensitive flag; bulk import CSV; proposal → approval workflow; preview "matches in last 7 days".
- Terms are masked by default (e.g., first letter + •••) with a reveal toggle for authorised roles.

### 2.8 Rules & thresholds (`/admin/rules`)
- YAML editor with schema validation + form view for thresholds per category × speaker × mode; **"simulate on last N calls"** button showing precision/recall deltas vs current config before saving; versioned with rollback.

### 2.9 Models (`/admin/models`)
- Active versions per component, golden-set metrics, shadow comparisons, promote/rollback (admin only, requires second approver).

### 2.10 System health (`/admin/health`)
- Embedded Grafana panels: bus lag, GPU utilisation, per-stage latency p50/p95, error rates, alert volume.

### 2.11 Audit log (`/admin/audit`)
- Filterable, export CSV.

---

## 3. API surface (FastAPI, OpenAPI → generated TS client with `openapi-typescript`)

```
GET    /api/v1/calls?filters…                 list
GET    /api/v1/calls/{id}                     detail (meta, scores)
GET    /api/v1/calls/{id}/segments            redacted transcript
GET    /api/v1/calls/{id}/events
GET    /api/v1/calls/{id}/timelines           loudness/anger/emotion series (downsampled)
GET    /api/v1/calls/{id}/audio-url           signed URL (audited)
POST   /api/v1/events/{id}/reviews            verdict
POST   /api/v1/events/{id}/ack                acknowledge alert
GET    /api/v1/review-queue
GET    /api/v1/analytics/agents/{id} | /teams/{id} | /queues
CRUD   /api/v1/lexicon
GET/PUT /api/v1/rules  ; POST /api/v1/rules/simulate
GET    /api/v1/models ; POST /api/v1/models/{component}/promote
GET    /api/v1/audit
POST   /api/v1/ingest/recordings              upload/batch register recordings (post-call mode)
WS     /ws/live                                active calls + alerts (filtered by role/team)
WS     /ws/calls/{id}                          live transcript, signals, events for one call
GET    /healthz /readyz /metrics
```

---

## 4. UX details that matter
- Show **provisional vs confirmed** clearly; never hide that an alert came only from the fast path.
- Every flag must show **why** (evidence span + top signals + LLM rationale) — supervisors trust explainable alerts.
- Bangla category labels: গালি/অশ্লীল ভাষা, অপমান, হুমকি, যৌন হয়রানি, বিদ্বেষমূলক মন্তব্য, অসম্মানজনক সম্বোধন, রূঢ় কণ্ঠস্বর, চিৎকার, ব্যঙ্গ/উপহাস, অভদ্র আচরণ, কথার মাঝে বাধা।
- Performance: wallboard must handle 1,000 active calls at 2 updates/s without jank (virtualised list, batched WS messages every 250 ms).
- Empty/error/loading states for every view; offline banner when WS disconnects with auto-reconnect + backoff.

## 5. Front-end testing
- Unit: Vitest + Testing Library. E2E: Playwright against docker-compose stack with the call simulator replaying fixture calls. Visual regression for wallboard and call detail. Lighthouse accessibility ≥ 90.
