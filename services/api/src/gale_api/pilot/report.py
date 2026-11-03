"""Weekly pilot metrics report (docs/06 §6, docs/02 §6; ADR-020).

    python -m gale_api pilot-report [--week 2026-W44] [--notify] [--out DIR]

``collect`` reads one ISO week (call-centre time zone) from Postgres, ``compute`` turns it into the
report payload (pure, unit-tested), ``render_markdown`` / ``render_html`` format it. Latency and
shadow-candidate agreement come from Prometheus when ``GALE_PROMETHEUS_URL`` is set; without it
those rows read "unknown". The payload holds aggregates only: no call ids, agent ids or text.

Definitions (ADR-020):
- an event's **verdict** is its latest QA/supervisor verdict review; ``correct`` and
  ``wrong_severity`` count as true positives, ``false_positive`` and ``wrong_category`` as false;
- **precision** = TP / (TP + FP) over reviewed events, with a 95 % Wilson interval;
- **alert acceptance** = accepted (TP) alerts / reviewed alerts;
- **drift** = PSI of each signal source's score histogram against the previous week;
- **QA coverage** = ended calls with a scorecard / ended calls (GAL-E reviews every call).
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import Float, and_, cast, func, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session

from gale_api.db import (
    Agent,
    Call,
    CallScore,
    EventRow,
    MissedEvent,
    PilotFeedback,
    ReviewRow,
    Segment,
    SignalRow,
    Team,
)
from gale_api.pilot import previous_week, week_bounds
from gale_api.pilot.stats import median, per_100, psi, rate, wilson
from galecore.pilot import PilotConfig

TP = frozenset({"correct", "wrong_severity"})
FP = frozenset({"false_positive", "wrong_category"})
DISMISSED = "dismissed_by_llm"
BINS = 10
STAGES = ("provisional_event", "confirmed_event", "llm_event", "alert")


# --- data ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallFact:
    call_id: str
    started_at: datetime
    queue: str | None = None
    direction: str | None = None
    duration_s: float | None = None
    team: str | None = None
    agent: str | None = None  # only used to count repeat offenders; never reported
    ended: bool = True
    scored: bool = False


@dataclass(frozen=True)
class EventFact:
    event_id: str
    call_id: str
    speaker: str
    category: str
    severity: int
    alert: bool
    status: str
    shadow: bool = False
    probability: float = 0.0
    ack_delay_s: float | None = None


@dataclass
class WeekData:
    week: str
    calls: list[CallFact] = field(default_factory=list)
    events: list[EventFact] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)  # event id → latest verdict
    disputes: int = 0
    missed: list[tuple[str, str]] = field(default_factory=list)  # (speaker, category)
    ratings: list[int] = field(default_factory=list)
    score_hist: dict[str, list[int]] = field(default_factory=dict)  # source → BINS counts
    prev_score_hist: dict[str, list[int]] = field(default_factory=dict)
    asr_conf: float | None = None
    prev_asr_conf: float | None = None
    prev_calls: int = 0
    prev_events_by_category: dict[str, int] = field(default_factory=dict)
    candidate_signals: dict[str, int] = field(default_factory=dict)  # "<version>|shadow" → n


@dataclass
class PromData:
    latency_p95: dict[str, float] = field(default_factory=dict)
    shadow_compare: dict[str, dict[str, float]] = field(default_factory=dict)  # comp:ver → counts
    available: bool = False


# --- collection (SQL) ---------------------------------------------------------------------------


def _hist(db: Session, start: datetime, end: datetime) -> dict[str, list[int]]:
    b = func.least(func.floor(SignalRow.score * BINS), BINS - 1)
    rows = db.execute(
        select(SignalRow.source, b, func.count())
        .where(
            SignalRow.call_started_at >= start,
            SignalRow.call_started_at < end,
            SignalRow.shadow.is_(False),
        )
        .group_by(SignalRow.source, b)
    ).all()
    out: dict[str, list[int]] = {}
    for source, idx, n in rows:
        out.setdefault(source, [0] * BINS)[int(idx)] += int(n)
    return out


def _asr_conf(db: Session, start: datetime, end: datetime) -> float | None:
    v = db.scalar(
        select(func.avg(cast(Segment.avg_conf, Float))).where(
            Segment.call_started_at >= start, Segment.call_started_at < end
        )
    )
    return None if v is None else round(float(v), 4)


def collect(db: Session, cfg: PilotConfig, week: str) -> WeekData:
    start, end = week_bounds(week, cfg.report.timezone)
    pstart, pend = week_bounds(previous_week(week), cfg.report.timezone)
    in_week = and_(Call.started_at >= start, Call.started_at < end)
    data = WeekData(week=week)

    scored = select(CallScore.call_id).where(CallScore.call_id == Call.id).exists()
    for c, team, sc in db.execute(
        select(Call, Team.name, scored)
        .outerjoin(Agent, Agent.id == Call.agent_id)
        .outerjoin(Team, Team.id == Agent.team_id)
        .where(in_week)
    ):
        data.calls.append(
            CallFact(
                call_id=str(c.id),
                started_at=c.started_at,
                queue=c.queue,
                direction=c.direction,
                duration_s=c.duration_s,
                team=team,
                agent=str(c.agent_id) if c.agent_id else None,
                ended=c.status == "ended",
                scored=bool(sc),
            )
        )
    for e in db.scalars(select(EventRow).join(Call, Call.id == EventRow.call_id).where(in_week)):
        delay = (e.acked_at - e.created_at).total_seconds() if e.acked_at else None
        data.events.append(
            EventFact(
                event_id=e.id,
                call_id=str(e.call_id),
                speaker=e.speaker,
                category=e.category,
                severity=e.severity,
                alert=e.alert,
                status=e.status,
                shadow=e.shadow,
                probability=e.probability,
                ack_delay_s=delay,
            )
        )
    latest = (
        select(ReviewRow.event_id, ReviewRow.verdict)
        .join(EventRow, EventRow.id == ReviewRow.event_id)
        .join(Call, Call.id == EventRow.call_id)
        .where(in_week, ReviewRow.kind == "verdict")
        .ext(distinct_on(ReviewRow.event_id))
        .order_by(ReviewRow.event_id, ReviewRow.created_at.desc())
    )
    data.verdicts = {eid: v for eid, v in db.execute(latest) if v}
    data.disputes = (
        db.scalar(
            select(func.count(ReviewRow.id))
            .join(EventRow, EventRow.id == ReviewRow.event_id)
            .join(Call, Call.id == EventRow.call_id)
            .where(in_week, ReviewRow.kind == "dispute")
        )
        or 0
    )
    data.missed = [
        (s, c)
        for s, c in db.execute(
            select(MissedEvent.speaker, MissedEvent.category)
            .join(Call, Call.id == MissedEvent.call_id)
            .where(in_week)
        )
    ]
    data.ratings = list(db.scalars(select(PilotFeedback.rating).where(PilotFeedback.week == week)))
    data.score_hist = _hist(db, start, end)
    data.prev_score_hist = _hist(db, pstart, pend)
    data.asr_conf, data.prev_asr_conf = _asr_conf(db, start, end), _asr_conf(db, pstart, pend)
    prev = and_(Call.started_at >= pstart, Call.started_at < pend)
    data.prev_calls = db.scalar(select(func.count(Call.id)).where(prev)) or 0
    data.prev_events_by_category = {
        cat: int(n)
        for cat, n in db.execute(
            select(EventRow.category, func.count())
            .join(Call, Call.id == EventRow.call_id)
            .where(prev, EventRow.status != DISMISSED)
            .group_by(EventRow.category)
        )
    }
    versions = {c.version for c in cfg.candidates.values()}
    if versions:
        for ver, shadow, n in db.execute(
            select(SignalRow.model_version, SignalRow.shadow, func.count())
            .where(
                SignalRow.call_started_at >= start,
                SignalRow.call_started_at < end,
                SignalRow.model_version.in_(versions),
            )
            .group_by(SignalRow.model_version, SignalRow.shadow)
        ):
            data.candidate_signals[f"{ver}|{'shadow' if shadow else 'live'}"] = int(n)
    return data


# --- Prometheus ---------------------------------------------------------------------------------


class PromClient:
    """Instant queries against Prometheus' HTTP API (in-cluster, no auth)."""

    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), transport=transport, timeout=30)

    def query(self, expr: str, at: datetime) -> list[tuple[dict[str, str], float]]:
        r = self.http.get("/api/v1/query", params={"query": expr, "time": at.timestamp()})
        r.raise_for_status()
        out = []
        for item in r.json().get("data", {}).get("result", []):
            v = float(item["value"][1])
            if v == v:  # drop NaN (no samples)
                out.append((dict(item.get("metric", {})), v))
        return out


def prom_data(prom: PromClient | None, end: datetime) -> PromData:
    if prom is None:
        return PromData()
    out = PromData(available=True)
    lat = (
        "histogram_quantile(0.95, sum by (le, stage) "
        "(increase(gale_stage_latency_seconds_bucket[7d])))"
    )
    for labels, v in prom.query(lat, end):
        if labels.get("stage") in STAGES:
            out.latency_p95[labels["stage"]] = round(v, 3)
    cmp_ = "sum by (component, version, outcome) (increase(gale_shadow_compare_total[7d]))"
    for labels, v in prom.query(cmp_, end):
        key = f"{labels.get('component')}:{labels.get('version')}"
        out.shadow_compare.setdefault(key, {})[labels.get("outcome", "?")] = round(v, 1)
    return out


# --- computation (pure) -------------------------------------------------------------------------


def _prf(events: Iterable[EventFact], verdicts: dict[str, str]) -> dict[str, Any]:
    tp = fp = 0
    for e in events:
        v = verdicts.get(e.event_id)
        tp += v in TP
        fp += v in FP
    n = tp + fp
    return {"reviewed": n, "tp": tp, "fp": fp, "precision": rate(tp, n), "ci95": wilson(tp, n)}


def _daypart(dt: datetime, tz: str) -> str:
    h = dt.astimezone(ZoneInfo(tz)).hour
    return "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"


def _duration(s: float | None) -> str:
    if s is None:
        return "unknown"
    return "<2m" if s < 120 else "2-5m" if s < 300 else "5-10m" if s < 600 else ">=10m"


def _fairness(data: WeekData, cfg: PilotConfig, overall: dict[str, Any]) -> dict[str, Any]:
    r = cfg.report
    dims: dict[str, Callable[[CallFact], str]] = {
        "team": lambda c: c.team or "unassigned",
        "queue": lambda c: c.queue or "unknown",
        "direction": lambda c: c.direction or "unknown",
        "daypart": lambda c: _daypart(c.started_at, r.timezone),
        "duration": lambda c: _duration(c.duration_s),
    }
    by_call: dict[str, list[EventFact]] = defaultdict(list)
    for e in data.events:
        by_call[e.call_id].append(e)
    all_alerts = sum(1 for e in data.events if e.alert and e.status != DISMISSED)
    overall_rate = per_100(all_alerts, len(data.calls))
    flags: list[dict[str, Any]] = []
    slices: dict[str, list[dict[str, Any]]] = {}
    for dim, key in dims.items():
        groups: dict[str, list[CallFact]] = defaultdict(list)
        for c in data.calls:
            groups[key(c)].append(c)
        rows = []
        for value, calls in sorted(groups.items()):
            evs = [e for c in calls for e in by_call[c.call_id]]
            alerts = sum(1 for e in evs if e.alert and e.status != DISMISSED)
            prf = _prf(evs, data.verdicts)
            row = {
                "value": value,
                "calls": len(calls),
                "alerts_per_100": per_100(alerts, len(calls)),
                **{k: prf[k] for k in ("reviewed", "precision")},
                "flags": [],
            }
            big = len(calls) >= r.min_slice_calls
            p, op = prf["precision"], overall["precision"]
            if (
                big
                and p is not None
                and op is not None
                and prf["reviewed"] >= 10
                and p < op - r.fairness_precision_gap
            ):
                row["flags"].append("precision_gap")
            a = row["alerts_per_100"]
            if big and a is not None and overall_rate and a > r.fairness_rate_ratio * overall_rate:
                row["flags"].append("alert_rate")
            for f in row["flags"]:
                flags.append({"dimension": dim, "value": value, "flag": f})
            rows.append(row)
        slices[dim] = rows
    return {
        "slices": slices,
        "red_flags": flags,
        "note": "agent gender / dialect slices wait on Q-05 (restricted attributes)",
    }


def _drift(data: WeekData, cfg: PilotConfig) -> dict[str, Any]:
    sources = {}
    for src in sorted(set(data.score_hist) | set(data.prev_score_hist)):
        cur, prev = data.score_hist.get(src), data.prev_score_hist.get(src)
        v = psi(prev, cur) if cur and prev else None
        sources[src] = {
            "psi": v,
            "n": sum(cur or []),
            "shift": v is not None and v > cfg.report.drift_psi,
        }
    cats: dict[str, dict[str, float | None]] = {}
    now_by_cat = Counter(e.category for e in data.events if e.status != DISMISSED)
    for cat in sorted(set(now_by_cat) | set(data.prev_events_by_category)):
        cats[cat] = {
            "per_100": per_100(now_by_cat.get(cat, 0), len(data.calls)),
            "prev_per_100": per_100(data.prev_events_by_category.get(cat, 0), data.prev_calls),
        }
    return {
        "signal_psi": sources,
        "asr_confidence": {"mean": data.asr_conf, "prev_mean": data.prev_asr_conf},
        "events_per_100_calls": cats,
    }


def _exit(
    cfg: PilotConfig,
    prec: dict[str, Any],
    acc: dict[str, Any],
    prom: PromData,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    crit = cfg.exit_criteria.get(cfg.stage, {})
    out: list[dict[str, Any]] = []

    def add(name: str, target: Any, value: Any, ok: bool | None) -> None:
        status = "unknown" if ok is None else "pass" if ok else "fail"
        out.append({"criterion": name, "target": target, "value": value, "status": status})

    for name, target in crit.items():
        if name == "precision":
            v = prec["precision"]
            add(name, f">= {target}", v, None if v is None else v >= float(target))  # type: ignore[arg-type]
        elif name == "min_reviewed":
            add(name, f">= {target}", prec["reviewed"], prec["reviewed"] >= int(target))  # type: ignore[call-overload]
        elif name == "alert_acceptance":
            v = acc["acceptance"]
            add(name, f">= {target}", v, None if v is None else v >= float(target))  # type: ignore[arg-type]
        elif name == "usefulness":
            v = extra["usefulness"]
            add(name, f">= {target}", v, None if v is None else v >= float(target))  # type: ignore[arg-type]
        elif name == "fairness_red_flags":
            v = extra["red_flags"]
            add(name, f"<= {target}", v, v <= int(target))  # type: ignore[call-overload]
        elif name == "latency_p95_s" and isinstance(target, dict):
            for stage, budget in target.items():
                v = prom.latency_p95.get(stage)
                add(f"latency_p95_s.{stage}", f"<= {budget}", v, None if v is None else v <= budget)
        elif name == "manual" and isinstance(target, list):
            for item in target:
                out.append(
                    {
                        "criterion": str(item),
                        "target": "pilot lead",
                        "value": None,
                        "status": "manual",
                    }
                )
        else:
            add(name, target, None, None)
    return out


def compute(data: WeekData, cfg: PilotConfig, prom: PromData | None = None) -> dict[str, Any]:
    prom = prom or PromData()
    live = [e for e in data.events if e.status != DISMISSED]
    alerts = [e for e in live if e.alert]
    precision = _prf(data.events, data.verdicts)
    by_cat = {
        cat: _prf([e for e in data.events if e.category == cat], data.verdicts)
        for cat in sorted({e.category for e in data.events})
    }
    by_spk = {
        spk: _prf([e for e in data.events if e.speaker == spk], data.verdicts)
        for spk in ("agent", "customer")
    }
    reviewed_alerts = [e for e in alerts if data.verdicts.get(e.event_id) in TP | FP]
    accepted = sum(data.verdicts.get(e.event_id) in TP for e in reviewed_alerts)
    sev_rev = [
        data.verdicts[e.event_id] for e in data.events if data.verdicts.get(e.event_id) in TP
    ]
    acceptance = {
        "alerts": len(alerts),
        "reviewed": len(reviewed_alerts),
        "accepted": accepted,
        "acceptance": rate(accepted, len(reviewed_alerts)),
        "ci95": wilson(accepted, len(reviewed_alerts)),
        "acked": sum(e.ack_delay_s is not None for e in alerts),
        "median_time_to_ack_s": median(
            [e.ack_delay_s for e in alerts if e.ack_delay_s is not None]
        ),
    }
    ended = [c for c in data.calls if c.ended]
    agent_of = {c.call_id: c.agent for c in data.calls}
    agent_incidents = [e for e in alerts if e.speaker == "agent"]
    per_agent = Counter(agent_of.get(e.call_id) for e in agent_incidents if agent_of.get(e.call_id))
    kpis = {
        "calls": len(data.calls),
        "qa_coverage": rate(sum(c.scored for c in ended), len(ended)),
        "agent_incidents_per_100_calls": per_100(len(agent_incidents), len(data.calls)),
        "customer_abuse_per_100_calls": per_100(
            sum(e.speaker == "customer" for e in alerts), len(data.calls)
        ),
        "agents_with_incidents": len(per_agent),
        "repeat_offender_agents": sum(
            n >= cfg.report.repeat_offender_incidents for n in per_agent.values()
        ),
        "disputes": data.disputes,
        "csat_correlation": None,  # no CSAT feed yet (docs/06 §6: KPI from day 1 → needs data)
    }
    fairness = _fairness(data, cfg, precision)
    usefulness = round(sum(data.ratings) / len(data.ratings), 2) if data.ratings else None
    shadow = {
        "events": sum(e.shadow for e in data.events),
        "share": rate(sum(e.shadow for e in data.events), len(data.events)),
        "candidates": [
            {
                "component": comp,
                "version": c.version,
                "role": c.role,
                "percent": c.percent,
                "signals_shadow": data.candidate_signals.get(f"{c.version}|shadow", 0),
                "signals_live": data.candidate_signals.get(f"{c.version}|live", 0),
                "agreement": prom.shadow_compare.get(f"{comp}:{c.version}"),
            }
            for comp, c in sorted(cfg.candidates.items())
        ],
    }
    extra = {"usefulness": usefulness, "red_flags": len(fairness["red_flags"])}
    exit_ = _exit(cfg, precision, acceptance, prom, extra)
    return {
        "week": data.week,
        "stage": cfg.stage,
        "headline": {
            "calls": len(data.calls),
            "events": len(live),
            "alerts": len(alerts),
            "precision": precision["precision"],
            "alert_acceptance": acceptance["acceptance"],
            "usefulness": usefulness,
            "fairness_red_flags": len(fairness["red_flags"]),
            "missed_events": len(data.missed),
        },
        "exit": exit_,
        "precision": {
            **precision,
            "severity_accuracy": rate(sum(v == "correct" for v in sev_rev), len(sev_rev)),
            "by_category": by_cat,
            "by_speaker": by_spk,
        },
        "acceptance": acceptance,
        "latency_p95_s": prom.latency_p95 if prom.available else None,
        "drift": _drift(data, cfg),
        "fairness": fairness,
        "kpis": kpis,
        "usefulness": {"mean": usefulness, "responses": len(data.ratings)},
        "missed": dict(Counter(f"{s}/{c}" for s, c in data.missed)),
        "shadow": shadow,
    }


# --- rendering ----------------------------------------------------------------------------------

ICON = {"pass": "✅", "fail": "❌", "unknown": "❔", "manual": "📝"}


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (list, tuple)):
        return "–".join(_fmt(x) for x in v)
    return str(v)


def _table(head: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(p: dict[str, Any]) -> str:
    h = p["headline"]
    parts = [
        f"# GAL-E pilot report — {p['week']} (stage {p['stage']})",
        "",
        "Aggregates only. Alerts are advisory (CLAUDE.md rule 4).",
        "",
        "## Exit criteria",
        _table(
            ["", "criterion", "target", "value"],
            [[ICON[x["status"]], x["criterion"], x["target"], x["value"]] for x in p["exit"]],
        ),
        "",
        "## Headline",
        _table(["metric", "value"], [[k, v] for k, v in h.items()]),
        "",
        "## Precision (reviewed events)",
        f"Overall {_fmt(p['precision']['precision'])} "
        f"(95 % CI {_fmt(p['precision']['ci95'])}, n = {p['precision']['reviewed']}); "
        f"severity accuracy {_fmt(p['precision']['severity_accuracy'])}.",
        "",
        _table(
            ["category", "reviewed", "precision", "95 % CI"],
            [
                [c, v["reviewed"], v["precision"], v["ci95"]]
                for c, v in p["precision"]["by_category"].items()
            ],
        ),
        "",
        "## Alert acceptance",
        _table(
            ["alerts", "reviewed", "accepted", "acceptance", "acked", "median time-to-ack (s)"],
            [
                [
                    p["acceptance"][k]
                    for k in (
                        "alerts",
                        "reviewed",
                        "accepted",
                        "acceptance",
                        "acked",
                        "median_time_to_ack_s",
                    )
                ]
            ],
        ),
        "",
        "## Latency (p95, s)",
        _table(["stage", "p95"], [[k, v] for k, v in (p["latency_p95_s"] or {}).items()])
        if p["latency_p95_s"]
        else "Prometheus not configured (GALE_PROMETHEUS_URL): unknown.",
        "",
        "## Drift",
        _table(
            ["signal source", "PSI vs last week", "signals", "shift"],
            [
                [s, v["psi"], v["n"], "⚠" if v["shift"] else ""]
                for s, v in p["drift"]["signal_psi"].items()
            ],
        ),
        "",
        f"ASR confidence: {_fmt(p['drift']['asr_confidence']['mean'])} "
        f"(last week {_fmt(p['drift']['asr_confidence']['prev_mean'])}).",
        "",
        "## Fairness slices",
        f"Red flags: {len(p['fairness']['red_flags'])}. {p['fairness']['note']}.",
        "",
    ]
    for dim, rows in p["fairness"]["slices"].items():
        parts += [
            f"### {dim}",
            _table(
                ["value", "calls", "alerts / 100", "reviewed", "precision", "flags"],
                [
                    [
                        r["value"],
                        r["calls"],
                        r["alerts_per_100"],
                        r["reviewed"],
                        r["precision"],
                        ", ".join(r["flags"]),
                    ]
                    for r in rows
                ],
            ),
            "",
        ]
    parts += [
        "## Business KPIs",
        _table(["kpi", "value"], [[k, v] for k, v in p["kpis"].items()]),
        "",
        "## Shadow / canary candidates",
        _table(
            ["component", "version", "role", "%", "shadow signals", "live signals", "agreement"],
            [
                [
                    c["component"],
                    c["version"],
                    c["role"],
                    c["percent"],
                    c["signals_shadow"],
                    c["signals_live"],
                    json.dumps(c["agreement"]) if c["agreement"] else None,
                ]
                for c in p["shadow"]["candidates"]
            ],
        )
        if p["shadow"]["candidates"]
        else "No candidates configured.",
        "",
        f"Usefulness rating: {_fmt(p['usefulness']['mean'])} "
        f"({p['usefulness']['responses']} responses). QA-found missed events: "
        f"{sum(p['missed'].values())}.",
        "",
    ]
    return "\n".join(parts)


def render_html(p: dict[str, Any]) -> str:
    """Self-contained HTML (no external assets): the Markdown tables as HTML tables."""
    body: list[str] = []
    in_table = False
    for line in render_markdown(p).splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(line) <= {"|", "-"}:
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                body.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
                continue
            body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line:
            body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")
    style = (
        "body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;"
        "color:#111;background:#fff}table{border-collapse:collapse;margin:.5rem 0 1rem}"
        "td,th{border:1px solid #ccc;padding:.25rem .5rem;text-align:left;font-size:.9rem}"
        "th{background:#f3f4f6}h2{margin-top:2rem}"
        "@media (prefers-color-scheme:dark){body{background:#111;color:#eee}th{background:#222}"
        "td,th{border-color:#444}}"
    )
    title = html.escape(f"GAL-E pilot report {p['week']}")
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<style>{style}</style></head><body>{''.join(body)}</body></html>"
    )


def summary_text(p: dict[str, Any], link: str = "") -> str:
    """Short plain-text summary for email / chat: numbers and pass/fail only."""
    h = p["headline"]
    lines = [
        f"GAL-E pilot {p['week']} (stage {p['stage']})",
        f"calls {h['calls']} · alerts {h['alerts']} · precision {_fmt(h['precision'])} · "
        f"acceptance {_fmt(h['alert_acceptance'])} · usefulness {_fmt(h['usefulness'])} · "
        f"fairness flags {h['fairness_red_flags']}",
    ]
    lines += [
        f"{ICON[x['status']]} {x['criterion']}: {_fmt(x['value'])} ({x['target']})"
        for x in p["exit"]
    ]
    if link:
        lines.append(link)
    return "\n".join(lines)
