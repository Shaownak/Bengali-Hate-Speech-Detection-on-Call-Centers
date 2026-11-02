"""Calls, transcript, events, timelines, audio, reviews, acknowledgements and the review queue."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response
from sqlalchemy import ColumnElement, Select, and_, exists, func, select
from sqlalchemy.orm import Session

from gale_api import repo
from gale_api.api_models import (
    AckOut,
    CallDetail,
    CallPage,
    CallScoreOut,
    CallSummary,
    EventOut,
    ReviewBrief,
    ReviewIn,
    ReviewOut,
    ReviewQueueItem,
    ReviewQueuePage,
    SegmentOut,
    SignalBrief,
    SignedUrls,
    Timelines,
)
from gale_api.auth import DB, Principal, Role, require
from gale_api.db import (
    Agent,
    Call,
    CallScore,
    EventRow,
    ReviewRow,
    Segment,
    SignalRow,
    now,
)
from gale_api.scope import call_scope, get_call, get_event
from galecore.bangla import normalize
from galecore.objectstore import LocalObjectStore
from galecore.schemas import Category, EventStatus

router = APIRouter(prefix="/api/v1", tags=["calls"])

LIVE = EventRow.status != EventStatus.DISMISSED_BY_LLM.value


def _event_stats() -> Any:
    return (
        select(
            EventRow.call_id.label("call_id"),
            func.count().filter(LIVE).label("event_count"),
            func.count().filter(EventRow.alert).label("alert_count"),
            func.max(EventRow.severity).filter(LIVE).label("max_severity"),
        )
        .group_by(EventRow.call_id)
        .subquery()
    )


def _summary_query() -> tuple[Select[Any], Any]:
    st = _event_stats()
    q = (
        select(
            Call,
            Agent.employee_code,
            Agent.team_id,
            func.coalesce(st.c.event_count, 0),
            func.coalesce(st.c.alert_count, 0),
            st.c.max_severity,
        )
        .outerjoin(Agent, Agent.id == Call.agent_id)
        .outerjoin(st, st.c.call_id == Call.id)
    )
    return q, st


def _summary(row: Any) -> CallSummary:
    call, code, team, n_ev, n_al, sev = row
    return CallSummary.model_validate(call).model_copy(
        update={
            "agent_code": code,
            "team_id": team,
            "event_count": n_ev,
            "alert_count": n_al,
            "max_severity": sev,
        }
    )


def _like(text: str) -> str:
    esc = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


@router.get("/calls", response_model=CallPage)
def list_calls(
    db: DB,
    p: Annotated[Principal, require("calls.read")],
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    agent_id: uuid.UUID | None = None,
    team_id: uuid.UUID | None = None,
    queue: str | None = None,
    category: Category | None = None,
    alert: bool | None = None,
    call_status: Annotated[str | None, Query(alias="status")] = None,
    mode: str | None = None,
    min_severity: Annotated[int | None, Query(ge=1, le=3)] = None,
    speaker: Literal["agent", "customer"] | None = None,
    reviewed: bool | None = None,
    agent_score_min: Annotated[int | None, Query(ge=1, le=5)] = None,
    agent_score_max: Annotated[int | None, Query(ge=1, le=5)] = None,
    hostility_min: Annotated[int | None, Query(ge=1, le=5)] = None,
    hostility_max: Annotated[int | None, Query(ge=1, le=5)] = None,
    q: Annotated[str | None, Query(max_length=200, description="transcript search")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CallPage:
    base, st = _summary_query()
    conds: list[ColumnElement[bool]] = [call_scope(p)]
    if date_from:
        conds.append(Call.started_at >= date_from)
    if date_to:
        conds.append(Call.started_at < date_to)
    if agent_id:
        conds.append(Call.agent_id == agent_id)
    if team_id:
        conds.append(Agent.team_id == team_id)
    if queue:
        conds.append(Call.queue == queue)
    if call_status:
        conds.append(Call.status == call_status)
    if mode:
        conds.append(Call.mode == mode)
    if min_severity:
        conds.append(st.c.max_severity >= min_severity)
    if speaker:
        conds.append(exists().where(EventRow.call_id == Call.id, EventRow.speaker == speaker, LIVE))
    if reviewed is not None:
        has_verdict = exists().where(
            ReviewRow.event_id == EventRow.id,
            ReviewRow.kind == "verdict",
            EventRow.call_id == Call.id,
        )
        conds.append(has_verdict if reviewed else ~has_verdict)
    scores = (agent_score_min, agent_score_max, hostility_min, hostility_max)
    if any(v is not None for v in scores):
        sc = select(CallScore.call_id).where(CallScore.call_id == Call.id)
        if agent_score_min is not None:
            sc = sc.where(CallScore.agent_score >= agent_score_min)
        if agent_score_max is not None:
            sc = sc.where(CallScore.agent_score <= agent_score_max)
        if hostility_min is not None:
            sc = sc.where(CallScore.customer_hostility >= hostility_min)
        if hostility_max is not None:
            sc = sc.where(CallScore.customer_hostility <= hostility_max)
        conds.append(sc.exists())
    if alert is not None:
        conds.append(
            func.coalesce(st.c.alert_count, 0) > 0 if alert else st.c.alert_count.is_(None)
        )
    if category:
        conds.append(
            exists().where(EventRow.call_id == Call.id, EventRow.category == category.value, LIVE)
        )
    if q:
        needle = normalize(q)
        conds.append(
            exists().where(
                Segment.call_id == Call.id, Segment.text_norm.ilike(_like(needle), escape="\\")
            )
        )
    where = and_(*conds)
    total = db.scalar(select(func.count()).select_from(base.where(where).subquery())) or 0
    rows = db.execute(
        base.where(where).order_by(Call.started_at.desc()).limit(limit).offset(offset)
    ).all()
    return CallPage(items=[_summary(r) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/calls/{call_id}", response_model=CallDetail)
def call_detail(call_id: str, db: DB, p: Annotated[Principal, require("calls.read")]) -> CallDetail:
    call = get_call(db, p, call_id)
    base, _ = _summary_query()
    summary = _summary(db.execute(base.where(Call.id == call.id)).one())
    sc = db.get(CallScore, call.id)
    scores = (
        CallScoreOut(
            agent_score=sc.agent_score,
            customer_hostility=sc.customer_hostility,
            escalation_curve=sc.escalation_curve_jsonb,
            scorecard=sc.scorecard_jsonb,
            model_version=sc.model_version,
        )
        if sc
        else None
    )
    return CallDetail(
        **summary.model_dump(),
        channel_layout=call.channel_layout,
        language_detected=call.language_detected,
        has_recording=bool(call.recording_uri) or call.mode == "realtime",
        scores=scores,
    )


@router.get("/calls/{call_id}/segments", response_model=list[SegmentOut])
def call_segments(
    call_id: str, db: DB, p: Annotated[Principal, require("transcript.read")]
) -> list[SegmentOut]:
    call = get_call(db, p, call_id)  # audited by AuditMiddleware
    rows = db.scalars(
        select(Segment)
        .where(Segment.call_id == call.id)
        .order_by(Segment.ts_start, Segment.channel)
    ).all()
    return [
        SegmentOut(
            id=r.id,
            channel=r.channel,
            ts_start=r.ts_start,
            ts_end=r.ts_end,
            text_redacted=r.text_redacted,
            words=r.words_jsonb,
            avg_conf=r.avg_conf,
        )
        for r in rows
    ]


def events_out(db: Session, rows: list[EventRow]) -> list[EventOut]:
    if not rows:
        return []
    sig_ids = {sid for r in rows for sid in r.contributing_signal_ids}
    call_ids = {r.call_id for r in rows}
    sigs = {
        s.id: s
        for s in db.scalars(
            select(SignalRow).where(SignalRow.call_id.in_(call_ids), SignalRow.id.in_(sig_ids))
        )
    }
    reviews: dict[str, list[ReviewRow]] = {}
    for rv in db.scalars(
        select(ReviewRow)
        .where(ReviewRow.event_id.in_([r.id for r in rows]))
        .order_by(ReviewRow.created_at)
    ):
        reviews.setdefault(rv.event_id, []).append(rv)
    out = []
    for r in rows:
        top = sorted(
            (sigs[i] for i in r.contributing_signal_ids if i in sigs),
            key=lambda s: s.score,
            reverse=True,
        )[:3]
        rvs = reviews.get(r.id, [])
        verdicts = [v.verdict for v in rvs if v.kind == "verdict"]
        flagged = any(v.kind == "flag" for v in rvs)
        out.append(
            EventOut(
                **repo.event_of(r).model_dump(),
                acked_by=r.acked_by,
                acked_at=r.acked_at,
                review_count=len(rvs),
                latest_verdict=verdicts[-1] if verdicts else None,
                flagged=flagged,
                reviews=[
                    ReviewBrief(
                        kind=v.kind, verdict=v.verdict, comment=v.comment, created_at=v.created_at
                    )
                    for v in rvs
                ],
                top_signals=[
                    SignalBrief(
                        signal_id=s.id,
                        source=s.source,
                        category=s.category,
                        score=s.score,
                        ts_start=s.ts_start,
                        ts_end=s.ts_end,
                        text_span=(s.evidence_jsonb or {}).get("text_span"),
                    )
                    for s in top
                ],
            )
        )
    return out


@router.get("/calls/{call_id}/events", response_model=list[EventOut])
def call_events(
    call_id: str, db: DB, p: Annotated[Principal, require("calls.read")]
) -> list[EventOut]:
    call = get_call(db, p, call_id)
    rows = list(
        db.scalars(select(EventRow).where(EventRow.call_id == call.id).order_by(EventRow.ts_start))
    )
    return events_out(db, rows)


SERIES = {
    "dsp_shout": "loudness",
    "ser": "anger",
    "audio_tox": "audio_toxicity",
    "text_clf": "text_toxicity",
}


@router.get("/calls/{call_id}/timelines", response_model=Timelines)
def call_timelines(
    call_id: str,
    db: DB,
    p: Annotated[Principal, require("calls.read")],
    points: Annotated[int, Query(ge=10, le=2000)] = 300,
) -> Timelines:
    call = get_call(db, p, call_id)
    rows = db.execute(
        select(
            SignalRow.channel,
            SignalRow.source,
            SignalRow.ts_start,
            SignalRow.ts_end,
            SignalRow.score,
        ).where(SignalRow.call_id == call.id, SignalRow.source.in_(list(SERIES)))
    ).all()
    duration = call.duration_s or max((r.ts_end for r in rows), default=0.0)
    bin_s = max(1.0, duration / points)
    n = int(duration // bin_s) + 1
    series: dict[str, dict[str, list[float | None]]] = {}
    for ch, source, t0, t1, score in rows:
        vals = series.setdefault(ch, {}).setdefault(SERIES[source], [None] * n)
        for b in range(int(t0 // bin_s), min(n, int(t1 // bin_s) + 1)):
            cur = vals[b]
            vals[b] = score if cur is None else max(cur, score)  # downsample by max
    return Timelines(bin_s=bin_s, duration_s=duration, series=series)


@router.get("/calls/{call_id}/audio-url", response_model=SignedUrls)
def call_audio_url(
    call_id: str, request: Request, db: DB, p: Annotated[Principal, require("audio.read")]
) -> SignedUrls:
    call = get_call(db, p, call_id)  # audited by AuditMiddleware
    store = request.app.state.store
    keys: dict[str, str] = {}
    if call.recording_uri and store.exists(call.recording_uri):
        keys["stereo"] = call.recording_uri
    for ch in ("agent", "customer"):  # gateway-retained live audio (P4)
        key = f"audio/{call.id}/{ch}.wav"
        if store.exists(key):
            keys[ch] = key
    if not keys:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no audio retained for this call")
    ttl = request.app.state.settings.signed_url_ttl_s
    return SignedUrls(
        urls={
            name: _public_url(request, store, store.signed_url(k, ttl)) for name, k in keys.items()
        },
        expires_in_s=ttl,
    )


def _public_url(request: Request, store: Any, url: str) -> str:
    if isinstance(store, LocalObjectStore):  # dev: served by this API at /api/v1/objects/…
        rest = url.split("local://objects/", 1)[1]
        return str(request.base_url).rstrip("/") + "/api/v1/objects/" + rest
    return url


@router.get("/objects/{key:path}", include_in_schema=False)
def get_object(key: str, request: Request) -> Response:
    """Dev/local object download behind an HMAC-signed URL (S3 serves its own in prod)."""
    store = request.app.state.store
    if not isinstance(store, LocalObjectStore):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    try:
        verified = store.verify(f"local://objects/{key}?{request.url.query}")
    except PermissionError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid or expired link") from None
    if not store.exists(verified):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    media = "audio/wav" if verified.endswith(".wav") else "application/octet-stream"
    return Response(store.get(verified), media_type=media, headers={"Cache-Control": "no-store"})


# --- reviews & acknowledgements ----------------------------------------------------------------

VERDICT_STATUS = {
    "correct": EventStatus.REVIEWED_TRUE,
    "wrong_category": EventStatus.REVIEWED_TRUE,
    "wrong_severity": EventStatus.REVIEWED_TRUE,
    "false_positive": EventStatus.REVIEWED_FALSE,
}


@router.post("/events/{event_id}/reviews", response_model=ReviewOut, status_code=201)
def review_event(
    event_id: str, body: ReviewIn, db: DB, p: Annotated[Principal, require("calls.read")]
) -> ReviewOut:
    perm = "events.dispute" if body.kind == "dispute" else "events.review"
    if not p.can(perm):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"role {p.role} lacks {perm}")
    ev, _call = get_event(db, p, event_id)  # agents: only events of their own calls
    if body.kind == "dispute" and ev.speaker != "agent":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "agents dispute agent events")
    rv = ReviewRow(
        event_id=ev.id,
        reviewer_user_id=p.user_id,
        kind=body.kind,
        verdict=body.verdict,
        corrected_category=body.corrected_category.value if body.corrected_category else None,
        corrected_severity=body.corrected_severity,
        comment=body.comment,
        created_at=now(),
    )
    db.add(rv)
    if body.kind == "verdict" and body.verdict:
        ev.status = VERDICT_STATUS[body.verdict].value
        ev.updated_at = now()
    db.flush()
    return ReviewOut.model_validate(rv)


@router.post("/events/{event_id}/ack", response_model=AckOut)
def ack_event(event_id: str, db: DB, p: Annotated[Principal, require("events.ack")]) -> AckOut:
    ev, _call = get_event(db, p, event_id)
    if not ev.alert:
        raise HTTPException(status.HTTP_409_CONFLICT, "only alerts can be acknowledged")
    acked_by, acked_at = ev.acked_by, ev.acked_at
    if acked_by is None or acked_at is None:  # first ack wins; later acks are no-ops
        acked_by, acked_at = p.user_id, now()
        ev.acked_by, ev.acked_at = acked_by, acked_at
    return AckOut(event_id=ev.id, acked_by=acked_by, acked_at=acked_at)


@router.get("/review-queue", response_model=ReviewQueuePage)
def review_queue(
    db: DB,
    p: Annotated[Principal, require("review_queue.read")],
    category: Category | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ReviewQueuePage:
    """Alerts, severity ≥ 2 and QA-flagged events without a verdict: worst first, then the most
    uncertain (probability nearest 0.5) first."""
    reviewed = exists().where(ReviewRow.event_id == EventRow.id, ReviewRow.kind == "verdict")
    flagged = exists().where(ReviewRow.event_id == EventRow.id, ReviewRow.kind == "flag")
    conds: list[ColumnElement[bool]] = [
        call_scope(p),
        ~reviewed,
        EventRow.status.in_([EventStatus.PROVISIONAL.value, EventStatus.CONFIRMED.value]),
        (EventRow.alert | (EventRow.severity >= 2) | flagged),
        EventRow.shadow.is_(False),
    ]
    if category:
        conds.append(EventRow.category == category.value)
    q = select(EventRow).join(Call, Call.id == EventRow.call_id).where(and_(*conds))
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = list(
        db.scalars(
            q.order_by(
                EventRow.severity.desc(),
                func.abs(EventRow.probability - 0.5),
                EventRow.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
    )
    base, _ = _summary_query()
    calls = {
        r[0].id: _summary(r) for r in db.execute(base.where(Call.id.in_({e.call_id for e in rows})))
    }
    return ReviewQueuePage(
        items=[
            ReviewQueueItem(event=e, call=calls[uuid.UUID(e.call_id)]) for e in events_out(db, rows)
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


__all__ = ["Role", "router"]
