"""Pilot & feedback-loop endpoints (docs/06 §6, docs/02 §6; ADR-020).

- ``GET /pilot/status``: stage, alert visibility, candidates, exit criteria (from the pilot config);
- ``GET /pilot/reports`` · ``/pilot/reports/{week}`` · ``…/{week}/html``: weekly reports;
- ``GET|POST /pilot/feedback``: a supervisor's usefulness rating for the current week (P-2 exit);
- ``GET|POST /calls/{id}/missed``: QA reports abuse GAL-E missed (false negatives, AL sampler).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from gale_api.api_models import (
    FeedbackIn,
    FeedbackOut,
    MissedEventIn,
    MissedEventOut,
    PilotCandidateOut,
    PilotReportBrief,
    PilotReportOut,
    PilotStatus,
)
from gale_api.auth import DB, Principal, require
from gale_api.db import MissedEvent, PilotFeedback, PilotReport
from gale_api.pilot import current_week, parse_week
from gale_api.scope import get_call
from galecore.pilot import PilotConfig, load_pilot, pilot_path

router = APIRouter(prefix="/api/v1", tags=["pilot"])


def _cfg(request: Request) -> PilotConfig:
    return load_pilot(pilot_path(request.app.state.settings))


@router.get("/pilot/status", response_model=PilotStatus)
def pilot_status(request: Request, p: Annotated[Principal, require("pilot.read")]) -> PilotStatus:
    cfg = _cfg(request)
    return PilotStatus(
        stage=cfg.stage,
        alerts_default=cfg.alerts.default,
        visible_queues=cfg.alerts.visible_queues,
        candidates=[
            PilotCandidateOut(component=k, **c.model_dump())
            for k, c in sorted(cfg.candidates.items())
        ],
        exit_criteria={k: dict(v) for k, v in cfg.exit_criteria.items()},
        week=current_week(datetime.now(UTC), cfg.report.timezone),
    )


def _brief(r: PilotReport) -> PilotReportBrief:
    return PilotReportBrief(
        week=r.week,
        stage=r.stage,
        created_at=r.created_at,
        exit=r.payload_jsonb.get("exit", []),
        headline=r.payload_jsonb.get("headline", {}),
        has_html=bool(r.html_key),
    )


@router.get("/pilot/reports", response_model=list[PilotReportBrief])
def list_reports(db: DB, p: Annotated[Principal, require("pilot.read")]) -> list[PilotReportBrief]:
    rows = db.scalars(select(PilotReport).order_by(PilotReport.week.desc()).limit(104))
    return [_brief(r) for r in rows]


def _report(db: DB, week: str) -> PilotReport:
    try:
        parse_week(week)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    row = db.scalar(select(PilotReport).where(PilotReport.week == week))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no report for that week")
    return row


@router.get("/pilot/reports/{week}", response_model=PilotReportOut)
def get_report(week: str, db: DB, p: Annotated[Principal, require("pilot.read")]) -> PilotReportOut:
    r = _report(db, week)
    return PilotReportOut(**_brief(r).model_dump(), payload=r.payload_jsonb)


@router.get("/pilot/reports/{week}/html", response_class=HTMLResponse)
def get_report_html(
    week: str, request: Request, db: DB, p: Annotated[Principal, require("pilot.read")]
) -> HTMLResponse:
    r = _report(db, week)
    store = request.app.state.store
    if not r.html_key or not store.exists(r.html_key):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "HTML report not stored")
    return HTMLResponse(
        store.get(r.html_key).decode("utf-8"),
        headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"},
    )


@router.get("/pilot/feedback", response_model=FeedbackOut | None)
def my_feedback(
    request: Request, db: DB, p: Annotated[Principal, require("pilot.feedback")]
) -> PilotFeedback | None:
    week = current_week(datetime.now(UTC), _cfg(request).report.timezone)
    return db.scalar(
        select(PilotFeedback).where(PilotFeedback.user_id == p.user_id, PilotFeedback.week == week)
    )


@router.post("/pilot/feedback", response_model=FeedbackOut)
def rate_week(
    body: FeedbackIn, request: Request, db: DB, p: Annotated[Principal, require("pilot.feedback")]
) -> PilotFeedback:
    """One rating per user and week; posting again replaces it."""
    week = current_week(datetime.now(UTC), _cfg(request).report.timezone)
    row = db.scalar(
        select(PilotFeedback).where(PilotFeedback.user_id == p.user_id, PilotFeedback.week == week)
    )
    if row is None:
        row = PilotFeedback(user_id=p.user_id, week=week)
        db.add(row)
    row.rating, row.comment, row.created_at = body.rating, body.comment, datetime.now(UTC)
    db.flush()
    return row


@router.post("/calls/{call_id}/missed", response_model=MissedEventOut, status_code=201)
def report_missed(
    call_id: str,
    body: MissedEventIn,
    db: DB,
    p: Annotated[Principal, require("events.report_missed")],
) -> MissedEvent:
    call = get_call(db, p, call_id)
    if call.duration_s is not None and body.ts_start > call.duration_s:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "span starts after the call ends"
        )
    row = MissedEvent(
        id=uuid.uuid4(),
        call_id=call.id,
        reporter_user_id=p.user_id,
        speaker=body.speaker,
        category=body.category.value,
        severity=body.severity,
        ts_start=body.ts_start,
        ts_end=body.ts_end,
        comment=body.comment,
        created_at=datetime.now(UTC),
    )
    db.add(row)
    db.flush()
    return row


@router.get("/calls/{call_id}/missed", response_model=list[MissedEventOut])
def list_missed(
    call_id: str, db: DB, p: Annotated[Principal, require("calls.read")]
) -> list[MissedEvent]:
    call = get_call(db, p, call_id)
    return list(
        db.scalars(
            select(MissedEvent).where(MissedEvent.call_id == call.id).order_by(MissedEvent.ts_start)
        )
    )
