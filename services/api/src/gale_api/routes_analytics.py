"""Agent, team and queue analytics (docs/04 §2.5–2.6). Alerts count non-dismissed alert events."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import ColumnElement, and_, cast, func, select
from sqlalchemy.types import Date

from gale_api.api_models import (
    AgentAnalytics,
    AgentRow,
    CategoryCount,
    DayPoint,
    QueueRow,
    TeamAnalytics,
)
from gale_api.auth import DB, Principal, Role, require
from gale_api.db import Agent, Call, CallScore, EventRow, ReviewRow, Team
from gale_api.scope import call_scope, can_see_agent, can_see_team

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

AnalyticsUser = Annotated[Principal, require("analytics.read")]


def _window(date_from: datetime | None, date_to: datetime | None) -> list[ColumnElement[bool]]:
    out: list[ColumnElement[bool]] = []
    if date_from:
        out.append(Call.started_at >= date_from)
    if date_to:
        out.append(Call.started_at < date_to)
    return out


def _alerts_by_call() -> Any:
    return (
        select(EventRow.call_id, func.count().label("n"))
        .where(EventRow.alert, EventRow.status != "dismissed_by_llm")
        .group_by(EventRow.call_id)
        .subquery()
    )


def _scope_for_reports(p: Principal) -> ColumnElement[bool]:
    # auditors read aggregate reports across the org, never call content
    return call_scope(p) if p.role != Role.AUDITOR else Call.id.is_not(None)


@router.get("/agents/{agent_id}", response_model=AgentAnalytics)
def agent_analytics(
    agent_id: uuid.UUID,
    db: DB,
    p: AnalyticsUser,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> AgentAnalytics:
    agent = db.get(Agent, agent_id)
    if agent is None or not can_see_agent(db, p, agent_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "agent not found")
    where = and_(Call.agent_id == agent_id, *_window(date_from, date_to))
    al = _alerts_by_call()
    calls, minutes, alerts = db.execute(
        select(
            func.count(Call.id),
            func.coalesce(func.sum(Call.duration_s), 0.0) / 60.0,
            func.coalesce(func.sum(al.c.n), 0),
        )
        .outerjoin(al, al.c.call_id == Call.id)
        .where(where)
    ).one()
    by_cat = db.execute(
        select(EventRow.category, EventRow.speaker, func.count())
        .join(Call, Call.id == EventRow.call_id)
        .where(where, EventRow.status != "dismissed_by_llm")
        .group_by(EventRow.category, EventRow.speaker)
        .order_by(func.count().desc())
    ).all()
    prof = db.scalar(
        select(func.avg(CallScore.agent_score))
        .join(Call, Call.id == CallScore.call_id)
        .where(where)
    )
    disputes = db.scalar(
        select(func.count(ReviewRow.id))
        .join(EventRow, EventRow.id == ReviewRow.event_id)
        .join(Call, Call.id == EventRow.call_id)
        .where(where, ReviewRow.kind == "dispute")
    )
    day = cast(Call.started_at, Date)
    trend = db.execute(
        select(day, func.count(Call.id), func.coalesce(func.sum(al.c.n), 0))
        .outerjoin(al, al.c.call_id == Call.id)
        .where(where)
        .group_by(day)
        .order_by(day)
    ).all()
    return AgentAnalytics(
        agent_id=agent.id,
        employee_code=agent.employee_code,
        team_id=agent.team_id,
        calls=calls,
        talk_minutes=round(float(minutes), 2),
        alerts=int(alerts),
        events_by_category=[CategoryCount(category=c, speaker=s, count=n) for c, s, n in by_cat],
        avg_professionalism=round(float(prof), 2) if prof is not None else None,
        disputes=disputes or 0,
        trend=[DayPoint(day=d.isoformat(), calls=c, alerts=int(a)) for d, c, a in trend],
    )


@router.get("/teams/{team_id}", response_model=TeamAnalytics)
def team_analytics(
    team_id: uuid.UUID,
    db: DB,
    p: AnalyticsUser,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> TeamAnalytics:
    team = db.get(Team, team_id)
    if team is None or not can_see_team(p, team_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "team not found")
    al = _alerts_by_call()
    sc = select(CallScore.call_id, CallScore.agent_score).subquery()
    rows = db.execute(
        select(
            Agent.id,
            Agent.employee_code,
            func.count(Call.id),
            func.coalesce(func.sum(al.c.n), 0),
            func.avg(sc.c.agent_score),
        )
        .outerjoin(Call, and_(Call.agent_id == Agent.id, *_window(date_from, date_to)))
        .outerjoin(al, al.c.call_id == Call.id)
        .outerjoin(sc, sc.c.call_id == Call.id)
        .where(Agent.team_id == team_id)
        .group_by(Agent.id, Agent.employee_code)
        .order_by(Agent.employee_code)
    ).all()
    agents = [
        AgentRow(
            agent_id=a,
            employee_code=code,
            calls=n,
            alerts=int(alerts),
            avg_professionalism=round(float(prof), 2) if prof is not None else None,
        )
        for a, code, n, alerts, prof in rows
    ]
    return TeamAnalytics(
        team_id=team.id,
        name=team.name,
        calls=sum(a.calls for a in agents),
        alerts=sum(a.alerts for a in agents),
        agents=agents,
    )


@router.get("/queues", response_model=list[QueueRow])
def queue_analytics(
    db: DB,
    p: AnalyticsUser,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[QueueRow]:
    if p.role == Role.AGENT:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "agents see only their own analytics")
    ev = (
        select(
            EventRow.call_id,
            func.count().filter(EventRow.speaker == "agent").label("agent_alerts"),
            func.count().filter(EventRow.speaker == "customer").label("customer_alerts"),
        )
        .where(EventRow.alert, EventRow.status != "dismissed_by_llm")
        .group_by(EventRow.call_id)
        .subquery()
    )
    qname = func.coalesce(Call.queue, "(none)")
    rows = db.execute(
        select(
            qname,
            func.count(Call.id),
            func.coalesce(func.sum(ev.c.agent_alerts), 0),
            func.coalesce(func.sum(ev.c.customer_alerts), 0),
            func.avg(CallScore.customer_hostility),
        )
        .outerjoin(ev, ev.c.call_id == Call.id)
        .outerjoin(CallScore, CallScore.call_id == Call.id)
        .where(_scope_for_reports(p), *_window(date_from, date_to))
        .group_by(qname)
        .order_by(qname)
    ).all()
    return [
        QueueRow(
            queue=q,
            calls=n,
            agent_alerts=int(a),
            customer_alerts=int(c),
            avg_customer_hostility=round(float(h), 2) if h is not None else None,
        )
        for q, n, a, c, h in rows
    ]
