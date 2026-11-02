"""Who am I, the team/agent directory (for filters and pickers), reviewer agreement."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import ColumnElement, false, select, true

from gale_api.api_models import AgentOut, Agreement, Directory, MeOut, TeamOut
from gale_api.auth import DB, PERMISSIONS, Principal, Role, current_principal, require
from gale_api.db import Agent, Call, EventRow, ReviewRow, Team
from gale_api.scope import call_scope

router = APIRouter(prefix="/api/v1", tags=["me"])


@router.get("/me", response_model=MeOut)
def me(p: Annotated[Principal, Depends(current_principal)]) -> MeOut:
    return MeOut(
        user_id=p.user_id,
        sub=p.sub,
        role=p.role.value,
        team_ids=sorted(p.team_ids, key=str),
        agent_id=p.agent_id,
        permissions=sorted(k for k in PERMISSIONS if p.can(k)),
    )


def _team_scope(p: Principal) -> ColumnElement[bool]:
    if p.role in (Role.ADMIN, Role.QA, Role.AUDITOR):
        return true()
    if p.role == Role.SUPERVISOR:
        return Team.id.in_(p.team_ids) if p.team_ids else false()
    return false()


def _agent_scope(p: Principal) -> ColumnElement[bool]:
    if p.role in (Role.ADMIN, Role.QA, Role.AUDITOR):
        return true()
    if p.role == Role.SUPERVISOR:
        return Agent.team_id.in_(p.team_ids) if p.team_ids else false()
    return Agent.id == p.agent_id if p.agent_id else false()


@router.get("/directory", response_model=Directory)
def directory(db: DB, p: Annotated[Principal, require("analytics.read")]) -> Directory:
    teams = db.scalars(select(Team).where(_team_scope(p)).order_by(Team.name)).all()
    agents = db.scalars(select(Agent).where(_agent_scope(p)).order_by(Agent.employee_code)).all()
    return Directory(
        teams=[TeamOut.model_validate(t) for t in teams],
        agents=[AgentOut.model_validate(a) for a in agents],
    )


@router.get("/reviews/agreement", response_model=Agreement)
def agreement(db: DB, p: Annotated[Principal, require("review_queue.read")]) -> Agreement:
    """Pairwise percent agreement between QA verdicts on the same event (docs/04 §2.4)."""
    rows = db.execute(
        select(ReviewRow.event_id, ReviewRow.reviewer_user_id, ReviewRow.verdict)
        .join(EventRow, EventRow.id == ReviewRow.event_id)
        .join(Call, Call.id == EventRow.call_id)
        .where(ReviewRow.kind == "verdict", call_scope(p))
        .order_by(ReviewRow.created_at)
    ).all()
    latest: dict[str, dict[object, str | None]] = defaultdict(dict)
    for event_id, reviewer, verdict in rows:
        latest[event_id][reviewer] = verdict  # a reviewer's latest verdict counts
    multi = {e: v for e, v in latest.items() if len(v) > 1}
    pairs = agree = 0
    for verdicts in multi.values():
        for a, b in itertools.combinations(verdicts.values(), 2):
            pairs += 1
            agree += a == b
    return Agreement(
        events_with_multiple_verdicts=len(multi),
        reviewers=len({r for v in latest.values() for r in v}),
        pairwise_agreement=round(agree / pairs, 3) if pairs else None,
    )
