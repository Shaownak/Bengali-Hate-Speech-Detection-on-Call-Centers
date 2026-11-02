"""Row-level scoping on top of role permissions (docs/04 §1).

admin / qa_analyst: every call. supervisor: calls of agents in their teams. agent: their own calls.
Anything else: none. A call outside the caller's scope answers 404, so its existence doesn't leak.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import ColumnElement, false, select, true
from sqlalchemy.orm import Session

from gale_api.auth import Principal, Role
from gale_api.db import Agent, Call, EventRow


def call_scope(p: Principal) -> ColumnElement[bool]:
    if p.sees_all_calls:
        return true()
    if p.role == Role.SUPERVISOR and p.team_ids:
        return Call.agent_id.in_(select(Agent.id).where(Agent.team_id.in_(p.team_ids)))
    if p.role == Role.AGENT and p.agent_id is not None:
        return Call.agent_id == p.agent_id
    return false()


def parse_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found") from None


def get_call(db: Session, p: Principal, call_id: str) -> Call:
    call = db.scalar(select(Call).where(Call.id == parse_uuid(call_id), call_scope(p)))
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "call not found")
    return call


def get_event(db: Session, p: Principal, event_id: str) -> tuple[EventRow, Call]:
    row = db.execute(
        select(EventRow, Call)
        .join(Call, Call.id == EventRow.call_id)
        .where(EventRow.id == event_id, call_scope(p))
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "event not found")
    return row[0], row[1]


def can_see_agent(db: Session, p: Principal, agent_id: uuid.UUID) -> bool:
    if p.role in (Role.ADMIN, Role.QA, Role.AUDITOR):
        return True
    if p.role == Role.AGENT:
        return p.agent_id == agent_id
    if p.role == Role.SUPERVISOR:
        team = db.scalar(select(Agent.team_id).where(Agent.id == agent_id))
        return team is not None and team in p.team_ids
    return False  # pragma: no cover - every role is covered above


def can_see_team(p: Principal, team_id: uuid.UUID) -> bool:
    if p.role in (Role.ADMIN, Role.QA, Role.AUDITOR):
        return True
    return p.role == Role.SUPERVISOR and team_id in p.team_ids
