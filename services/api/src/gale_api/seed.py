"""Dev seed (``python -m gale_api seed-dev``): teams, agents, one user per role, model versions.

Idempotent (fixed uuid5 ids). Dev users sign in with ``POST /api/v1/dev/token`` in
``GALE_AUTH_MODE=dev`` only. Production users come from Keycloak and are linked by admins.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from gale_api.db import Agent, ModelVersion, Team, User, now
from galecore.config import Settings, load_models

NS = uuid.UUID("6a1e0000-0000-4000-8000-000000000000")


def sid(name: str) -> uuid.UUID:
    return uuid.uuid5(NS, name)


TEAMS = {"team-dhaka": "Team Dhaka", "team-ctg": "Team Chattogram"}
AGENTS = {"AG-001": "team-dhaka", "AG-002": "team-dhaka", "AG-003": "team-ctg"}
# sub → (role, team, agent code)
USERS: dict[str, tuple[str, str | None, str | None]] = {
    "dev-admin": ("admin", None, None),
    "dev-admin2": ("admin", None, None),  # second admin: two-person model promotion
    "dev-qa": ("qa_analyst", None, None),
    "dev-supervisor": ("supervisor", "team-dhaka", None),
    "dev-supervisor-ctg": ("supervisor", "team-ctg", None),
    "dev-agent": ("agent", "team-dhaka", "AG-001"),
    "dev-agent-3": ("agent", "team-ctg", "AG-003"),
    "dev-auditor": ("auditor", None, None),
}
SUPERVISES = {"team-dhaka": "dev-supervisor", "team-ctg": "dev-supervisor-ctg"}


def _ins(db: Session, model: Any, rows: list[dict[str, Any]]) -> None:
    if rows:
        db.execute(insert(model).values(rows).on_conflict_do_nothing())


def seed_dev(db: Session, settings: Settings) -> dict[str, int]:
    _ins(db, Team, [{"id": sid(k), "name": v} for k, v in TEAMS.items()])
    _ins(
        db,
        Agent,
        [
            {
                "id": sid(code),
                "employee_code": code,
                "name": code,
                "team_id": sid(t),
                "active": True,
            }
            for code, t in AGENTS.items()
        ],
    )
    _ins(
        db,
        User,
        [
            {
                "id": sid(sub),
                "oidc_sub": sub,
                "email": f"{sub}@gale.local",
                "role": role,
                "team_id": sid(team) if team else None,
                "agent_id": sid(agent) if agent else None,
                "created_at": now(),
            }
            for sub, (role, team, agent) in USERS.items()
        ],
    )
    for team, sub in SUPERVISES.items():
        t = db.get(Team, sid(team))
        if t is not None and t.supervisor_user_id is None:
            t.supervisor_user_id = sid(sub)
    models = load_models(settings.config_dir)
    _ins(
        db,
        ModelVersion,
        [
            {
                "id": sid(f"model:{comp}:{m.version}"),
                "component": comp,
                "name": f"{comp}-{m.backend}",
                "version": m.version,
                "metrics_jsonb": {},
                "deployed_at": now(),
                "active": True,
                "research_only": False,
            }
            for comp, m in models.components.items()
        ],
    )
    return {"teams": len(TEAMS), "agents": len(AGENTS), "users": len(USERS)}
