"""Authentication (OIDC JWT from Keycloak, or the dev issuer) and RBAC (docs/04 §1).

The token decides the role, since Keycloak is the source of truth. The ``users`` row, created on
first sight, holds team and agent links, which admins manage. Supervisors see the teams they
supervise plus their own team; agents see only calls of their own ``agents`` row.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from gale_api.db import Team, User
from galecore.config import Settings

DEV_ISSUER = "gale-dev"


class Role(StrEnum):
    ADMIN = "admin"
    QA = "qa_analyst"
    SUPERVISOR = "supervisor"
    AGENT = "agent"
    AUDITOR = "auditor"


# strongest first: a token carrying several roles gets the first match
ROLE_ORDER = (Role.ADMIN, Role.QA, Role.SUPERVISOR, Role.AUDITOR, Role.AGENT)

A, Q, S, G, U = Role.ADMIN, Role.QA, Role.SUPERVISOR, Role.AGENT, Role.AUDITOR

# docs/04 §1 → permission table. Data scoping (own team / own calls) is applied on top.
PERMISSIONS: dict[str, frozenset[Role]] = {
    "calls.read": frozenset({A, Q, S, G}),
    "transcript.read": frozenset({A, Q, S, G}),
    "audio.read": frozenset({A, Q, S, G}),
    "events.review": frozenset({A, Q, S}),
    "events.dispute": frozenset({G}),
    "events.ack": frozenset({A, S}),
    "review_queue.read": frozenset({A, Q, S}),
    "analytics.read": frozenset({A, Q, S, G, U}),
    "lexicon.read": frozenset({A, Q}),
    "lexicon.propose": frozenset({A, Q}),
    "lexicon.write": frozenset({A}),
    "rules.read": frozenset({A, Q, U}),
    "rules.write": frozenset({A}),
    "models.read": frozenset({A, U}),
    "models.promote": frozenset({A}),
    "audit.read": frozenset({A, U}),
    "ingest.write": frozenset({A, Q}),
    "live.read": frozenset({A, Q, S}),
    "live_call.read": frozenset({A, Q, S, G}),
}


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    sub: str
    role: Role
    team_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)
    agent_id: uuid.UUID | None = None

    def can(self, perm: str) -> bool:
        return self.role in PERMISSIONS[perm]

    @property
    def sees_all_calls(self) -> bool:
        return self.role in (Role.ADMIN, Role.QA)


class AuthError(Exception):
    pass


def _claim(claims: dict[str, Any], path: str) -> Any:
    cur: Any = claims
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def role_from_claims(claims: dict[str, Any], role_claim: str) -> Role:
    raw = _claim(claims, role_claim)
    roles = {raw} if isinstance(raw, str) else set(raw or [])
    for r in ROLE_ORDER:
        if r.value in roles:
            return r
    raise AuthError("token carries no GAL-E role")


class TokenVerifier:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._jwks: jwt.PyJWKClient | None = None

    def jwks(self) -> jwt.PyJWKClient:  # pragma: no cover - needs a live Keycloak
        if self._jwks is None:
            url = self.s.oidc_jwks_url or (
                self.s.oidc_issuer.rstrip("/") + "/protocol/openid-connect/certs"
            )
            self._jwks = jwt.PyJWKClient(url, cache_keys=True)
        return self._jwks

    def verify(self, token: str) -> dict[str, Any]:
        try:
            if self.s.auth_mode == "dev":
                claims: dict[str, Any] = jwt.decode(
                    token,
                    self.s.dev_jwt_secret,
                    algorithms=["HS256"],
                    audience=self.s.oidc_audience,
                    issuer=DEV_ISSUER,
                    options={"require": ["exp", "sub"]},
                )
            else:  # pragma: no cover - exercised against Keycloak in staging
                key = self.jwks().get_signing_key_from_jwt(token).key
                claims = jwt.decode(
                    token,
                    key,
                    algorithms=["RS256", "ES256"],
                    audience=self.s.oidc_audience,
                    issuer=self.s.oidc_issuer,
                    options={"require": ["exp", "sub"]},
                )
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid token: {type(e).__name__}") from None
        return claims


def issue_dev_token(settings: Settings, sub: str, role: Role, ttl_s: int = 8 * 3600) -> str:
    """Dev-mode JWT (ADR-009). Refused outside dev/test by the settings guard."""
    if settings.auth_mode != "dev":
        raise AuthError("dev tokens only in GALE_AUTH_MODE=dev")
    now = int(time.time())
    claims = {
        "iss": DEV_ISSUER,
        "aud": settings.oidc_audience,
        "sub": sub,
        "iat": now,
        "exp": now + ttl_s,
        "realm_access": {"roles": [role.value]},
    }
    return jwt.encode(claims, settings.dev_jwt_secret, algorithm="HS256")


def principal_for(db: Session, claims: dict[str, Any], role_claim: str) -> Principal:
    role = role_from_claims(claims, role_claim)
    sub = str(claims["sub"])
    user = db.scalar(select(User).where(User.oidc_sub == sub))
    if user is None:
        user = User(oidc_sub=sub, email=claims.get("email"), role=role.value)
        db.add(user)
        db.flush()
    elif user.role != role.value:
        user.role = role.value
    teams: set[uuid.UUID] = set()
    if role == Role.SUPERVISOR:
        teams = set(
            db.scalars(
                select(Team.id).where(
                    or_(Team.supervisor_user_id == user.id, Team.id == user.team_id)
                )
            )
        )
    return Principal(
        user_id=user.id,
        sub=sub,
        role=role,
        team_ids=frozenset(teams),
        agent_id=user.agent_id if role == Role.AGENT else None,
    )


def authenticate(request: Request, db: Session, token: str) -> Principal:
    app_state = request.app.state
    try:
        claims = app_state.verifier.verify(token)
        p = principal_for(db, claims, app_state.settings.oidc_role_claim)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from None
    request.state.principal = p
    return p


bearer = HTTPBearer(auto_error=False)


def get_db(request: Request) -> Any:
    with request.app.state.sessions() as s, s.begin():
        yield s


DB = Annotated[Session, Depends(get_db)]


def current_principal(
    request: Request,
    db: DB,
    cred: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if cred is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token", {"WWW-Authenticate": "Bearer"}
        )
    return authenticate(request, db, cred.credentials)


def require(perm: str) -> Any:
    if perm not in PERMISSIONS:
        raise KeyError(perm)

    def dep(p: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if not p.can(perm):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role {p.role} lacks {perm}")
        return p

    return Depends(dep)
