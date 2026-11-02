"""Audit logging (docs/04 §1): every view of audio or transcript, and every change, writes a row.

The middleware covers HTTP. ``/ws/calls/{id}`` audits itself when the socket opens. Rows hold who,
what and which object: never transcript text or audio (CLAUDE.md rule 1).
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from gale_api.db import AuditLog

# reads of call content: (pattern, action, object type)
AUDITED_READS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^/api/v1/calls/([^/]+)/segments$"), "transcript.view", "call"),
    (re.compile(r"^/api/v1/calls/([^/]+)/audio-url$"), "audio.url", "call"),
    (re.compile(r"^/api/v1/objects/(.+)$"), "audio.download", "object"),
)
UNAUDITED_WRITES = ("/api/v1/dev/token",)


def write_audit(
    sessions: sessionmaker[Session],
    principal: Any,
    action: str,
    object_type: str,
    object_id: str | None,
    ip: str | None,
    details: dict[str, Any],
) -> None:
    with sessions() as s, s.begin():
        s.add(
            AuditLog(
                user_id=getattr(principal, "user_id", None),
                action=action,
                object_type=object_type,
                object_id=object_id,
                ip=ip,
                details_jsonb=details,
            )
        )


def classify(method: str, path: str, route: str | None) -> tuple[str, str, str | None] | None:
    if method == "GET":
        for pat, action, otype in AUDITED_READS:
            m = pat.match(path)
            if m:
                return action, otype, m.group(1)
        return None
    if not path.startswith("/api/v1/") or path in UNAUDITED_WRITES:
        return None
    parts = path.removeprefix("/api/v1/").split("/")
    template = route or path
    return f"{method} {template}", parts[0], parts[1] if len(parts) > 1 else None


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        route = request.scope.get("route")
        what = classify(request.method, request.url.path, getattr(route, "path", None))
        if what is None or response.status_code == 401:
            return response  # anonymous noise isn't audit-worthy; 401s are in access metrics
        action, otype, oid = what
        principal = getattr(request.state, "principal", None)
        ip = request.client.host if request.client else None
        await run_in_threadpool(
            write_audit,
            request.app.state.sessions,
            principal,
            action,
            otype,
            oid,
            ip,
            {"status": response.status_code},
        )
        return response
