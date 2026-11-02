"""Dev-only sign-in (ADR-009): mounted only when ``GALE_AUTH_MODE=dev``, which the settings
guard refuses outside dev/test. Production signs in via Keycloak (next-auth in the web app)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from gale_api.api_models import DevTokenIn, DevTokenOut
from gale_api.auth import Role, issue_dev_token

router = APIRouter(prefix="/api/v1/dev", tags=["dev"])


@router.post("/token", response_model=DevTokenOut)
def dev_token(body: DevTokenIn, request: Request) -> DevTokenOut:
    return DevTokenOut(
        access_token=issue_dev_token(request.app.state.settings, body.sub, Role(body.role))
    )
