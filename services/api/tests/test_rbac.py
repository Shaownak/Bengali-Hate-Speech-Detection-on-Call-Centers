"""RBAC matrix (docs/04 §1): every REST endpoint × every role, plus data scoping.

Users (dev seed): admin, qa, supervisor of Team Dhaka, supervisor of Team Chattogram, agent AG-001
(Dhaka), agent AG-003 (Chattogram), auditor. Call A belongs to AG-001, call B to AG-003. A call
or event outside the caller's scope answers 404, so its existence doesn't leak.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from gale_api.db import ModelPromotion, ModelVersion, make_sessionmaker, now, session_scope
from gale_api.seed import sid

from .conftest import World

ADMIN, QA, SUP, SUPC, AGENT, AGENT3, AUD = (
    "dev-admin",
    "dev-qa",
    "dev-supervisor",
    "dev-supervisor-ctg",
    "dev-agent",
    "dev-agent-3",
    "dev-auditor",
)
# mutations by the admin run last, so earlier denials see the original state
ORDER = (AUD, AGENT3, AGENT, SUPC, SUP, QA, ADMIN)


def expect(default: int, **by_role: int) -> dict[str, int]:
    names = {
        "admin": ADMIN,
        "qa": QA,
        "sup": SUP,
        "supc": SUPC,
        "agent": AGENT,
        "agent3": AGENT3,
        "aud": AUD,
    }
    out = dict.fromkeys(ORDER, default)
    for k, v in by_role.items():
        out[names[k]] = v
    return out


# role sees call A: admin/qa/supervisor(Dhaka)/agent AG-001; others: out of scope or no permission
CALL_A = expect(200, supc=404, agent3=404, aud=403)


@dataclass
class Case:
    name: str
    method: str
    path: Callable[[World, dict[str, Any]], str]
    expected: dict[str, int]
    body: Callable[[World, dict[str, Any]], dict[str, Any]] | None = None
    files: bool = False
    route: str = ""  # route template, for the coverage check
    extra: dict[str, Any] = field(default_factory=dict)


def _wav() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros((1600, 2), dtype="int16"), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _lex() -> dict[str, Any]:
    return {"term": "<GALI_9>", "category": "insult", "severity": 2}


CASES = [
    Case(
        "list calls",
        "GET",
        lambda w, c: "/api/v1/calls",
        expect(200, aud=403),
        route="/api/v1/calls",
    ),
    Case(
        "call detail",
        "GET",
        lambda w, c: f"/api/v1/calls/{w.call_a}",
        CALL_A,
        route="/api/v1/calls/{call_id}",
    ),
    Case(
        "segments",
        "GET",
        lambda w, c: f"/api/v1/calls/{w.call_a}/segments",
        CALL_A,
        route="/api/v1/calls/{call_id}/segments",
    ),
    Case(
        "events",
        "GET",
        lambda w, c: f"/api/v1/calls/{w.call_a}/events",
        CALL_A,
        route="/api/v1/calls/{call_id}/events",
    ),
    Case(
        "timelines",
        "GET",
        lambda w, c: f"/api/v1/calls/{w.call_a}/timelines",
        CALL_A,
        route="/api/v1/calls/{call_id}/timelines",
    ),
    Case(
        "audio url",
        "GET",
        lambda w, c: f"/api/v1/calls/{w.call_a}/audio-url",
        CALL_A,
        route="/api/v1/calls/{call_id}/audio-url",
    ),
    Case(
        "review verdict",
        "POST",
        lambda w, c: f"/api/v1/events/{w.ev_agent_a}/reviews",
        expect(403, admin=201, qa=201, sup=201, supc=404),
        body=lambda w, c: {"kind": "verdict", "verdict": "correct"},
        route="/api/v1/events/{event_id}/reviews",
    ),
    Case(
        "review dispute",
        "POST",
        lambda w, c: f"/api/v1/events/{w.ev_agent_a}/reviews",
        expect(403, agent=201, agent3=404),
        body=lambda w, c: {"kind": "dispute", "comment": "I was quoting the customer"},
        route="/api/v1/events/{event_id}/reviews",
    ),
    Case(
        "ack",
        "POST",
        lambda w, c: f"/api/v1/events/{w.ev_agent_a}/ack",
        expect(403, admin=200, sup=200, supc=404),
        route="/api/v1/events/{event_id}/ack",
    ),
    Case(
        "review queue",
        "GET",
        lambda w, c: "/api/v1/review-queue",
        expect(200, agent=403, agent3=403, aud=403),
        route="/api/v1/review-queue",
    ),
    Case(
        "agent analytics",
        "GET",
        lambda w, c: f"/api/v1/analytics/agents/{sid('AG-001')}",
        expect(200, supc=404, agent3=404),
        route="/api/v1/analytics/agents/{agent_id}",
    ),
    Case(
        "team analytics",
        "GET",
        lambda w, c: f"/api/v1/analytics/teams/{sid('team-dhaka')}",
        expect(200, supc=404, agent=404, agent3=404),
        route="/api/v1/analytics/teams/{team_id}",
    ),
    Case(
        "queue analytics",
        "GET",
        lambda w, c: "/api/v1/analytics/queues",
        expect(200, agent=403, agent3=403),
        route="/api/v1/analytics/queues",
    ),
    Case(
        "lexicon list",
        "GET",
        lambda w, c: "/api/v1/lexicon",
        expect(403, admin=200, qa=200),
        route="/api/v1/lexicon",
    ),
    Case(
        "lexicon create",
        "POST",
        lambda w, c: "/api/v1/lexicon",
        expect(403, admin=201, qa=201),
        body=lambda w, c: _lex(),
        route="/api/v1/lexicon",
    ),
    Case(
        "lexicon update",
        "PUT",
        lambda w, c: f"/api/v1/lexicon/{c['lex_id']}",
        expect(403, admin=200),
        body=lambda w, c: _lex(),
        route="/api/v1/lexicon/{entry_id}",
    ),
    Case(
        "lexicon delete",
        "DELETE",
        lambda w, c: f"/api/v1/lexicon/{c['lex_id']}",
        expect(403, admin=204),
        route="/api/v1/lexicon/{entry_id}",
    ),
    Case(
        "rules get",
        "GET",
        lambda w, c: "/api/v1/rules",
        expect(403, admin=200, qa=200, aud=200),
        route="/api/v1/rules",
    ),
    Case(
        "rules put",
        "PUT",
        lambda w, c: "/api/v1/rules",
        expect(403, admin=200),
        body=lambda w, c: {"yaml": c["rules_yaml"]},
        route="/api/v1/rules",
    ),
    Case(
        "rules simulate",
        "POST",
        lambda w, c: "/api/v1/rules/simulate",
        expect(403, admin=200),
        body=lambda w, c: {"yaml": c["rules_yaml"]},
        route="/api/v1/rules/simulate",
    ),
    Case(
        "models",
        "GET",
        lambda w, c: "/api/v1/models",
        expect(403, admin=200, aud=200),
        route="/api/v1/models",
    ),
    Case(
        "promotions",
        "GET",
        lambda w, c: "/api/v1/models/promotions",
        expect(403, admin=200, aud=200),
        route="/api/v1/models/promotions",
    ),
    Case(
        "promote",
        "POST",
        lambda w, c: "/api/v1/models/asr_postcall/promote",
        expect(403, admin=202),
        body=lambda w, c: {"model_version_id": str(c["candidate"])},
        route="/api/v1/models/{component}/promote",
    ),
    Case(
        "approve",
        "POST",
        lambda w, c: f"/api/v1/models/promotions/{c['promotion']}/approve",
        expect(403, admin=200),
        route="/api/v1/models/promotions/{promotion_id}/approve",
    ),
    Case(
        "reject",
        "POST",
        lambda w, c: f"/api/v1/models/promotions/{c['promotion2']}/reject",
        expect(403, admin=200),
        route="/api/v1/models/promotions/{promotion_id}/reject",
    ),
    Case(
        "thresholds",
        "GET",
        lambda w, c: "/api/v1/thresholds",
        expect(403, admin=200, qa=200, aud=200),
        route="/api/v1/thresholds",
    ),
    Case("me", "GET", lambda w, c: "/api/v1/me", expect(200), route="/api/v1/me"),
    Case(
        "directory",
        "GET",
        lambda w, c: "/api/v1/directory",
        expect(200),
        route="/api/v1/directory",
    ),
    Case(
        "agreement",
        "GET",
        lambda w, c: "/api/v1/reviews/agreement",
        expect(200, agent=403, agent3=403, aud=403),
        route="/api/v1/reviews/agreement",
    ),
    Case(
        "review note",
        "POST",
        lambda w, c: f"/api/v1/events/{w.ev_agent_a}/reviews",
        expect(403, admin=201, qa=201, sup=201, supc=404),
        body=lambda w, c: {"kind": "note", "comment": "coached on register"},
        route="/api/v1/events/{event_id}/reviews",
    ),
    Case(
        "audit",
        "GET",
        lambda w, c: "/api/v1/audit",
        expect(403, admin=200, aud=200),
        route="/api/v1/audit",
    ),
    Case(
        "ingest",
        "POST",
        lambda w, c: "/api/v1/ingest/recordings",
        expect(403, admin=202, qa=202),
        files=True,
        route="/api/v1/ingest/recordings",
    ),
]


def _context(w: World) -> dict[str, Any]:
    """Objects some cases act on: a lexicon entry, candidate model versions, pending promotions
    requested by the second admin (so ``dev-admin`` may decide them), the rules YAML, audio."""
    w.app.state.store.put(f"audio/{w.call_a}/agent.wav", _wav())
    r = w.client.post("/api/v1/lexicon", json=_lex(), headers=w.h(ADMIN))
    assert r.status_code == 201
    ctx: dict[str, Any] = {"lex_id": r.json()["id"]}
    ctx["rules_yaml"] = w.client.get("/api/v1/rules", headers=w.h(ADMIN)).json()[0]["yaml"]
    with session_scope(make_sessionmaker(w.engine)) as s:
        ids = []
        for v in ("cand-1", "cand-2", "cand-3"):
            mv = ModelVersion(
                component="asr_postcall",
                name="asr-cand",
                version=v,
                metrics_jsonb={},
                active=False,
                research_only=False,
            )
            s.add(mv)
            s.flush()
            ids.append(mv.id)
        ctx["candidate"] = ids[0]
        for key, mv_id in (("promotion", ids[1]), ("promotion2", ids[2])):
            p = ModelPromotion(
                component="asr_postcall",
                model_version_id=mv_id,
                requested_by=sid("dev-admin2"),
                status="pending",
                created_at=now(),
            )
            s.add(p)
            s.flush()
            ctx[key] = p.id
    return ctx


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rbac_matrix(world: World, case: Case) -> None:
    ctx = _context(world)
    url = case.path(world, ctx)
    got: dict[str, int] = {}

    def call(headers: dict[str, str]) -> int:
        kw: dict[str, Any] = {"headers": headers}
        if case.body is not None:
            kw["json"] = case.body(world, ctx)
        if case.files:
            kw["files"] = {"file": ("x.wav", _wav(), "audio/wav")}
            kw["data"] = {"queue": "billing"}
        return world.client.request(case.method, url, **kw).status_code

    assert call({}) == 401, "anonymous must be rejected"
    for sub in ORDER:
        got[sub] = call(world.h(sub))
    assert got == case.expected


def test_matrix_covers_every_endpoint(world: World) -> None:
    covered = {(c.method, c.route) for c in CASES}
    routes = {
        (m, r.path)
        for r in world.app.routes
        if getattr(r, "path", "").startswith("/api/v1/")
        and not r.path.startswith(("/api/v1/dev/", "/api/v1/objects/"))
        for m in getattr(r, "methods", set()) - {"HEAD", "OPTIONS"}
    }
    assert routes - covered == set(), "add RBAC cases for new endpoints"


def test_invalid_and_foreign_tokens_are_rejected(world: World) -> None:
    import jwt

    bad = jwt.encode(
        {"sub": "x", "aud": "gale-api", "iss": "gale-dev", "exp": 9_999_999_999},
        "wrong-secret-wrong-secret-wrong-secret",
        algorithm="HS256",
    )
    assert (
        world.client.get("/api/v1/calls", headers={"Authorization": f"Bearer {bad}"}).status_code
        == 401
    )
    no_role = jwt.encode(
        {"sub": "x", "aud": "gale-api", "iss": "gale-dev", "exp": 9_999_999_999},
        world.settings.dev_jwt_secret,
        algorithm="HS256",
    )
    r = world.client.get("/api/v1/calls", headers={"Authorization": f"Bearer {no_role}"})
    assert r.status_code == 401
    assert "role" in r.json()["detail"]


def test_scoped_lists(world: World) -> None:
    def ids(sub: str) -> set[str]:
        r = world.client.get("/api/v1/calls", headers=world.h(sub))
        return {c["id"] for c in r.json()["items"]}

    both = {world.call_a, world.call_b}
    assert ids(ADMIN) == both
    assert ids(QA) == both
    assert ids(SUP) == {world.call_a}
    assert ids(SUPC) == {world.call_b}
    assert ids(AGENT) == {world.call_a}
    assert ids(AGENT3) == {world.call_b}
    q = world.client.get("/api/v1/review-queue", headers=world.h(SUPC)).json()
    assert {i["call"]["id"] for i in q["items"]} == {world.call_b}
    unknown = str(uuid.uuid4())
    assert world.client.get(f"/api/v1/calls/{unknown}", headers=world.h(ADMIN)).status_code == 404
    assert world.client.get("/api/v1/calls/not-a-uuid", headers=world.h(ADMIN)).status_code == 404
