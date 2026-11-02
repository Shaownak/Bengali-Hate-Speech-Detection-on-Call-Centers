"""Behaviour of the REST endpoints beyond RBAC: search, reviews, acks, promotion, rules, audit."""

from __future__ import annotations

import io
import uuid

import numpy as np
import soundfile as sf
import yaml
from sqlalchemy import select

from gale_api.db import AuditLog, ModelVersion, make_sessionmaker, session_scope
from gale_api.seed import sid
from galecore.bus import POSTCALL_JOBS
from galecore.schemas import PostcallJob

from .conftest import World


def _audit(w: World) -> list[AuditLog]:
    with session_scope(make_sessionmaker(w.engine)) as s:
        return list(s.scalars(select(AuditLog).order_by(AuditLog.id)))


def test_call_list_filters_and_transcript_search(world: World) -> None:
    h = world.h("dev-admin")

    def ids(**params: object) -> set[str]:
        r = world.client.get("/api/v1/calls", params=params, headers=h)
        assert r.status_code == 200
        return {c["id"] for c in r.json()["items"]}

    assert ids(q="বাজে") == {world.call_a}  # normalised substring search on redacted text
    assert ids(q="খারাপ") == {world.call_b}
    assert ids(q="100%_") == set()  # LIKE wildcards are escaped
    assert ids(queue="billing") == {world.call_a}
    assert ids(category="slang_profanity") == {world.call_a, world.call_b}
    assert ids(alert=True) == {world.call_a, world.call_b}
    assert ids(alert=False) == set()
    assert ids(team_id=str(sid("team-ctg"))) == {world.call_b}
    assert ids(agent_id=str(sid("AG-001"))) == {world.call_a}
    page = world.client.get("/api/v1/calls", params={"limit": 1}, headers=h).json()
    assert page["total"] == 2
    assert len(page["items"]) == 1
    item = page["items"][0]
    assert item["alert_count"] == 1
    assert item["max_severity"] == 3
    assert item["status"] == "ended"


def test_detail_events_timelines_segments(world: World) -> None:
    h = world.h("dev-qa")
    d = world.client.get(f"/api/v1/calls/{world.call_a}", headers=h).json()
    assert d["agent_code"] == "AG-001"
    assert d["duration_s"] == 30.0
    assert d["scores"]["agent_score"] == 2
    assert d["scores"]["escalation_curve"] == [[0.0, 1.0, 1.0]]
    events = world.client.get(f"/api/v1/calls/{world.call_a}/events", headers=h).json()
    agent_ev = next(e for e in events if e["event_id"] == world.ev_agent_a)
    assert [s["source"] for s in agent_ev["top_signals"]] == ["lexicon", "dsp_shout"]  # why
    assert agent_ev["top_signals"][0]["text_span"] == "<GALI_1>"
    tl = world.client.get(f"/api/v1/calls/{world.call_a}/timelines", headers=h).json()
    assert tl["bin_s"] == 1.0
    assert max(v for v in tl["series"]["agent"]["loudness"] if v is not None) == 0.7
    assert "anger" in tl["series"]["customer"]
    segs = world.client.get(f"/api/v1/calls/{world.call_a}/segments", headers=h).json()
    assert [s["channel"] for s in segs] == ["customer", "agent"]
    view = [a for a in _audit(world) if a.action == "transcript.view"]
    assert len(view) == 1
    assert view[0].object_id == world.call_a
    assert view[0].user_id == sid("dev-qa")


def test_signed_audio_url_roundtrip(world: World) -> None:
    store = world.app.state.store
    store.put(f"audio/{world.call_a}/agent.wav", b"RIFF-agent")
    r = world.client.get(
        f"/api/v1/calls/{world.call_a}/audio-url", headers=world.h("dev-supervisor")
    )
    assert r.status_code == 200
    url = r.json()["urls"]["agent"]
    assert set(r.json()["urls"]) == {"agent"}
    path = url.split("://", 1)[1].split("/", 1)[1]
    got = world.client.get("/" + path)
    assert got.status_code == 200
    assert got.content == b"RIFF-agent"
    assert world.client.get("/" + path.replace("sig=", "sig=0")).status_code == 403
    actions = [a.action for a in _audit(world)]
    assert actions.count("audio.url") == 1
    assert actions.count("audio.download") == 2  # the tampered attempt is audited too
    none = world.client.get(f"/api/v1/calls/{world.call_b}/audio-url", headers=world.h("dev-admin"))
    assert none.status_code == 404


def test_reviews_change_status_and_validate(world: World) -> None:
    url = f"/api/v1/events/{world.ev_agent_a}/reviews"
    bad = world.client.post(url, json={"kind": "verdict"}, headers=world.h("dev-qa"))
    assert bad.status_code == 422
    bad = world.client.post(
        url, json={"kind": "verdict", "verdict": "wrong_category"}, headers=world.h("dev-qa")
    )
    assert bad.status_code == 422
    ok = world.client.post(
        url, json={"kind": "verdict", "verdict": "false_positive"}, headers=world.h("dev-qa")
    )
    assert ok.status_code == 201
    ev = next(
        e
        for e in world.client.get(
            f"/api/v1/calls/{world.call_a}/events", headers=world.h("dev-qa")
        ).json()
        if e["event_id"] == world.ev_agent_a
    )
    assert ev["status"] == "reviewed_false"
    assert ev["latest_verdict"] == "false_positive"
    # reviewed events leave the queue
    q = world.client.get("/api/v1/review-queue", headers=world.h("dev-qa")).json()
    assert world.ev_agent_a not in {i["event"]["event_id"] for i in q["items"]}
    # agents dispute only agent events, with a comment
    cust = f"/api/v1/events/{world.ev_customer_a}/reviews"
    r = world.client.post(
        cust, json={"kind": "dispute", "comment": "x"}, headers=world.h("dev-agent")
    )
    assert r.status_code == 422
    r = world.client.post(
        url, json={"kind": "dispute", "comment": " "}, headers=world.h("dev-agent")
    )
    assert r.status_code == 422
    a = world.client.get(f"/api/v1/analytics/agents/{sid('AG-001')}", headers=world.h("dev-agent"))
    assert a.json()["disputes"] == 0


def test_ack_only_alerts_and_idempotent(world: World) -> None:
    h = world.h("dev-supervisor")
    r1 = world.client.post(f"/api/v1/events/{world.ev_agent_a}/ack", headers=h)
    r2 = world.client.post(f"/api/v1/events/{world.ev_agent_a}/ack", headers=world.h("dev-admin"))
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["acked_by"] == r2.json()["acked_by"] == str(sid("dev-supervisor"))
    r = world.client.post(f"/api/v1/events/{world.ev_customer_a}/ack", headers=h)
    assert r.status_code == 409


def test_analytics_numbers(world: World) -> None:
    h = world.h("dev-admin")
    a = world.client.get(f"/api/v1/analytics/agents/{sid('AG-001')}", headers=h).json()
    assert (a["calls"], a["alerts"], a["avg_professionalism"]) == (1, 1, 2.0)
    assert a["talk_minutes"] == 0.5
    assert a["trend"][0]["calls"] == 1
    t = world.client.get(f"/api/v1/analytics/teams/{sid('team-dhaka')}", headers=h).json()
    assert {x["employee_code"]: x["calls"] for x in t["agents"]} == {"AG-001": 1, "AG-002": 0}
    queues = world.client.get("/api/v1/analytics/queues", headers=world.h("dev-auditor")).json()
    assert {q["queue"]: q["agent_alerts"] for q in queues} == {"billing": 1, "support": 1}
    assert world.client.get(
        "/api/v1/analytics/queues", headers=world.h("dev-supervisor")
    ).json() == [q for q in queues if q["queue"] == "billing"]


def test_lexicon_proposal_and_soft_delete(world: World) -> None:
    body = {"term": "<GALI_7>", "category": "insult", "severity": 2, "active": True}
    r = world.client.post("/api/v1/lexicon", json=body, headers=world.h("dev-qa"))
    assert r.status_code == 201
    assert r.json()["active"] is False  # a QA analyst only proposes
    eid = r.json()["id"]
    body["notes"] = "approved"
    up = world.client.put(f"/api/v1/lexicon/{eid}", json=body, headers=world.h("dev-admin"))
    assert up.json()["active"] is True
    assert (
        world.client.delete(f"/api/v1/lexicon/{eid}", headers=world.h("dev-admin")).status_code
        == 204
    )
    lst = world.client.get("/api/v1/lexicon", params={"active": False}, headers=world.h("dev-qa"))
    assert [e["id"] for e in lst.json()] == [eid]
    assert world.client.get(
        "/api/v1/lexicon", params={"q": "gali_7"}, headers=world.h("dev-qa")
    ).json()
    missing = world.client.put(
        f"/api/v1/lexicon/{uuid.uuid4()}", json=body, headers=world.h("dev-admin")
    )
    assert missing.status_code == 404


def test_rules_versioning_validation_and_simulation(world: World) -> None:
    h = world.h("dev-admin")
    current = world.client.get("/api/v1/rules", headers=h).json()[0]
    assert current["version"] == 1
    bad = world.client.put(
        "/api/v1/rules", json={"yaml": "version: 1\nrules: [{name: nope}]"}, headers=h
    )
    assert bad.status_code == 422
    assert world.client.put("/api/v1/rules", json={"yaml": ": : :"}, headers=h).status_code == 422
    cfg = yaml.safe_load(current["yaml"])
    lex = next(r for r in cfg["rules"] if r["name"] == "lexicon_fast_path")
    lex["when"]["score_gte"] = 0.99  # stricter than the fixtures' 0.95 lexicon hits
    cand = yaml.safe_dump(cfg, allow_unicode=True)
    sim = world.client.post("/api/v1/rules/simulate", json={"yaml": cand}, headers=h)
    assert sim.status_code == 200, sim.text
    rep = sim.json()
    assert rep["calls"] == 2
    assert rep["candidate_alerts"] <= rep["baseline_alerts"]
    same = world.client.post("/api/v1/rules/simulate", json={"yaml": current["yaml"]}, headers=h)
    assert same.json()["changed_calls"] == []
    put = world.client.put("/api/v1/rules", json={"yaml": cand}, headers=h)
    assert put.json()["version"] == 2
    hist = world.client.get("/api/v1/rules", params={"history": True}, headers=h).json()
    assert [(r["version"], r["active"]) for r in hist] == [(2, True), (1, False)]
    actions = [a.action for a in _audit(world)]
    assert "PUT /api/v1/rules" in actions
    assert "POST /api/v1/rules/simulate" in actions


def test_two_person_model_promotion(world: World) -> None:
    with session_scope(make_sessionmaker(world.engine)) as s:
        cand = ModelVersion(
            component="text_clf", name="textclf", version="v2", metrics_jsonb={"f1": 0.8}
        )
        research = ModelVersion(
            component="text_clf", name="textclf", version="open-v1", research_only=True
        )
        s.add_all([cand, research])
        s.flush()
        cand_id, research_id = str(cand.id), str(research.id)
    a1, a2 = world.h("dev-admin"), world.h("dev-admin2")
    url = "/api/v1/models/text_clf/promote"
    r = world.client.post(url, json={"model_version_id": research_id}, headers=a1)
    assert r.status_code == 409  # ADR-012: research_only never promotable
    r = world.client.post(url, json={"model_version_id": cand_id, "reason": "better"}, headers=a1)
    assert r.status_code == 202
    pid = r.json()["id"]
    again = world.client.post(url, json={"model_version_id": cand_id}, headers=a2)
    assert again.json()["id"] == pid  # one pending request per version
    wrong = world.client.post(
        "/api/v1/models/asr_postcall/promote", json={"model_version_id": cand_id}, headers=a1
    )
    assert wrong.status_code == 404
    self_ok = world.client.post(f"/api/v1/models/promotions/{pid}/approve", headers=a1)
    assert self_ok.status_code == 403  # the requester cannot approve
    ok = world.client.post(f"/api/v1/models/promotions/{pid}/approve", headers=a2)
    assert ok.status_code == 200
    assert ok.json()["approved_by"] == str(sid("dev-admin2"))
    twice = world.client.post(f"/api/v1/models/promotions/{pid}/approve", headers=a2)
    assert twice.status_code == 409
    models = world.client.get("/api/v1/models", params={"component": "text_clf"}, headers=a1).json()
    assert {m["version"]: m["active"] for m in models} == {
        "stub-0.1": False,
        "v2": True,
        "open-v1": False,
    }
    assert world.client.post(url, json={"model_version_id": cand_id}, headers=a1).status_code == 409
    gone = world.client.post(f"/api/v1/models/promotions/{uuid.uuid4()}/approve", headers=a2)
    assert gone.status_code == 404
    pend = world.client.get("/api/v1/models/promotions", params={"status": "approved"}, headers=a1)
    assert [p["id"] for p in pend.json()] == [pid]


def test_ingest_recording_queues_postcall_job(world: World) -> None:
    import asyncio

    buf = io.BytesIO()
    sf.write(buf, np.zeros((16000, 2), dtype="int16"), 16000, format="WAV", subtype="PCM_16")
    h = world.h("dev-qa")
    files = {"file": ("call.wav", buf.getvalue(), "audio/wav")}
    r = world.client.post(
        "/api/v1/ingest/recordings",
        files=files,
        data={"queue": "billing", "external_ref": "crm-1"},
        headers=h,
    )
    assert r.status_code == 202, r.text
    out = r.json()
    assert world.app.state.store.exists(out["uri"])
    rows = asyncio.run(world.bus().redis.xrange(POSTCALL_JOBS))
    from galecore.bus import decode

    job = decode(rows[-1][1], PostcallJob)
    assert (job.call_id, job.uri, job.queue) == (out["call_id"], out["uri"], "billing")
    d = world.client.get(f"/api/v1/calls/{out['call_id']}", headers=h).json()
    assert (d["status"], d["mode"], d["external_call_id"]) == ("queued", "postcall", "crm-1")
    mono = {"file": ("m.wav", buf.getvalue(), "audio/wav")}
    r = world.client.post(
        "/api/v1/ingest/recordings", files=mono, data={"channel_layout": "mono"}, headers=h
    )
    assert r.status_code == 422
    junk = {"file": ("x.wav", b"not audio", "audio/wav")}
    assert world.client.post("/api/v1/ingest/recordings", files=junk, headers=h).status_code == 415


def test_audit_log_filters(world: World) -> None:
    world.client.get(f"/api/v1/calls/{world.call_a}/segments", headers=world.h("dev-agent"))
    world.client.get(f"/api/v1/calls/{world.call_b}/segments", headers=world.h("dev-agent"))  # 404
    r = world.client.get(
        "/api/v1/audit", params={"action": "transcript.view"}, headers=world.h("dev-auditor")
    )
    rows = r.json()
    assert [(x["object_id"], x["details"]["status"]) for x in rows] == [
        (world.call_b, 404),
        (world.call_a, 200),
    ]
    assert (
        world.client.get(
            "/api/v1/audit",
            params={"user_id": str(sid("dev-admin"))},
            headers=world.h("dev-auditor"),
        ).json()
        == []
    )


def test_dev_token_endpoint(world: World) -> None:
    r = world.client.post("/api/v1/dev/token", json={"sub": "dev-qa", "role": "qa_analyst"})
    tok = r.json()["access_token"]
    assert (
        world.client.get("/api/v1/lexicon", headers={"Authorization": f"Bearer {tok}"}).status_code
        == 200
    )
    # a brand-new subject is provisioned from the token on first sight
    r = world.client.post("/api/v1/dev/token", json={"sub": "new-qa", "role": "qa_analyst"})
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert world.client.get("/api/v1/calls", headers=h).status_code == 200
