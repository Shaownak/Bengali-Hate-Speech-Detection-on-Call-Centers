"""Pilot & feedback loop (ADR-020): weekly report, shadow visibility, usefulness ratings, missed
events, active-learning sampler, feedback export, candidate registration, notifications."""

from __future__ import annotations

import json
import smtplib
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import select, update

from gale_api.db import AlSample, EventRow, ModelVersion, make_sessionmaker, session_scope
from gale_api.pilot import (
    current_week,
    iso_week,
    last_full_week,
    parse_week,
    previous_week,
    week_bounds,
)
from gale_api.pilot.export import export_feedback
from gale_api.pilot.notify import notify, send_email, send_webhook
from gale_api.pilot.register import Candidate, RegistrationError, register_model
from gale_api.pilot.report import (
    CallFact,
    EventFact,
    PromClient,
    WeekData,
    compute,
    prom_data,
    render_html,
    render_markdown,
    summary_text,
)
from gale_api.pilot.run import al_sample, export_feedback_cmd, pilot_report, register_model_cmd
from gale_api.pilot.sampler import (
    GoldenManifestMissingError,
    Item,
    disagreement,
    golden_call_ids,
    mine_terms,
    run_sampler,
    select_items,
)
from gale_api.pilot.stats import median, per_100, psi, rate, wilson
from galecore.labelstudio import LabelStudioClient
from galecore.pilot import EmailNotify, Notify, PilotConfig, SamplerConfig, WebhookNotify

from .conftest import World

WEEK = iso_week(datetime.fromtimestamp(1_790_000_000, UTC).astimezone(ZoneInfo("Asia/Dhaka")))
CFG = PilotConfig.model_validate(
    {
        "stage": "P-2",
        "exit_criteria": {
            "P-2": {"alert_acceptance": 0.75, "usefulness": 4.0, "fairness_red_flags": 0},
            "P-1": {"precision": 0.8, "latency_p95_s": {"provisional_event": 2.0}},
        },
        "sampler": {
            "budget": 20,
            "quotas": {
                "false_positive": 0.3,
                "false_negative": 0.25,
                "low_confidence": 0.2,
                "disagreement": 0.15,
                "new_terms": 0.1,
            },
            "new_term_min_count": 1,
            "new_term_min_calls": 1,
        },
    }
)


# --- weeks & stats ------------------------------------------------------------------------------


def test_week_helpers() -> None:
    assert parse_week("2026-W44").isoformat() == "2026-10-26"
    assert previous_week("2026-W01") == "2025-W52"
    start, end = week_bounds("2026-W44", "Asia/Dhaka")
    assert start == datetime(2026, 10, 25, 18, tzinfo=UTC)  # Monday 00:00 in Dhaka (UTC+6)
    assert end - start == timedelta(days=7)
    now = datetime(2026, 11, 2, 3, tzinfo=UTC)  # Monday morning in Dhaka
    assert current_week(now, "Asia/Dhaka") == "2026-W45"
    assert last_full_week(now, "Asia/Dhaka") == "2026-W44"
    with pytest.raises(ValueError, match="ISO week"):
        parse_week("2026-44")


def test_stats() -> None:
    lo, hi = wilson(8, 10) or (0, 0)
    assert lo < 0.8 < hi
    assert wilson(0, 0) is None
    assert rate(1, 4) == 0.25
    assert per_100(3, 200) == 1.5
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0]) == 1.5
    assert median([]) is None
    assert psi([10, 10], [10, 10]) == 0
    assert (psi([90, 10], [10, 90]) or 0) > 0.2
    assert psi([0, 0], [1, 1]) is None


# --- report computation (pure) ------------------------------------------------------------------


def _week_data() -> WeekData:
    t = datetime(2026, 9, 21, 4, tzinfo=UTC)
    calls = [
        CallFact(f"c{i}", t, queue="billing" if i < 40 else "sales", team="A", agent=f"a{i % 3}")
        for i in range(80)
    ]
    calls[0] = CallFact("c0", t, queue="billing", team="A", agent="a0", scored=True)
    events, verdicts = [], {}
    for i in range(40):  # billing: 40 alerts, 30 reviewed, 27 correct
        eid = f"e{i}"
        events.append(
            EventFact(eid, f"c{i}", "agent", "insult", 2, True, "confirmed", True, 0.9, 30)
        )
        if i < 30:
            verdicts[eid] = "correct" if i < 27 else "false_positive"
    for i in range(40, 52):  # sales: 12 alerts, all reviewed, half wrong
        eid = f"e{i}"
        events.append(EventFact(eid, f"c{i}", "customer", "threat", 3, True, "confirmed"))
        verdicts[eid] = "correct" if i % 2 else "wrong_category"
    events.append(EventFact("d1", "c1", "agent", "insult", 1, False, "dismissed_by_llm"))
    return WeekData(
        week="2026-W39",
        calls=calls,
        events=events,
        verdicts=verdicts,
        disputes=1,
        missed=[("agent", "insult")],
        ratings=[4, 5, 3],
        score_hist={"ser": [90, 10] + [0] * 8, "lexicon": [5] * 10},
        prev_score_hist={"ser": [10, 90] + [0] * 8, "lexicon": [5] * 10},
        asr_conf=0.8,
        prev_asr_conf=0.82,
        prev_calls=50,
        prev_events_by_category={"insult": 10},
    )


def test_compute_precision_acceptance_fairness_drift_and_exit() -> None:
    p = compute(_week_data(), CFG)
    pr = p["precision"]
    assert (pr["reviewed"], pr["tp"], pr["fp"]) == (42, 33, 9)
    assert pr["precision"] == round(33 / 42, 4)
    assert pr["by_category"]["threat"]["precision"] == 0.5
    acc = p["acceptance"]
    assert (acc["alerts"], acc["reviewed"], acc["accepted"]) == (52, 42, 33)
    assert acc["median_time_to_ack_s"] == 30
    assert p["kpis"]["agent_incidents_per_100_calls"] == 50.0
    assert p["kpis"]["repeat_offender_agents"] == 3  # a0/a1/a2 each ≥ 3 agent incidents
    assert p["kpis"]["qa_coverage"] == round(1 / 80, 4)
    # sales: precision 0.5 vs 0.79 overall (gap > 0.10) → red flag; slices need ≥ 30 calls
    flags = p["fairness"]["red_flags"]
    assert {"dimension": "queue", "value": "sales", "flag": "precision_gap"} in flags
    assert p["drift"]["signal_psi"]["ser"]["shift"] is True
    assert p["drift"]["signal_psi"]["lexicon"]["shift"] is False
    assert p["drift"]["events_per_100_calls"]["insult"]["prev_per_100"] == 20.0
    ex = {x["criterion"]: x["status"] for x in p["exit"]}
    assert ex == {"alert_acceptance": "pass", "usefulness": "pass", "fairness_red_flags": "fail"}
    assert p["shadow"]["events"] == 40
    assert p["missed"] == {"agent/insult": 1}


def test_exit_unknown_without_prometheus_and_with_it() -> None:
    cfg = CFG.model_copy(update={"stage": "P-1"})
    ex = compute(_week_data(), cfg)["exit"]
    assert [x["status"] for x in ex] == ["fail", "unknown"]  # precision 0.79 < 0.8; no latency

    def handler(req: httpx.Request) -> httpx.Response:
        q = req.url.params["query"]
        if "gale_stage_latency" in q:
            result = [
                {"metric": {"stage": "provisional_event"}, "value": [0, "1.5"]},
                {"metric": {"stage": "alert"}, "value": [0, "NaN"]},
            ]
        else:
            result = [
                {
                    "metric": {"component": "text_clf", "version": "v2", "outcome": "agree"},
                    "value": [0, "12"],
                }
            ]
        return httpx.Response(200, json={"status": "success", "data": {"result": result}})

    prom = prom_data(
        PromClient("http://prom:9090", transport=httpx.MockTransport(handler)), datetime.now(UTC)
    )
    assert prom.latency_p95 == {"provisional_event": 1.5}
    assert prom.shadow_compare == {"text_clf:v2": {"agree": 12.0}}
    ex = compute(_week_data(), cfg, prom)["exit"]
    assert ex[1] == {
        "criterion": "latency_p95_s.provisional_event",
        "target": "<= 2.0",
        "value": 1.5,
        "status": "pass",
    }


def test_rendering_has_no_ids_and_is_self_contained() -> None:
    p = compute(_week_data(), CFG)
    md, page, text = render_markdown(p), render_html(p), summary_text(p, "https://x/pilot")
    for out in (md, page, text):
        assert "c17" not in out
        assert "e3" not in out.split()
        assert "a0" not in out
    assert "## Exit criteria" in md
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    assert "http" not in page.split("<body>")[0]  # no external assets in <head>
    assert "❌ fairness_red_flags" in text


# --- endpoints, CLI and DB ---------------------------------------------------------------------


def _review(w: World, event_id: str, verdict: str, who: str = "dev-qa") -> None:
    body: dict[str, Any] = {"verdict": verdict}
    if verdict == "wrong_category":
        body["corrected_category"] = "insult"
    r = w.client.post(f"/api/v1/events/{event_id}/reviews", json=body, headers=w.h(who))
    assert r.status_code == 201, r.text


def _settings(w: World, tmp: Path, cfg: PilotConfig = CFG) -> Any:
    f = tmp / "pilot.yaml"
    f.write_text(json.dumps(cfg.model_dump(mode="json")), "utf-8")
    return w.settings.model_copy(update={"pilot_config": f, "data_dir": tmp})


def test_weekly_report_job_and_endpoints(world: World, tmp_path: Path) -> None:
    _review(world, world.ev_agent_a, "correct")
    _review(world, world.ev_agent_b, "false_positive")
    r = world.client.post(
        "/api/v1/pilot/feedback", json={"rating": 2}, headers=world.h("dev-supervisor")
    )
    assert r.status_code == 200
    s = _settings(world, tmp_path)
    out = pilot_report(s, ["--week", WEEK, "--out", str(tmp_path / "out")])
    assert out["week"] == WEEK
    assert out["exit"]["fairness_red_flags"] == "pass"
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == [f"pilot_{WEEK}.html", f"pilot_{WEEK}.json", f"pilot_{WEEK}.md"]
    # idempotent per week
    pilot_report(s, ["--week", WEEK])
    lst = world.client.get("/api/v1/pilot/reports", headers=world.h("dev-qa")).json()
    assert [x["week"] for x in lst] == [WEEK]
    rep = world.client.get(f"/api/v1/pilot/reports/{WEEK}", headers=world.h("dev-auditor")).json()
    h = rep["payload"]["headline"]
    assert (h["calls"], h["precision"]) == (2, 0.5)
    raw = json.dumps(rep)
    assert world.call_a not in raw
    assert world.ev_agent_a not in raw
    page = world.client.get(f"/api/v1/pilot/reports/{WEEK}/html", headers=world.h("dev-admin"))
    assert page.status_code == 200
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert (
        world.client.get("/api/v1/pilot/reports/2026-W01", headers=world.h("dev-qa")).status_code
        == 404
    )
    assert (
        world.client.get("/api/v1/pilot/reports/bad", headers=world.h("dev-qa")).status_code == 422
    )
    st = world.client.get("/api/v1/pilot/status", headers=world.h("dev-qa")).json()
    assert st["stage"] == "P-3"  # the app reads the repo default config (dev: rollout)
    assert st["alerts_default"] == "visible"


def test_shadow_events_hidden_from_supervisors_visible_to_qa(world: World) -> None:
    with session_scope(make_sessionmaker(world.engine)) as s:
        s.execute(update(EventRow).where(EventRow.id == world.ev_agent_a).values(shadow=True))
    sup, qa = world.h("dev-supervisor"), world.h("dev-qa")
    evs = world.client.get(f"/api/v1/calls/{world.call_a}/events", headers=sup).json()
    assert world.ev_agent_a not in {e["event_id"] for e in evs}
    detail = world.client.get(f"/api/v1/calls/{world.call_a}", headers=sup).json()
    assert detail["alert_count"] == 0
    q = world.client.get("/api/v1/review-queue", headers=sup).json()
    assert world.ev_agent_a not in {i["event"]["event_id"] for i in q["items"]}
    r = world.client.post(
        f"/api/v1/events/{world.ev_agent_a}/reviews", json={"verdict": "correct"}, headers=sup
    )
    assert r.status_code == 404
    # QA samples shadow events in the review queue (P-1)
    q = world.client.get("/api/v1/review-queue?shadow=true", headers=qa).json()
    assert [i["event"]["event_id"] for i in q["items"]] == [world.ev_agent_a]
    assert q["items"][0]["event"]["shadow"] is True
    q = world.client.get("/api/v1/review-queue?shadow=false", headers=qa).json()
    assert world.ev_agent_a not in {i["event"]["event_id"] for i in q["items"]}
    evs = world.client.get(f"/api/v1/calls/{world.call_a}/events", headers=qa).json()
    assert world.ev_agent_a in {e["event_id"] for e in evs}
    # dashboards never count shadow events
    team = world.client.get("/api/v1/analytics/queues", headers=world.h("dev-admin")).json()
    billing = next(r for r in team if r["queue"] == "billing")
    assert billing["agent_alerts"] == 0


async def test_live_hub_drops_shadow_events_for_supervisors(world: World) -> None:
    from gale_api.auth import principal_for
    from gale_api.ws import LiveHub, Subscriber
    from galecore.bus import ALERTS, EVENTS_FUSED
    from galecore.schemas import FusedEvent

    hub = LiveHub(world.bus(), make_sessionmaker(world.engine))
    with make_sessionmaker(world.engine)() as s:
        claim = world.settings.oidc_role_claim
        sup = principal_for(
            s, world.app.state.verifier.verify(world.token("dev-supervisor")), claim
        )
        qa = principal_for(s, world.app.state.verifier.verify(world.token("dev-qa")), claim)
    a, b = Subscriber(sup, frozenset({EVENTS_FUSED, ALERTS})), Subscriber(qa, frozenset({ALERTS}))
    hub.subs |= {a, b}
    ev = FusedEvent(
        event_id="x",
        call_id=world.call_a,
        speaker="agent",
        category="insult",
        severity=2,
        probability=0.9,
        ts_start=1,
        ts_end=2,
        alert=True,
        shadow=True,
    )
    await hub.dispatch(ALERTS, ev)
    assert a.take() == []
    assert len(b.take()) == 1
    await hub.dispatch(ALERTS, ev.model_copy(update={"shadow": False}))
    assert len(a.take()) == 1


def test_feedback_and_missed_events(world: World) -> None:
    sup = world.h("dev-supervisor")
    assert world.client.get("/api/v1/pilot/feedback", headers=sup).json() is None
    world.client.post("/api/v1/pilot/feedback", json={"rating": 3}, headers=sup)
    r = world.client.post(
        "/api/v1/pilot/feedback", json={"rating": 5, "comment": "ok"}, headers=sup
    )
    assert r.json()["rating"] == 5
    assert world.client.get("/api/v1/pilot/feedback", headers=sup).json()["comment"] == "ok"
    bad = world.client.post("/api/v1/pilot/feedback", json={"rating": 6}, headers=sup)
    assert bad.status_code == 422
    url = f"/api/v1/calls/{world.call_a}/missed"
    body = {"speaker": "customer", "category": "threat", "severity": 3, "ts_start": 1, "ts_end": 3}
    assert world.client.post(url, json=body, headers=world.h("dev-qa")).status_code == 201
    inverted = {**body, "ts_start": 5, "ts_end": 4}
    assert world.client.post(url, json=inverted, headers=world.h("dev-qa")).status_code == 422
    late = {**body, "ts_start": 999, "ts_end": 1000}
    assert world.client.post(url, json=late, headers=world.h("dev-qa")).status_code == 422
    lst = world.client.get(url, headers=world.h("dev-agent")).json()
    assert [(m["category"], m["severity"]) for m in lst] == [("threat", 3)]


# --- active-learning sampler --------------------------------------------------------------------


def _item(bucket: str, call: str, n: int, prio: float = 0) -> Item:
    return Item(f"{bucket}:{call}:{n}", bucket, call, 0, 1, "agent", priority=prio)


def test_select_items_quotas_caps_leftovers_golden_and_seen() -> None:
    cfg = SamplerConfig(
        budget=10,
        per_call_max=2,
        quotas={"false_positive": 0.5, "low_confidence": 0.5},
    )
    cands = {
        "false_positive": [_item("false_positive", f"c{i}", 0, prio=i) for i in range(3)],
        "low_confidence": [_item("low_confidence", f"d{i // 3}", i) for i in range(30)],
    }
    got = select_items(
        cands, cfg, "2026-W40", seen_refs={"false_positive:c1:0"}, golden_calls={"c0"}
    )
    assert len(got) == 10
    fp = [i for i in got if i.bucket == "false_positive"]
    assert [i.call_id for i in fp] == ["c2"]  # c1 seen, c0 golden; highest priority first
    per_call: dict[str, int] = {}
    for i in got:
        per_call[i.call_id] = per_call.get(i.call_id, 0) + 1
    assert max(per_call.values()) <= 2
    assert select_items(cands, cfg, "2026-W40") == select_items(cands, cfg, "2026-W40")
    assert select_items(cands, cfg, "2026-W40", budget=0) == []


def test_disagreement_and_term_mining() -> None:
    assert disagreement([("ser", 0.9), ("text_clf", 0.1)]) == pytest.approx(0.8)
    assert disagreement([("ser", 0.9)]) is None
    flagged = [("a", "ওই ফালতু লোক <GALI_1>"), ("b", "ফালতু কথা"), ("b", "ফালতু")]
    background = [("a", "আপনার বিল"), ("b", "লোক লোক লোক")]
    terms = mine_terms(flagged, background, lexicon={"কথা"}, min_count=2, min_calls=2)
    assert [t[0] for t in terms] == ["ফালতু"]
    assert terms[0][1:3] == (3, 2)


def _ls_transport(seen: list[Any]) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(
                200, json={"results": [{"id": 7, "title": CFG.sampler.labelstudio.project}]}
            )
        seen.append(json.loads(req.content))
        return httpx.Response(201, json={"task_count": len(seen[-1])})

    return httpx.MockTransport(handler)


def test_sampler_on_the_database(world: World, tmp_path: Path) -> None:
    _review(world, world.ev_agent_a, "false_positive")
    url = f"/api/v1/calls/{world.call_b}/missed"
    body = {"speaker": "agent", "category": "insult", "severity": 2, "ts_start": 4, "ts_end": 6}
    world.client.post(url, json=body, headers=world.h("dev-qa"))
    with session_scope(make_sessionmaker(world.engine)) as s:  # one uncertain event
        s.execute(
            update(EventRow).where(EventRow.id == world.ev_customer_a).values(probability=0.5)
        )
    pushed: list[Any] = []
    client = LabelStudioClient("http://ls:8080", "t", transport=_ls_transport(pushed))
    with session_scope(make_sessionmaker(world.engine)) as s:
        rep = run_sampler(s, CFG, WEEK, set(), push=client, out_dir=tmp_path)
    assert rep.candidates["false_positive"] == 1
    assert rep.candidates["false_negative"] == 1
    assert rep.candidates["low_confidence"] == 1
    assert rep.candidates["new_terms"] >= 1
    assert rep.pushed == sum(rep.selected.values()) == len(pushed[0])
    task = next(t for t in pushed[0] if t["data"]["bucket"] == "false_positive")
    assert task["data"]["event_id"] == world.ev_agent_a
    assert task["data"]["audio"].startswith("/data/local-files/?d=")
    assert task["predictions"][0]["result"][0]["from_name"] == "category"
    assert any("candidate_term" in t["data"] for t in pushed[0])
    assert (tmp_path / f"al_{WEEK}.jsonl").exists()
    # next run: nothing twice (only new-term items the per-call cap held back are left)
    with session_scope(make_sessionmaker(world.engine)) as s:
        again = run_sampler(s, CFG, WEEK, set())
        assert set(again.selected) <= {"new_terms"}
        assert len(list(s.scalars(select(AlSample)))) == rep.pushed
    # golden calls are never sampled
    with session_scope(make_sessionmaker(world.engine)) as s:
        s.execute(AlSample.__table__.delete())
    with session_scope(make_sessionmaker(world.engine)) as s:
        gold = run_sampler(s, CFG, WEEK, {world.call_a, world.call_b})
    assert sum(gold.selected.values()) == 0
    assert gold.golden_excluded >= 3


def test_sampler_cli_golden_manifest_gate(world: World, tmp_path: Path) -> None:
    s = _settings(world, tmp_path)
    with pytest.raises(GoldenManifestMissingError):
        al_sample(s, ["--week", WEEK])
    manifest = tmp_path / CFG.sampler.golden_manifest
    manifest.parent.mkdir(parents=True)
    rows = [
        {"call_id": world.call_a, "split": "golden"},
        {"call_id": "pbx-42", "split": "golden"},
        {"call_id": world.call_b, "split": "train"},
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    assert golden_call_ids(manifest) == {world.call_a, "pbx-42"}
    out = al_sample(s, ["--week", WEEK, "--out", str(tmp_path / "al")])
    assert out["week"] == WEEK
    assert al_sample(s, ["--week", WEEK, "--no-golden-manifest"])["week"] == WEEK
    with pytest.raises(SystemExit, match="dev/test only"):
        al_sample(s.model_copy(update={"env": "staging"}), ["--no-golden-manifest"])
    with pytest.raises(SystemExit, match="GALE_LS_TOKEN"):
        al_sample(s, ["--push"])


# --- export, registration, promotion ------------------------------------------------------------


def test_feedback_export(world: World, tmp_path: Path) -> None:
    _review(world, world.ev_agent_a, "correct")
    _review(world, world.ev_agent_b, "wrong_category")
    world.client.post(
        f"/api/v1/calls/{world.call_a}/missed",
        json={
            "speaker": "customer",
            "category": "threat",
            "severity": 3,
            "ts_start": 1,
            "ts_end": 2,
        },
        headers=world.h("dev-qa"),
    )
    since, until = datetime.now(UTC) - timedelta(hours=1), datetime.now(UTC) + timedelta(hours=1)
    with session_scope(make_sessionmaker(world.engine)) as s:
        rows, stats = export_feedback(s, since, until, set())
        _, gstats = export_feedback(s, since, until, {world.call_a})
    assert (stats.rows, stats.positives, stats.negatives) == (3, 2, 1)
    by = {r["uid"].split(":")[0] + ":" + str(r["label"]): r for r in rows}
    pos = next(r for r in rows if r["event_id"] == world.ev_agent_a)
    assert pos["label"] == 1
    assert "<GALI_1>" in pos["text"]
    assert pos["signals"][0]["source"] in ("lexicon", "dsp_shout")
    neg = next(r for r in rows if r["event_id"] == world.ev_agent_b)
    assert (neg["label"], neg["category"]) == (0, "insult")  # corrected category kept
    assert by["missed:1"]["verdict"] == "missed"
    assert gstats.golden_dropped == 2
    s2 = _settings(world, tmp_path)
    out = export_feedback_cmd(
        s2,
        [
            "--since",
            since.date().isoformat(),
            "--until",
            (until + timedelta(days=1)).date().isoformat(),
            "--out",
            str(tmp_path / "fb.jsonl"),
            "--no-golden-manifest",
        ],
    )
    assert out["rows"] == 3
    assert world.app.state.store.exists(out["key"])


def _cand(**kw: Any) -> Candidate:
    base: dict[str, Any] = {
        "component": "text_clf",
        "name": "BanglaBERT textclf",
        "version": "textclf-2026-12",
        "artifact_uri": "s3://gale/models/candidates/text_clf/textclf-2026-12/",
        "golden": {"macro_f1": 0.7},
        "gate": {"passed": True, "regressions": []},
        "training": {"rows": 1000, "golden_overlap": 0},
    }
    return Candidate.model_validate(base | kw)


def test_register_and_promote_candidate(world: World, tmp_path: Path) -> None:
    sessions = make_sessionmaker(world.engine)
    with session_scope(sessions) as s:
        with pytest.raises(RegistrationError, match="gate failed"):
            register_model(s, _cand(gate={"passed": False, "regressions": [{"metric": "x"}]}))
        with pytest.raises(RegistrationError, match="golden"):
            register_model(s, _cand(training={"golden_overlap": 3}))
    f = tmp_path / "candidate.json"
    f.write_text(_cand().model_dump_json(), "utf-8")
    out = register_model_cmd(world.settings, [str(f)])
    assert out == {
        "component": "text_clf",
        "version": "textclf-2026-12",
        "created": True,
        "shadow": True,
    }
    assert register_model_cmd(world.settings, [str(f)])["created"] is False
    with sessions() as s:
        mv = s.scalar(select(ModelVersion).where(ModelVersion.version == "textclf-2026-12"))
        assert mv is not None
        mv_id = str(mv.id)
    models = world.client.get(
        "/api/v1/models?component=text_clf", headers=world.h("dev-admin")
    ).json()
    assert next(m for m in models if m["id"] == mv_id)["shadow"] is True
    r = world.client.post(
        "/api/v1/models/text_clf/promote",
        json={"model_version_id": mv_id},
        headers=world.h("dev-admin2"),
    )
    assert r.status_code == 202
    ok = world.client.post(
        f"/api/v1/models/promotions/{r.json()['id']}/approve", headers=world.h("dev-admin")
    )
    assert ok.status_code == 200
    with sessions() as s:
        mv2 = s.get(ModelVersion, uuid.UUID(mv_id))
        assert mv2 is not None
        assert (mv2.active, mv2.shadow) == (True, False)
        failed = ModelVersion(
            component="fusion",
            name="calib",
            version="bad",
            metrics_jsonb={"gate": {"passed": False}},
            active=False,
            research_only=False,
        )
        s.add(failed)
        s.commit()
        failed_id = str(failed.id)
    r = world.client.post(
        "/api/v1/models/fusion/promote",
        json={"model_version_id": failed_id},
        headers=world.h("dev-admin"),
    )
    assert r.status_code == 409
    assert "gate" in r.json()["detail"]


# --- notifications ------------------------------------------------------------------------------


def test_webhook_and_email(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = compute(_week_data(), CFG)
    sent: list[Any] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(200)

    t = httpx.MockTransport(handler)
    assert send_webhook(WebhookNotify(enabled=True), "hi", payload, t) is False  # no URL
    monkeypatch.setenv("GALE_REPORT_WEBHOOK_URL", "https://hooks.example.invalid/x")
    assert send_webhook(WebhookNotify(enabled=True), "hi", payload, t)
    assert send_webhook(WebhookNotify(enabled=True, format="json"), "hi", payload, t)
    assert sent[0] == {"text": "hi"}
    assert set(sent[1]) == {"week", "stage", "headline", "exit"}

    class FakeSMTP:
        msgs: ClassVar[list[Any]] = []

        def __init__(self, *a: Any, **kw: Any) -> None:
            self.calls: list[str] = []

        def __enter__(self) -> FakeSMTP:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def starttls(self) -> None:
            self.calls.append("tls")

        def login(self, u: str, p: str) -> None:
            self.calls.append("login")

        def send_message(self, m: Any) -> None:
            FakeSMTP.msgs.append(m)

    cfg = EmailNotify(enabled=True, to=["qa@gale.internal"])
    assert send_email(EmailNotify(enabled=True), "s", "t", "<p>h</p>") is False  # no recipients
    monkeypatch.setenv("GALE_SMTP_USER", "u")
    monkeypatch.setenv("GALE_SMTP_PASSWORD", "p")
    assert send_email(cfg, "s", "t", "<p>h</p>", smtp=FakeSMTP)  # type: ignore[arg-type]
    assert FakeSMTP.msgs[0]["To"] == "qa@gale.internal"

    def boom(*a: Any, **kw: Any) -> Any:
        raise smtplib.SMTPException("down")

    monkeypatch.setattr("gale_api.pilot.notify.smtplib.SMTP", boom)
    monkeypatch.setenv("GALE_REPORT_WEBHOOK_URL", "")
    res = notify(Notify(email=cfg, webhook=WebhookNotify(enabled=True)), payload, "t", "<p/>")
    assert res == {"email": False, "webhook": False}
