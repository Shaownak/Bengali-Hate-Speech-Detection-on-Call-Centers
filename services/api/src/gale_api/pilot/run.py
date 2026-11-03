"""CLI entry points of the pilot jobs (``python -m gale_api <command>``; Helm CronJobs, ADR-020).

All of them print counts and pass/fail only: never call ids, agents, terms or text.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from gale_api.db import PilotReport, make_engine, make_sessionmaker, session_scope
from gale_api.pilot import last_full_week, week_bounds
from galecore.config import Settings
from galecore.objectstore import store_from_settings
from galecore.pilot import load_pilot, pilot_path


def _sessions(s: Settings) -> Any:
    return make_sessionmaker(make_engine(s.db_url))


def pilot_report(s: Settings, argv: list[str]) -> dict[str, Any]:
    from gale_api.pilot.notify import notify
    from gale_api.pilot.report import (
        PromClient,
        collect,
        compute,
        prom_data,
        render_html,
        render_markdown,
        summary_text,
    )

    ap = argparse.ArgumentParser(prog="python -m gale_api pilot-report")
    ap.add_argument("--week", help="ISO week, e.g. 2026-W44 (default: last full week)")
    ap.add_argument("--notify", action="store_true", help="send email / webhook per pilot config")
    ap.add_argument("--out", type=Path, help="also write JSON/Markdown/HTML here")
    a = ap.parse_args(argv)
    cfg = load_pilot(pilot_path(s))
    week = a.week or last_full_week(datetime.now(UTC), cfg.report.timezone)
    _, end = week_bounds(week, cfg.report.timezone)
    prom_url = os.environ.get("GALE_PROMETHEUS_URL")
    prom = prom_data(PromClient(prom_url) if prom_url else None, min(end, datetime.now(UTC)))
    store = store_from_settings(s)
    with session_scope(_sessions(s)) as db:
        payload = compute(collect(db, cfg, week), cfg, prom)
        md, page = render_markdown(payload), render_html(payload)
        key = f"reports/pilot/{week}/report.html"
        store.put(key, page.encode("utf-8"))
        store.put(f"reports/pilot/{week}/report.json", json.dumps(payload).encode("utf-8"))
        row = db.scalar(select(PilotReport).where(PilotReport.week == week))
        if row is None:
            row = PilotReport(week=week)
            db.add(row)
        row.stage, row.payload_jsonb, row.html_key = cfg.stage, payload, key
        row.created_at = datetime.now(UTC)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"pilot_{week}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (a.out / f"pilot_{week}.md").write_text(md, encoding="utf-8")
        (a.out / f"pilot_{week}.html").write_text(page, encoding="utf-8")
    sent = {}
    if a.notify:
        link = cfg.report.notify.dashboard_url
        sent = notify(cfg.report.notify, payload, summary_text(payload, link), page)
    return {
        "week": week,
        "stage": cfg.stage,
        "exit": {x["criterion"]: x["status"] for x in payload["exit"]},
        "notified": sent,
    }


def al_sample(s: Settings, argv: list[str]) -> dict[str, Any]:
    from gale_api.pilot.sampler import golden_call_ids, report_dict, run_sampler
    from galecore.labelstudio import LabelStudioClient

    ap = argparse.ArgumentParser(prog="python -m gale_api al-sample")
    ap.add_argument("--week", help="ISO week (default: last full week)")
    ap.add_argument("--budget", type=int, help="items (default: sampler.budget)")
    ap.add_argument("--push", action="store_true", help="import the tasks into Label Studio")
    ap.add_argument("--out", type=Path, help="write the tasks as JSONL here")
    ap.add_argument(
        "--no-golden-manifest",
        action="store_true",
        help="dev/test only: run without the golden manifest",
    )
    a = ap.parse_args(argv)
    cfg = load_pilot(pilot_path(s))
    week = a.week or last_full_week(datetime.now(UTC), cfg.report.timezone)
    if a.no_golden_manifest:
        if s.env not in ("dev", "test"):
            raise SystemExit(
                "--no-golden-manifest is for GALE_ENV=dev/test only (CLAUDE.md rule 3)"
            )
        golden: set[str] = set()
    else:
        golden = golden_call_ids(s.data_dir / cfg.sampler.golden_manifest)
    client = None
    if a.push:
        token = os.environ.get("GALE_LS_TOKEN")
        if not token:
            raise SystemExit("set GALE_LS_TOKEN for our self-hosted Label Studio")
        client = LabelStudioClient(
            os.environ.get("GALE_LS_URL", cfg.sampler.labelstudio.url), token
        )
    with session_scope(_sessions(s)) as db:
        rep = run_sampler(db, cfg, week, golden, budget=a.budget, push=client, out_dir=a.out)
    return report_dict(rep)


def export_feedback_cmd(s: Settings, argv: list[str]) -> dict[str, Any]:
    from gale_api.pilot.export import export_feedback, to_jsonl
    from gale_api.pilot.sampler import golden_call_ids

    ap = argparse.ArgumentParser(prog="python -m gale_api export-feedback")
    ap.add_argument("--since", help="ISO date (default: yesterday 00:00 UTC)")
    ap.add_argument("--until", help="ISO date (default: today 00:00 UTC)")
    ap.add_argument("--out", type=Path, help="also write the JSONL here")
    ap.add_argument("--no-golden-manifest", action="store_true", help="dev/test only")
    a = ap.parse_args(argv)
    cfg = load_pilot(pilot_path(s))
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    until = datetime.fromisoformat(a.until).replace(tzinfo=UTC) if a.until else today
    since = (
        datetime.fromisoformat(a.since).replace(tzinfo=UTC)
        if a.since
        else until - timedelta(days=1)
    )
    if a.no_golden_manifest and s.env not in ("dev", "test"):
        raise SystemExit("--no-golden-manifest is for GALE_ENV=dev/test only (CLAUDE.md rule 3)")
    golden = (
        set() if a.no_golden_manifest else golden_call_ids(s.data_dir / cfg.sampler.golden_manifest)
    )
    with session_scope(_sessions(s)) as db:
        rows, stats = export_feedback(db, since, until, golden)
    data = to_jsonl(rows)
    key = f"datasets/feedback/{since.date()}_{until.date()}.jsonl"
    store_from_settings(s).put(key, data)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(data)
    return {"key": key, **stats.__dict__}


def register_model_cmd(s: Settings, argv: list[str]) -> dict[str, Any]:
    from gale_api.pilot.register import Candidate, register_model

    ap = argparse.ArgumentParser(prog="python -m gale_api register-model")
    ap.add_argument("candidate", type=Path, help="candidate.json from `python -m ml.retrain gate`")
    a = ap.parse_args(argv)
    cand = Candidate.model_validate_json(a.candidate.read_text(encoding="utf-8"))
    with session_scope(_sessions(s)) as db:
        row, created = register_model(db, cand)
        return {
            "component": row.component,
            "version": row.version,
            "created": created,
            "shadow": row.shadow,
        }


COMMANDS = {
    "pilot-report": pilot_report,
    "al-sample": al_sample,
    "export-feedback": export_feedback_cmd,
    "register-model": register_model_cmd,
}
