"""Pilot routing through the whole streaming stack (ADR-020): shadow mode hides every event of a
hidden queue, shadow candidates publish tagged signals and never change the stable events, and a
canary fusion calibration still yields the same events on stub models."""

from __future__ import annotations

import uuid
from pathlib import Path

import yaml

from galecore import metrics as m
from galecore.bus import SIGNALS
from galecore.config import Settings
from galecore.schemas import EventStatus, Signal
from tests.integration.streaming_harness import SETTINGS, Stack, play_call, wait_quiet
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav

WORKERS = ("asr", "audio", "text", "llm", "fusion")


def _settings(tmp: Path, cfg: dict[str, object]) -> Settings:
    f = tmp / "pilot.yaml"
    f.write_text(yaml.safe_dump(cfg), "utf-8")
    return SETTINGS.model_copy(update={"pilot_config": f})


async def _run(settings: Settings, fixture: str = "demo_01") -> tuple[Stack, str]:
    st = Stack(settings=settings)
    for n in WORKERS:
        await st.start(n)
    cid = str(uuid.uuid4())
    await play_call(st, ensure_wav(CALLS_DIR / fixture), cid, f"fixture:{fixture}", speed=0)
    await wait_quiet(st, 90, settle=4.0)
    return st, cid


def _count(component: str, version: str, outcome: str) -> float:
    v = m.SHADOW_COMPARE.labels(component, version, outcome)._value.get()
    return float(v)


async def test_shadow_mode_hides_events_and_candidates_never_drive_them(tmp_path: Path) -> None:
    before = _count("text_clf", "tc-shadow", "agree") + _count("llm_judge", "lj-shadow", "agree")
    settings = _settings(
        tmp_path,
        {
            "stage": "P-1",
            "alerts": {"default": "hidden"},
            "candidates": {
                "text_clf": {"version": "tc-shadow", "role": "shadow", "percent": 100},
                "llm_judge": {"version": "lj-shadow", "role": "shadow", "percent": 100},
                "fusion": {"version": "fu-shadow", "role": "shadow", "percent": 100},
            },
        },
    )
    st, cid = await _run(settings)
    events = [e for e in (await st.events()).values() if e.call_id == cid]
    alerts = [a for a in await st.alerts() if a.call_id == cid]
    signals = [s for s in await st.read(SIGNALS, Signal) if s.call_id == cid]
    await st.shutdown()

    assert events
    assert alerts
    assert all(e.shadow for e in events)
    assert all(a.shadow for a in alerts)
    shadow_sigs = [s for s in signals if s.shadow]
    assert shadow_sigs
    assert {s.model_version for s in shadow_sigs} <= {"tc-shadow", "lj-shadow"}
    assert all(s.signal_id.endswith(("~tc-shadow", "~lj-shadow")) for s in shadow_sigs)
    # no event was built from a shadow signal
    shadow_ids = {s.signal_id for s in shadow_sigs}
    assert not any(set(e.contributing_signals) & shadow_ids for e in events)
    after = _count("text_clf", "tc-shadow", "agree") + _count("llm_judge", "lj-shadow", "agree")
    assert after > before  # stub candidate == stable model: they agree
    assert _count("fusion", "fu-shadow", "agree") > 0


async def test_canary_and_visible_queue_match_the_baseline(tmp_path: Path) -> None:
    base, cid0 = await _run(SETTINGS)
    want = sorted(
        (e.speaker, e.category, e.status, e.alert)
        for e in (await base.events()).values()
        if e.call_id == cid0
    )
    await base.shutdown()
    settings = _settings(
        tmp_path,
        {
            "alerts": {"default": "visible"},
            "candidates": {
                "text_clf": {"version": "tc-canary", "role": "canary", "percent": 100},
                "fusion": {"version": "fu-canary", "role": "canary", "percent": 100},
                "llm_judge": {"version": "lj-canary", "role": "canary", "percent": 100},
            },
        },
    )
    st, cid = await _run(settings)
    events = [e for e in (await st.events()).values() if e.call_id == cid]
    signals = [s for s in await st.read(SIGNALS, Signal) if s.call_id == cid]
    await st.shutdown()
    got = sorted((e.speaker, e.category, e.status, e.alert) for e in events)
    assert got == want  # stub candidates behave like the stable stubs
    assert not any(e.shadow for e in events)
    assert not any(s.shadow for s in signals)
    clf = {s.model_version for s in signals if s.source.value == "text_clf"}
    assert clf == {"tc-canary"}
    assert any(e.status == EventStatus.CONFIRMED for e in events)
