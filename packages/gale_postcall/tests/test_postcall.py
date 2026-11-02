"""Integration: the stub pipeline on every synthetic fixture matches ``expected_events.json``.

Event matching follows docs/06 §2: same speaker and category, and temporal IoU ≥ 0.3 or the
predicted centre within the gold span ± 2 s; one-to-one, greedy in time order.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from gale_llm_judge.client import JudgeUnavailableError
from gale_postcall.__main__ import main
from gale_postcall.models import build_models
from gale_postcall.pipeline import PostCallResult, run_postcall
from gale_postcall.report import to_html, to_json
from galecore.config import Settings
from galecore.models import FixtureOracle
from galecore.models.base import WindowInput
from galecore.schemas import EventStatus, FusedEvent, WindowJudgement
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav

SETTINGS = Settings(env="test", bus_backend="memory")
CALLS = sorted(p.name for p in CALLS_DIR.glob("demo_*"))


def matches(p: FusedEvent, g: dict[str, Any]) -> bool:
    if (p.speaker.value, p.category.value) != (g["speaker"], g["category"]):
        return False
    inter = max(0.0, min(p.ts_end, g["t_end"]) - max(p.ts_start, g["t_start"]))
    union = max(p.ts_end, g["t_end"]) - min(p.ts_start, g["t_start"])
    centre = (p.ts_start + p.ts_end) / 2
    return inter / union >= 0.3 or g["t_start"] - 2 <= centre <= g["t_end"] + 2


@pytest.fixture(scope="module")
def results() -> dict[str, PostCallResult]:
    return {c: run_postcall(ensure_wav(CALLS_DIR / c), SETTINGS) for c in CALLS}


@pytest.mark.parametrize("call", CALLS)
def test_events_match_ground_truth(results: dict[str, PostCallResult], call: str) -> None:
    r = results[call]
    gold = json.loads((CALLS_DIR / call / "expected_events.json").read_text(encoding="utf-8"))[
        "events"
    ]
    free = list(r.events)
    for g in sorted(gold, key=lambda g: g["t_start"]):
        hit = next((p for p in free if matches(p, g)), None)
        assert hit is not None, f"{call}: missed {g}"
        free.remove(hit)
        dismissed = hit.status == EventStatus.DISMISSED_BY_LLM
        assert dismissed == (not g["alert"]), f"{call}: {g} → {hit.status}"
    assert free == [], f"{call}: false positives {[(e.category.value, e.ts_start) for e in free]}"


def test_result_contents(results: dict[str, PostCallResult]) -> None:
    r = results["demo_01"]
    assert r.layout == "stereo_agent_left"
    assert r.sample_rate_in == 8000
    assert r.llm_status == "ok"
    assert r.scorecard is not None
    assert r.scorecard.customer.abuse_toward_agent is True
    assert set(r.stages_s) == {
        "load",
        "vad",
        "asr",
        "audio",
        "text",
        "fusion",
        "llm_window",
        "llm_call",
    }
    assert all(s.text_redacted == " ".join(w.word for w in s.words) for s in r.segments)
    assert {"loudness", "shout_p", "anger_p", "escalation", "overtalk"} <= set(r.timelines)
    assert max(p for _, p in r.timelines["escalation"]["customer"]) > 0.8
    assert results["demo_02"].events == []
    assert any(e.alert for e in r.events)
    assert results["demo_04"].timelines["overtalk"]


def test_report_json_and_html(results: dict[str, PostCallResult]) -> None:
    r = results["demo_03"]
    j = to_json(r)
    assert json.loads(json.dumps(j))["call_id"] == r.call_id
    assert len(j["events"]) == len(r.events)
    assert "audio_16k" not in j
    page = to_html(r)
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    assert "disrespect_register" in page
    assert "<svg" in page
    gali = to_html(results["demo_01"])
    assert "&lt;GALI_4&gt;" in gali  # transcript text is HTML-escaped…
    assert "<GALI_4>" not in gali  # …never raw markup
    assert "Advisory only" in page


class OfflineJudge:
    version = "llm:offline"

    def judge_window(self, window: WindowInput) -> WindowJudgement:
        raise JudgeUnavailableError("down")

    def judge_call(self, *_: object) -> object:  # pragma: no cover - never reached
        raise JudgeUnavailableError("down")


def test_llm_offline_keeps_events_provisional() -> None:
    wav = ensure_wav(CALLS_DIR / "demo_06")
    models = build_models(SETTINGS, FixtureOracle.for_audio(wav))
    models.llm = OfflineJudge()  # type: ignore[assignment]
    r = run_postcall(wav, SETTINGS, models=models)
    assert r.llm_status == "offline"
    assert r.scorecard is None
    assert r.events
    assert all(e.status == EventStatus.PROVISIONAL for e in r.events)
    assert "LLM judge offline" in r.notes[0]
    assert "Scorecard unavailable" in to_html(r)


def test_mono_recording(tmp_path: Path) -> None:
    sr = 8000
    x = np.zeros(sr * 4, np.float32)
    x[sr : 3 * sr] = 0.2 * np.sin(2 * np.pi * 180 * np.arange(2 * sr) / sr)
    wav = tmp_path / "mono.wav"
    sf.write(wav, x, sr, subtype="PCM_16")
    r = run_postcall(wav, SETTINGS)
    assert r.layout == "mono"
    assert r.events == []
    assert "mono recording" in r.notes[0]
    assert "M</text>" in to_html(r)


def test_cli_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ensure_wav(CALLS_DIR / "demo_07")
    assert (
        main(
            [
                "report",
                str(CALLS_DIR / "demo_07" / "call.wav"),
                "--out",
                str(tmp_path),
                "--call-id",
                "8f14e45f-ceea-467a-9e4b-2b1c3d4e5f60",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "threat" in out
    assert "ALERT" in out
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["call_id"].startswith(
        "8f14e45f"
    )
    assert (tmp_path / "report.html").is_file()
    assert main(["report", str(tmp_path / "missing.wav")]) == 2
