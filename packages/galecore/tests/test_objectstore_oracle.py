import json
from pathlib import Path

import numpy as np
import pytest

from galecore.config import Settings
from galecore.models import FixtureOracle, RealModelsUnavailableError, load_models
from galecore.models.base import AudioChunk, WindowInput
from galecore.models.oracle import oracle_for_ref
from galecore.models.stubs import StubVAD
from galecore.objectstore import LocalObjectStore
from galecore.schemas import Category, Channel, EventStatus, FusedEvent

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "calls"


def test_local_object_store(tmp_path: Path) -> None:
    now = [1000.0]
    st = LocalObjectStore(tmp_path, secret=b"k" * 32, clock=lambda: now[0])
    st.put("reports/c1/report.json", b"{}")
    assert st.exists("reports/c1/report.json")
    assert st.get("reports/c1/report.json") == b"{}"
    assert not st.exists("reports/c1/nope")
    url = st.signed_url("reports/c1/report.json", ttl_s=60)
    assert st.verify(url) == "reports/c1/report.json"
    with pytest.raises(PermissionError):
        st.verify(url.replace("sig=", "sig=0"))
    with pytest.raises(PermissionError):
        st.verify("local://objects/reports/c1/report.json")
    now[0] += 61
    with pytest.raises(PermissionError, match="expired"):
        st.verify(url)
    for bad in ("../x", "/etc/passwd", "a\\b", ""):
        with pytest.raises(ValueError, match="invalid object key"):
            st.put(bad, b"")


def test_oracle_reads_fixture() -> None:
    o = FixtureOracle.for_audio(FIXTURES / "demo_03" / "call.wav")
    assert o is not None
    assert o.call_id == "demo_03"
    assert o.turns[0].channel is Channel.AGENT
    assert o.turns[0].words[0].token == "আসসালামু"
    ann = o.turn_annotations()
    assert list(ann.values()) == [[(Category.DISMISSIVE_RUDE, 2)]]
    assert FixtureOracle.for_audio(Path("elsewhere/call.wav")) is None


def test_oracle_refuses_non_synthetic(tmp_path: Path) -> None:
    (tmp_path / "script.json").write_text(
        json.dumps({"call_id": "x", "turns": []}), encoding="utf-8"
    )
    (tmp_path / "expected_events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic"):
        FixtureOracle.from_dir(tmp_path)


def test_oracle_for_ref_resolves_names_and_foreign_paths(tmp_path: Path) -> None:
    assert oracle_for_ref(None) is None
    assert oracle_for_ref("pbx:12345", FIXTURES) is None  # real calls never get an oracle
    by_name = oracle_for_ref("fixture:demo_03", FIXTURES)
    assert by_name is not None
    local = oracle_for_ref(f"fixture:{(FIXTURES / 'demo_03').resolve()}", tmp_path)
    assert local == by_name  # an existing absolute dir wins
    # the simulator ran on another host (CI runner → cluster): same fixture name under calls_dir
    foreign = "fixture:/home/runner/work/x/tests/fixtures/calls/demo_03"
    assert oracle_for_ref(foreign, FIXTURES) == by_name
    assert oracle_for_ref("fixture:/elsewhere/no_such_call", FIXTURES) is None


def test_stub_asr_and_llm_with_oracle() -> None:
    o = FixtureOracle.from_dir(FIXTURES / "demo_09")
    m = load_models(Settings(env="test"), o)
    t = o.turns[1]
    x = np.zeros(int((t.end + 1) * 16000), np.float32)
    segs = m.asr.transcribe(AudioChunk("c", t.channel, x[int(t.start * 16000) :], offset_s=t.start))
    assert segs[0].text == t.text
    assert segs[0].ts_start == t.start
    g = o.events[0]
    event = FusedEvent(
        event_id="e",
        call_id="c",
        speaker=g.speaker,
        category=g.category,
        severity=1,
        probability=0.9,
        ts_start=g.start,
        ts_end=g.end,
    )
    other = event.model_copy(update={"event_id": "f", "ts_start": 500.0, "ts_end": 501.0})
    j = m.llm.judge_window(WindowInput("c", [], [], [event, other]))
    assert [d.decision for d in j.provisional_decisions] == ["dismiss", "confirm"]
    assert "reported_speech" in (j.provisional_decisions[0].reason_bn or "")
    card = m.llm.judge_call(
        "c",
        [],
        [
            other.model_copy(update={"status": EventStatus.CONFIRMED}),
            event.model_copy(update={"status": EventStatus.DISMISSED_BY_LLM}),
        ],
    )
    assert card.customer.abuse_toward_agent is True
    assert card.escalation.who_escalated_first == "customer"
    assert card.agent.respectful_register_maintained is True


def test_stub_vad_and_real_mode_guard() -> None:
    sr = 16000
    x = np.random.default_rng(0).normal(0, 1e-4, sr * 4).astype(np.float32)
    x[sr : 2 * sr] += 0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr).astype(np.float32)
    regions = StubVAD().speech_regions(AudioChunk("c", Channel.AGENT, x, offset_s=10))
    assert len(regions) == 1
    assert regions[0][0] == pytest.approx(11.0, abs=0.05)
    assert StubVAD().speech_regions(AudioChunk("c", Channel.AGENT, np.zeros(10, np.float32))) == []
    with pytest.raises(RealModelsUnavailableError):
        load_models(Settings(env="test", model_mode="real"))
