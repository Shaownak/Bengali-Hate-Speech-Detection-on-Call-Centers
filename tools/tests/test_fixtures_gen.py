import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from galecore.audio import rms_dbfs
from galecore.schemas import Category, Speaker
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav, main
from tools.fixtures_gen.compose import compose, load_fake_lexicon, render
from tools.fixtures_gen.models import ExpectedEvents, Script
from tools.fixtures_gen.scenarios import SCENARIOS, Fill, Line, Scenario
from tools.fixtures_gen.tts import StubTTS, load_tts

TTS = StubTTS()


def _load(call_id: str) -> tuple[Script, ExpectedEvents]:
    d = CALLS_DIR / call_id
    return (
        Script.model_validate_json((d / "script.json").read_text(encoding="utf-8")),
        ExpectedEvents.model_validate_json(
            (d / "expected_events.json").read_text(encoding="utf-8")
        ),
    )


def test_committed_fixtures_match_generator(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check"]) == 0, capsys.readouterr().out


def test_ten_calls_of_two_to_four_minutes() -> None:
    assert len(SCENARIOS) == 10
    assert len({s.call_id for s in SCENARIOS}) == 10
    for sc in SCENARIOS:
        script, _ = _load(sc.call_id)
        assert 120 <= script.duration_s <= 240, sc.call_id
        assert script.synthetic is True
        assert script.sample_rate == 8000


def test_only_placeholders_from_the_fake_lexicon() -> None:
    lexicon = load_fake_lexicon()
    for sc in SCENARIOS:
        script, _ = _load(sc.call_id)
        for turn in script.turns:
            for tag in re.findall(r"<[^>]*>", turn.text):
                assert tag in lexicon, f"{sc.call_id}: {tag}"


def test_events_are_consistent_with_turns() -> None:
    for sc in SCENARIOS:
        script, expected = _load(sc.call_id)
        turns = {t.idx: t for t in script.turns}
        for e in expected.events:
            t = turns[e.turn_idx]  # the turn where the event starts
            assert e.speaker == t.speaker
            assert t.t_start - 1e-6 <= e.t_start < e.t_end <= script.duration_s, (sc.call_id, e)
            # merged / overtalk events may span turns, but end inside one of the speaker's turns
            assert any(
                u.speaker == e.speaker and u.t_start - 1e-6 <= e.t_end <= u.t_end + 1e-6
                for u in script.turns
            ), (sc.call_id, e)
            if not e.alert:
                assert e.context_flags, "non-alerting ground truth must say why"


def test_ground_truth_merges_like_fusion() -> None:
    # same speaker + category closer than merge_gap_s (5 s) is one event (thresholds.yaml)
    for sc in SCENARIOS:
        _, expected = _load(sc.call_id)
        evs = sorted(expected.events, key=lambda e: (e.speaker, e.category, e.t_start))
        for a, b in itertools.pairwise(evs):
            if (a.speaker, a.category, a.alert) == (b.speaker, b.category, b.alert):
                assert b.t_start - a.t_end >= 5.0, (sc.call_id, a, b)
    shouts = [e for e in _load("demo_05")[1].events if e.category is Category.SHOUTING]
    assert len(shouts) == 1
    assert shouts[0].evidence["merged"] == 3


def test_category_coverage_and_negative_control() -> None:
    seen: set[Category] = set()
    for sc in SCENARIOS:
        _, expected = _load(sc.call_id)
        seen |= {e.category for e in expected.events}
    assert {
        Category.SLANG_PROFANITY,
        Category.INSULT,
        Category.THREAT,
        Category.SEXUAL_HARASSMENT,
        Category.DISRESPECT_REGISTER,
        Category.SHOUTING,
        Category.INTERRUPTION_OVERTALK,
        Category.DISMISSIVE_RUDE,
    } <= seen
    assert _load("demo_02")[1].events == []
    by_speaker = {e.speaker for sc in SCENARIOS for e in _load(sc.call_id)[1].events}
    assert by_speaker == {Speaker.AGENT, Speaker.CUSTOMER}


def test_register_drop_events() -> None:
    _, expected = _load("demo_03")
    reg = [e for e in expected.events if e.category is Category.DISRESPECT_REGISTER]
    assert [(e.evidence["from"], e.evidence["to"], e.severity) for e in reg] == [
        ("apni", "tumi", 2),
        ("apni", "tui", 3),
    ]


def test_overtalk_needs_three_interruptions_in_a_minute() -> None:
    for call_id in ("demo_04", "demo_08"):
        _, expected = _load(call_id)
        ot = [e for e in expected.events if e.category is Category.INTERRUPTION_OVERTALK]
        assert len(ot) == 1, call_id
        assert ot[0].speaker is Speaker.CUSTOMER
        assert ot[0].evidence["interruptions"] == 3
        assert ot[0].t_end - ot[0].t_start <= 60
    # demo_08: the agent interrupts once, which is not an event (docs/02 §3.4)
    agent_ot = [
        e
        for e in _load("demo_08")[1].events
        if e.category is Category.INTERRUPTION_OVERTALK and e.speaker is Speaker.AGENT
    ]
    assert agent_ot == []


def test_render_is_deterministic_stereo_with_louder_shouting(tmp_path: Path) -> None:
    script, _ = _load("demo_01")
    a = render(script, TTS)
    assert a.dtype == np.int16
    assert a.shape == (round(script.duration_s * 8000), 2)
    assert np.array_equal(a, render(script, TTS))
    x = a.astype(np.float32) / 32768

    def level(turn_style: str) -> float:
        ts = [t for t in script.turns if t.speaker is Speaker.CUSTOMER and t.style == turn_style]
        return float(
            np.mean([rms_dbfs(x[round(t.t_start * 8000) : round(t.t_end * 8000), 1]) for t in ts])
        )

    assert level("shout") - level("normal") == pytest.approx(12.0, abs=1.5)
    assert np.abs(a).max() < 32767  # soft limiter, no hard clipping
    # the far channel only carries crosstalk while the customer talks
    t = next(t for t in script.turns if t.speaker is Speaker.CUSTOMER)
    seg = x[round(t.t_start * 8000) : round(t.t_end * 8000)]
    assert rms_dbfs(seg[:, 1]) - rms_dbfs(seg[:, 0]) > 25

    # ensure_wav renders a missing call.wav identical to render()
    d = tmp_path / "demo_01"
    d.mkdir()
    (d / "script.json").write_text(
        (CALLS_DIR / "demo_01" / "script.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    wav = ensure_wav(d)
    data, sr = sf.read(wav, dtype="int16")
    assert sr == 8000
    assert np.array_equal(data, a)


def test_compose_validates_register_and_overtalk() -> None:
    base = dict(call_id="t", title="t", description="t", seed=1, agent_f0=200, customer_f0=120)
    bad_pronoun = Scenario(**base, lines=[Line(Speaker.AGENT, "জি, বলুন।", register="tumi")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no such pronoun"):
        compose(bad_pronoun, TTS)
    bad_overtalk = Scenario(**base, lines=[Line(Speaker.AGENT, "জি।", gap=-0.5)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overtalk"):
        compose(bad_overtalk, TTS)
    unknown = Scenario(**base, lines=[Line(Speaker.AGENT, "<GALI_99> জি।")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fake lexicon"):
        compose(unknown, TTS)
    ok = Scenario(**base, lines=[Fill(2)])  # type: ignore[arg-type]
    script, events = compose(ok, TTS)
    assert len(script.turns) == 4
    assert events.events == []


def test_gen_cli_writes_three_files(tmp_path: Path) -> None:
    assert main(["gen", "--only", "demo_09", "--out", str(tmp_path), "--tts", "stub"]) == 0
    d = tmp_path / "demo_09"
    assert sorted(p.name for p in d.iterdir()) == [
        "call.wav",
        "expected_events.json",
        "script.json",
    ]
    events = json.loads((d / "expected_events.json").read_text(encoding="utf-8"))["events"]
    assert all(e["alert"] is False for e in events)


def test_load_tts(tmp_path: Path) -> None:
    assert load_tts("auto", None).name == "stub"
    assert load_tts("auto", tmp_path).name == "stub"  # no checkpoint there
    with pytest.raises(FileNotFoundError):
        load_tts("local", tmp_path)
