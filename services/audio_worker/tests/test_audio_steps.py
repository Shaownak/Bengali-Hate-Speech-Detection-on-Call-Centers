import numpy as np
import pytest

from gale_audio_worker.dsp import f0_track, speech_ratio, window_features, windows
from gale_audio_worker.overtalk import overtalk_signals, own_speech_regions
from gale_audio_worker.shout import DSPShoutDetector, rule_score, suppress_line_artefacts
from galecore.models.base import AudioChunk
from galecore.models.stubs import StubVAD
from galecore.schemas import Category, Channel, Signal, SignalSource

SR = 16000


def voice(seconds: float, f0: float, db: float, seed: int = 0, pauses: bool = True) -> np.ndarray:
    """Harmonic voice with a 4 Hz syllable envelope at roughly ``db`` dBFS RMS."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6)) / k for k in range(1, 12))
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t) ** 2)  # sin² at 2 Hz: a 4 Hz syllable rate
    x = x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)
    if pauses:  # 0.4 s pause every 2 s, like turn-internal breathing pauses
        x[(t % 2.0) > 1.6] = 0.0
    return (x + rng.normal(0, 10 ** (-65 / 20), len(x))).astype(np.float32)


def test_f0_track_finds_the_pitch() -> None:
    f0 = f0_track(voice(1.0, 150.0, -20, pauses=False), SR)
    assert np.median(f0[f0 > 0]) == pytest.approx(150.0, rel=0.05)
    assert f0_track(np.zeros(SR, np.float32), SR).max() == 0.0


def test_window_features_levels() -> None:
    x = voice(1.5, 150.0, -20, pauses=False)
    f = window_features(x, SR, f0_track(x, SR))
    assert f["rms_db"] == pytest.approx(-20, abs=0.5)
    assert f["clip_ratio"] == 0.0
    assert 3 <= f["rate_hz"] <= 5
    assert speech_ratio(0, 2, [(0.5, 1.5)]) == 0.5
    ws = windows(np.concatenate([np.zeros(2 * SR, np.float32), x]), SR, [(2.0, 3.5)])
    assert ws[0].speech_ratio == 0.0
    assert ws[2].speech_ratio == pytest.approx(1 / 3, abs=0.01)
    assert ws[0].feats["f0_med_st"] == 0.0  # no speech features there…
    assert ws[0].feats["rms_db"] < -100  # …but the level is kept for the loudness timeline


def test_rule_score_monotonic() -> None:
    z = dict.fromkeys(["rms_db", "f0_med_st"], 0.0)
    assert rule_score(z) < 0.05
    assert rule_score({"rms_db": 6.0, "f0_med_st": 5.0}) > 0.95
    assert rule_score({"rms_db": 3.0, "f0_med_st": 0.0}) < 0.5  # loud handset alone is not shouting


def test_dsp_shout_detects_raised_voice_against_baseline() -> None:
    calm = voice(30.0, 120.0, -28)
    shout = voice(4.0, 162.0, -16, seed=1)  # +12 dB, ×1.35 pitch
    x = np.concatenate([calm, shout, voice(6.0, 120.0, -28, seed=2)])
    det = DSPShoutDetector(StubVAD())
    sig = det.detect(AudioChunk("c", Channel.CUSTOMER, x))
    assert sig, "no shouting found"
    assert all(s.source is SignalSource.DSP_SHOUT and s.category is Category.SHOUTING for s in sig)
    hot = [s for s in sig if s.score >= 0.8]
    assert min(s.ts_start for s in hot) >= 28.0
    assert max(s.ts_end for s in hot) <= 36.0
    assert hot[0].evidence.features["loudness_z"] > 3
    assert det.version == "dsp-shout-rule-0.1"
    tl = det.timeline[("c", Channel.CUSTOMER)]
    assert len(tl) == len(windows(x, SR, [(0, 40)]))


def test_dsp_shout_quiet_or_short_calls() -> None:
    det = DSPShoutDetector(StubVAD())
    assert det.detect(AudioChunk("c", Channel.AGENT, np.zeros(SR * 5, np.float32))) == []
    # too little speech for a baseline: no scores at all
    assert det.detect(AudioChunk("d", Channel.AGENT, voice(2.0, 120, -28))) == []


def test_suppress_line_artefacts() -> None:
    def s(ch: Channel, t: float) -> Signal:
        return Signal(
            call_id="c",
            channel=ch,
            signal_id=f"{ch}{t}",
            source=SignalSource.DSP_SHOUT,
            category=Category.SHOUTING,
            score=0.9,
            ts_start=t,
            ts_end=t + 1.5,
            model_version="t",
        )

    a, b = suppress_line_artefacts(
        [s(Channel.AGENT, 10), s(Channel.AGENT, 20)], [s(Channel.CUSTOMER, 10.1)]
    )
    assert [x.ts_start for x in a] == [20]
    assert b == []


def test_overtalk_and_crosstalk_guard() -> None:
    n = 20 * SR
    agent = np.zeros(n, np.float32)
    cust = np.zeros(n, np.float32)
    agent[2 * SR : 8 * SR] = voice(6.0, 200, -26)
    cust[5 * SR : 10 * SR] = voice(5.0, 120, -26, seed=3)  # starts 3 s into the agent's turn
    cust[12 * SR : 16 * SR] = voice(4.0, 120, -26, seed=4)
    agent[12 * SR : 16 * SR] = 10 ** (-32 / 20) * cust[12 * SR : 16 * SR]  # far-end echo only
    vad = StubVAD()
    regions = {
        ch: vad.speech_regions(AudioChunk("c", ch, x))
        for ch, x in ((Channel.AGENT, agent), (Channel.CUSTOMER, cust))
    }
    sig = overtalk_signals("c", regions, {Channel.AGENT: agent, Channel.CUSTOMER: cust}, SR)
    assert [(s.channel, round(s.ts_start)) for s in sig] == [(Channel.CUSTOMER, 5)]
    # agent speaks until 8 s, but its last 0.4 s is a pause: 7.6 - 5.0 ≈ 2.6 s of real overlap
    assert sig[0].evidence.features["overlap_s"] == pytest.approx(2.6, abs=0.15)
    # the echo on the agent line is not the agent speaking
    raw = [(12.0, 16.0)]
    assert own_speech_regions(raw, agent, cust, SR) == []
    assert own_speech_regions(raw, cust, agent, SR)[0][0] == pytest.approx(12.0, abs=0.05)
