import numpy as np
import pytest

from galecore.config import Settings
from galecore.models import RealModelsUnavailableError, load_models
from galecore.models.base import AudioChunk, TextContext, WindowInput
from galecore.schemas import AsrSegment, Category, Channel, FusedEvent

SR = 16000


def tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def call_audio() -> np.ndarray:
    # 4 s normal speech level, 2 s silence, 2 s +20 dB "shout", 2 s normal
    return np.concatenate(
        [tone(4, 0.02), np.zeros(2 * SR, np.float32), tone(2, 0.2), tone(2, 0.02)]
    )


def seg(text: str, channel: Channel = Channel.AGENT, i: int = 0) -> AsrSegment:
    return AsrSegment(
        call_id="c", channel=channel, segment_id=f"s{i}", ts_start=i, ts_end=i + 1,
        text_redacted=text, is_final=True, avg_conf=0.9, model_version="t",
    )  # fmt: skip


@pytest.fixture
def models():  # type: ignore[no-untyped-def]
    return load_models(Settings(env="test", model_mode="stub"))


def test_real_mode_not_yet_available() -> None:
    with pytest.raises(RealModelsUnavailableError):
        load_models(Settings(env="test", model_mode="real"))


def test_stub_asr_is_deterministic_and_finds_voiced_regions(models) -> None:  # type: ignore[no-untyped-def]
    chunk = AudioChunk("c", Channel.AGENT, call_audio())
    a, b = models.asr.transcribe(chunk), models.asr.transcribe(chunk)
    assert a == b
    assert len(a) == 2  # silence gap splits the call into two regions
    assert a[0].ts_start == 0.0
    assert all(w.start <= w.end for s in a for w in s.words)


def test_stub_shout_and_ser_fire_only_on_loud_part(models) -> None:  # type: ignore[no-untyped-def]
    chunk = AudioChunk("c", Channel.CUSTOMER, call_audio(), offset_s=100.0)
    shouts = models.shout.detect(chunk)
    assert shouts
    assert all(s.category == Category.SHOUTING for s in shouts)
    # loud part sits at 106-108 s; every flagged 1.5 s window must overlap it, none elsewhere
    assert all(s.ts_end > 106.0 and s.ts_start < 108.0 for s in shouts)
    assert min(s.ts_start for s in shouts) >= 104.5
    assert models.shout.detect(chunk) == shouts  # deterministic ids too
    assert {s.category for s in models.ser.detect(chunk)} == {Category.HARSH_TONE}


def test_stub_shout_silence(models) -> None:  # type: ignore[no-untyped-def]
    assert models.shout.detect(AudioChunk("c", Channel.AGENT, np.zeros(SR * 3, np.float32))) == []
    assert models.kws.detect(AudioChunk("c", Channel.AGENT, call_audio())) == []
    assert models.audio_tox.detect(AudioChunk("c", Channel.AGENT, call_audio())) == []


def test_stub_text_classifier(models) -> None:  # type: ignore[no-untyped-def]
    out = models.text_clf.classify(TextContext("তুই একটা <GALI_3>", Channel.AGENT))
    assert out.scores[Category.SLANG_PROFANITY] > 0.9
    assert out.scores[Category.DISRESPECT_REGISTER] == 0.6
    assert out.severity == 2
    clean = models.text_clf.classify(TextContext("আপনার রিচার্জ হয়েছে", Channel.AGENT))
    assert max(clean.scores.values()) < 0.1
    assert clean.severity == 0


def test_stub_register_agent_drop(models) -> None:  # type: ignore[no-untyped-def]
    reg = models.register
    assert reg.observe(seg("আপনি একটু অপেক্ষা করেন", i=0)) == []
    sig = reg.observe(seg("তুমি বুঝো না?", i=1))
    assert len(sig) == 1
    assert sig[0].category == Category.DISRESPECT_REGISTER
    assert sig[0].evidence.features == {"from_level": 0, "to_level": 1}
    assert reg.observe(seg("তুমি শোনো", i=2)) == []  # same level: no new signal


def test_stub_register_customer_tumi_is_not_flagged(models) -> None:  # type: ignore[no-untyped-def]
    reg = models.register
    assert reg.observe(seg("আপনি", Channel.CUSTOMER, 0)) == []
    assert reg.observe(seg("তুমি", Channel.CUSTOMER, 1)) == []
    assert len(reg.observe(seg("তুই", Channel.CUSTOMER, 2))) == 1


def test_stub_llm(models) -> None:  # type: ignore[no-untyped-def]
    ev = FusedEvent(
        event_id="e1", call_id="c", speaker="agent", category="insult", severity=2,
        probability=0.8, ts_start=0, ts_end=1,
    )  # fmt: skip
    j = models.llm.judge_window(WindowInput("c", [seg("x")], [], [ev]))
    assert [d.decision for d in j.provisional_decisions] == ["confirm"]
    card = models.llm.judge_call("c", [seg("x")])
    assert card.escalation.who_escalated_first == "none"
