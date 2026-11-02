import numpy as np

from gale_asr_worker.steps import to_asr_segment, transcribe, utterance_chunks
from galecore.models.base import AudioChunk, RawSegment, RawWord
from galecore.models.stubs import StubASR
from galecore.schemas import Channel


def raw(words: list[str]) -> RawSegment:
    ws = [RawWord(w, 1.0 + i, 1.5 + i, 0.9 - 0.1 * (i % 2)) for i, w in enumerate(words)]
    return RawSegment(Channel.CUSTOMER, 1.0, ws[-1].end, " ".join(words), ws, True, 0.85, "t")


def test_utterance_chunks_join_pad_and_cap() -> None:
    assert utterance_chunks([(1.0, 2.0), (2.5, 3.0), (5.0, 6.0)], 6.1) == [(0.8, 3.2), (4.8, 6.1)]
    long = [(float(i), i + 0.9) for i in range(0, 40)]
    chunks = utterance_chunks(long, 41.0)
    assert all(b - a <= 30.0 + 0.4 for a, b in chunks)
    assert len(chunks) == 2


def test_redaction_merges_pii_words_and_keeps_timing() -> None:
    seg = to_asr_segment(raw(["আমার", "নম্বর", "০১৭১২", "৩৪৫৬৭৮,", "ঠিক", "আছে"]), "c1")
    assert seg.text_redacted == "আমার নম্বর <PHONE>, ঠিক আছে"
    assert seg.text_redacted == " ".join(w.word for w in seg.words)
    phone = seg.words[2]
    assert (phone.word, phone.start, phone.end, phone.conf) == ("<PHONE>,", 3.0, 4.5, 0.8)
    assert "01712" not in seg.model_dump_json()
    assert (
        seg.segment_id == to_asr_segment(raw(["x"]), "c1").segment_id
    )  # stable per call/channel/start


def test_redaction_without_pii_normalises_only() -> None:
    seg = to_asr_segment(raw(["Hello", "আপনি"]), "c1")
    assert seg.text_redacted == "hello আপনি"
    assert [w.start for w in seg.words] == [1.0, 2.0]


def test_transcribe_with_stub() -> None:
    sr = 16000
    x = np.zeros(sr * 3, np.float32)
    x[sr : 2 * sr] = 0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    segs = transcribe(StubASR(), AudioChunk("c1", Channel.AGENT, x), "c1")
    assert len(segs) == 1
    assert segs[0].text_redacted.startswith("স্টাব")
