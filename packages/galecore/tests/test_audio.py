import warnings

import numpy as np
import pytest

from galecore.audio import (
    decode_alaw,
    decode_ulaw,
    encode_alaw,
    encode_ulaw,
    float_to_s16le,
    frames,
    resample,
    rms_dbfs,
    s16le_to_float,
    split_channels,
)
from galecore.schemas import Channel

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # stdlib reference implementation (removed in 3.13; we're on 3.12)

ALL_BYTES = bytes(range(256))


def sine(freq: float, sr: int, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestG711:
    def test_ulaw_decode_matches_reference(self) -> None:
        ref = np.frombuffer(audioop.ulaw2lin(ALL_BYTES, 2), dtype="<i2")
        assert np.array_equal(decode_ulaw(ALL_BYTES), ref)

    def test_alaw_decode_matches_reference(self) -> None:
        ref = np.frombuffer(audioop.alaw2lin(ALL_BYTES, 2), dtype="<i2")
        assert np.array_equal(decode_alaw(ALL_BYTES), ref)

    def test_ulaw_encode_matches_reference(self) -> None:
        pcm = np.arange(-32768, 32768, 7, dtype=np.int16)
        assert encode_ulaw(pcm) == audioop.lin2ulaw(pcm.astype("<i2").tobytes(), 2)

    def test_alaw_encode_matches_reference(self) -> None:
        pcm = np.arange(-32768, 32768, 7, dtype=np.int16)
        assert encode_alaw(pcm) == audioop.lin2alaw(pcm.astype("<i2").tobytes(), 2)

    @pytest.mark.parametrize(
        ("enc", "dec"), [(encode_ulaw, decode_ulaw), (encode_alaw, decode_alaw)]
    )
    def test_round_trip_snr(self, enc, dec) -> None:  # type: ignore[no-untyped-def]
        x = (sine(440, 8000, 0.5) * 32767).astype(np.int16)
        y = dec(enc(x)).astype(np.float64)
        noise = y - x
        snr = 10 * np.log10(np.mean(x.astype(np.float64) ** 2) / np.mean(noise**2))
        assert snr > 30  # G.711 gives ~38 dB on a loud tone


class TestPcmAndResample:
    def test_s16le_round_trip(self) -> None:
        x = sine(300, 16000, 0.1)
        y = s16le_to_float(float_to_s16le(x))
        assert np.max(np.abs(x - y)) < 1e-4

    def test_s16le_clips(self) -> None:
        y = s16le_to_float(float_to_s16le(np.array([2.0, -2.0], dtype=np.float32)))
        assert y[0] == pytest.approx(32767 / 32768)
        assert y[1] == -1.0

    def test_resample_8k_to_16k_length_and_frequency(self) -> None:
        x = sine(1000, 8000, 1.0)
        y = resample(x, 8000, 16000)
        assert y.dtype == np.float32
        assert abs(len(y) - 16000) <= 2
        spectrum = np.abs(np.fft.rfft(y))
        peak_hz = np.argmax(spectrum) * 16000 / len(y)
        assert abs(peak_hz - 1000) < 5

    def test_resample_noop(self) -> None:
        x = sine(100, 16000, 0.1)
        assert np.array_equal(resample(x, 16000, 16000), x)

    def test_resample_stereo(self) -> None:
        x = np.stack([sine(300, 8000, 0.5), sine(600, 8000, 0.5)], axis=1)
        assert resample(x, 8000, 16000).shape[1] == 2


class TestChannelsAndFrames:
    def test_split_agent_left(self) -> None:
        st = np.stack([np.ones(10, np.float32), -np.ones(10, np.float32)], axis=1)
        ch = split_channels(st)
        assert ch[Channel.AGENT][0] == 1.0
        assert ch[Channel.CUSTOMER][0] == -1.0

    def test_split_agent_right(self) -> None:
        st = np.stack([np.ones(10, np.float32), -np.ones(10, np.float32)], axis=1)
        assert split_channels(st, "stereo_agent_right")[Channel.AGENT][0] == -1.0

    def test_mono(self) -> None:
        assert list(split_channels(np.zeros(5, np.float32))) == [Channel.MIXED]
        st = np.stack([np.ones(4, np.float32), np.zeros(4, np.float32)], axis=1)
        assert split_channels(st, "mono")[Channel.MIXED][0] == 0.5

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="stereo"):
            split_channels(np.zeros((4, 3), np.float32))

    def test_frames_200ms(self) -> None:
        x = np.zeros(16000 + 1600, np.float32)  # 1.1 s
        fs = list(frames(x, offset_s=10.0))
        assert len(fs) == 6
        assert fs[0].ts_start == 10.0
        assert fs[0].ts_end == pytest.approx(10.2)
        assert len(fs[0].samples) == 3200
        assert len(fs[-1].samples) == 1600  # last frame not padded
        assert [f.seq for f in fs] == list(range(6))

    def test_frames_rejects_stereo(self) -> None:
        with pytest.raises(ValueError, match="mono"):
            list(frames(np.zeros((10, 2), np.float32)))

    def test_rms_dbfs(self) -> None:
        assert rms_dbfs(np.zeros(100, np.float32)) == -120.0
        assert rms_dbfs(np.array([], np.float32)) == -120.0
        assert rms_dbfs(np.ones(100, np.float32)) == pytest.approx(0.0)
        assert rms_dbfs(sine(100, 16000, 1, amp=1.0)) == pytest.approx(-3.01, abs=0.05)
