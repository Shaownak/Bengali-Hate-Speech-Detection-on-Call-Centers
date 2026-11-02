"""Audio utilities: G.711 codecs, resampling, channel split, framing, PCM conversion.

Model audio is 16 kHz mono float32 in [-1, 1] per channel (docs/01 §2). On the bus it travels
as 16-bit little-endian PCM (D-02).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import soxr

from galecore.schemas import Channel

FloatArray = npt.NDArray[np.float32]
MODEL_SR = 16000
FRAME_MS = 200

ChannelLayout = Literal["stereo_agent_left", "stereo_agent_right", "mono"]


# ---------------------------------------------------------------------------
# G.711 µ-law / A-law (ITU-T G.711), table-driven
# ---------------------------------------------------------------------------


def _build_ulaw_table() -> npt.NDArray[np.int16]:
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        table[i] = -sample if sign else sample
    return table


def _build_alaw_table() -> npt.NDArray[np.int16]:
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        a = i ^ 0x55
        sign = a & 0x80
        exponent = (a >> 4) & 0x07
        mantissa = a & 0x0F
        if exponent == 0:
            sample = (mantissa << 4) + 8
        else:
            sample = ((mantissa << 4) + 0x108) << (exponent - 1)
        table[i] = sample if sign else -sample
    return table


_ULAW_DECODE = _build_ulaw_table()
_ALAW_DECODE = _build_alaw_table()


def decode_ulaw(data: bytes) -> npt.NDArray[np.int16]:
    """G.711 µ-law bytes → int16 PCM."""
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def decode_alaw(data: bytes) -> npt.NDArray[np.int16]:
    """G.711 A-law bytes → int16 PCM."""
    return _ALAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


# Segment end points from the CCITT reference implementation (g711.c)
_SEG_UEND = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_SEG_AEND = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def encode_ulaw(pcm: npt.NDArray[np.int16]) -> bytes:
    """int16 PCM → G.711 µ-law bytes (bit-exact with the CCITT reference)."""
    x = pcm.astype(np.int32) >> 2  # 14-bit
    mask = np.where(x < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(x), 8159) + 33
    seg = np.searchsorted(_SEG_UEND, mag, side="left")
    segc = np.minimum(seg, 7)
    uval = (segc << 4) | ((mag >> (segc + 1)) & 0x0F)
    out = np.where(seg >= 8, 0x7F, uval) ^ mask
    return out.astype(np.uint8).tobytes()


def encode_alaw(pcm: npt.NDArray[np.int16]) -> bytes:
    """int16 PCM → G.711 A-law bytes (bit-exact with the CCITT reference)."""
    x = pcm.astype(np.int32) >> 3  # 13-bit
    mask = np.where(x >= 0, 0xD5, 0x55)
    mag = np.where(x >= 0, x, -x - 1)
    seg = np.searchsorted(_SEG_AEND, mag, side="left")
    segc = np.minimum(seg, 7)
    quant = np.where(segc < 2, mag >> 1, mag >> segc) & 0x0F
    aval = (segc << 4) | quant
    out = np.where(seg >= 8, 0x7F, aval) ^ mask
    return out.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# PCM conversion & resampling
# ---------------------------------------------------------------------------


def int16_to_float(pcm: npt.NDArray[np.int16]) -> FloatArray:
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def float_to_int16(x: FloatArray) -> npt.NDArray[np.int16]:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def float_to_s16le(x: FloatArray) -> bytes:
    return float_to_int16(x).astype("<i2").tobytes()


def s16le_to_float(data: bytes) -> FloatArray:
    return int16_to_float(np.frombuffer(data, dtype="<i2").astype(np.int16))


def resample(x: FloatArray, sr_in: int, sr_out: int = MODEL_SR) -> FloatArray:
    """High-quality resample (soxr HQ). Works on 1-D (mono) or 2-D (samples × channels)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    y = soxr.resample(x, sr_in, sr_out, quality="HQ")
    return np.asarray(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Channels & framing
# ---------------------------------------------------------------------------


def split_channels(
    audio: FloatArray, layout: ChannelLayout = "stereo_agent_left"
) -> dict[Channel, FloatArray]:
    """Split audio (samples × channels, or 1-D mono) into per-speaker channels."""
    if audio.ndim == 1 or layout == "mono":
        mono = audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
        return {Channel.MIXED: mono}
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"expected stereo audio with shape (n, 2), got {audio.shape}")
    left = np.ascontiguousarray(audio[:, 0])
    right = np.ascontiguousarray(audio[:, 1])
    if layout == "stereo_agent_left":
        return {Channel.AGENT: left, Channel.CUSTOMER: right}
    return {Channel.AGENT: right, Channel.CUSTOMER: left}


@dataclass(frozen=True, slots=True)
class Frame:
    seq: int
    ts_start: float
    ts_end: float
    samples: FloatArray


def frames(
    x: FloatArray, sr: int = MODEL_SR, frame_ms: int = FRAME_MS, offset_s: float = 0.0
) -> Iterator[Frame]:
    """Cut mono audio into consecutive frames. The last frame may be shorter (not padded)."""
    if x.ndim != 1:
        raise ValueError("frames() expects mono audio")
    step = sr * frame_ms // 1000
    for seq, start in enumerate(range(0, len(x), step)):
        chunk = x[start : start + step]
        yield Frame(
            seq=seq,
            ts_start=offset_s + start / sr,
            ts_end=offset_s + (start + len(chunk)) / sr,
            samples=chunk,
        )


def rms_dbfs(x: FloatArray) -> float:
    """RMS level in dBFS (−inf-safe: silence returns −120)."""
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return max(-120.0, 20.0 * np.log10(rms)) if rms > 0 else -120.0
