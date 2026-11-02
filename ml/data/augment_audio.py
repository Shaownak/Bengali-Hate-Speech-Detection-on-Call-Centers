"""Audio augmentation library (docs/02 §2.8). Every function takes an explicit
``np.random.Generator``, so a seed reproduces the exact output.

Codecs: G.711 µ-law / A-law in-process (``galecore.audio``); AMR-NB and GSM-FR through ffmpeg
(``libopencore_amrnb`` / ``libgsm``) or sox when installed. ``codec_available`` tells which.
Gain augmentation must not destroy shouting labels: ``gain(..., shout=True)`` refuses.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from galecore.audio import (
    decode_alaw,
    decode_ulaw,
    encode_alaw,
    encode_ulaw,
    float_to_int16,
    int16_to_float,
    resample,
)

Float = npt.NDArray[np.float32]
FFMPEG_CODECS = {
    "amr_nb": ("libopencore_amrnb", "amr", 8000, ["-ar", "8000", "-ac", "1", "-b:a", "12.2k"]),
    "gsm": ("libgsm", "gsm", 8000, ["-ar", "8000", "-ac", "1"]),
}


def g711(x: Float, sr: int, law: str = "ulaw") -> Float:
    y = resample(x, sr, 8000)
    pcm = float_to_int16(np.clip(y, -1, 1))
    y = int16_to_float(
        decode_ulaw(encode_ulaw(pcm)) if law == "ulaw" else decode_alaw(encode_alaw(pcm))
    )
    return resample(y, 8000, sr)


def codec_available(codec: str) -> bool:
    if codec in ("ulaw", "alaw"):
        return True
    return shutil.which("ffmpeg") is not None or (
        codec == "gsm" and shutil.which("sox") is not None
    )


def ffmpeg_codec(x: Float, sr: int, codec: str) -> Float:  # pragma: no cover - needs ffmpeg/sox
    """Round-trip through AMR-NB or GSM-FR with ffmpeg (or sox for GSM)."""
    enc, ext, csr, args = FFMPEG_CODECS[codec]
    with tempfile.TemporaryDirectory() as d:
        src, mid, dst = Path(d) / "in.wav", Path(d) / f"mid.{ext}", Path(d) / "out.wav"
        sf.write(src, resample(x, sr, csr), csr, subtype="PCM_16")
        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-c:a",
                    enc,
                    *args,
                    str(mid),
                ],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mid), str(dst)], check=True
            )
        else:
            subprocess.run(["sox", str(src), str(mid)], check=True)
            subprocess.run(["sox", str(mid), str(dst)], check=True)
        y, _ = sf.read(dst, dtype="float32")
    return resample(y, csr, sr)


def codec(x: Float, sr: int, name: str) -> Float:
    if name in ("ulaw", "alaw"):
        return g711(x, sr, name)
    if not codec_available(name):
        raise RuntimeError(f"{name} needs ffmpeg (or sox) on PATH")
    return ffmpeg_codec(x, sr, name)  # pragma: no cover


def packet_loss(
    x: Float,
    sr: int,
    rng: np.random.Generator,
    rate: float = 0.03,
    frame_ms: int = 20,
    conceal: str = "zero",
) -> Float:
    """Drop 20 ms frames with probability ``rate`` (1–5 %, docs/02); zero-fill or repeat."""
    y = x.copy()
    n = sr * frame_ms // 1000
    for i in range(0, len(y) - n + 1, n):
        if rng.random() < rate:
            y[i : i + n] = y[i - n : i] if conceal == "repeat" and i >= n else 0.0
    return y


def band_limit(x: Float, sr: int, lo: float = 300.0, hi: float = 3400.0) -> Float:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x)).astype(np.float32)


def add_noise(
    x: Float, rng: np.random.Generator, snr_db: float, noise: Float | None = None
) -> Float:
    """Mix ``noise`` (or white noise) at ``snr_db`` (docs/02: 5–20 dB)."""
    if noise is None:
        noise = rng.standard_normal(len(x)).astype(np.float32)
    elif len(noise) < len(x):
        noise = np.tile(noise, len(x) // len(noise) + 1)
    start = int(rng.integers(0, len(noise) - len(x) + 1))
    n = noise[start : start + len(x)]
    ps, pn = float(np.mean(x**2)) + 1e-12, float(np.mean(n**2)) + 1e-12
    return (x + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def speed(x: Float, sr: int, factor: float) -> Float:
    """Speed perturbation (0.9 / 1.0 / 1.1): resample and play at the original rate."""
    return x if factor == 1.0 else resample(x, round(sr * factor), sr)


def gain(x: Float, db: float, shout: bool = False) -> Float:
    if shout:
        raise ValueError("gain augmentation on shouting samples destroys the label (docs/02 §2.8)")
    return np.clip(x * 10 ** (db / 20), -1, 1).astype(np.float32)


def telephony_chain(x: Float, sr: int, rng: np.random.Generator, shout: bool = False) -> Float:
    """A random but seeded call-channel chain: band-limit → codec → packet loss → noise (→ gain)."""
    y = band_limit(x, sr)
    y = codec(y, sr, str(rng.choice(["ulaw", "alaw"])))
    y = packet_loss(y, sr, rng, rate=float(rng.uniform(0.01, 0.05)))
    y = add_noise(y, rng, snr_db=float(rng.uniform(5, 20)))
    if not shout:
        y = gain(y, float(rng.uniform(-10, 10)))
    return y
