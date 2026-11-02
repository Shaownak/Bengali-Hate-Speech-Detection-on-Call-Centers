"""KWS front end (numpy): telephony simulation and log-mel features (docs/03 §2b).

BAAD clips are clean isolated words; the call channel is 8 kHz G.711 with noise, so every training
clip goes through ``telephony`` (resample 16→8→16 kHz, µ-law or A-law, line noise, gain jitter).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from galecore.audio import (
    decode_alaw,
    decode_ulaw,
    encode_alaw,
    encode_ulaw,
    float_to_int16,
    int16_to_float,
    resample,
)

SR = 16000
N_FFT, HOP, N_MELS = 400, 160, 40
CLIP_S = 1.5
Float = npt.NDArray[np.float32]


def mel_filterbank(
    sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = 60, fmax: float = 3800
) -> Float:
    """Triangular mel filters, limited to the telephone band."""

    def hz2mel(f: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return 2595 * np.log10(1 + np.asarray(f) / 700)

    def mel2hz(m: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return 700 * (10 ** (np.asarray(m) / 2595) - 1)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            fb[m - 1, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            fb[m - 1, k] = (c - k) / max(1, c - b)
    return fb


_FB = mel_filterbank()


def log_mel(x: Float) -> Float:
    """(frames, N_MELS) log-mel energies, mean-normalised per clip."""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = np.log(power @ _FB.T + 1e-6)
    return (mel - mel.mean(axis=0, keepdims=True)).astype(np.float32)


def fix_length(x: Float, seconds: float = CLIP_S, rng: np.random.Generator | None = None) -> Float:
    n = int(seconds * SR)
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1)) if rng is not None else (len(x) - n) // 2
        return x[start : start + n]
    pad = n - len(x)
    left = int(rng.integers(0, pad + 1)) if rng is not None else pad // 2
    return np.pad(x, (left, pad - left)).astype(np.float32)


def telephony(x: Float, rng: np.random.Generator) -> Float:
    """16 kHz → 8 kHz G.711 (random law) with line noise and gain jitter → 16 kHz."""
    y = resample(x * float(10 ** (rng.uniform(-6, 6) / 20)), SR, 8000)
    pcm = float_to_int16(np.clip(y, -1, 1))
    y = int16_to_float(
        decode_ulaw(encode_ulaw(pcm)) if rng.random() < 0.5 else decode_alaw(encode_alaw(pcm))
    )
    y = y + rng.normal(0, 10 ** (rng.uniform(-60, -35) / 20), len(y)).astype(np.float32)
    return resample(y, 8000, SR)
