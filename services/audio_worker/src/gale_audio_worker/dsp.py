"""DSP features for shouting (docs/03 §3): per 1.5 s window, 0.5 s hop, per channel.

Loudness (RMS dBFS; a LUFS proxy at telephone bandwidth), peak and clipping ratio, F0 (median,
p90, range in semitones, autocorrelation pitch on 40 ms frames), spectral tilt, HF energy ratio
(2–4 kHz over 0.3–1 kHz), spectral flatness and a syllable-rate proxy (envelope modulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
WIN_S = 1.5
HOP_S = 0.5
FEATURES = (
    "rms_db",
    "peak_db",
    "clip_ratio",
    "f0_med_st",
    "f0_p90_st",
    "f0_range_st",
    "tilt_db_oct",
    "hf_ratio_db",
    "flatness",
    "rate_hz",
)


@dataclass(frozen=True, slots=True)
class Window:
    t_start: float
    t_end: float
    speech_ratio: float  # share of the window inside VAD speech regions
    feats: dict[str, float]


def _db(x: float) -> float:
    return float(10 * np.log10(max(x, 1e-12)))


F0_HOP_S = 0.01


def f0_track(x: FloatArray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> FloatArray:
    """Autocorrelation pitch per 40 ms frame (10 ms hop), FFT-based and vectorised; 0 where
    unvoiced. Computed once per channel and sliced per window."""
    n, hop = int(0.04 * sr), int(F0_HOP_S * sr)
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 1)
    count = max(0, (len(x) - n) // hop + 1)
    out = np.zeros(count, dtype=np.float32)
    idx = np.arange(n)
    for b in range(0, count, 2048):
        starts = np.arange(b, min(count, b + 2048)) * hop
        fr = x[starts[:, None] + idx].astype(np.float64)
        fr -= fr.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(fr, 2 * n, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
        energy = ac[:, 0]
        k = np.argmax(ac[:, lo:hi], axis=1) + lo
        peak = ac[np.arange(len(k)), k]
        voiced = (energy > 1e-6) & (peak > 0.45 * np.maximum(energy, 1e-12))
        out[b : b + len(k)] = np.where(voiced, sr / k, 0.0)
    return out


def window_features(x: FloatArray, sr: int, f0: FloatArray) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1 / sr)

    def band(a: float, b: float) -> float:
        m = (freqs >= a) & (freqs < b)
        return float(spec[m].sum()) if m.any() else 1e-12

    sel = (freqs >= 100) & (freqs <= min(4000, sr / 2 - 1))
    slope = (
        float(np.polyfit(np.log2(freqs[sel]), 10 * np.log10(spec[sel]), 1)[0])
        if sel.sum() > 2
        else 0.0
    )
    voiced = f0[f0 > 0]
    st = 12 * np.log2(voiced / 100.0) if len(voiced) else np.zeros(1)
    env = np.abs(x)
    k = max(1, int(0.02 * sr))
    env = np.convolve(env, np.ones(k) / k, mode="same")[::k]
    env_spec = np.abs(np.fft.rfft(env - env.mean())) if len(env) > 4 else np.zeros(2)
    env_freqs = np.fft.rfftfreq(len(env), k / sr) if len(env) > 4 else np.zeros(2)
    m = (env_freqs >= 2) & (env_freqs <= 10)
    rate = float(env_freqs[m][np.argmax(env_spec[m])]) if m.any() else 0.0
    return {
        "rms_db": _db(rms**2),
        "peak_db": _db(peak**2),
        "clip_ratio": float(np.mean(np.abs(x) > 0.98)) if len(x) else 0.0,
        "f0_med_st": float(np.median(st)),
        "f0_p90_st": float(np.percentile(st, 90)),
        "f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "tilt_db_oct": slope,
        "hf_ratio_db": _db(band(2000, 4000) / band(300, 1000)),
        "flatness": float(np.exp(np.mean(np.log(spec[sel]))) / np.mean(spec[sel]))
        if sel.any()
        else 0.0,
        "rate_hz": rate,
    }


def speech_ratio(t0: float, t1: float, regions: list[tuple[float, float]]) -> float:
    covered = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in regions)
    return covered / max(1e-9, t1 - t0)


def windows(
    x: FloatArray, sr: int, regions: list[tuple[float, float]], offset_s: float = 0.0
) -> list[Window]:
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    f0 = f0_track(x, sr)
    per_s = 1 / F0_HOP_S
    out = []
    for i in range(0, max(1, len(x) - win + 1), hop):
        t0 = offset_s + i / sr
        t1 = t0 + min(win, len(x) - i) / sr
        ratio = speech_ratio(t0, t1, regions)
        seg = x[i : i + win]
        f0_win = f0[int(i / sr * per_s) : int((i + win) / sr * per_s)]
        if ratio > 0:
            feats = window_features(seg, sr, f0_win)
        else:  # no speech: only the level is meaningful (kept for the loudness timeline)
            feats = dict.fromkeys(FEATURES, 0.0)
            feats["rms_db"] = (
                _db(float(np.mean(np.square(seg, dtype=np.float64)))) if len(seg) else -120.0
            )
        out.append(Window(round(t0, 3), round(t1, 3), round(ratio, 3), feats))
    return out
