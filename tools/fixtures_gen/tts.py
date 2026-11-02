"""Pluggable TTS for synthetic fixture calls.

``StubTTS`` needs no model: it makes a deterministic speech-like signal (harmonic voice with a
falling f0 contour, syllable-rate envelope and consonant noise bursts) whose word timings follow a
simple duration model. ``LocalVitsTTS`` uses a Bangla VITS checkpoint (e.g. MMS-TTS) from a local
directory when one is configured; it never downloads anything.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from tools.fixtures_gen.models import Voice

FloatArray = npt.NDArray[np.float32]
SR = 8000
PLACEHOLDER = re.compile(r"^<[A-Z][A-Z0-9_]*>$")
_PUNCT = "।,?!.;:\"'()"


def clean_word(token: str) -> str:
    return token.strip(_PUNCT)


def _letters(word: str) -> int:
    return sum(1 for ch in word if unicodedata.category(ch).startswith("L"))


class TTSBackend(Protocol):
    name: str

    def synthesize(
        self, words: list[str], voice: Voice, *, pitch_factor: float, seed: int
    ) -> tuple[FloatArray, list[tuple[float, float]]]:
        """Return 8 kHz mono audio and one (t_start, t_end) per word, relative to the turn start."""
        ...


def word_durations(tokens: list[str]) -> list[tuple[float, float]]:
    """Stub duration model: (spoken length, pause after) per whitespace token."""
    out = []
    for tok in tokens:
        w = clean_word(tok)
        dur = 0.45 if PLACEHOLDER.match(w) else float(np.clip(0.16 + 0.07 * _letters(w), 0.2, 0.85))
        pause = 0.2 if tok and tok[-1] in "।,?!.;" else 0.07
        out.append((dur, pause))
    return out


class StubTTS:
    name = "stub"

    def synthesize(
        self, words: list[str], voice: Voice, *, pitch_factor: float, seed: int
    ) -> tuple[FloatArray, list[tuple[float, float]]]:
        rng = np.random.default_rng(seed)
        plan = word_durations(words)
        spans: list[tuple[float, float]] = []
        t = 0.0
        for dur, pause in plan:
            spans.append((round(t, 4), round(t + dur, 4)))
            t += dur + pause
        total = spans[-1][1] + 0.05
        n = round(total * SR)
        time = np.arange(n) / SR

        # Carrier: harmonics of a falling, slowly wobbling f0 (continuous phase across words)
        f0 = voice.f0_hz * pitch_factor
        contour = (
            f0
            * (1.06 - 0.12 * time / total)
            * (1 + 0.04 * np.sin(2 * np.pi * 2.3 * time + rng.uniform(0, 6.3)))
        )
        phase = 2 * np.pi * np.cumsum(contour) / SR
        carrier = np.zeros(n)
        for k in range(1, int(3600 // (f0 * 1.1)) + 1):
            fk = k * f0
            weight = (
                1.0
                + 2.5 * np.exp(-(((fk - 600) / 180) ** 2))
                + 1.8 * np.exp(-(((fk - 1500) / 260) ** 2))
                + 1.1 * np.exp(-(((fk - 2600) / 320) ** 2))
            ) / k
            carrier += weight * np.sin(k * phase + rng.uniform(0, 6.3))

        # Envelope: raised-cosine syllables inside each word, noise bursts at syllable onsets
        env = np.zeros(n)
        noise = np.zeros(n)
        for t0, t1 in spans:
            n_syl = max(1, round((t1 - t0) / 0.17))
            edges = np.linspace(t0, t1, n_syl + 1)
            for a, b in itertools.pairwise(edges):
                i0, i1 = round(a * SR), round(b * SR)
                env[i0:i1] = np.maximum(env[i0:i1], rng.uniform(0.65, 1.0) * np.hanning(i1 - i0))
                burst = min(i1, i0 + round(0.03 * SR))
                noise[i0:burst] += np.diff(rng.standard_normal(burst - i0 + 1)) * 0.25
        y = carrier * env + noise
        active = env > 0.05
        rms = np.sqrt(np.mean(y[active] ** 2)) if active.any() else 1.0
        return (y / rms).astype(np.float32), spans  # unit RMS while speaking; renderer sets level


class LocalVitsTTS:  # pragma: no cover - needs torch, transformers and local weights
    """Bangla VITS checkpoint (transformers ``VitsModel``) loaded from a local directory.

    Placeholders (<GALI_n>) are not speakable, so they are rendered with the stub voice. Word
    timings inside a spoken segment are spread by letter count (approximate).
    """

    name = "local"

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self._torch = torch
        self.model_dir = model_dir
        self._tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        # Any: the same code type-checks with and without the optional transformers install
        model: Any = VitsModel.from_pretrained(model_dir, local_files_only=True)
        self._model = model.eval()
        self._sr = int(self._model.config.sampling_rate)
        self._stub = StubTTS()

    def _speak(self, text: str) -> FloatArray:
        from galecore.audio import resample

        with self._torch.no_grad():
            wav = self._model(**self._tok(text, return_tensors="pt")).waveform[0].numpy()
        return resample(wav.astype(np.float32), self._sr, SR)

    def synthesize(
        self, words: list[str], voice: Voice, *, pitch_factor: float, seed: int
    ) -> tuple[FloatArray, list[tuple[float, float]]]:
        pieces: list[FloatArray] = []
        spans: list[tuple[float, float]] = []
        t = 0.0
        i = 0
        while i < len(words):
            if PLACEHOLDER.match(clean_word(words[i])):
                audio, _ = self._stub.synthesize(
                    [words[i]], voice, pitch_factor=pitch_factor, seed=seed + i
                )
                group = [words[i]]
                i += 1
            else:
                group = []
                while i < len(words) and not PLACEHOLDER.match(clean_word(words[i])):
                    group.append(words[i])
                    i += 1
                audio = self._speak(" ".join(group))
                if (
                    pitch_factor != 1.0
                ):  # crude pitch raise: fewer samples played at SR (also faster)
                    from galecore.audio import resample

                    audio = resample(audio, SR, round(SR / pitch_factor))
            seg = len(audio) / SR
            weights = np.array([max(1, _letters(clean_word(w))) for w in group], dtype=float)
            edges = t + np.concatenate([[0.0], np.cumsum(weights / weights.sum() * seg)])
            spans += [(round(a, 4), round(b, 4)) for a, b in itertools.pairwise(edges)]
            pieces.append(audio)
            t += seg
        y = np.concatenate(pieces)
        return (y / max(1e-6, float(np.sqrt(np.mean(y**2))))).astype(np.float32), spans


TTSChoice = Literal["auto", "stub", "local"]


def load_tts(choice: TTSChoice, tts_dir: Path | None) -> TTSBackend:
    """``auto``: the local model if ``tts_dir`` holds a checkpoint (config.json), else the stub."""
    has_model = tts_dir is not None and (tts_dir / "config.json").is_file()
    if choice == "local" and not has_model:
        raise FileNotFoundError(f"no TTS checkpoint (config.json) in {tts_dir}; set GALE_TTS_DIR")
    if choice == "local" or (choice == "auto" and has_model):
        return LocalVitsTTS(tts_dir)  # type: ignore[arg-type]  # has_model implies a dir
    return StubTTS()
