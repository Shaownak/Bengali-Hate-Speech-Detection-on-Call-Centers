"""Two-channel VAD overlap → ``vad_overtalk`` signals (D-05), one per interruption.

Speaker B interrupts A when B starts talking while A has been talking for at least
``min_prior_s`` and the overlap lasts at least ``min_overlap_s``. B's onset must be a turn
start (B silent for ``min_turn_gap_s`` before), not a breath inside B's own turn.
Crosstalk guard: B's onset
only counts if B is no more than ``bleed_db`` quieter than A in that moment (a far-end echo of
A's own voice is ~30 dB down), applied to the VAD regions first (``own_speech_regions``)
and again at the onset. Fusion turns ≥ 3 interruptions in 60 s into an event (rules.yaml).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from galecore.models.stubs import stable_id
from galecore.schemas import Category, Channel, Evidence, Signal, SignalSource

VERSION = "vad-overtalk-0.1"


def _db(x: npt.NDArray[np.float32]) -> float:
    return (
        float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12)) if len(x) else -120.0
    )


def own_speech_regions(
    regions: list[tuple[float, float]],
    mine: npt.NDArray[np.float32],
    other: npt.NDArray[np.float32],
    sr: int,
    *,
    bleed_db: float = 20.0,
    frame_s: float = 0.03,
    min_gap_s: float = 0.3,
    min_len_s: float = 0.15,
) -> list[tuple[float, float]]:
    """Trim VAD regions to frames where this speaker is not just the other line's echo
    (more than ``bleed_db`` below it). A shouting far end leaks enough to open a VAD."""
    n = int(frame_s * sr)
    out: list[tuple[float, float]] = []
    for a, b in regions:
        i0, i1 = int(a * sr), min(int(b * sr), len(mine), len(other))
        k = max(0, (i1 - i0) // n)
        if k == 0:
            continue
        fm = mine[i0 : i0 + k * n].astype(np.float64).reshape(k, n)
        fo = other[i0 : i0 + k * n].astype(np.float64).reshape(k, n)
        db_m = 10 * np.log10(np.mean(fm**2, axis=1) + 1e-12)
        db_o = 10 * np.log10(np.mean(fo**2, axis=1) + 1e-12)
        for j in np.flatnonzero(db_m >= db_o - bleed_db):
            t0, t1 = a + j * frame_s, a + (j + 1) * frame_s
            if out and t0 - out[-1][1] <= min_gap_s:
                out[-1] = (out[-1][0], t1)
            else:
                out.append((t0, t1))
    return [(round(x, 3), round(y, 3)) for x, y in out if y - x >= min_len_s]


def _turns(regions: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in regions:
        if out and a - out[-1][1] < gap_s:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def overtalk_signals(
    call_id: str,
    regions: dict[Channel, list[tuple[float, float]]],
    audio: dict[Channel, npt.NDArray[np.float32]],
    sr: int,
    *,
    min_prior_s: float = 0.5,
    min_overlap_s: float = 0.5,
    min_turn_gap_s: float = 1.0,
    bleed_db: float = 20.0,
) -> list[Signal]:
    out = []
    pairs = [(Channel.AGENT, Channel.CUSTOMER), (Channel.CUSTOMER, Channel.AGENT)]
    # own speech only (no echo), then regions closer than ``min_turn_gap_s`` form one turn
    turns = {
        b: _turns(
            own_speech_regions(regions.get(b, []), audio[b], audio[a], sr, bleed_db=bleed_db),
            min_turn_gap_s,
        )
        for b, a in pairs
    }
    for b, a in pairs:  # b interrupts a
        for b0, b1 in turns[b]:
            for a0, a1 in turns[a]:
                if not (a0 + min_prior_s <= b0 < a1):
                    continue
                overlap = min(a1, b1) - b0
                if overlap < min_overlap_s:
                    continue
                i0, i1 = int(b0 * sr), int((b0 + min_overlap_s) * sr)
                if _db(audio[b][i0:i1]) < _db(audio[a][i0:i1]) - bleed_db:
                    continue  # crosstalk of A on B's line, not B speaking
                out.append(
                    Signal(
                        call_id=call_id,
                        channel=b,
                        signal_id=stable_id(call_id, b, "overtalk", round(b0, 2)),
                        source=SignalSource.VAD_OVERTALK,
                        category=Category.INTERRUPTION_OVERTALK,
                        score=round(min(1.0, overlap / 1.0), 3),
                        ts_start=round(b0, 3),
                        ts_end=round(b0 + overlap, 3),
                        evidence=Evidence(
                            features={"overlap_s": round(overlap, 3), "prior_s": round(b0 - a0, 3)}
                        ),
                        model_version=VERSION,
                    )
                )
                break
    return sorted(out, key=lambda s: s.ts_start)
