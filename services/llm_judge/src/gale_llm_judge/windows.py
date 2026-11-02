"""Window building and the user-message format of docs/08 §3.

Utterance ids are ``u_<index>`` in call order. Only redacted text is ever rendered (D-01).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from galecore.models.base import WindowInput
from galecore.schemas import AsrSegment, Channel, FusedEvent, Signal, SignalSource

WINDOW_S = 30.0  # docs/08 §1: last ≤ 30 s of both channels
_LABEL = {Channel.AGENT: "AGENT", Channel.CUSTOMER: "CUSTOMER", Channel.MIXED: "MIXED"}
ACOUSTIC = {SignalSource.DSP_SHOUT, SignalSource.SER, SignalSource.AUDIO_TOX}
TEXTUAL = {SignalSource.KWS, SignalSource.LEXICON, SignalSource.TEXT_CLF, SignalSource.REGISTER}


def system_prompt(mode: str = "window") -> str:
    return (resources.files("gale_llm_judge") / "prompts" / f"{mode}_system.md").read_text(
        encoding="utf-8"
    )


def utterance_ids(segments: list[AsrSegment]) -> dict[str, str]:
    """segment_id → stable utterance id (``u_<n>``) in call order."""
    ordered = sorted(segments, key=lambda s: (s.ts_start, s.channel.value))
    return {s.segment_id: f"u_{i}" for i, s in enumerate(ordered)}


@dataclass(frozen=True, slots=True)
class Window:
    start: float
    end: float
    input: WindowInput


def windows_for_events(
    call_id: str,
    segments: list[AsrSegment],
    signals: list[Signal],
    events: list[FusedEvent],
    queue: str | None = None,
) -> list[Window]:
    """Post-call: one window per cluster of provisional events, ending ≤ 5 s after the last one
    and reaching back ≤ 30 s, so every event is judged with its context."""
    out: list[Window] = []
    for e in sorted(events, key=lambda e: e.ts_start):
        if out and e.ts_end <= out[-1].end:
            out[-1].input.provisional.append(e)
            continue
        end = e.ts_end + 5.0
        start = max(0.0, end - WINDOW_S)
        segs = [s for s in segments if s.ts_end > start and s.ts_start < end]
        sigs = [s for s in signals if s.ts_end > start and s.ts_start < end]
        out.append(Window(start, end, WindowInput(call_id, segs, sigs, [e], queue)))
    return out


def _line(uid: str, s: AsrSegment) -> str:
    return f"{uid} | {_LABEL[s.channel]} | {s.ts_start:.1f}-{s.ts_end:.1f} | {s.text_redacted}"


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def render_user_message(window: WindowInput, ids: dict[str, str]) -> str:
    segs = sorted(window.segments, key=lambda s: (s.ts_start, s.channel.value))
    lines = [
        f"CALL_META: queue={window.queue or 'unknown'}; "
        f"minute={_mmss(segs[0].ts_start if segs else 0)}",
        "TRANSCRIPT (utterance_id | speaker | start-end | text):",
    ]
    for s in segs:
        lines.append(_line(ids[s.segment_id], s))

    def seg_of(sig: Signal) -> AsrSegment | None:
        return next(
            (
                s
                for s in segs
                if s.channel == sig.channel
                and s.ts_start <= sig.ts_end
                and sig.ts_start <= s.ts_end
            ),
            None,
        )

    acoustic: dict[str, dict[str, float]] = {}
    textual: dict[str, list[str]] = {}
    for sig in window.signals:
        seg = seg_of(sig)
        if seg is None:
            continue
        key = f"{ids[seg.segment_id]} {_LABEL[seg.channel]}"
        if sig.source in ACOUSTIC:
            name = {SignalSource.DSP_SHOUT: "shout_p", SignalSource.SER: "anger_p"}.get(
                sig.source, "tox_p"
            )
            acoustic.setdefault(key, {})[name] = max(
                acoustic.get(key, {}).get(name, 0.0), sig.score
            )
            if "loudness_z" in sig.evidence.features:
                acoustic[key]["loudness_z"] = max(
                    acoustic[key].get("loudness_z", 0.0), sig.evidence.features["loudness_z"]
                )
        elif sig.source in TEXTUAL:
            tag = sig.evidence.term_id or sig.category.value
            textual.setdefault(key, []).append(f"{sig.source.value}:{tag} p={sig.score:.2f}")
    lines.append("ACOUSTIC_SIGNALS (per utterance):")
    lines += [
        f"{k} " + " ".join(f"{n}={v:.2f}" for n, v in sorted(d.items()))
        for k, d in acoustic.items()
    ]
    lines.append("TEXT_SIGNALS:")
    lines += [f"{k} " + " ".join(v) for k, v in textual.items()]
    lines.append("PROVISIONAL_EVENTS:")
    for e in window.provisional:
        seg = next(
            (
                s
                for s in segs
                if s.ts_start <= e.ts_end
                and e.ts_start <= s.ts_end
                and s.channel.value == e.speaker.value
            ),
            None,
        )
        uid = ids[seg.segment_id] if seg else "-"
        lines.append(
            f"{e.event_id} | {e.speaker.value.upper()} | {e.category.value} | "
            f"sev {e.severity} | {uid}"
        )
    return "\n".join(lines)


def render_call_message(
    segments: list[AsrSegment], events: list[FusedEvent], ids: dict[str, str]
) -> str:
    segs = sorted(segments, key=lambda s: (s.ts_start, s.channel.value))
    lines = ["TRANSCRIPT (utterance_id | speaker | start-end | text):"]
    lines += [_line(ids[s.segment_id], s) for s in segs]
    lines.append("EVENTS (speaker | category | severity | status | start-end):")
    lines += [
        f"{e.speaker.value.upper()} | {e.category.value} | {e.severity} | {e.status.value} | "
        f"{e.ts_start:.1f}-{e.ts_end:.1f}"
        for e in sorted(events, key=lambda e: e.ts_start)
    ]
    return "\n".join(lines)
