"""Scenario → timed script + ground-truth events, and script → stereo 8 kHz audio."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

from galecore.config import REPO_ROOT, LexiconConfig, LexiconEntry, load_rules, load_thresholds
from galecore.schemas import Category, Speaker, Target
from tools.fixtures_gen.models import (
    ExpectedEvent,
    ExpectedEvents,
    Register,
    Script,
    Turn,
    Voice,
    WordSpan,
)
from tools.fixtures_gen.scenarios import AGENT_FILLERS, CUSTOMER_FILLERS, Fill, Line, Scenario
from tools.fixtures_gen.tts import PLACEHOLDER, SR, TTSBackend, clean_word

FAKE_LEXICON = REPO_ROOT / "configs" / "lexicon" / "fake_lexicon.yaml"

# Renderer parameters (recorded in script.json so a render is reproducible)
RENDER = {
    "speech_dbfs": -26.0,  # RMS while speaking, normal style
    "shout_gain_db": 12.0,  # prompt 2: shouting = +12 dB gain …
    "shout_pitch": 1.35,  # … and raised pitch
    "noise_dbfs": -60.0,  # line noise floor per channel
    "bleed_db": -32.0,  # far-end crosstalk into each channel
}

# ground truth follows the same conventions as fusion (configs/thresholds.yaml, rules.yaml)
MERGE_GAP_S = load_thresholds(REPO_ROOT / "configs").hysteresis.merge_gap_s
_OVERTALK = next(r for r in load_rules(REPO_ROOT / "configs").rules if r.name == "overtalk").when
OVERTALK_MIN = int(_OVERTALK["count_gte"])
OVERTALK_WINDOW_S = float(_OVERTALK["window_s"])

PRONOUNS: dict[Register, tuple[str, ...]] = {
    "apni": ("আপনি", "আপনার", "আপনাকে", "আপনারা", "আপনাদের", "apni", "apnake", "apnar"),
    "tumi": ("তুমি", "তোমার", "তোমাকে", "tumi", "tomar", "tomake"),
    "tui": ("তুই", "তোর", "তোকে", "tui", "tor", "toke"),
}
_RANK: dict[Register, int] = {"apni": 0, "tumi": 1, "tui": 2}
# severity of a register drop: (speaker, new register) → 1..3 (agent misconduct weighs more)
_REGISTER_SEVERITY: dict[tuple[Speaker, Register], int] = {
    (Speaker.AGENT, "tumi"): 2,
    (Speaker.AGENT, "tui"): 3,
    (Speaker.CUSTOMER, "tumi"): 1,
    (Speaker.CUSTOMER, "tui"): 2,
}


def load_fake_lexicon(path: Path = FAKE_LEXICON) -> dict[str, LexiconEntry]:
    """Placeholder → lexicon entry. Fixtures only ever use the fake lexicon (rule 2)."""
    import yaml

    lex = LexiconConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not lex.fake:
        raise ValueError("fixtures must be generated from the fake placeholder lexicon")
    return {term: e for e in lex.entries for term in [e.term, *e.variants]}


def _other(s: Speaker) -> Target:
    return Target.CUSTOMER if s is Speaker.AGENT else Target.AGENT


def _expand(scenario: Scenario, rng: np.random.Generator) -> list[Line]:
    lines: list[Line] = []
    pools = {Speaker.AGENT: list(AGENT_FILLERS), Speaker.CUSTOMER: list(CUSTOMER_FILLERS)}
    last: dict[Speaker, str] = {}
    for item in scenario.lines:
        if not isinstance(item, Fill):
            lines.append(item)
            continue
        for _ in range(item.n):
            for spk in (Speaker.AGENT, Speaker.CUSTOMER):
                choices = [t for t in pools[spk] if t != last.get(spk)]
                text = choices[int(rng.integers(len(choices)))]
                last[spk] = text
                lines.append(Line(spk, text))
    return lines


def compose(scenario: Scenario, tts: TTSBackend) -> tuple[Script, ExpectedEvents]:
    rng = np.random.default_rng(scenario.seed)
    lexicon = load_fake_lexicon()
    voices = {
        Speaker.AGENT: Voice(f0_hz=scenario.agent_f0, seed=scenario.seed * 10 + 1),
        Speaker.CUSTOMER: Voice(f0_hz=scenario.customer_f0, seed=scenario.seed * 10 + 2),
    }
    turns: list[Turn] = []
    events: list[ExpectedEvent] = []
    baseline: dict[Speaker, Register] = {}
    interruptions: list[tuple[Speaker, int, float, float, bool]] = []
    cursor = 0.6  # leading silence

    for idx, line in enumerate(_expand(scenario, rng)):
        tokens = line.text.split()
        pitch = RENDER["shout_pitch"] if line.style == "shout" else 1.0
        audio, rel = tts.synthesize(
            tokens,
            voices[line.speaker],
            pitch_factor=pitch,
            seed=voices[line.speaker].seed * 1000 + idx,
        )
        gap = line.gap if line.gap is not None else float(rng.uniform(0.35, 1.1))
        prev = turns[-1] if turns else None
        if gap < 0:
            if prev is None or prev.speaker is line.speaker:
                raise ValueError(
                    f"{scenario.call_id}: overtalk line {idx} must follow the other speaker"
                )
            t_start = round(prev.t_end + gap, 3)
            if t_start < prev.t_start + 0.3:
                raise ValueError(f"{scenario.call_id}: overtalk line {idx} starts too early")
        else:
            t_start = round(cursor + gap, 3)
        dur = len(audio) / SR
        words = [
            WordSpan(w=clean_word(tok), t_start=round(t_start + a, 3), t_end=round(t_start + b, 3))
            for tok, (a, b) in zip(tokens, rel, strict=True)
        ]
        turn = Turn(
            idx=idx,
            speaker=line.speaker,
            text=line.text,
            style=line.style,
            pronoun_register=line.register,
            t_start=t_start,
            t_end=round(t_start + dur, 3),
            words=words,
        )
        turns.append(turn)
        cursor = max(cursor, turn.t_end)
        target = line.target or _other(line.speaker)

        def add(
            cat: Category,
            sev: int,
            t0: float,
            t1: float,
            tgt: Target = target,
            _line: Line = line,
            _idx: int = idx,
            ev: dict[str, object] | None = None,
        ) -> None:
            events.append(
                ExpectedEvent(
                    category=cat,
                    speaker=_line.speaker,
                    target=tgt,
                    t_start=t0,
                    t_end=t1,
                    severity=sev,
                    turn_idx=_idx,
                    alert=_line.alert,
                    context_flags=list(_line.flags),
                    evidence=dict(ev or {}),
                )
            )

        for w in words:  # lexicon placeholders
            if PLACEHOLDER.match(w.w):
                if w.w not in lexicon:
                    raise ValueError(f"{scenario.call_id}: {w.w} is not in the fake lexicon")
                e = lexicon[w.w]
                add(
                    e.category,
                    e.severity,
                    w.t_start,
                    w.t_end,
                    ev={"term": w.w, "lexicon_id": e.id, "source": "lexicon"},
                )
        for cat, sev in line.extra:
            add(cat, sev, turn.t_start, turn.t_end, ev={"source": "annotation"})
        if line.style == "shout":
            add(
                Category.SHOUTING,
                2,
                turn.t_start,
                turn.t_end,
                tgt=_other(line.speaker),
                ev={
                    "gain_db": RENDER["shout_gain_db"],
                    "pitch_factor": RENDER["shout_pitch"],
                    "source": "dsp",
                },
            )
        if line.register is not None:
            pron = next((w for w in words if w.w.lower() in PRONOUNS[line.register]), None)
            if pron is None:
                raise ValueError(
                    f"{scenario.call_id}: line {idx} is {line.register} but has no such pronoun"
                )
            base = baseline.setdefault(line.speaker, line.register)
            if _RANK[line.register] > _RANK[base]:
                add(
                    Category.DISRESPECT_REGISTER,
                    _REGISTER_SEVERITY[(line.speaker, line.register)],
                    pron.t_start,
                    pron.t_end,
                    tgt=_other(line.speaker),
                    ev={"from": base, "to": line.register, "pronoun": pron.w, "source": "register"},
                )
        if gap < 0 and prev is not None:
            overlap = round(min(prev.t_end, turn.t_end) - t_start, 3)
            interruptions.append(
                (line.speaker, idx, t_start, round(t_start + overlap, 3), line.alert)
            )

    events += _overtalk_events(interruptions)
    events = _merge_events(events)
    duration = round(cursor + 0.8, 3)
    script = Script(
        call_id=scenario.call_id,
        title=scenario.title,
        description=scenario.description,
        duration_s=duration,
        tts={"backend": tts.name},
        render=dict(RENDER),
        voices=voices,
        turns=turns,
    )
    return script, ExpectedEvents(call_id=scenario.call_id, events=events)


def _overtalk_events(
    interruptions: list[tuple[Speaker, int, float, float, bool]],
) -> list[ExpectedEvent]:
    """docs/02 §3.4: overtalk is ≥ OVERTALK_MIN interruptions by one speaker within 60 s.

    A single overlap is not an event. One event spans the first to the last interruption.
    """
    out = []
    for spk in Speaker:
        mine = [i for i in interruptions if i[0] is spk]
        k = 0
        while k < len(mine):
            group = [m for m in mine[k:] if m[2] - mine[k][2] <= OVERTALK_WINDOW_S]
            if len(group) >= OVERTALK_MIN:
                out.append(
                    ExpectedEvent(
                        category=Category.INTERRUPTION_OVERTALK,
                        speaker=spk,
                        target=_other(spk),
                        t_start=group[0][2],
                        t_end=group[-1][3],
                        severity=1,
                        turn_idx=group[0][1],
                        alert=all(g[4] for g in group),
                        evidence={"interruptions": len(group), "source": "vad_overtalk"},
                    )
                )
                k += len(group)
            else:
                k += 1
    return out


def _merge_events(events: list[ExpectedEvent]) -> list[ExpectedEvent]:
    """Annotation convention = fusion hysteresis: same speaker + category spans closer than
    ``merge_gap_s`` (configs/thresholds.yaml) are one event, with the higher severity."""
    events = sorted(events, key=lambda e: (e.speaker.value, e.category.value, e.t_start))
    out: list[ExpectedEvent] = []
    for e in events:
        last = out[-1] if out else None
        if (
            last is not None
            and (last.speaker, last.category, last.alert) == (e.speaker, e.category, e.alert)
            and e.t_start - last.t_end < MERGE_GAP_S
        ):
            out[-1] = last.model_copy(
                update={
                    "t_end": max(last.t_end, e.t_end),
                    "severity": max(last.severity, e.severity),
                    "context_flags": sorted({*last.context_flags, *e.context_flags}),
                    "evidence": {**last.evidence, "merged": last.evidence.get("merged", 1) + 1},
                }
            )
        else:
            out.append(e)
    return sorted(out, key=lambda e: (e.t_start, e.category.value))


def render(script: Script, tts: TTSBackend) -> npt.NDArray[np.int16]:
    """Stereo int16 (samples × 2, agent left) at 8 kHz. Deterministic for the stub backend."""
    n = round(script.duration_s * SR)
    clean = {Speaker.AGENT: np.zeros(n), Speaker.CUSTOMER: np.zeros(n)}
    p = script.render
    level = 10 ** (p["speech_dbfs"] / 20)
    for turn in script.turns:
        voice = script.voices[turn.speaker]
        shout = turn.style == "shout"
        audio, _ = tts.synthesize(
            turn.text.split(),
            voice,
            pitch_factor=p["shout_pitch"] if shout else 1.0,
            seed=voice.seed * 1000 + turn.idx,
        )
        gain = level * (10 ** (p["shout_gain_db"] / 20) if shout else 1.0)
        i0 = round(turn.t_start * SR)
        seg = audio[: max(0, min(len(audio), n - i0))] * gain
        clean[turn.speaker][i0 : i0 + len(seg)] += seg
    rng = np.random.default_rng(zlib.crc32(script.call_id.encode()))
    bleed = 10 ** (p["bleed_db"] / 20)
    delay = round(0.001 * SR)
    out = np.zeros((n, 2))
    for ch, (me, other) in enumerate(
        [(Speaker.AGENT, Speaker.CUSTOMER), (Speaker.CUSTOMER, Speaker.AGENT)]
    ):
        noise = np.convolve(rng.standard_normal(n), np.ones(4) / 2, mode="same") * 10 ** (
            p["noise_dbfs"] / 20
        )
        out[:, ch] = (
            clean[me] + bleed * np.concatenate([np.zeros(delay), clean[other][:-delay]]) + noise
        )
    knee = 0.9  # soft limiter: shouting peaks must not hard-clip
    mag = np.abs(out)
    out = np.where(
        mag > knee, np.sign(out) * (knee + (1 - knee) * np.tanh((mag - knee) / (1 - knee))), out
    )
    return np.clip(np.round(out * 32767), -32768, 32767).astype(np.int16)
