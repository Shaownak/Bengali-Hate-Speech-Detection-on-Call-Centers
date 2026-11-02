"""Fixture oracle: what the stub models "know" about a synthetic fixture call.

Prompt 3: stub models are deterministic and keyed to the fixture scripts, so the stub
pipeline reproduces a perfect model's output on ``tests/fixtures/calls/*`` and the
integration test exercises the plumbing (timing, fusion, hysteresis, LLM decisions, report).
Only used in ``GALE_MODEL_MODE=stub``, and only when ``script.json`` sits next to the WAV.

What is oracle-keyed:
- ASR: the script's words and timings (stand-in for a perfect recogniser),
- text classifier: turn-level annotated categories that words alone don't carry
  (e.g. ``dismissive_rude``),
- LLM judge: context flags on ground-truth events (reported speech, self-directed).

Everything else (lexicon matching, register rules, DSP shouting, VAD overtalk, fusion) runs the
real code on the stub's outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from galecore.schemas import Category, Channel, ContextFlag, Speaker


@dataclass(frozen=True, slots=True)
class OracleWord:
    token: str  # as written in the script (may carry punctuation)
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class OracleTurn:
    idx: int
    channel: Channel
    text: str
    start: float
    end: float
    words: tuple[OracleWord, ...]


@dataclass(frozen=True, slots=True)
class OracleEvent:
    category: Category
    speaker: Speaker
    start: float
    end: float
    severity: int
    alert: bool
    context_flags: tuple[ContextFlag, ...]
    source: str
    turn_idx: int


@dataclass(frozen=True, slots=True)
class FixtureOracle:
    call_id: str
    turns: tuple[OracleTurn, ...]
    events: tuple[OracleEvent, ...]

    @classmethod
    def for_audio(cls, wav: Path) -> FixtureOracle | None:
        """The oracle for ``<dir>/call.wav`` if ``<dir>/script.json`` exists, else None."""
        d = wav.parent
        if not (d / "script.json").is_file() or not (d / "expected_events.json").is_file():
            return None
        return cls.from_dir(d)

    @classmethod
    def from_dir(cls, d: Path) -> FixtureOracle:
        script = json.loads((d / "script.json").read_text(encoding="utf-8"))
        expected = json.loads((d / "expected_events.json").read_text(encoding="utf-8"))
        if script.get("synthetic") is not True:
            raise ValueError(f"{d}: the oracle only reads synthetic fixtures")
        turns = []
        for t in script["turns"]:
            tokens = t["text"].split()
            words = tuple(
                OracleWord(tok, w["t_start"], w["t_end"])
                for tok, w in zip(tokens, t["words"], strict=True)
            )
            turns.append(
                OracleTurn(
                    idx=t["idx"],
                    channel=Channel(t["speaker"]),
                    text=t["text"],
                    start=t["t_start"],
                    end=t["t_end"],
                    words=words,
                )
            )
        events = tuple(
            OracleEvent(
                category=Category(e["category"]),
                speaker=Speaker(e["speaker"]),
                start=e["t_start"],
                end=e["t_end"],
                severity=e["severity"],
                alert=e["alert"],
                context_flags=tuple(ContextFlag(f) for f in e["context_flags"]),
                source=e["evidence"].get("source", ""),
                turn_idx=e["turn_idx"],
            )
            for e in expected["events"]
        )
        return cls(call_id=script["call_id"], turns=tuple(turns), events=events)

    def turn_annotations(self) -> dict[str, list[tuple[Category, int]]]:
        """Turn text → turn-level annotated categories (``source == "annotation"``)."""
        by_idx = {t.idx: t for t in self.turns}
        out: dict[str, list[tuple[Category, int]]] = {}
        for e in self.events:
            if e.source == "annotation":
                out.setdefault(by_idx[e.turn_idx].text, []).append((e.category, e.severity))
        return out
