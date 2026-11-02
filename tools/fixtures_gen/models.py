"""File formats for synthetic fixture calls (ADR-010).

``script.json`` is the source of truth; ``call.wav`` is rendered from it and
``expected_events.json`` holds the ground truth that pipeline tests compare against.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from galecore.schemas import Category, ContextFlag, Speaker, Target

FIXTURE_SCHEMA_VERSION = "1"
Register = Literal["apni", "tumi", "tui"]
Style = Literal["normal", "shout"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WordSpan(_Strict):
    w: str
    t_start: float
    t_end: float


class Voice(_Strict):
    f0_hz: float = Field(gt=50, lt=400)
    seed: int


class Turn(_Strict):
    idx: int
    speaker: Speaker
    text: str  # synthetic Bangla; abusive words are only <GALI_n> placeholders (CLAUDE.md rule 2)
    style: Style = "normal"
    pronoun_register: Register | None = None  # pronoun register of this turn, None = neutral
    t_start: float
    t_end: float
    words: list[WordSpan]

    @model_validator(mode="after")
    def _check(self) -> Turn:
        if self.t_end <= self.t_start:
            raise ValueError("turn must have positive duration")
        return self


class Script(_Strict):
    schema_version: str = FIXTURE_SCHEMA_VERSION
    call_id: str
    title: str
    description: str
    synthetic: Literal[True] = True
    sample_rate: Literal[8000] = 8000
    layout: Literal["stereo_agent_left"] = "stereo_agent_left"
    duration_s: float
    tts: dict[str, Any]  # {"backend": "stub"} or {"backend": "local", "model_dir_name": ...}
    render: dict[str, float]  # level / noise / bleed / shout parameters used by the renderer
    voices: dict[Speaker, Voice]
    turns: list[Turn]


class ExpectedEvent(_Strict):
    category: Category
    speaker: Speaker
    target: Target
    t_start: float
    t_end: float
    severity: Literal[1, 2, 3]
    turn_idx: int
    alert: bool = True  # False → an annotator would tag it but no alert should fire
    context_flags: list[ContextFlag] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ExpectedEvents(_Strict):
    schema_version: str = FIXTURE_SCHEMA_VERSION
    call_id: str
    tolerance_s: float = 0.5
    events: list[ExpectedEvent]
