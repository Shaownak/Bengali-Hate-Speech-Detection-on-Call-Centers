"""Unified open-data schema (docs/09 §4) and the raw-row shape loaders produce."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LABELS = (
    "slang_profanity",
    "insult",
    "threat",
    "sexual_harassment",
    "hate_identity",
    "sarcasm_mockery",
    "dismissive_rude",
    "aggression",
)
Script = Literal["bangla", "roman", "mixed", "other"]
Split = Literal["train", "dev", "test"]


@dataclass
class RawRow:
    """What a loader yields: the text plus raw source fields, before label mapping."""

    source: str
    source_id: str
    text: str
    split: Split | None  # official split if the source has one
    fields: dict[str, Any] = field(default_factory=dict)  # normalised field name → raw value
    context: list[str] = field(default_factory=list)
    parent_id: str | None = None  # BIDWESH → BD-SHS source id
    dialect: str = "standard"
    text_bn: str | None = None  # an official Bangla-script version (Code-mixed Chaos)


class UnifiedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    source: str
    source_split: Split | None
    source_id: str
    parent_uid: str | None = None
    text_orig: str
    script_orig: Script
    text_bn: str
    dialect: str = "standard"
    context: list[str] = Field(default_factory=list)
    labels: dict[str, int | None]
    label_confidence: dict[str, float] = Field(
        default_factory=dict
    )  # < 1 = weak mapping (soft target)
    toxic_any: int | None
    severity: int | None = Field(default=None, ge=0, le=3)
    target: str | None = None
    context_flags: list[str] = Field(default_factory=list)
    emoji_present: bool = False
    licence_id: str
    commercial_ok: bool
    research_only: bool
    weight: float = Field(default=1.0, ge=0.0)
    split: Split | None = None  # final split after leakage control
    synthetic: bool = False
    aug: str | None = None  # augmentation that produced this row (asr_noise, …)


class AudioRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    source: str
    path: str
    label: str  # BAAD: the term (folder name); A2–A4: toxic / non_toxic …
    speaker: str | None = None
    severity: int | None = None
    licence_id: str
    research_only: bool
    split: Split | None = None
