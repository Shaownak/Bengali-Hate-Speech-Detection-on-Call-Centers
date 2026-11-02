"""Authoritative message and API types for GAL-E.

Spec: docs/01_ARCHITECTURE.md §3, docs/02_DATA_PLAN.md §3.4, docs/08_LLM_JUDGE_SPEC.md §2,
plus accepted decisions D-01..D-09 in docs/PROGRESS.md.

TypeScript types in ``web/src/types/generated`` are generated from this module
(``make gen-types``). Never hand-edit the generated files.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"


class GaleModel(BaseModel):
    """Base for every bus/API model: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    # Fields carried as raw bytes on the bus instead of inside the JSON body (D-02).
    bus_binary_fields: ClassVar[tuple[str, ...]] = ()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Channel(StrEnum):
    AGENT = "agent"
    CUSTOMER = "customer"
    MIXED = "mixed"  # only before diarization; never reaches fusion (D-15)


class Speaker(StrEnum):
    AGENT = "agent"
    CUSTOMER = "customer"


class Category(StrEnum):
    """Annotation/alert taxonomy. Must match docs/02_DATA_PLAN.md §3.4."""

    SLANG_PROFANITY = "slang_profanity"
    INSULT = "insult"
    THREAT = "threat"
    SEXUAL_HARASSMENT = "sexual_harassment"
    HATE_IDENTITY = "hate_identity"
    DISRESPECT_REGISTER = "disrespect_register"
    HARSH_TONE = "harsh_tone"
    SHOUTING = "shouting"
    SARCASM_MOCKERY = "sarcasm_mockery"
    DISMISSIVE_RUDE = "dismissive_rude"
    INTERRUPTION_OVERTALK = "interruption_overtalk"


class Target(StrEnum):
    """Who a behaviour is aimed at (D-03)."""

    AGENT = "agent"
    CUSTOMER = "customer"
    COMPANY = "company"
    THIRD_PARTY = "third_party"
    SELF = "self"
    NONE = "none"


class ContextFlag(StrEnum):
    """Shared by annotation, text models, LLM judge and fusion (D-04)."""

    BANTER = "banter"
    REPORTED_SPEECH = "reported_speech"
    SELF_DIRECTED = "self_directed"
    PARTIAL = "partial"
    QUOTED_POLICY = "quoted_policy"
    NEUTRAL_REGISTER = "neutral_register"
    UNSURE = "unsure"


class SignalSource(StrEnum):
    DSP_SHOUT = "dsp_shout"
    SER = "ser"
    AUDIO_TOX = "audio_tox"
    KWS = "kws"
    LEXICON = "lexicon"
    REGISTER = "register"
    TEXT_CLF = "text_clf"
    LLM_WINDOW = "llm_window"
    LLM_CALL = "llm_call"
    VAD_OVERTALK = "vad_overtalk"  # D-05
    CALL_EVENT = "call_event"  # D-05


class EventStatus(StrEnum):
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    DISMISSED_BY_LLM = "dismissed_by_llm"
    REVIEWED_TRUE = "reviewed_true"
    REVIEWED_FALSE = "reviewed_false"


Severity = Literal[1, 2, 3]
Probability = float


# ---------------------------------------------------------------------------
# Audio & ASR
# ---------------------------------------------------------------------------


class AudioFrame(GaleModel):
    """200 ms of 16 kHz mono 16-bit little-endian PCM for one channel (D-02)."""

    bus_binary_fields: ClassVar[tuple[str, ...]] = ("pcm_s16le",)

    call_id: str
    channel: Channel
    seq: int = Field(ge=0)
    ts_start: float = Field(ge=0)
    ts_end: float = Field(ge=0)
    sample_rate: Literal[16000] = 16000
    pcm_s16le: bytes

    @model_validator(mode="after")
    def _check(self) -> AudioFrame:
        if self.ts_end < self.ts_start:
            raise ValueError("ts_end must be >= ts_start")
        if len(self.pcm_s16le) % 2:
            raise ValueError("pcm_s16le must hold whole 16-bit samples")
        return self


class Word(GaleModel):
    """One recognised word. ``word`` is already redacted (PII words replaced by tags)."""

    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    conf: float = Field(ge=0, le=1)


class AsrSegment(GaleModel):
    """ASR output as published on the bus.

    D-01: asr-worker redacts in-process before publishing, so this model has no raw-text
    field at all. Un-redacted text never leaves the asr-worker process.
    """

    call_id: str
    channel: Channel
    segment_id: str
    ts_start: float = Field(ge=0)
    ts_end: float = Field(ge=0)
    text_redacted: str
    words: list[Word] = Field(default_factory=list)
    is_final: bool
    avg_conf: float = Field(ge=0, le=1)
    model_version: str


# ---------------------------------------------------------------------------
# Signals & events
# ---------------------------------------------------------------------------


class Evidence(GaleModel):
    """Why a signal fired. Text is always redacted; features are numeric only."""

    text_span: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    word_ids: list[int] = Field(default_factory=list)
    segment_id: str | None = None
    term_id: str | None = None
    # numeric features, e.g. loudness_z, f0_p90, aggression (D-06)
    features: dict[str, float] = Field(default_factory=dict)


class Signal(GaleModel):
    call_id: str
    channel: Channel
    signal_id: str
    source: SignalSource
    category: Category
    score: float = Field(ge=0, le=1)
    ts_start: float = Field(ge=0)
    ts_end: float = Field(ge=0)
    evidence: Evidence = Field(default_factory=Evidence)
    model_version: str
    shadow: bool = False  # D-07


class FusedEvent(GaleModel):
    event_id: str
    call_id: str
    speaker: Speaker
    category: Category
    severity: Severity
    probability: float = Field(ge=0, le=1)
    ts_start: float = Field(ge=0)
    ts_end: float = Field(ge=0)
    contributing_signals: list[str] = Field(default_factory=list)
    explanation_bn: str | None = None
    explanation_en: str | None = None
    status: EventStatus = EventStatus.PROVISIONAL
    alert: bool = False
    shadow: bool = False  # D-07


class CallLifecycle(GaleModel):
    call_id: str
    event: Literal["started", "ended", "hold", "resume", "transfer", "hangup"]
    ts: float = Field(ge=0, description="seconds from call start")
    by: Speaker | None = None  # who hung up / put on hold (feeds call_event signals, D-05)
    queue: str | None = None
    direction: Literal["inbound", "outbound"] | None = None
    agent_id: str | None = None
    channel_layout: Literal["stereo_agent_left", "stereo_agent_right", "mono"] | None = None
    mode: Literal["realtime", "postcall"] | None = None
    # PBX / recorder id of the call (``external_call_id`` in the DB); synthetic fixtures use
    # "fixture:<dir>", which stub models key off (galecore.models.oracle)
    external_ref: str | None = None
    wall_clock: float | None = Field(
        default=None, description="unix time at `ts` (latency measurement)"
    )


class Review(GaleModel):
    """Human verdict on an event, or an agent dispute (D-08)."""

    event_id: str
    reviewer_user_id: str
    kind: Literal["verdict", "dispute"] = "verdict"
    verdict: Literal["correct", "wrong_category", "false_positive", "wrong_severity"] | None = None
    corrected_category: Category | None = None
    corrected_severity: Severity | None = None
    comment: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Review:
        if self.kind == "verdict" and self.verdict is None:
            raise ValueError("a verdict review needs a verdict")
        if self.verdict == "wrong_category" and self.corrected_category is None:
            raise ValueError("wrong_category needs corrected_category")
        if self.verdict == "wrong_severity" and self.corrected_severity is None:
            raise ValueError("wrong_severity needs corrected_severity")
        return self


# ---------------------------------------------------------------------------
# LLM judge outputs (docs/08 §2, tightened per D-09)
# ---------------------------------------------------------------------------


class JudgedEvent(GaleModel):
    speaker: Speaker
    category: Category
    severity: Severity
    target: Target | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(description="verbatim substring of the input utterance(s)")
    utterance_id: str | None = None
    context_flags: list[ContextFlag] = Field(default_factory=list)
    rationale_bn: str = Field(max_length=200)
    rationale_en: str | None = Field(default=None, max_length=200)


class ProvisionalDecision(GaleModel):
    event_id: str
    decision: Literal["confirm", "dismiss", "change_category", "change_severity"]
    new_category: Category | None = None
    new_severity: Severity | None = None
    reason_bn: str | None = Field(default=None, max_length=160)


class WindowJudgement(GaleModel):
    events: list[JudgedEvent]
    provisional_decisions: list[ProvisionalDecision]


class AgentScore(GaleModel):
    professionalism: int = Field(ge=1, le=5)
    empathy: int = Field(ge=1, le=5)
    de_escalation: int = Field(ge=1, le=5)
    respectful_register_maintained: bool
    violations: list[str]


class CustomerScore(GaleModel):
    hostility: int = Field(ge=1, le=5)
    abuse_toward_agent: bool
    main_issue_bn: str


class Escalation(GaleModel):
    escalated: bool
    who_escalated_first: Literal["agent", "customer", "none", "unclear"]
    turning_point_utterance_id: str | None


class CallScorecard(GaleModel):
    agent: AgentScore
    customer: CustomerScore
    escalation: Escalation
    summary_bn: str = Field(max_length=600)
    summary_en: str = Field(max_length=600)
    coaching_points_bn: list[str] = Field(max_length=5)


class LlmDecision(GaleModel):
    """One LLM-judge decision on a provisional event (stream ``llm.decisions``)."""

    call_id: str
    decision: ProvisionalDecision
    model_version: str


class CallScoreMessage(GaleModel):
    """The whole-call scorecard after hang-up (stream ``call.scores``)."""

    call_id: str
    scorecard: CallScorecard
    model_version: str


class PostcallJob(GaleModel):
    """A recording to process in post-call mode (stream ``postcall.jobs``)."""

    call_id: str
    uri: str  # object-store key or local path of the stereo recording
    channel_layout: Literal["stereo_agent_left", "stereo_agent_right", "mono"] = "stereo_agent_left"
    external_ref: str | None = None
    queue: str | None = None


# Models exported to JSON Schema / TypeScript by `make gen-types`.
EXPORTED_MODELS: tuple[type[GaleModel], ...] = (
    AudioFrame,
    Word,
    AsrSegment,
    Evidence,
    Signal,
    FusedEvent,
    CallLifecycle,
    Review,
    JudgedEvent,
    ProvisionalDecision,
    WindowJudgement,
    CallScorecard,
    LlmDecision,
    CallScoreMessage,
    PostcallJob,
)


def json_schemas() -> dict[str, Any]:
    """JSON Schema for every exported model, keyed by model name."""
    return {m.__name__: m.model_json_schema() for m in EXPORTED_MODELS}
