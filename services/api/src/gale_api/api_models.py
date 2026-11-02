"""Request/response models of the REST API (OpenAPI → ``web/src/types/generated/api.ts``)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from galecore.schemas import CallScorecard, Category, FusedEvent, Severity, Word


class _M(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CallSummary(_M):
    id: uuid.UUID
    external_call_id: str | None
    started_at: datetime
    ended_at: datetime | None
    duration_s: float | None
    queue: str | None
    direction: str | None
    agent_id: uuid.UUID | None
    agent_code: str | None = None
    team_id: uuid.UUID | None = None
    mode: str
    status: str
    event_count: int = 0
    alert_count: int = 0
    max_severity: int | None = None


class CallPage(_M):
    items: list[CallSummary]
    total: int
    limit: int
    offset: int


class CallScoreOut(_M):
    agent_score: int | None
    customer_hostility: int | None
    escalation_curve: list[list[float]]
    scorecard: CallScorecard
    model_version: str


class CallDetail(CallSummary):
    channel_layout: str | None
    language_detected: str | None
    has_recording: bool
    scores: CallScoreOut | None = None


class SegmentOut(_M):
    id: str
    channel: str
    ts_start: float
    ts_end: float
    text_redacted: str
    words: list[Word]
    avg_conf: float


class SignalBrief(_M):
    signal_id: str
    source: str
    category: str
    score: float
    ts_start: float
    ts_end: float
    text_span: str | None = None


class ReviewBrief(_M):
    kind: str
    verdict: str | None
    comment: str | None
    created_at: datetime


class EventOut(FusedEvent):
    acked_by: uuid.UUID | None = None
    acked_at: datetime | None = None
    review_count: int = 0
    latest_verdict: str | None = None
    flagged: bool = False
    top_signals: list[SignalBrief] = Field(default_factory=list)
    reviews: list[ReviewBrief] = Field(default_factory=list)


class Timelines(_M):
    bin_s: float
    duration_s: float
    # channel → series name (loudness, anger, audio_toxicity, text_toxicity) → value per bin
    series: dict[str, dict[str, list[float | None]]]


class SignedUrls(_M):
    # "stereo" for an uploaded recording; "agent" / "customer" for gateway-retained live audio
    urls: dict[str, str]
    expires_in_s: int


class ReviewIn(BaseModel):
    # verdict: QA label · dispute: the agent contests · note: supervisor/coaching note ·
    # flag: send to the QA review queue
    kind: Literal["verdict", "dispute", "note", "flag"] = "verdict"
    verdict: Literal["correct", "wrong_category", "false_positive", "wrong_severity"] | None = None
    corrected_category: Category | None = None
    corrected_severity: Severity | None = None
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _shape(self) -> ReviewIn:
        if self.kind == "verdict" and self.verdict is None:
            raise ValueError("a verdict review needs `verdict`")
        if self.kind in ("dispute", "note") and not (self.comment or "").strip():
            raise ValueError(f"a {self.kind} needs a comment")
        if self.kind != "verdict" and self.verdict is not None:
            raise ValueError("only verdict reviews carry `verdict`")
        if self.verdict == "wrong_category" and self.corrected_category is None:
            raise ValueError("wrong_category needs corrected_category")
        if self.verdict == "wrong_severity" and self.corrected_severity is None:
            raise ValueError("wrong_severity needs corrected_severity")
        return self


class ReviewOut(_M):
    id: uuid.UUID
    event_id: str
    reviewer_user_id: uuid.UUID
    kind: str
    verdict: str | None
    corrected_category: str | None
    corrected_severity: int | None
    comment: str | None
    created_at: datetime


class AckOut(_M):
    event_id: str
    acked_by: uuid.UUID
    acked_at: datetime


class ReviewQueueItem(_M):
    event: EventOut
    call: CallSummary


class ReviewQueuePage(_M):
    items: list[ReviewQueueItem]
    total: int
    limit: int
    offset: int


class CategoryCount(_M):
    category: str
    speaker: str
    count: int


class DayPoint(_M):
    day: str
    calls: int
    alerts: int


class CoachingMoment(_M):
    call_id: uuid.UUID
    event_id: str
    category: str
    severity: int
    ts_start: float
    explanation_bn: str | None


class AgentAnalytics(_M):
    agent_id: uuid.UUID
    employee_code: str
    team_id: uuid.UUID | None
    calls: int
    talk_minutes: float
    alerts: int
    events_by_category: list[CategoryCount]
    avg_professionalism: float | None
    disputes: int
    trend: list[DayPoint]
    alerts_per_100_calls: float | None = None
    team_median_alerts_per_100_calls: float | None = None
    coaching_moments: list[CoachingMoment] = Field(default_factory=list)
    coaching_points_bn: list[str] = Field(default_factory=list)


class AgentRow(_M):
    agent_id: uuid.UUID
    employee_code: str
    calls: int
    alerts: int
    avg_professionalism: float | None


class HeatCell(_M):
    weekday: int  # 0 = Monday (call-centre local time)
    hour: int
    calls: int
    agent_incidents: int
    avg_customer_hostility: float | None


class TeamAnalytics(_M):
    team_id: uuid.UUID
    name: str
    calls: int
    alerts: int
    agents: list[AgentRow]
    heatmap: list[HeatCell] = Field(default_factory=list)


class QueueRow(_M):
    queue: str
    calls: int
    agent_alerts: int
    customer_alerts: int
    avg_customer_hostility: float | None


class LexiconIn(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    variants: list[str] = Field(default_factory=list)
    romanized: list[str] = Field(default_factory=list)
    category: Category
    severity: Severity
    context_sensitive: bool = False
    dialect: str | None = None
    notes: str | None = Field(default=None, max_length=2000)
    active: bool = True


class LexiconOut(_M):
    id: uuid.UUID
    term: str
    normalized: str
    variants: list[str]
    romanized: list[str]
    category: str
    severity: int
    context_sensitive: bool
    dialect: str | None
    notes: str | None
    created_by: uuid.UUID | None
    updated_at: datetime
    active: bool


class RulesIn(BaseModel):
    yaml: str = Field(min_length=1, max_length=200_000)


class RulesOut(_M):
    id: uuid.UUID
    name: str
    version: int
    yaml: str
    active: bool
    updated_by: uuid.UUID | None
    updated_at: datetime


class SimulateIn(BaseModel):
    yaml: str = Field(min_length=1, max_length=200_000)
    call_ids: list[uuid.UUID] | None = Field(default=None, max_length=500)
    limit: int = Field(default=100, ge=1, le=500)


class ModelVersionOut(_M):
    id: uuid.UUID
    component: str
    name: str
    version: str
    mlflow_run_id: str | None
    metrics: dict[str, Any] = Field(validation_alias="metrics_jsonb")
    deployed_at: datetime | None
    active: bool
    research_only: bool


class PromoteIn(BaseModel):
    model_version_id: uuid.UUID
    reason: str | None = Field(default=None, max_length=2000)


class PromotionOut(_M):
    id: uuid.UUID
    component: str
    model_version_id: uuid.UUID
    requested_by: uuid.UUID
    approved_by: uuid.UUID | None
    status: str
    reason: str | None
    created_at: datetime
    decided_at: datetime | None


class AuditOut(_M):
    id: int
    user_id: uuid.UUID | None
    action: str
    object_type: str
    object_id: str | None
    ip: str | None
    ts: datetime
    details: dict[str, Any] = Field(validation_alias="details_jsonb")


class IngestOut(_M):
    call_id: uuid.UUID
    uri: str
    queued: bool


class DevTokenIn(BaseModel):
    sub: str = Field(min_length=1, max_length=255)
    role: Literal["admin", "qa_analyst", "supervisor", "agent", "auditor"]


class DevTokenOut(_M):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 (not a secret)


class MeOut(_M):
    user_id: uuid.UUID
    sub: str
    role: str
    team_ids: list[uuid.UUID]
    agent_id: uuid.UUID | None
    permissions: list[str]


class TeamOut(_M):
    id: uuid.UUID
    name: str
    supervisor_user_id: uuid.UUID | None


class AgentOut(_M):
    id: uuid.UUID
    employee_code: str
    name: str | None
    team_id: uuid.UUID | None
    active: bool


class Directory(_M):
    teams: list[TeamOut]
    agents: list[AgentOut]


class Agreement(_M):
    events_with_multiple_verdicts: int
    reviewers: int
    pairwise_agreement: float | None
