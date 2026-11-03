"""Lexicon, rules (+ simulate), models (two-person promotion), audit log and recording ingest."""

from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

import soundfile as sf
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import ValidationError
from sqlalchemy import ColumnElement, and_, func, select, update
from sqlalchemy.orm import Session

from gale_api import repo
from gale_api.api_models import (
    AuditOut,
    IngestOut,
    LexiconIn,
    LexiconOut,
    ModelVersionOut,
    PromoteIn,
    PromotionOut,
    RulesIn,
    RulesOut,
    SimulateIn,
)
from gale_api.auth import DB, Principal, Role, require
from gale_api.db import (
    AuditLog,
    Call,
    EventRow,
    LexiconEntry,
    ModelPromotion,
    ModelVersion,
    RuleSet,
    SignalRow,
    now,
)
from gale_api.retention import DsarIn, DsarOut, run_dsar
from gale_fusion_engine.rules import parse_rules
from gale_fusion_engine.simulate import SimulationReport, simulate
from galecore.bangla import normalize
from galecore.bus import POSTCALL_JOBS
from galecore.config import RulesConfig, ThresholdsConfig, load_thresholds
from galecore.schemas import Category, PostcallJob

router = APIRouter(prefix="/api/v1", tags=["admin"])

RULES_NAME = "fusion"
MAX_UPLOAD = 200 * 1024 * 1024


# --- lexicon ------------------------------------------------------------------------------------


@router.get("/lexicon", response_model=list[LexiconOut])
def list_lexicon(
    db: DB,
    p: Annotated[Principal, require("lexicon.read")],
    category: Category | None = None,
    active: bool | None = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[LexiconEntry]:
    conds: list[ColumnElement[bool]] = []
    if category:
        conds.append(LexiconEntry.category == category.value)
    if active is not None:
        conds.append(LexiconEntry.active.is_(active))
    if q:
        conds.append(LexiconEntry.normalized.icontains(normalize(q), autoescape=True))
    stmt = select(LexiconEntry).where(and_(*conds)) if conds else select(LexiconEntry)
    return list(db.scalars(stmt.order_by(LexiconEntry.normalized).limit(limit).offset(offset)))


def _lex_values(body: LexiconIn) -> dict[str, Any]:
    return {
        "term": body.term,
        "normalized": normalize(body.term),
        "variants": body.variants,
        "romanized": body.romanized,
        "category": body.category.value,
        "severity": body.severity,
        "context_sensitive": body.context_sensitive,
        "dialect": body.dialect,
        "notes": body.notes,
        "active": body.active,
        "updated_at": now(),
    }


@router.post("/lexicon", response_model=LexiconOut, status_code=201)
def create_lexicon(
    body: LexiconIn, db: DB, p: Annotated[Principal, require("lexicon.propose")]
) -> LexiconEntry:
    values = _lex_values(body)
    if p.role != Role.ADMIN:
        values["active"] = False  # QA analysts propose; an admin activates (docs/04 §1)
    row = LexiconEntry(**values, created_by=p.user_id)
    db.add(row)
    db.flush()
    return row


def _lex(db: Session, entry_id: uuid.UUID) -> LexiconEntry:
    row = db.get(LexiconEntry, entry_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "lexicon entry not found")
    return row


@router.put("/lexicon/{entry_id}", response_model=LexiconOut)
def update_lexicon(
    entry_id: uuid.UUID, body: LexiconIn, db: DB, p: Annotated[Principal, require("lexicon.write")]
) -> LexiconEntry:
    row = _lex(db, entry_id)
    for k, v in _lex_values(body).items():
        setattr(row, k, v)
    return row


@router.delete("/lexicon/{entry_id}", status_code=204)
def delete_lexicon(
    entry_id: uuid.UUID, db: DB, p: Annotated[Principal, require("lexicon.write")]
) -> None:
    """Soft delete: the entry is deactivated, so history and audit stay intact."""
    row = _lex(db, entry_id)
    row.active, row.updated_at = False, now()


# --- rules --------------------------------------------------------------------------------------


def _validate_rules(text: str) -> RulesConfig:
    try:
        cfg = RulesConfig.model_validate(yaml.safe_load(text))
        parse_rules(cfg)
    except (yaml.YAMLError, ValidationError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"invalid rules: {type(e).__name__}: {e}"
        ) from None
    return cfg


def active_rules(db: Session, request: Request) -> RuleSet:
    row = db.scalar(select(RuleSet).where(RuleSet.name == RULES_NAME, RuleSet.active))
    if row is None:  # first use: seed version 1 from configs/rules.yaml
        path = request.app.state.settings.config_dir / "rules.yaml"
        row = RuleSet(
            name=RULES_NAME,
            yaml=path.read_text(encoding="utf-8"),
            version=1,
            active=True,
            updated_at=now(),
        )
        db.add(row)
        db.flush()
    return row


@router.get("/rules", response_model=list[RulesOut])
def get_rules(
    request: Request,
    db: DB,
    p: Annotated[Principal, require("rules.read")],
    history: bool = False,
) -> list[RuleSet]:
    current = active_rules(db, request)
    if not history:
        return [current]
    return list(
        db.scalars(
            select(RuleSet).where(RuleSet.name == RULES_NAME).order_by(RuleSet.version.desc())
        )
    )


@router.put("/rules", response_model=RulesOut)
def put_rules(
    body: RulesIn, request: Request, db: DB, p: Annotated[Principal, require("rules.write")]
) -> RuleSet:
    """New active version. Workers load rules at start; see ADR-016 for hot reload."""
    _validate_rules(body.yaml)
    active_rules(db, request)  # seeds v1 from configs/ if this is the first change
    latest = db.scalar(select(func.max(RuleSet.version)).where(RuleSet.name == RULES_NAME)) or 0
    db.execute(update(RuleSet).where(RuleSet.name == RULES_NAME).values(active=False))
    row = RuleSet(
        name=RULES_NAME,
        yaml=body.yaml,
        version=latest + 1,
        active=True,
        updated_by=p.user_id,
        updated_at=now(),
    )
    db.add(row)
    db.flush()
    return row


@router.post("/rules/simulate", response_model=SimulationReport)
def simulate_rules(
    body: SimulateIn, request: Request, db: DB, p: Annotated[Principal, require("rules.write")]
) -> SimulationReport:
    candidate = parse_rules(_validate_rules(body.yaml))
    baseline = parse_rules(_validate_rules(active_rules(db, request).yaml))
    if body.call_ids:
        ids = list(body.call_ids)
    else:
        ids = list(
            db.scalars(
                select(Call.id)
                .where(Call.status == "ended")
                .order_by(Call.started_at.desc())
                .limit(body.limit)
            )
        )
    calls: dict[str, tuple[list[Any], list[Any]]] = {str(i): ([], []) for i in ids}
    for s in db.scalars(select(SignalRow).where(SignalRow.call_id.in_(ids))):
        calls[str(s.call_id)][0].append(repo.signal_of(s))
    for e in db.scalars(select(EventRow).where(EventRow.call_id.in_(ids))):
        calls[str(e.call_id)][1].append(repo.event_of(e))
    thresholds = load_thresholds(request.app.state.settings.config_dir)
    return simulate(calls, baseline, candidate, thresholds)


@router.get("/thresholds", response_model=ThresholdsConfig)
def get_thresholds(
    request: Request, p: Annotated[Principal, require("rules.read")]
) -> ThresholdsConfig:
    """Alert thresholds per category × speaker × mode (read-only; edited in configs/, ADR-017)."""
    return load_thresholds(request.app.state.settings.config_dir)


# --- models -------------------------------------------------------------------------------------


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(
    db: DB, p: Annotated[Principal, require("models.read")], component: str | None = None
) -> list[ModelVersion]:
    stmt = select(ModelVersion)
    if component:
        stmt = stmt.where(ModelVersion.component == component)
    return list(db.scalars(stmt.order_by(ModelVersion.component, ModelVersion.version)))


@router.get("/models/promotions", response_model=list[PromotionOut])
def list_promotions(
    db: DB,
    p: Annotated[Principal, require("models.read")],
    promotion_status: Annotated[str | None, Query(alias="status")] = None,
) -> list[ModelPromotion]:
    stmt = select(ModelPromotion)
    if promotion_status:
        stmt = stmt.where(ModelPromotion.status == promotion_status)
    return list(db.scalars(stmt.order_by(ModelPromotion.created_at.desc()).limit(200)))


@router.post("/models/{component}/promote", response_model=PromotionOut, status_code=202)
def request_promotion(
    component: str, body: PromoteIn, db: DB, p: Annotated[Principal, require("models.promote")]
) -> ModelPromotion:
    """Step 1 of two-person approval: one admin requests, a different admin approves."""
    mv = db.get(ModelVersion, body.model_version_id)
    if mv is None or mv.component != component:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found for component")
    if mv.research_only:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "research_only model versions are not promotable (ADR-012)"
        )
    if mv.active:
        raise HTTPException(status.HTTP_409_CONFLICT, "already active")
    gate = (mv.metrics_jsonb or {}).get("gate")
    if gate is not None and not gate.get("passed"):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "failed the golden-set promotion gate (ADR-020)"
        )
    pending = db.scalar(
        select(ModelPromotion).where(
            ModelPromotion.model_version_id == mv.id, ModelPromotion.status == "pending"
        )
    )
    if pending is not None:
        return pending
    row = ModelPromotion(
        component=component,
        model_version_id=mv.id,
        requested_by=p.user_id,
        status="pending",
        reason=body.reason,
        created_at=now(),
    )
    db.add(row)
    db.flush()
    return row


def _decide(
    db: Session, p: Principal, promotion_id: uuid.UUID, decision: Literal["approved", "rejected"]
) -> ModelPromotion:
    row = db.get(ModelPromotion, promotion_id, with_for_update=True)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "promotion not found")
    if row.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, f"promotion already {row.status}")
    if row.requested_by == p.user_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "two-person rule: the requester cannot decide"
        )
    row.status, row.approved_by, row.decided_at = decision, p.user_id, now()
    if decision == "approved":
        db.execute(
            update(ModelVersion).where(ModelVersion.component == row.component).values(active=False)
        )
        mv = db.get(ModelVersion, row.model_version_id)
        if mv is None:  # pragma: no cover - FK guarantees it
            raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
        mv.active, mv.shadow, mv.deployed_at = True, False, now()
    return row


@router.post("/models/promotions/{promotion_id}/approve", response_model=PromotionOut)
def approve_promotion(
    promotion_id: uuid.UUID, db: DB, p: Annotated[Principal, require("models.promote")]
) -> ModelPromotion:
    return _decide(db, p, promotion_id, "approved")


@router.post("/models/promotions/{promotion_id}/reject", response_model=PromotionOut)
def reject_promotion(
    promotion_id: uuid.UUID, db: DB, p: Annotated[Principal, require("models.promote")]
) -> ModelPromotion:
    return _decide(db, p, promotion_id, "rejected")


# --- audit --------------------------------------------------------------------------------------


@router.get("/audit", response_model=list[AuditOut])
def list_audit(
    db: DB,
    p: Annotated[Principal, require("audit.read")],
    user_id: uuid.UUID | None = None,
    action: str | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditLog]:
    conds: list[ColumnElement[bool]] = []
    if user_id:
        conds.append(AuditLog.user_id == user_id)
    if action:
        conds.append(AuditLog.action == action)
    if object_type:
        conds.append(AuditLog.object_type == object_type)
    if object_id:
        conds.append(AuditLog.object_id == object_id)
    if date_from:
        conds.append(AuditLog.ts >= date_from)
    if date_to:
        conds.append(AuditLog.ts < date_to)
    stmt = select(AuditLog).where(and_(*conds)) if conds else select(AuditLog)
    return list(db.scalars(stmt.order_by(AuditLog.id.desc()).limit(limit).offset(offset)))


# --- ingest -------------------------------------------------------------------------------------


@router.post("/ingest/recordings", response_model=IngestOut, status_code=202)
async def ingest_recording(
    request: Request,
    p: Annotated[Principal, require("ingest.write")],
    file: Annotated[UploadFile, File(description="stereo WAV (agent left by default)")],
    external_ref: Annotated[str | None, Form(max_length=255)] = None,
    queue: Annotated[str | None, Form(max_length=100)] = None,
    channel_layout: Annotated[
        Literal["stereo_agent_left", "stereo_agent_right", "mono"], Form()
    ] = "stereo_agent_left",
) -> IngestOut:
    """Upload one recording for post-call processing (it becomes a ``postcall.jobs`` entry)."""
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "recording over 200 MB")
    try:
        info = sf.info(io.BytesIO(data))
    except RuntimeError:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "not a readable WAV") from None
    if info.channels not in (1, 2) or (info.channels == 1) != (channel_layout == "mono"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "channel count does not match channel_layout"
        )
    call_id = uuid.uuid4()  # D-18
    key = f"recordings/{call_id}.wav"
    state = request.app.state
    await _to_thread(state.store.put, key, data)

    def register() -> None:
        with state.sessions() as s, s.begin():
            s.add(
                Call(
                    id=call_id,
                    external_call_id=external_ref,
                    started_at=now(),
                    queue=queue,
                    channel_layout=channel_layout,
                    recording_uri=key,
                    mode="postcall",
                    status="queued",
                    created_at=now(),
                )
            )

    await _to_thread(register)
    job = PostcallJob(
        call_id=str(call_id),
        uri=key,
        channel_layout=channel_layout,
        external_ref=external_ref,
        queue=queue,
    )
    await state.bus.publish(POSTCALL_JOBS, job, idem_key=str(call_id))
    return IngestOut(call_id=call_id, uri=key, queued=True)


async def _to_thread(fn: Any, *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)


# --- DSAR (docs/05 §7) --------------------------------------------------------------------------


@router.post("/dsar", response_model=DsarOut)
def dsar(
    body: DsarIn, request: Request, db: DB, p: Annotated[Principal, require("dsar.execute")]
) -> DsarOut:
    """Erase a data subject's calls (audio, transcripts, signals, events, reviews, scores).
    Defaults to a dry run; runbook docs/runbooks/dsar-deletion.md."""
    return run_dsar(db, request.app.state.store, body, p.user_id)
