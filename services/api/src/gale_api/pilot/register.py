"""Register a retrained model as a shadow candidate (ADR-020; last step of the retrain workflow).

    python -m gale_api register-model candidate.json

``candidate.json`` is written by ``ml.retrain gate``. Registration never activates anything:
the version lands in ``model_versions`` with ``shadow=true``; ops route a share of calls to it in
the pilot config, and promotion stays the two-person approval in /admin/models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from gale_api.db import ModelVersion


class GateResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    passed: bool
    regressions: list[dict[str, Any]] = Field(default_factory=list)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    component: Literal["text_clf", "fusion", "llm_judge"]
    name: str
    version: str = Field(min_length=1, max_length=200)
    artifact_uri: str
    research_only: bool = False
    mlflow_run_id: str | None = None
    baseline_version: str | None = None
    golden: dict[str, Any]
    gate: GateResult
    training: dict[str, Any] = Field(default_factory=dict)


class RegistrationError(ValueError):
    pass


def register_model(db: Session, cand: Candidate) -> tuple[ModelVersion, bool]:
    """(row, created). Refuses a candidate that failed the gate or touched the golden set."""
    if not cand.gate.passed:
        raise RegistrationError("promotion gate failed: not registering (see the gate report)")
    if int(cand.training.get("golden_overlap", 0)):
        raise RegistrationError("training data overlaps the golden set (CLAUDE.md rule 3)")
    existing = db.scalar(
        select(ModelVersion).where(
            ModelVersion.component == cand.component, ModelVersion.version == cand.version
        )
    )
    if existing is not None:
        return existing, False
    row = ModelVersion(
        component=cand.component,
        name=cand.name,
        version=cand.version,
        mlflow_run_id=cand.mlflow_run_id,
        metrics_jsonb={
            "golden": cand.golden,
            "gate": cand.gate.model_dump(),
            "training": cand.training,
            "artifact_uri": cand.artifact_uri,
            "baseline_version": cand.baseline_version,
        },
        active=False,
        research_only=cand.research_only,
        shadow=True,
    )
    db.add(row)
    db.flush()
    return row, True
