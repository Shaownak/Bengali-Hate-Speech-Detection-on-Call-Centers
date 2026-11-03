"""Pilot policy (docs/06 §6, ADR-020): alert visibility (shadow mode) and candidate-model routing.

``configs/pilot.yaml`` is the single source of truth: the ML workers have no database access, so
routing cannot live in Postgres. Every decision is deterministic per (component, call), so all
replicas and restarts route a call the same way.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from galecore.config import REPO_ROOT, Settings

Stage = Literal["P-0", "P-1", "P-2", "P-3"]
COMPONENTS = ("text_clf", "fusion", "llm_judge")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlertVisibility(_Strict):
    default: Literal["hidden", "visible"] = "hidden"
    visible_queues: list[str] = Field(default_factory=list)


class Candidate(_Strict):
    version: str
    role: Literal["shadow", "canary"]
    percent: float = Field(default=100, ge=0, le=100)
    queues: list[str] = Field(default_factory=list)


class EmailNotify(_Strict):
    enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 587
    starttls: bool = True
    sender: str = "gale-reports@localhost"
    to: list[str] = Field(default_factory=list)


class WebhookNotify(_Strict):
    enabled: bool = False
    format: Literal["slack", "json"] = "slack"


class Notify(_Strict):
    email: EmailNotify = EmailNotify()
    webhook: WebhookNotify = WebhookNotify()
    dashboard_url: str = ""


class ReportConfig(_Strict):
    timezone: str = "Asia/Dhaka"
    out_dir: str = "reports/pilot"
    min_slice_calls: int = 30
    fairness_precision_gap: float = 0.10
    fairness_rate_ratio: float = 1.5
    drift_psi: float = 0.2
    repeat_offender_incidents: int = 3
    notify: Notify = Notify()


class LabelStudioTarget(_Strict):
    url: str = "http://127.0.0.1:8080"
    project: str = "GAL-E T2 active learning"
    audio_prefix: str = "/data/local-files/?d="


class SamplerConfig(_Strict):
    budget: int = Field(default=300, ge=0)
    per_call_max: int = Field(default=3, ge=1)
    quotas: dict[str, float] = Field(default_factory=dict)
    low_confidence_band: tuple[float, float] = (0.4, 0.65)
    disagreement_gap: float = 0.5
    new_term_min_count: int = 3
    new_term_min_calls: int = 2
    golden_manifest: str = "data/manifests/calls.jsonl"
    labelstudio: LabelStudioTarget = LabelStudioTarget()


class PilotConfig(_Strict):
    version: int = 1
    stage: Stage = "P-1"
    alerts: AlertVisibility = AlertVisibility()
    candidates: dict[str, Candidate] = Field(default_factory=dict)
    exit_criteria: dict[str, dict[str, object]] = Field(default_factory=dict)
    report: ReportConfig = ReportConfig()
    sampler: SamplerConfig = SamplerConfig()

    def model_post_init(self, _: object) -> None:
        unknown = set(self.candidates) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"candidates for unknown components: {sorted(unknown)}")


def pilot_path(settings: Settings | None = None) -> Path:
    if settings is not None and settings.pilot_config is not None:
        return settings.pilot_config
    return (settings.config_dir if settings else REPO_ROOT / "configs") / "pilot.yaml"


@lru_cache(maxsize=8)
def load_pilot(path: Path) -> PilotConfig:
    if not path.exists():
        return PilotConfig()
    return PilotConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def bucket(component: str, call_id: str) -> float:
    """Stable position of a call in [0, 100) for one component (salted so the canaries of
    different components don't land on the same calls)."""
    h = hashlib.sha256(f"{component}:{call_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 * 100


class PilotPolicy:
    def __init__(self, cfg: PilotConfig) -> None:
        self.cfg = cfg

    @classmethod
    def load(cls, settings: Settings | None = None) -> PilotPolicy:
        return cls(load_pilot(pilot_path(settings)))

    def alerts_visible(self, queue: str | None) -> bool:
        a = self.cfg.alerts
        return a.default == "visible" or (queue is not None and queue in a.visible_queues)

    def route(self, component: str, call_id: str, queue: str | None) -> Candidate | None:
        """The candidate that handles this call for ``component`` (shadow or canary), or None."""
        c = self.cfg.candidates.get(component)
        if c is None or c.percent <= 0:
            return None
        if c.queues and queue not in c.queues:
            return None
        return c if bucket(component, call_id) < c.percent else None
