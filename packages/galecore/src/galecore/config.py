"""Settings (env, prefix ``GALE_``) and validated YAML config from ``configs/``.

Thresholds and rules are config, never code (CLAUDE.md).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from galecore.schemas import Category, Severity, Speaker

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GALE_", env_file=None, extra="ignore")

    env: Literal["dev", "test", "staging", "prod"] = "dev"
    model_mode: Literal["stub", "real"] = "stub"
    model_dir: Path = REPO_ROOT / ".gale" / "models"
    lexicon_path: Path | None = None  # real lexicon, outside the repo (CLAUDE.md hard rule 2)
    config_dir: Path = REPO_ROOT / "configs"
    # pilot policy override (Helm ConfigMap); default <config_dir>/pilot.yaml (ADR-020)
    pilot_config: Path | None = None
    data_dir: Path = REPO_ROOT / ".gale"
    # synthetic fixture calls the stub models read their script from (`fixture:<name>` refs);
    # in a cluster, a dev-only ConfigMap mount (Helm `devFixtures`)
    fixture_calls_dir: Path = REPO_ROOT / "tests" / "fixtures" / "calls"

    bus_url: str = "redis://127.0.0.1:6380/0"
    bus_backend: Literal["redis", "memory"] = "redis"
    db_url: str = "postgresql+psycopg://postgres@127.0.0.1:5433/gale"
    object_store: Literal["local", "s3"] = "local"
    object_store_dir: Path = REPO_ROOT / ".gale" / "objects"

    # LLM judge: our own vLLM (OpenAI-compatible) server, never an external API (CLAUDE.md rule 1)
    llm_base_url: str = "http://127.0.0.1:8100/v1"
    llm_model: str = "gale-llm-judge"
    # docs/08 §5: 4 s window / 60 s call against vLLM; the local transformers server (ADR-021,
    # `make llm-serve`) needs longer
    llm_window_timeout_s: float = Field(default=4.0, gt=0)
    llm_call_timeout_s: float = Field(default=60.0, gt=0)

    # Call-affine sharding of the stateful real-time workers (ADR-018): replica `shard_index` of
    # `shard_count` handles the calls whose audio shard % shard_count == shard_index. The index
    # defaults to the StatefulSet ordinal at the end of HOSTNAME (gale-asr-worker-3 → 3).
    shard_count: int = Field(default=1, ge=1, le=8)
    shard_index: int | None = None
    workers: bool = True  # run the service's bus consumer (off for API-only test apps)

    # S3 / MinIO (object_store="s3"); SSE-KMS is configured on the bucket (docs/05)
    s3_endpoint: str = "127.0.0.1:9000"
    s3_secure: bool = False
    s3_bucket: str = "gale"
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    # HMAC key for local signed URLs; random per process if unset (then only that process verifies)
    object_url_secret: str = ""
    signed_url_ttl_s: int = 300

    auth_mode: Literal["oidc", "dev"] = "oidc"
    # OIDC (Keycloak): issuer = https://kc/realms/<realm>; JWKS defaults to the realm certs URL
    oidc_issuer: str = "http://127.0.0.1:8080/realms/gale"
    oidc_audience: str = "gale-api"
    oidc_jwks_url: str | None = None
    oidc_role_claim: str = "realm_access.roles"
    dev_jwt_secret: str = "gale-dev-only-secret-not-for-prod-0000"  # noqa: S105 (dev/test only)
    display_tz: str = "Asia/Dhaka"  # call-centre local time for hour-of-day analytics
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    db_auto_migrate: bool = False  # api runs `alembic upgrade head` at startup (dev stack)
    log_level: str = "INFO"
    log_json: bool = True

    service_name: str = "gale"
    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("db_url")
    @classmethod
    def _psycopg_driver(cls, v: str) -> str:
        """Operators (CloudNativePG) hand out ``postgresql://`` URIs; use the psycopg 3 driver."""
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix) :]
        return v

    @model_validator(mode="after")
    def _guards(self) -> Settings:
        # ADR-009: the local dev auth issuer must never run outside dev/test.
        if self.auth_mode == "dev" and self.env not in ("dev", "test"):
            raise ValueError("GALE_AUTH_MODE=dev is only allowed when GALE_ENV is dev or test")
        if self.env == "prod" and self.model_mode == "stub":
            raise ValueError("stub models are not allowed in prod")
        return self


def shard_of(settings: Settings) -> tuple[int, int] | None:
    """(index, count) for this replica, or None when unsharded."""
    if settings.shard_count <= 1:
        return None
    index = settings.shard_index
    if index is None:
        import os
        import re

        m = re.search(r"-(\d+)$", os.environ.get("HOSTNAME", ""))
        if m is None:
            raise ValueError(
                "GALE_SHARD_COUNT > 1 needs GALE_SHARD_INDEX or a StatefulSet hostname"
            )
        index = int(m.group(1))
    if not 0 <= index < settings.shard_count:
        raise ValueError(f"shard index {index} outside 0..{settings.shard_count - 1}")
    return index, settings.shard_count


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# YAML configs
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpeakerThresholds(_Strict):
    agent: float = Field(ge=0, le=1)
    customer: float = Field(ge=0, le=1)

    def for_speaker(self, speaker: Speaker) -> float:
        return self.agent if speaker == Speaker.AGENT else self.customer


class CategoryThresholds(_Strict):
    log: float = Field(ge=0, le=1)
    alert_realtime: SpeakerThresholds
    alert_postcall: SpeakerThresholds


class Hysteresis(_Strict):
    cooldown_s: float = Field(gt=0)
    merge_gap_s: float = Field(ge=0)
    escalate_repeats: int = Field(ge=2)
    escalate_window_s: float = Field(gt=0)


class ThresholdsConfig(_Strict):
    version: int
    hysteresis: Hysteresis
    categories: dict[Category, CategoryThresholds]

    @model_validator(mode="after")
    def _all_categories(self) -> ThresholdsConfig:
        missing = set(Category) - set(self.categories)
        if missing:
            raise ValueError(f"thresholds missing categories: {sorted(missing)}")
        return self


class Rule(_Strict):
    name: str
    description: str = ""
    when: dict[str, Any]
    then: dict[str, Any]


class RulesConfig(_Strict):
    version: int
    rules: list[Rule]


class ComponentModel(_Strict):
    backend: str
    version: str
    revision: str | None = None  # HF commit hash (CLAUDE.md hard rule 6)


class ModelsConfig(_Strict):
    version: int
    components: dict[str, ComponentModel]


class LexiconEntry(_Strict):
    id: str
    term: str
    variants: list[str] = Field(default_factory=list)
    romanized: list[str] = Field(default_factory=list)
    category: Category
    severity: Severity
    context_sensitive: bool = False
    dialect: str = "standard"


class LexiconConfig(_Strict):
    version: int
    fake: bool = False
    entries: list[LexiconEntry]


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_thresholds(config_dir: Path | None = None) -> ThresholdsConfig:
    d = config_dir or get_settings().config_dir
    return ThresholdsConfig.model_validate(_load_yaml(d / "thresholds.yaml"))


def load_rules(config_dir: Path | None = None) -> RulesConfig:
    d = config_dir or get_settings().config_dir
    return RulesConfig.model_validate(_load_yaml(d / "rules.yaml"))


def load_models(config_dir: Path | None = None) -> ModelsConfig:
    d = config_dir or get_settings().config_dir
    return ModelsConfig.model_validate(_load_yaml(d / "models.yaml"))


def load_canonical(config_dir: Path | None = None) -> dict[str, str]:
    """Loanword canonical-form map for ``bangla.normalize(canonical=...)``."""
    d = config_dir or get_settings().config_dir
    data = _load_yaml(d / "bangla_canonical.yaml") or {}
    return {str(k): str(v) for k, v in (data.get("canonical") or {}).items()}


def load_lexicon(settings: Settings | None = None) -> LexiconConfig:
    """Real lexicon from ``GALE_LEXICON_PATH`` if set, else the fake placeholder lexicon.

    In prod the bootstrap seed is imported into Postgres and workers read a DB snapshot (D-10).
    """
    s = settings or get_settings()
    path = s.lexicon_path or (s.config_dir / "lexicon" / "fake_lexicon.yaml")
    lex = LexiconConfig.model_validate(_load_yaml(path))
    if s.env == "prod" and lex.fake:
        raise ValueError("the fake lexicon must not be used in prod")
    return lex
