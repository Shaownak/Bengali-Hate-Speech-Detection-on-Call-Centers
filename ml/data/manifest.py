"""In-house call manifests (docs/02 §2.1) and pseudonymisation (docs/02 §1.2).

One JSONL row per call channel: ``ml/data/manifests/<name>.jsonl`` (real manifests live under
``$GALE_DATA_DIR/data/manifests``; they hold no audio and no customer numbers, only salted hashes).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Split = Literal["train", "dev", "test", "golden"]
_NS = uuid.UUID("9d7f1d5e-4a1b-4c55-9a3e-6a1e6a1e0002")


class ChannelRow(BaseModel):
    """One channel of one call, after ingest + processing."""

    model_config = ConfigDict(extra="forbid")

    call_id: str  # GAL-E UUID (D-18), stable for the same source recording
    channel: Literal["agent", "customer", "mixed"]
    source: str  # adapter + location, e.g. "folder:/mnt/rec" or "s3:bucket/prefix"
    source_ref: str  # path/key of the original recording inside that source
    audio_path: str | None = None  # native-rate channel copy (8 kHz telephony)
    audio_16k_path: str | None = None  # model copy
    duration: float = Field(ge=0)
    sample_rate: int
    codec: str = "pcm_s16le"
    layout: Literal["stereo_agent_left", "stereo_agent_right", "mono"] = "stereo_agent_left"
    queue: str | None = None
    date: str | None = None  # ISO date of the call
    direction: Literal["inbound", "outbound"] | None = None
    agent_id: str | None = None  # kept for coaching, access-controlled
    customer_hash: str | None = None  # salted HMAC of the customer number, never the number
    dialect: str | None = None
    split: Split | None = None
    quality: dict[str, float] = Field(default_factory=dict)
    dropped: str | None = None  # reason when a quality filter removed this channel
    synthetic: bool = False


class PseudonymSaltError(RuntimeError):
    pass


def load_salt(env: str = "GALE_PSEUDONYM_SALT") -> bytes:
    """The salt is a secret: ``$GALE_PSEUDONYM_SALT`` or the file ``$GALE_PSEUDONYM_SALT_FILE``."""
    if os.environ.get(f"{env}_FILE"):
        salt = Path(os.environ[f"{env}_FILE"]).read_bytes().strip()
    else:
        salt = os.environ.get(env, "").encode()
    if len(salt) < 16:
        raise PseudonymSaltError(f"set {env} (or {env}_FILE) to a secret of at least 16 bytes")
    return salt


def pseudonymise(customer_number: str | None, salt: bytes) -> str | None:
    """Salted HMAC-SHA256 of the digits only, so +880 / 0 prefixes and spacing hash the same."""
    if not customer_number:
        return None
    digits = "".join(ch for ch in customer_number if ch.isdigit())
    if digits.startswith("880"):
        digits = "0" + digits[3:]
    return "c_" + hmac.new(salt, digits.encode(), hashlib.sha256).hexdigest()[:24]


def call_uuid(source: str, recording_id: str) -> str:
    """D-18: GAL-E mints UUID call ids; uuid5 makes re-ingesting a recording idempotent."""
    return str(uuid.uuid5(_NS, f"{source}|{recording_id}"))


def write_manifest(rows: Iterable[ChannelRow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_manifest(path: Path) -> list[ChannelRow]:
    with path.open(encoding="utf-8") as f:
        return [ChannelRow.model_validate(json.loads(line)) for line in f if line.strip()]
