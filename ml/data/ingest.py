"""Ingestion adapters → ``Recording`` records (docs/02 §2.1, Prompt 4).

- ``FolderSource``: stereo WAVs under a folder, optional ``<name>.json`` CDR sidecar per file.
- ``S3Source``: an S3 / MinIO bucket prefix (boto3, optional ``data-engine`` group); files are
  fetched to a local cache on demand.
- ``RecorderCsvSource``: a recorder export = CSV of CDR rows + a folder of WAVs. Column names are
  configurable (``RecorderColumns``) because every recorder names them differently.

Customer numbers are pseudonymised at ingest (salted HMAC) and never stored or logged in clear.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ml.data.manifest import call_uuid, pseudonymise

AUDIO_EXT = (".wav", ".flac")


@dataclass
class Recording:
    call_id: str
    source: str
    source_ref: str
    local_path: Path
    meta: dict[str, Any] = field(default_factory=dict)  # queue, date, agent_id, customer_hash, …


class Source(Protocol):
    name: str

    def recordings(self) -> Iterator[Recording]: ...


CDR_FIELDS = ("queue", "date", "direction", "agent_id", "dialect", "layout")


def _meta(raw: dict[str, Any], salt: bytes) -> dict[str, Any]:
    meta = {k: raw[k] for k in CDR_FIELDS if raw.get(k) not in (None, "")}
    if raw.get("customer_number"):
        meta["customer_hash"] = pseudonymise(str(raw["customer_number"]), salt)
    return meta


class FolderSource:
    def __init__(self, root: Path, salt: bytes) -> None:
        self.root, self.salt = root, salt
        self.name = f"folder:{root.as_posix()}"

    def recordings(self) -> Iterator[Recording]:
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXT or not p.is_file():
                continue
            ref = p.relative_to(self.root).as_posix()
            side = p.with_suffix(".json")
            raw = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
            yield Recording(call_uuid(self.name, ref), self.name, ref, p, _meta(raw, self.salt))


class S3Source:
    """``client``: a boto3 S3 client (anything with ``list_objects_v2`` + ``download_file``)."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache: Path,
        salt: bytes,
        client: Any = None,
        endpoint_url: str | None = None,
    ) -> None:
        if client is None:  # pragma: no cover - needs boto3 + credentials
            import boto3

            client = boto3.client("s3", endpoint_url=endpoint_url)
        self.client, self.bucket, self.prefix, self.cache, self.salt = (
            client,
            bucket,
            prefix,
            cache,
            salt,
        )
        self.name = f"s3:{bucket}/{prefix}"

    def _keys(self) -> Iterator[str]:
        token: str | None = None
        while True:
            kw: dict[str, Any] = {"Bucket": self.bucket, "Prefix": self.prefix}
            if token:
                kw["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kw)
            for obj in page.get("Contents", []):
                yield obj["Key"]
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")

    def recordings(self) -> Iterator[Recording]:
        keys = sorted(self._keys())
        sidecars = {k for k in keys if k.endswith(".json")}
        for key in keys:
            if not key.lower().endswith(AUDIO_EXT):
                continue
            local = self.cache / self.bucket / key
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                self.client.download_file(self.bucket, key, str(local))
            raw: dict[str, Any] = {}
            side = key.rsplit(".", 1)[0] + ".json"
            if side in sidecars:
                side_local = local.with_suffix(".json")
                self.client.download_file(self.bucket, side, str(side_local))
                raw = json.loads(side_local.read_text(encoding="utf-8"))
            yield Recording(call_uuid(self.name, key), self.name, key, local, _meta(raw, self.salt))


@dataclass
class RecorderColumns:
    """Recorder CSV column names (defaults are common Asterisk/recorder export names)."""

    file: str = "recording_file"
    recording_id: str = "recording_id"
    customer_number: str = "caller_number"
    agent_id: str = "agent_id"
    queue: str = "queue"
    date: str = "start_time"
    direction: str = "direction"
    layout: str = "channel_layout"


class RecorderCsvSource:
    def __init__(
        self, csv_path: Path, audio_dir: Path, salt: bytes, columns: RecorderColumns | None = None
    ) -> None:
        self.csv_path, self.audio_dir, self.salt = csv_path, audio_dir, salt
        self.cols = columns or RecorderColumns()
        self.name = f"recorder:{csv_path.as_posix()}"
        self.missing: list[str] = []  # recording ids whose file is not in audio_dir

    def recordings(self) -> Iterator[Recording]:
        c = self.cols
        with self.csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                path = self.audio_dir / row[c.file]
                rid = row.get(c.recording_id) or row[c.file]
                if not path.is_file():
                    self.missing.append(rid)
                    continue
                raw = {
                    "customer_number": row.get(c.customer_number),
                    "agent_id": row.get(c.agent_id),
                    "queue": row.get(c.queue),
                    "date": (row.get(c.date) or "")[:10] or None,
                    "direction": (row.get(c.direction) or "").lower() or None,
                    "layout": row.get(c.layout),
                }
                yield Recording(
                    call_uuid(self.name, rid), self.name, rid, path, _meta(raw, self.salt)
                )
