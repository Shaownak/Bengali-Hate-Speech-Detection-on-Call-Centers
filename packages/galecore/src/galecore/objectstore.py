"""Object store interface with the local-folder dev backend (ADR-009).

Prod/staging use S3/MinIO with SSE-KMS on the bucket (docs/05): ``S3ObjectStore`` hands out real
pre-signed GET URLs. ``LocalObjectStore`` keeps objects under ``GALE_OBJECT_STORE_DIR`` and emulates
pre-signed URLs with an HMAC over (key, expiry); the API serves them at ``/api/v1/objects/…``.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import secrets
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse

if TYPE_CHECKING:
    from galecore.config import Settings


class ObjectStore(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def signed_url(self, key: str, ttl_s: int = 900) -> str: ...


def _safe_key(key: str) -> PurePosixPath:
    p = PurePosixPath(key)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in key:
        raise ValueError(f"invalid object key: {key!r}")
    return p


class LocalObjectStore(ObjectStore):
    def __init__(
        self, root: Path, secret: bytes | None = None, clock: Callable[[], float] = time.time
    ) -> None:
        self.root = root
        self._secret = secret or secrets.token_bytes(32)
        self._clock = clock

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*_safe_key(key).parts)

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def _sig(self, key: str, expires: int) -> str:
        return hmac.new(self._secret, f"{key}|{expires}".encode(), hashlib.sha256).hexdigest()

    def signed_url(self, key: str, ttl_s: int = 900) -> str:
        _safe_key(key)
        expires = int(self._clock()) + ttl_s
        return f"local://objects/{key}?" + urlencode(
            {"expires": expires, "sig": self._sig(key, expires)}
        )

    def verify(self, url: str) -> str:
        """Key of a valid, unexpired URL from ``signed_url``; else ``PermissionError``."""
        u = urlparse(url)
        key = u.path.lstrip("/")
        q = parse_qs(u.query)
        try:
            expires = int(q["expires"][0])
            sig = q["sig"][0]
        except (KeyError, ValueError, IndexError):
            raise PermissionError("malformed signed URL") from None
        if not hmac.compare_digest(sig, self._sig(key, expires)) or expires < self._clock():
            raise PermissionError("invalid or expired signed URL")
        return key


class S3ObjectStore(ObjectStore):
    """S3 / MinIO backend. Pre-signing is computed locally (no network call)."""

    def __init__(self, client: Any, bucket: str) -> None:
        self.client, self.bucket = client, bucket

    @classmethod
    def from_settings(cls, s: Settings) -> S3ObjectStore:
        from minio import Minio

        client = Minio(
            s.s3_endpoint,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            secure=s.s3_secure,
            region=s.s3_region,
        )
        return cls(client, s.s3_bucket)

    def put(self, key: str, data: bytes) -> None:
        _safe_key(key)
        self.client.put_object(self.bucket, key, io.BytesIO(data), len(data))

    def get(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, str(_safe_key(key)))
        try:
            return bytes(resp.read())
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self.client.stat_object(self.bucket, str(_safe_key(key)))
        except S3Error:
            return False
        return True

    def signed_url(self, key: str, ttl_s: int = 900) -> str:
        return str(
            self.client.presigned_get_object(
                self.bucket, str(_safe_key(key)), expires=timedelta(seconds=ttl_s)
            )
        )


def store_from_settings(s: Settings) -> ObjectStore:
    if s.object_store == "s3":
        return S3ObjectStore.from_settings(s)
    secret = s.object_url_secret.encode() or None
    return LocalObjectStore(s.object_store_dir, secret=secret)
