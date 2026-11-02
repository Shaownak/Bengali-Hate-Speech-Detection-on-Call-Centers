"""Object store interface with the local-folder dev backend (ADR-009).

Prod/staging use S3/MinIO with SSE-KMS (docs/05); the S3 backend arrives with the API in P5.
``LocalObjectStore`` keeps objects under ``GALE_OBJECT_STORE_DIR`` and emulates pre-signed
URLs with an HMAC over (key, expiry), so code paths that hand out audio links work in dev.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlencode, urlparse


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
