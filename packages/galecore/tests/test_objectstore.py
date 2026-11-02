"""S3 / MinIO object store backend."""

from typing import Any

import pytest


def test_s3_store_presigns_offline_and_wraps_client() -> None:
    from galecore.config import Settings
    from galecore.objectstore import S3ObjectStore, store_from_settings

    s = Settings(
        env="test",
        object_store="s3",
        s3_endpoint="minio.internal:9000",
        s3_access_key="ak",
        s3_secret_key="sk",
        s3_bucket="calls",
    )
    store = store_from_settings(s)
    assert isinstance(store, S3ObjectStore)
    url = store.signed_url("audio/c1/agent.wav", ttl_s=60)
    assert url.startswith("http://minio.internal:9000/calls/audio/c1/agent.wav?")
    assert "X-Amz-Expires=60" in url
    with pytest.raises(ValueError, match="invalid object key"):
        store.signed_url("../etc/passwd")

    class FakeResp:
        def read(self) -> bytes:
            return b"data"

        def close(self) -> None: ...

        def release_conn(self) -> None: ...

    class FakeClient:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put_object(self, bucket: str, key: str, data: Any, length: int) -> None:
            self.objects[key] = data.read(length)

        def get_object(self, bucket: str, key: str) -> FakeResp:
            return FakeResp()

        def remove_object(self, bucket: str, key: str) -> None:
            del self.objects[key]

        def stat_object(self, bucket: str, key: str) -> None:
            from minio.error import S3Error

            if key not in self.objects:
                raise S3Error("NoSuchKey", "missing", key, "r", "h", None)  # type: ignore[arg-type]

    fake = S3ObjectStore(FakeClient(), "calls")
    fake.put("a/b.wav", b"data")
    assert fake.exists("a/b.wav")
    assert not fake.exists("a/c.wav")
    assert fake.get("a/b.wav") == b"data"
    assert fake.delete("a/b.wav") is True
    assert fake.delete("a/b.wav") is False
