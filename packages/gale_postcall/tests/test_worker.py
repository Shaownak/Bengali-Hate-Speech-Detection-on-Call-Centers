"""Post-call worker: jobs from the API upload (object-store key) publish a ``postcall.results``."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fakeredis import FakeAsyncRedis

from gale_postcall.worker import PostcallWorker
from galecore.bus import CALL_SCORES, EVENTS_FUSED, POSTCALL_RESULTS, BusClient, decode
from galecore.config import Settings
from galecore.objectstore import LocalObjectStore
from galecore.schemas import PostcallJob, PostcallResult
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav


async def test_job_from_object_store_key(tmp_path: Path) -> None:
    fixture = CALLS_DIR / "demo_01"
    ensure_wav(fixture)
    store = LocalObjectStore(tmp_path / "objects")
    key = "recordings/demo.wav"
    store.put(key, (fixture / "call.wav").read_bytes())
    s = Settings(env="test", bus_backend="memory", data_dir=tmp_path, log_level="WARNING")
    bus = BusClient(FakeAsyncRedis())
    worker = PostcallWorker(bus, s, store=store, min_idle_ms=0)
    cid = str(uuid.uuid4())
    await worker.on_job(
        PostcallJob(call_id=cid, uri=key, external_ref=f"fixture:{fixture}", queue="billing")
    )
    rows = await bus.redis.xrange(POSTCALL_RESULTS)
    res = decode(rows[0][1], PostcallResult)
    assert (res.call_id, res.recording_uri, res.queue) == (cid, key, "billing")
    assert res.segments
    assert all(g.is_final for g in res.segments)
    assert res.signals
    assert await bus.redis.xlen(EVENTS_FUSED) > 0
    assert await bus.redis.xlen(CALL_SCORES) == 1
    assert list((tmp_path / "tmp").glob("*.wav")) == []  # temp copy removed
    assert store.exists(f"reports/{cid}/report.json")
    with pytest.raises(FileNotFoundError):
        worker.fetch("recordings/missing.wav")
