"""Message bus over Redis Streams (ADR-002).

- JSON body validated by the shared pydantic models; binary fields (audio PCM) travel as raw
  stream fields, not base64 (D-02).
- Consumer groups, at-least-once delivery, explicit ack.
- Failed messages stay pending and are re-claimed after ``min_idle_ms``; after
  ``max_retries`` deliveries they go to ``<stream>.dlq`` and are acked.
- Idempotency: messages carry an ``idem`` key (``signal_id`` / ``event_id`` …); a group skips
  keys it already processed successfully.

The ``memory`` backend is an in-process fakeredis instance with the same semantics, used by
unit tests and the offline report CLI (ADR-009).
"""

from __future__ import annotations

import asyncio
import json
import zlib
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from prometheus_client import Counter
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from galecore.config import Settings
from galecore.logging import get_logger
from galecore.schemas import GaleModel

M = TypeVar("M", bound=GaleModel)
Handler = Callable[[M], Awaitable[None]]

# Stream names (docs/01 §3)
AUDIO_FRAMES = "audio.frames.{shard}"
ASR_SEGMENTS = "asr.segments"
SIGNALS = "signals"
EVENTS_FUSED = "events.fused"
ALERTS = "alerts"
CALL_LIFECYCLE = "call.lifecycle"
FEEDBACK = "feedback"
LLM_DECISIONS = "llm.decisions"
CALL_SCORES = "call.scores"
POSTCALL_JOBS = "postcall.jobs"
POSTCALL_RESULTS = "postcall.results"
DLQ_SUFFIX = ".dlq"

DEFAULT_MAXLEN = 100_000
AUDIO_SHARDS = 8

_log = get_logger(__name__)

BUS_PUBLISHED = Counter("gale_bus_published_total", "Messages published", ["stream"])
BUS_CONSUMED = Counter("gale_bus_consumed_total", "Messages handled OK", ["stream", "group"])
BUS_FAILED = Counter("gale_bus_failed_total", "Handler failures", ["stream", "group"])
BUS_DLQ = Counter("gale_bus_dead_lettered_total", "Messages dead-lettered", ["stream", "group"])
BUS_DUPLICATE = Counter("gale_bus_duplicates_total", "Duplicates skipped", ["stream", "group"])


def audio_streams(shards: int = AUDIO_SHARDS) -> list[str]:
    return [AUDIO_FRAMES.format(shard=i) for i in range(shards)]


def audio_stream(call_id: str, shards: int = AUDIO_SHARDS) -> str:
    """Stable shard for a call's audio frames."""
    return AUDIO_FRAMES.format(shard=zlib.crc32(call_id.encode()) % shards)


def encode(msg: GaleModel, idem_key: str | None = None) -> dict[str, bytes]:
    binary = type(msg).bus_binary_fields
    body = msg.model_dump_json(exclude=set(binary))
    fields: dict[str, bytes] = {
        "type": type(msg).__name__.encode(),
        "data": body.encode(),
    }
    for name in binary:
        fields[f"bin:{name}"] = bytes(getattr(msg, name))
    if idem_key:
        fields["idem"] = idem_key.encode()
    return fields


def decode(fields: dict[bytes, bytes], model: type[M]) -> M:
    data: dict[str, Any] = json.loads(fields[b"data"])
    for name in model.bus_binary_fields:
        data[name] = fields[f"bin:{name}".encode()]
    return model.model_validate(data)


class BusClient:
    def __init__(self, redis: Redis, *, maxlen: int | None = DEFAULT_MAXLEN) -> None:
        self.redis = redis
        self.maxlen = maxlen

    @classmethod
    def from_settings(cls, settings: Settings) -> BusClient:
        if settings.bus_backend == "memory":
            try:
                from fakeredis import FakeAsyncRedis
            except ImportError as e:  # pragma: no cover - depends on install
                raise RuntimeError("bus_backend=memory needs the 'fakeredis' package") from e
            return cls(FakeAsyncRedis())
        return cls(Redis.from_url(settings.bus_url))

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:
            return False

    async def close(self) -> None:
        await self.redis.aclose()

    # -- publish ------------------------------------------------------------

    async def publish(self, stream: str, msg: GaleModel, *, idem_key: str | None = None) -> str:
        msg_id = await self.redis.xadd(
            stream,
            encode(msg, idem_key),  # type: ignore[arg-type]
            maxlen=self.maxlen,
            approximate=True,
        )
        BUS_PUBLISHED.labels(stream).inc()
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    # -- consume ------------------------------------------------------------

    async def ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _handle(
        self,
        stream: str,
        group: str,
        msg_id: bytes,
        fields: dict[bytes, bytes],
        model: type[M],
        handler: Handler[M],
        max_retries: int,
    ) -> None:
        idem = fields.get(b"idem")
        idem_key = f"idem:{group}:{idem.decode()}" if idem else None
        if idem_key and await self.redis.exists(idem_key):
            BUS_DUPLICATE.labels(stream, group).inc()
            await self.redis.xack(stream, group, msg_id)
            return
        try:
            await handler(decode(fields, model))
        except Exception as e:
            BUS_FAILED.labels(stream, group).inc()
            deliveries = await self._delivery_count(stream, group, msg_id)
            _log.warning(
                "bus_handler_failed",
                stream=stream,
                group=group,
                msg_id=msg_id.decode(),
                error=type(e).__name__,
                deliveries=deliveries,
            )
            if deliveries >= max_retries:
                dead = dict(fields)
                dead[b"error"] = type(e).__name__.encode()
                dead[b"origin_id"] = msg_id
                await self.redis.xadd(stream + DLQ_SUFFIX, dead)  # type: ignore[arg-type]
                await self.redis.xack(stream, group, msg_id)
                BUS_DLQ.labels(stream, group).inc()
            return
        if idem_key:
            await self.redis.set(idem_key, b"1", ex=24 * 3600)
        await self.redis.xack(stream, group, msg_id)
        BUS_CONSUMED.labels(stream, group).inc()

    async def _delivery_count(self, stream: str, group: str, msg_id: bytes) -> int:
        pending = await self.redis.xpending_range(stream, group, min=msg_id, max=msg_id, count=1)
        return int(pending[0]["times_delivered"]) if pending else 1

    async def poll_once(
        self,
        stream: str,
        group: str,
        consumer: str,
        model: type[M],
        handler: Handler[M],
        *,
        count: int = 10,
        block_ms: int | None = 1000,
        max_retries: int = 5,
        min_idle_ms: int = 30_000,
    ) -> int:
        """Re-claim stale pending messages, then read new ones. Returns messages handled."""
        handled = 0
        _next, claimed, *_ = await self.redis.xautoclaim(
            stream, group, consumer, min_idle_time=min_idle_ms, start_id="0-0", count=count
        )
        for msg_id, fields in claimed:
            if fields:  # entries trimmed by maxlen come back empty
                await self._handle(stream, group, msg_id, fields, model, handler, max_retries)
                handled += 1
        resp = cast(
            "list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] | None",
            await self.redis.xreadgroup(
                group, consumer, {stream: ">"}, count=count, block=block_ms
            ),
        )
        for _stream, entries in resp or []:
            for msg_id, fields in entries:
                await self._handle(stream, group, msg_id, fields, model, handler, max_retries)
                handled += 1
        return handled

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        model: type[M],
        handler: Handler[M],
        *,
        stop: asyncio.Event,
        **kwargs: Any,
    ) -> None:
        """Run ``poll_once`` until ``stop`` is set."""
        await self.ensure_group(stream, group)
        while not stop.is_set():
            await self.poll_once(stream, group, consumer, model, handler, **kwargs)
