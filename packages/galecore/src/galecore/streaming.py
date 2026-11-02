"""Shared machinery for the streaming workers (docs/01 §3, §5; Prompt 11).

- ``Worker``: consumer-group polling over several streams (at-least-once, retries, DLQ and
  idempotency come from ``BusClient``), cooperative stop, and per-call **replay**: a worker that
  meets a call mid-way (restart after a crash) rebuilds its state from the call's earlier messages.
- ``CallClock``: wall-clock start of every call from ``call.lifecycle``, so each stage can record
  its real latency: ``now − (call start + media time of the evidence)``.
- ``STAGE_LATENCY``: Prometheus histogram per service and stage (docs/01 §5 budget).
- ``span``: OpenTelemetry span with ``gale.call_id`` (a no-op until an SDK is configured).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from opentelemetry import trace
from prometheus_client import Histogram

from galecore.bus import CALL_LIFECYCLE, BusClient, decode
from galecore.logging import get_logger
from galecore.schemas import CallLifecycle, GaleModel

log = get_logger(__name__)
STAGE_LATENCY = Histogram(
    "gale_stage_latency_seconds",
    "Wall-clock latency from the evidence's media time to this stage",
    ["service", "stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0),
)
_tracer = trace.get_tracer("gale")


@contextmanager
def span(name: str, call_id: str, **attrs: Any) -> Iterator[None]:
    with _tracer.start_as_current_span(name, attributes={"gale.call_id": call_id, **attrs}):
        yield


def trace_id_for(call_id: str) -> int:
    """128-bit OTel trace id derived from the call UUID (trace id = call_id, Prompt 11)."""
    try:
        return uuid.UUID(call_id).int
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_URL, call_id).int


class CallClock:
    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.refs: dict[str, str | None] = {}
        self.ended: dict[str, float] = {}

    def observe(self, ev: CallLifecycle) -> None:
        if ev.event == "started":
            self.started[ev.call_id] = (ev.wall_clock or time.time()) - ev.ts
            self.refs[ev.call_id] = ev.external_ref
        elif ev.event in ("ended", "hangup"):
            self.ended[ev.call_id] = ev.ts

    def lag(self, call_id: str, media_ts: float, now: float | None = None) -> float | None:
        start = self.started.get(call_id)
        return None if start is None else (now or time.time()) - (start + media_ts)

    def observe_latency(
        self, service: str, stage: str, call_id: str, media_ts: float
    ) -> float | None:
        lag = self.lag(call_id, media_ts)
        if lag is not None:
            STAGE_LATENCY.labels(service, stage).observe(max(0.0, lag))
        return lag


Handler = Callable[[Any], Awaitable[None]]


@dataclass
class Subscription:
    stream: str
    model: type[GaleModel]
    handler: Handler


class Worker:
    """Base for bus consumers. Subclasses add subscriptions and per-call state."""

    service = "worker"

    def __init__(
        self,
        bus: BusClient,
        *,
        consumer: str | None = None,
        min_idle_ms: int = 30_000,
        batch: int = 50,
    ) -> None:
        self.bus = bus
        self.group = self.service
        self.consumer = consumer or f"{self.service}-{uuid.uuid4().hex[:8]}"
        self.min_idle_ms, self.batch = min_idle_ms, batch
        self.clock = CallClock()
        self._looked_up: set[str] = set()
        self.subs: list[Subscription] = [
            Subscription(CALL_LIFECYCLE, CallLifecycle, self._on_lifecycle)
        ]

    def subscribe(self, stream: str, model: type[GaleModel], handler: Handler) -> None:
        async def wrapped(msg: Any) -> None:
            call_id = getattr(msg, "call_id", None)
            if (
                call_id is not None
                and call_id not in self.clock.started
                and call_id not in self._looked_up
            ):
                await self.ensure_call(call_id)
            await handler(msg)

        self.subs.append(Subscription(stream, model, wrapped))

    async def ensure_call(self, call_id: str) -> None:
        """A call this worker hasn't seen start (crash restart, new replica): restore its lifecycle
        (external ref, wall-clock start) from the bus before handling its messages."""
        self._looked_up.add(call_id)  # once per call, even if it never had a lifecycle
        for ev in await self.replay(CALL_LIFECYCLE, call_id, CallLifecycle):
            self.clock.observe(ev)

    async def _on_lifecycle(self, ev: CallLifecycle) -> None:
        self.clock.observe(ev)
        await self.on_lifecycle(ev)

    async def on_lifecycle(self, ev: CallLifecycle) -> None:  # override
        return None

    async def tick(self) -> None:  # override: timers (debounce, flushes)
        return None

    async def setup(self) -> None:
        for s in self.subs:
            await self.bus.ensure_group(s.stream, self.group)

    async def poll(self) -> int:
        n = 0
        for s in self.subs:
            n += await self.bus.poll_once(
                s.stream,
                self.group,
                self.consumer,
                s.model,
                s.handler,
                count=self.batch,
                block_ms=None,
                min_idle_ms=self.min_idle_ms,
            )
        await self.tick()
        return n

    async def run(self, stop: asyncio.Event, idle_sleep: float = 0.02) -> None:
        await self.setup()
        while not stop.is_set():
            try:
                n = await self.poll()
            except Exception as e:  # keep the worker alive; BusClient already retried the message
                log.warning("worker_poll_failed", service=self.service, error=type(e).__name__)
                n = 0
            if not n:
                await asyncio.sleep(idle_sleep)

    async def replay[M: GaleModel](
        self, stream: str, call_id: str, model: type[M], before_ts: float | None = None
    ) -> list[M]:
        """A call's earlier messages on ``stream`` (rebuild state after a restart)."""
        out: list[M] = []
        start = "-"
        while True:
            rows = cast(
                "list[tuple[bytes, dict[bytes, bytes]]]",
                await self.bus.redis.xrange(stream, min=start, max="+", count=1000),
            )
            if not rows:
                break
            for _msg_id, fields in rows:
                if not fields or b"data" not in fields:
                    continue
                msg = decode(fields, model)
                if getattr(msg, "call_id", None) != call_id:
                    continue
                ts = getattr(msg, "ts_start", None)
                if before_ts is None or ts is None or ts < before_ts:
                    out.append(msg)
            last = rows[-1][0].decode()
            ms, seq = last.split("-")
            start = f"{ms}-{int(seq) + 1}"
            if len(rows) < 1000:
                break
        return out
