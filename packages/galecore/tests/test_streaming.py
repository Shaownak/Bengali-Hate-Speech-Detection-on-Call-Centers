import asyncio
import time
import uuid

from fakeredis import FakeAsyncRedis

from galecore.bus import CALL_LIFECYCLE, SIGNALS, BusClient
from galecore.schemas import CallLifecycle, Category, Channel, Signal, SignalSource
from galecore.streaming import CallClock, Worker, span, trace_id_for


def sig(call_id: str, t: float) -> Signal:
    return Signal(
        call_id=call_id,
        channel=Channel.AGENT,
        signal_id=f"{call_id}:{t}",
        source=SignalSource.LEXICON,
        category=Category.INSULT,
        score=0.9,
        ts_start=t,
        ts_end=t + 0.5,
        model_version="t",
    )


def test_call_clock_and_trace_id() -> None:
    clock = CallClock()
    now = time.time()
    clock.observe(
        CallLifecycle(call_id="c", event="started", ts=0.0, wall_clock=now - 10, external_ref="x")
    )
    lag = clock.lag("c", 8.0, now=now)
    assert lag is not None
    assert abs(lag - 2.0) < 1e-6
    assert clock.refs["c"] == "x"
    assert clock.lag("unknown", 1.0) is None
    clock.observe(CallLifecycle(call_id="c", event="ended", ts=30.0))
    assert clock.ended["c"] == 30.0
    u = str(uuid.uuid4())
    assert trace_id_for(u) == uuid.UUID(u).int
    assert trace_id_for("not-a-uuid") == trace_id_for("not-a-uuid")
    with span("x", "c"):
        pass


class Collector(Worker):
    service = "collector"

    def __init__(self, bus: BusClient, fail_once: bool = False) -> None:
        super().__init__(bus, min_idle_ms=0)
        self.seen: list[str] = []
        self.ticks = 0
        self.fail_once = fail_once
        self.subscribe(SIGNALS, Signal, self.on_signal)

    async def on_signal(self, s: Signal) -> None:
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("boom")
        self.seen.append(s.signal_id)

    async def tick(self) -> None:
        self.ticks += 1


async def test_worker_runs_retries_and_restores_unknown_calls() -> None:
    redis = FakeAsyncRedis()
    bus = BusClient(redis)
    await bus.publish(
        CALL_LIFECYCLE,
        CallLifecycle(
            call_id="c1",
            event="started",
            ts=0.0,
            wall_clock=time.time(),
            external_ref="fixture:demo_01",
        ),
    )
    for t in (1.0, 2.0, 3.0):
        await bus.publish(SIGNALS, sig("c1", t))
    # a worker that never saw call.lifecycle (its group starts at the end of the stream)
    w = Collector(BusClient(redis), fail_once=True)
    await w.setup()
    await redis.xgroup_setid(CALL_LIFECYCLE, "collector", "$")
    stop = asyncio.Event()
    task = asyncio.create_task(w.run(stop, idle_sleep=0.01))
    for _ in range(100):
        if len(w.seen) == 3:
            break
        await asyncio.sleep(0.02)
    stop.set()
    await task
    assert sorted(w.seen) == ["c1:1.0", "c1:2.0", "c1:3.0"]  # the failed one was retried
    assert w.clock.refs["c1"] == "fixture:demo_01"  # lifecycle restored by replay
    assert w.ticks > 0
    old = await w.replay(SIGNALS, "c1", Signal, before_ts=2.5)
    assert [s.ts_start for s in old] == [1.0, 2.0]
    assert await w.replay(SIGNALS, "other", Signal) == []
