"""In-process streaming stack for integration tests: one shared fake Redis, the gateway registry +
publisher, and every worker as an asyncio task (stub models keyed to fixture calls)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fakeredis import FakeAsyncRedis, FakeServer

from gale_asr_worker.stream import AsrStream
from gale_audio_worker.stream import AudioStream
from gale_fusion_engine.stream import FusionStream
from gale_ingest_gateway.gateway import GatewaySettings, Registry, WsStart
from gale_ingest_gateway.publisher import BusPublisher
from gale_llm_judge.stream import LlmStream
from gale_text_worker.stream import TextStream
from galecore.bus import ALERTS, CALL_SCORES, EVENTS_FUSED, BusClient
from galecore.config import Settings
from galecore.schemas import CallScoreMessage, FusedEvent
from galecore.streaming import Worker

SETTINGS = Settings(env="test", bus_backend="memory", log_level="WARNING")


@dataclass
class Stack:
    server: FakeServer = field(default_factory=FakeServer)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    workers: dict[str, Worker] = field(default_factory=dict)
    tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    min_idle_ms: int = 30_000

    def bus(self) -> BusClient:
        return BusClient(FakeAsyncRedis(server=self.server))

    def make(self, name: str, shard: tuple[int, int] | None = None) -> Worker:
        kinds: dict[str, Any] = {
            "asr": AsrStream,
            "audio": AudioStream,
            "text": TextStream,
            "llm": LlmStream,
            "fusion": FusionStream,
        }
        kw: dict[str, Any] = {"min_idle_ms": self.min_idle_ms, "shard": shard}
        if name == "llm":
            kw["debounce_s"] = 3.0
        return kinds[name](self.bus(), SETTINGS, **kw)  # type: ignore[no-any-return]

    async def start(self, name: str, shard: tuple[int, int] | None = None) -> Worker:
        w = self.make(name, shard)
        stop = asyncio.Event()
        key = f"{name}#{shard[0]}" if shard else name
        self.workers[key] = w
        self.tasks[key] = asyncio.create_task(w.run(stop))
        self.tasks[key].stop_event = stop  # type: ignore[attr-defined]
        await asyncio.sleep(0)
        return w

    async def kill(self, name: str) -> None:
        """Crash a worker: cancel mid-flight, no clean shutdown (unacked messages stay pending)."""
        t = self.tasks.pop(name)
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t

    async def shutdown(self) -> None:
        for t in self.tasks.values():
            t.stop_event.set()  # type: ignore[attr-defined]
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    async def read(self, stream: str, model: Any) -> list[Any]:
        from galecore.bus import decode

        rows = await self.bus().redis.xrange(stream, "-", "+")
        return [decode(f, model) for _, f in rows if f and b"data" in f]

    async def events(self) -> dict[str, FusedEvent]:
        """Latest state of every published event."""
        out: dict[str, FusedEvent] = {}
        for e in await self.read(EVENTS_FUSED, FusedEvent):
            out[e.event_id] = e
        return out

    async def alerts(self) -> list[FusedEvent]:
        return await self.read(ALERTS, FusedEvent)  # type: ignore[no-any-return]

    async def scores(self) -> list[CallScoreMessage]:
        return await self.read(CALL_SCORES, CallScoreMessage)  # type: ignore[no-any-return]


async def play_call(
    stack: Stack,
    wav: Path,
    call_id: str,
    external_ref: str,
    speed: float = 1.0,
    on_progress: Any = None,
) -> BusPublisher:
    """Feed a stereo fixture WAV through the gateway (WS path) into the bus, paced at ``speed``×
    real time (0 = as fast as possible)."""
    pub = BusPublisher(stack.bus())
    registry = Registry(GatewaySettings(rtp_port=0), pub.frame, pub)
    pub_stop = asyncio.Event()
    pub_task = asyncio.create_task(pub.run(pub_stop))
    x, sr = sf.read(wav, dtype="int16", always_2d=True)
    start = WsStart(event="start", sample_rate=sr, channels=2, external_ref=external_ref)
    session = registry.open_ws(call_id, start)
    step = sr // 50  # 20 ms
    t0 = time.monotonic()
    n = len(x) // step
    for k in range(n):
        registry.on_ws_bytes(
            session, start, x[k * step : (k + 1) * step].astype("<i2").tobytes(), b""
        )
        if on_progress is not None:
            await on_progress(k / n)
        if speed > 0:
            delay = t0 + (k + 1) * 0.02 / speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        elif k % 50 == 0:
            await asyncio.sleep(0)
    registry.close(call_id)
    pub_stop.set()
    await pub_task
    return pub


async def wait_quiet(stack: Stack, seconds: float, settle: float = 4.0) -> None:
    """Wait until no stream grows for ``settle`` seconds (or ``seconds`` pass)."""
    deadline = time.monotonic() + seconds
    last, quiet_since = -1, time.monotonic()
    redis = stack.bus().redis
    while time.monotonic() < deadline:
        total = 0
        for key in await redis.keys("*"):
            if (await redis.type(key)) == b"stream":
                total += await redis.xlen(key)
        if total != last:
            last, quiet_since = total, time.monotonic()
        elif time.monotonic() - quiet_since >= settle:
            return
        await asyncio.sleep(0.25)


def gold(fixture_dir: Path) -> list[dict[str, Any]]:
    import json

    return json.loads((fixture_dir / "expected_events.json").read_text(encoding="utf-8"))["events"]


def matches(e: FusedEvent, g: dict[str, Any]) -> bool:
    if (e.speaker.value, e.category.value) != (g["speaker"], g["category"]):
        return False
    inter = max(0.0, min(e.ts_end, g["t_end"]) - max(e.ts_start, g["t_start"]))
    union = max(e.ts_end, g["t_end"]) - min(e.ts_start, g["t_start"])
    centre = (e.ts_start + e.ts_end) / 2
    return inter / union >= 0.3 or g["t_start"] - 2 <= centre <= g["t_end"] + 2


def p95(xs: list[float]) -> float:
    return float(np.percentile(xs, 95)) if xs else 0.0
