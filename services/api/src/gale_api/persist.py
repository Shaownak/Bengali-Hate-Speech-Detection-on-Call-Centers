"""Bus → Postgres persister (consumer group ``api_persister``).

Stores calls, final redacted segments, signals, events (latest state), scorecards and post-call
results, so the API and dashboard read from Postgres. Writes are idempotent upserts. Each message
is written in its own transaction in a worker thread, and acknowledged only after the commit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from sqlalchemy.orm import Session, sessionmaker

from gale_api import repo
from galecore.bus import (
    ASR_SEGMENTS,
    CALL_SCORES,
    EVENTS_FUSED,
    POSTCALL_RESULTS,
    SIGNALS,
    BusClient,
)
from galecore.schemas import (
    AsrSegment,
    CallLifecycle,
    CallScoreMessage,
    FusedEvent,
    PostcallResult,
    Signal,
)
from galecore.streaming import STAGE_LATENCY, Worker


class Persister(Worker):
    service = "api_persister"

    def __init__(self, bus: BusClient, sessions: sessionmaker[Session], **kw: object) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.sessions = sessions
        self.parts = repo.Partitions()
        self.subscribe(POSTCALL_RESULTS, PostcallResult, self.on_postcall)
        self.subscribe(ASR_SEGMENTS, AsrSegment, self.on_segment)
        self.subscribe(SIGNALS, Signal, self.on_signal)
        self.subscribe(EVENTS_FUSED, FusedEvent, self.on_event)
        self.subscribe(CALL_SCORES, CallScoreMessage, self.on_score)

    async def _write(self, fn: Callable[[Session], None]) -> None:
        def run() -> None:
            with self.sessions() as s, s.begin():
                fn(s)

        await asyncio.to_thread(run)

    async def on_lifecycle(self, ev: CallLifecycle) -> None:
        await self._write(lambda s: repo.apply_lifecycle(s, ev))

    async def on_segment(self, seg: AsrSegment) -> None:
        if seg.is_final:
            await self._write(lambda s: repo.upsert_segment(s, self.parts, seg))

    async def on_signal(self, sig: Signal) -> None:
        await self._write(lambda s: repo.upsert_signal(s, self.parts, sig))

    async def on_event(self, ev: FusedEvent) -> None:
        await self._write(lambda s: repo.upsert_event(s, ev))
        lag = self.clock.lag(ev.call_id, ev.ts_end)
        if lag is not None:
            STAGE_LATENCY.labels("api", "event_persisted").observe(max(lag, 0.0))

    async def on_score(self, msg: CallScoreMessage) -> None:
        await self._write(
            lambda s: repo.upsert_scorecard(s, msg.call_id, msg.scorecard, msg.model_version)
        )

    async def on_postcall(self, res: PostcallResult) -> None:
        await self._write(lambda s: repo.apply_postcall(s, self.parts, res))
