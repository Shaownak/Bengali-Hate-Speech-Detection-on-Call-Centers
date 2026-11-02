"""Post-call job worker (P4): ``postcall.jobs`` → the post-call pipeline → ``events.fused``,
``call.scores`` and a report in the object store.

Priority: real-time work pre-empts batch. Before taking a job the worker checks the backlog of the
real-time consumers (frames delivered-but-unacked plus not yet delivered, for the ASR and audio
groups); above ``backlog_limit`` it leaves the job queued and checks again later. Jobs run in a
thread so a long call never stalls the event loop.

    python -m gale_postcall worker
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from gale_postcall.models import build_models
from gale_postcall.pipeline import PostCallResult, run_postcall
from gale_postcall.report import to_html, to_json_text
from galecore.bus import CALL_SCORES, EVENTS_FUSED, POSTCALL_JOBS, BusClient, audio_streams
from galecore.config import Settings
from galecore.logging import get_logger
from galecore.models.oracle import oracle_for_ref
from galecore.objectstore import LocalObjectStore, ObjectStore
from galecore.schemas import CallScoreMessage, PostcallJob
from galecore.streaming import Worker, span

log = get_logger(__name__)
REALTIME_GROUPS = ("asr_worker", "audio_worker")


class PostcallWorker(Worker):
    service = "postcall_worker"

    def __init__(
        self,
        bus: BusClient,
        settings: Settings,
        *,
        backlog_limit: int = 200,
        store: ObjectStore | None = None,
        **kw: object,
    ) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.settings, self.backlog_limit = settings, backlog_limit
        self.store = store or (
            LocalObjectStore(settings.object_store_dir)
            if settings.object_store == "local"
            else None
        )
        self.yielded = 0
        self.done: list[str] = []
        self.batch = 1  # one call at a time
        self.subscribe(POSTCALL_JOBS, PostcallJob, self.on_job)

    async def realtime_backlog(self) -> int:
        total = 0
        for stream in audio_streams():
            if not await self.bus.redis.exists(stream):
                continue
            groups: list[dict[str, Any]] = await self.bus.redis.xinfo_groups(stream)
            for g in groups:
                name = g["name"].decode() if isinstance(g["name"], bytes) else g["name"]
                if name in REALTIME_GROUPS:
                    total += int(g.get("pending") or 0) + int(g.get("lag") or 0)
        return total

    async def poll(self) -> int:
        if await self.realtime_backlog() > self.backlog_limit:
            self.yielded += 1
            return 0  # real time first: leave batch jobs queued
        return await super().poll()

    async def on_job(self, job: PostcallJob) -> None:
        oracle = oracle_for_ref(job.external_ref) if self.settings.model_mode == "stub" else None
        with span("postcall.job", job.call_id):
            res: PostCallResult = await asyncio.to_thread(
                run_postcall,
                Path(job.uri),
                self.settings,
                call_id=job.call_id,
                layout=job.channel_layout,
                models=build_models(self.settings, oracle),
                queue=job.queue,
            )
        for e in res.events:
            await self.bus.publish(EVENTS_FUSED, e, idem_key=f"{e.event_id}:postcall")
        if res.scorecard is not None:
            await self.bus.publish(
                CALL_SCORES,
                CallScoreMessage(
                    call_id=job.call_id,
                    scorecard=res.scorecard,
                    model_version=res.models.get("mode", "?"),
                ),
                idem_key=f"{job.call_id}:scorecard",
            )
        if self.store is not None:
            self.store.put(f"reports/{job.call_id}/report.json", to_json_text(res).encode())
            self.store.put(f"reports/{job.call_id}/report.html", to_html(res).encode())
        self.done.append(job.call_id)
        log.info("postcall_job_done", call_id=job.call_id, events=len(res.events))


def build_app(settings: Settings | None = None) -> FastAPI:
    from contextlib import asynccontextmanager

    from galecore.config import get_settings
    from galecore.service import create_app

    s = settings or get_settings()
    bus = BusClient.from_settings(s)
    stop = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        task = asyncio.create_task(PostcallWorker(bus, s).run(stop)) if s.workers else None
        yield
        stop.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    return create_app(
        "postcall_worker", ready_checks={"bus": bus.ping}, settings=s, lifespan=lifespan
    )
