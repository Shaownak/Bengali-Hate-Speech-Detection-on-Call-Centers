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
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from gale_postcall.models import build_models
from gale_postcall.pipeline import PostCallResult, run_postcall
from gale_postcall.report import to_html, to_json_text
from galecore import metrics as m
from galecore.bus import (
    CALL_SCORES,
    EVENTS_FUSED,
    POSTCALL_JOBS,
    POSTCALL_RESULTS,
    BusClient,
    audio_streams,
)
from galecore.config import Settings
from galecore.logging import get_logger
from galecore.models.oracle import oracle_for_ref
from galecore.objectstore import ObjectStore, store_from_settings
from galecore.schemas import CallScoreMessage, PostcallJob, PostcallResult
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
        self.store = store or store_from_settings(settings)
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

    def fetch(self, uri: str) -> tuple[Path, bool]:
        """A local path as is (drop folder); an object-store key (API upload) via a temp copy."""
        path = Path(uri)
        if path.is_file():
            return path, False
        if self.store is None or not self.store.exists(uri):
            raise FileNotFoundError(f"recording not found: {uri}")
        tmp_dir = self.settings.data_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir, delete=False) as f:
            f.write(self.store.get(uri))
        return Path(f.name), True

    async def on_job(self, job: PostcallJob) -> None:
        oracle = oracle_for_ref(job.external_ref) if self.settings.model_mode == "stub" else None
        wav, tmp = await asyncio.to_thread(self.fetch, job.uri)
        try:
            with span("postcall.job", job.call_id):
                res: PostCallResult = await asyncio.to_thread(
                    run_postcall,
                    wav,
                    self.settings,
                    call_id=job.call_id,
                    layout=job.channel_layout,
                    models=build_models(self.settings, oracle),
                    queue=job.queue,
                )
        finally:
            if tmp:
                wav.unlink(missing_ok=True)
        await self.bus.publish(
            POSTCALL_RESULTS,
            PostcallResult(
                call_id=job.call_id,
                duration_s=res.duration_s,
                channel_layout=job.channel_layout,
                external_ref=job.external_ref,
                queue=job.queue,
                recording_uri=job.uri,
                segments=[g for g in res.segments if g.is_final],
                signals=res.signals,
            ),
            idem_key=f"{job.call_id}:result",
        )
        for e in res.events:
            await self.bus.publish(EVENTS_FUSED, e, idem_key=f"{e.event_id}:postcall")
            stage = "dismissed" if e.status.value == "dismissed_by_llm" else "confirmed"
            m.EVENTS.labels(e.category.value, e.speaker.value, stage).inc()
            if e.alert:
                m.ALERTS.labels(e.category.value, e.speaker.value).inc()
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
