"""Bus publishing for the ingest-gateway (P4): ``audio.frames.{shard}`` + ``call.lifecycle``, and
call audio retention in the object store.

The RTP / WebSocket receivers are synchronous callbacks, so frames go through a bounded queue to
one publisher task. If the bus falls behind and the queue fills, frames are dropped and counted
(``gale_ingest_frames_dropped_total``) rather than blocking the receive path: real-time audio is
not worth stalling every other call for.

Retention: each call's 16 kHz channels are kept in memory and written to the object store at
hang-up (``audio/<call_id>/<channel>.wav``). Prod uses MinIO with SSE-KMS (docs/05); the local
dev store is a plain folder (ADR-009).
"""

from __future__ import annotations

import asyncio
import io
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
from prometheus_client import Counter

from galecore.audio import MODEL_SR
from galecore.bus import CALL_LIFECYCLE, BusClient, audio_stream
from galecore.logging import get_logger
from galecore.objectstore import ObjectStore
from galecore.schemas import AudioFrame, CallLifecycle, Channel, GaleModel
from galecore.streaming import STAGE_LATENCY

log = get_logger(__name__)
DROPPED = Counter(
    "gale_ingest_frames_dropped_total", "Frames dropped because the bus queue was full"
)


class BusPublisher:
    def __init__(
        self, bus: BusClient, store: ObjectStore | None = None, queue_size: int = 20_000
    ) -> None:
        self.bus, self.store = bus, store
        self.queue: asyncio.Queue[tuple[str, GaleModel, str | None]] = asyncio.Queue(queue_size)
        self.wall_start: dict[str, float] = {}
        self._audio: dict[str, dict[Channel, list[bytes]]] = defaultdict(lambda: defaultdict(list))
        self.published = 0

    # --- callbacks from the registry (sync) ------------------------------------------------------

    def frame(self, frame: AudioFrame) -> None:
        self._audio[frame.call_id][frame.channel].append(frame.pcm_s16le)
        self._put(
            audio_stream(frame.call_id), frame, f"{frame.call_id}:{frame.channel.value}:{frame.seq}"
        )

    def started(
        self,
        call_id: str,
        external_ref: str | None,
        layout: str | None,
        *,
        agent_id: str | None = None,
        queue: str | None = None,
    ) -> None:
        self.wall_start[call_id] = time.time()
        self._put(
            CALL_LIFECYCLE,
            CallLifecycle(
                call_id=call_id,
                event="started",
                ts=0.0,
                mode="realtime",
                external_ref=external_ref,
                channel_layout=layout,
                agent_id=agent_id,
                queue=queue,
                wall_clock=self.wall_start[call_id],
            ),
            f"{call_id}:started",
        )

    def ended(self, call_id: str, duration_s: float) -> None:
        now = time.time()
        self._put(
            CALL_LIFECYCLE,
            CallLifecycle(
                call_id=call_id,
                event="ended",
                ts=round(duration_s, 3),
                mode="realtime",
                wall_clock=now,
            ),
            f"{call_id}:ended",
        )
        self._retain(call_id)

    def _put(self, stream: str, msg: GaleModel, idem: str | None) -> None:
        try:
            self.queue.put_nowait((stream, msg, idem))
        except asyncio.QueueFull:
            DROPPED.inc()

    def _retain(self, call_id: str) -> None:
        chans = self._audio.pop(call_id, {})
        if self.store is None:
            return
        for ch, parts in chans.items():
            pcm = np.frombuffer(b"".join(parts), dtype="<i2")
            buf = io.BytesIO()
            sf.write(buf, pcm, MODEL_SR, format="WAV", subtype="PCM_16")
            self.store.put(f"audio/{call_id}/{ch.value}.wav", buf.getvalue())

    # --- publisher task --------------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        while not (stop.is_set() and self.queue.empty()):
            try:
                stream, msg, idem = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            await self.bus.publish(stream, msg, idem_key=idem)
            self.published += 1
            if isinstance(msg, AudioFrame) and msg.call_id in self.wall_start:
                lag = time.time() - (self.wall_start[msg.call_id] + msg.ts_end)
                STAGE_LATENCY.labels("ingest_gateway", "frame_on_bus").observe(max(0.0, lag))

    async def drain(self) -> None:
        while not self.queue.empty():
            stream, msg, idem = self.queue.get_nowait()
            await self.bus.publish(stream, msg, idem_key=idem)
            self.published += 1
