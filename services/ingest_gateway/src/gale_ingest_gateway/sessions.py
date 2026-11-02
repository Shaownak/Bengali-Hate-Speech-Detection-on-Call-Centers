"""Call sessions in the ingest-gateway: decode → 16 kHz → 200 ms ``AudioFrame``s (docs/01 §2).

P0 stub: frames go to a ``FrameSink`` that only counts them. P4 swaps in a sink that publishes
to ``audio.frames`` on the bus. Nothing here logs or stores audio (CLAUDE.md rule 1).
"""

from __future__ import annotations

import time
import zlib
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import soxr
from prometheus_client import Counter
from pydantic import BaseModel

from galecore.audio import MODEL_SR, float_to_s16le, int16_to_float
from galecore.rtp import SeqTracker
from galecore.schemas import AudioFrame, Channel

FRAMES = Counter("gale_ingest_frames_total", "200 ms frames emitted", ["transport", "channel"])
PACKETS = Counter("gale_ingest_rtp_packets_total", "RTP packets accepted", ["channel"])
LOST = Counter("gale_ingest_rtp_lost_total", "RTP packets lost (sequence gaps)", ["channel"])
REJECTED = Counter("gale_ingest_rtp_rejected_total", "RTP packets dropped", ["reason"])

Transport = Literal["rtp", "ws"]
FRAME_SAMPLES = MODEL_SR * 200 // 1000  # 3200


class FrameSink(Protocol):
    def __call__(self, frame: AudioFrame) -> None: ...


class ChannelStats(BaseModel):
    packets: int = 0
    lost: int = 0
    out_of_order: int = 0
    duplicates: int = 0
    samples_in: int = 0  # at the input sample rate
    frames_out: int = 0  # 200 ms frames at 16 kHz
    duration_s: float = 0.0
    pcm_crc32: int = 0  # CRC-32 of the decoded input PCM (s16le), for round-trip checks


class SessionStats(BaseModel):
    call_id: str
    transport: Transport
    sample_rate_in: int
    closed: bool
    channels: dict[Channel, ChannelStats]


class ChannelAssembler:
    """Per-channel streaming resampler and 200 ms framer."""

    def __init__(
        self, call_id: str, channel: Channel, sr_in: int, sink: FrameSink, transport: Transport
    ):
        self.call_id, self.channel, self.sr_in, self.sink, self.transport = (
            call_id,
            channel,
            sr_in,
            sink,
            transport,
        )
        self.stats = ChannelStats()
        self.seq = SeqTracker()
        self._stream = (
            soxr.ResampleStream(sr_in, MODEL_SR, 1, dtype="float32", quality="HQ")
            if sr_in != MODEL_SR
            else None
        )
        self._buf = np.empty(0, dtype=np.float32)
        self._crc = 0

    def push(self, pcm: npt.NDArray[np.int16], *, last: bool = False) -> None:
        self._crc = zlib.crc32(pcm.astype("<i2").tobytes(), self._crc)
        self.stats.samples_in += len(pcm)
        x = int16_to_float(pcm)
        if self._stream is not None:
            x = np.asarray(self._stream.resample_chunk(x, last=last), dtype=np.float32)
        self._buf = np.concatenate([self._buf, x])
        while len(self._buf) >= FRAME_SAMPLES:
            self._emit(self._buf[:FRAME_SAMPLES])
            self._buf = self._buf[FRAME_SAMPLES:]
        if last and len(self._buf):
            self._emit(self._buf)
            self._buf = self._buf[:0]

    def _emit(self, samples: npt.NDArray[np.float32]) -> None:
        start = self.stats.frames_out * FRAME_SAMPLES / MODEL_SR
        self.sink(
            AudioFrame(
                call_id=self.call_id,
                channel=self.channel,
                seq=self.stats.frames_out,
                ts_start=start,
                ts_end=start + len(samples) / MODEL_SR,
                pcm_s16le=float_to_s16le(samples),
            )
        )
        self.stats.frames_out += 1
        FRAMES.labels(self.transport, self.channel.value).inc()

    def finish(self) -> ChannelStats:
        if self._stream is not None:
            self.push(np.empty(0, dtype=np.int16), last=True)
        elif len(self._buf):
            self._emit(self._buf)
            self._buf = self._buf[:0]
        s = self.stats
        s.lost, s.out_of_order, s.duplicates = (
            self.seq.lost,
            self.seq.out_of_order,
            self.seq.duplicates,
        )
        s.duration_s = round(s.samples_in / self.sr_in, 3)
        s.pcm_crc32 = self._crc
        return s


class Session:
    def __init__(
        self,
        call_id: str,
        transport: Transport,
        sr_in: int,
        channels: list[Channel],
        sink: FrameSink,
    ):
        self.call_id, self.transport, self.sr_in = call_id, transport, sr_in
        self.created = time.monotonic()
        self.closed = False
        self.channels = {c: ChannelAssembler(call_id, c, sr_in, sink, transport) for c in channels}

    def close(self) -> SessionStats:
        if not self.closed:
            for a in self.channels.values():
                a.finish()
            self.closed = True
        return self.stats()

    def stats(self) -> SessionStats:
        return SessionStats(
            call_id=self.call_id,
            transport=self.transport,
            sample_rate_in=self.sr_in,
            closed=self.closed,
            channels={c: a.stats.model_copy() for c, a in self.channels.items()},
        )


class CountingSink:
    """P0 stub sink: counts frames per (call, channel) and keeps nothing else."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, Channel], int] = {}

    def __call__(self, frame: AudioFrame) -> None:
        key = (frame.call_id, frame.channel)
        self.counts[key] = self.counts.get(key, 0) + 1
