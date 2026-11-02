"""Streaming asr-worker (P4): ``audio.frames.*`` → ``asr.segments`` (partial + final).

Endpointing on the 200 ms frames: a frame is speech when its level is 12 dB over the channel's
running noise floor (and above −45 dBFS, which keeps far-end crosstalk out). An utterance starts
with one frame of pre-roll, emits a partial every ``partial_every_s`` while it grows, and is
finalised after ``silence_s`` of silence, at ``max_utt_s``, or at hang-up. The final segment is
normalised and redacted in-process (D-01) by the same step as post-call.

This is the endpoint-then-decode policy that works for any ``ASRBackend``; Whisper SimulStreaming
and NeMo cache-aware streaming can replace the partials when those backends are real (P3/P4).
Note: the utterance buffer lives in memory; a crash loses the utterance in progress (the gateway
keeps the full call audio, so post-call mode can recover it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gale_asr_worker.steps import to_asr_segment
from galecore.audio import MODEL_SR, rms_dbfs, s16le_to_float
from galecore.bus import ASR_SEGMENTS, BusClient
from galecore.config import Settings
from galecore.metrics import ASR_CONFIDENCE
from galecore.models import RealModelsUnavailableError, load_models
from galecore.models.base import ASRBackend, AudioChunk
from galecore.models.oracle import oracle_for_ref
from galecore.schemas import AudioFrame, CallLifecycle, Channel
from galecore.streaming import Worker, span

FLOOR_MARGIN_DB = 12.0
ABS_MIN_DB = -45.0


@dataclass
class Utterance:
    start: float
    chunks: list[np.ndarray] = field(default_factory=list)
    last_speech: float = 0.0
    last_partial: float = 0.0
    partials: int = 0


@dataclass
class ChannelState:
    floor: float = -60.0
    preroll: tuple[float, np.ndarray] | None = None
    utt: Utterance | None = None


class AsrStream(Worker):
    service = "asr_worker"

    def __init__(
        self,
        bus: BusClient,
        settings: Settings,
        *,
        silence_s: float = 0.6,
        partial_every_s: float = 1.0,
        max_utt_s: float = 25.0,
        **kw: object,
    ) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.settings = settings
        self.silence_s, self.partial_every_s, self.max_utt_s = silence_s, partial_every_s, max_utt_s
        self.state: dict[tuple[str, Channel], ChannelState] = {}
        self._asr: dict[str, ASRBackend] = {}
        for stream in self.owned_audio_streams():
            self.subscribe(stream, AudioFrame, self.on_frame)

    def asr_for(self, call_id: str) -> ASRBackend:
        if call_id not in self._asr:
            if self.settings.model_mode != "stub":  # pragma: no cover - needs weights
                raise RealModelsUnavailableError(
                    "real streaming ASR: configure GALE_MODEL_DIR (P3)"
                )
            oracle = oracle_for_ref(self.clock.refs.get(call_id))
            self._asr[call_id] = load_models(self.settings, oracle).asr
        return self._asr[call_id]

    async def on_frame(self, f: AudioFrame) -> None:
        st = self.state.setdefault((f.call_id, f.channel), ChannelState())
        x = s16le_to_float(f.pcm_s16le)
        db = rms_dbfs(x)
        speech = db > max(st.floor + FLOOR_MARGIN_DB, ABS_MIN_DB)
        if not speech:  # the floor follows quiet frames quickly and loud ones never
            st.floor = db if db < st.floor else 0.9 * st.floor + 0.1 * db
        if speech and st.utt is None:
            st.utt = Utterance(
                start=st.preroll[0] if st.preroll else f.ts_start, last_partial=f.ts_start
            )
            if st.preroll:
                st.utt.chunks.append(st.preroll[1])
        st.preroll = (f.ts_start, x)
        if st.utt is None:
            return
        st.utt.chunks.append(x)
        if speech:
            st.utt.last_speech = f.ts_end
        if (
            f.ts_end - st.utt.last_speech >= self.silence_s
            or f.ts_end - st.utt.start >= self.max_utt_s
        ):
            await self._finish(f.call_id, f.channel, st, final=True)
        elif f.ts_end - st.utt.last_partial >= self.partial_every_s:
            await self._finish(f.call_id, f.channel, st, final=False)
            st.utt.last_partial = f.ts_end

    async def _finish(self, call_id: str, ch: Channel, st: ChannelState, final: bool) -> None:
        utt = st.utt
        if utt is None:
            return
        audio = np.concatenate(utt.chunks) if utt.chunks else np.zeros(0, np.float32)
        if final:
            st.utt = None
        if len(audio) < MODEL_SR // 10:
            return
        with span("asr.decode", call_id, final=final):
            raws = self.asr_for(call_id).transcribe(
                AudioChunk(call_id, ch, audio, offset_s=utt.start)
            )
        for raw in raws:
            seg = to_asr_segment(raw, call_id)
            if not final:
                seg = seg.model_copy(update={"is_final": False})
                utt.partials += 1
            await self.bus.publish(
                ASR_SEGMENTS,
                seg,
                idem_key=f"{seg.segment_id}:{'f' if final else f'p{utt.partials}'}",
            )
            if final:
                ASR_CONFIDENCE.observe(seg.avg_conf)
                self.clock.observe_latency(self.service, "asr_final", call_id, seg.ts_end)

    async def on_lifecycle(self, ev: CallLifecycle) -> None:
        if ev.event in ("ended", "hangup"):
            for (cid, ch), st in list(self.state.items()):
                if cid == ev.call_id:
                    await self._finish(cid, ch, st, final=True)
                    del self.state[(cid, ch)]
            self._asr.pop(ev.call_id, None)
