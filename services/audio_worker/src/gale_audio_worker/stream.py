"""Streaming audio-worker (P4): ``audio.frames.*`` → ``signals`` (shouting, SER, audio-tox, KWS,
overtalk).

Per (call, channel) a 20 s ring buffer. Every ``shout_hop_s`` the newest 1.5 s window is scored
against the speaker's baseline (``DSPShoutDetector.score_window``: the first 20 s of speech build
it, as in docs/03 §3); speech ratio comes from VAD over the whole ring, so the VAD has context.
Every ``window_hop_s`` SER / audio-tox / KWS run on the newest 3 s. Every ``overtalk_hop_s`` the
two channels' rings are checked for interruptions; an onset is published once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gale_audio_worker.dsp import WIN_S, Window, f0_track, speech_ratio, window_features
from gale_audio_worker.overtalk import overtalk_signals
from gale_audio_worker.shout import DSPShoutDetector
from galecore.audio import MODEL_SR, s16le_to_float
from galecore.bus import SIGNALS, BusClient, audio_streams
from galecore.config import Settings
from galecore.models import ModelSet, RealModelsUnavailableError, load_models
from galecore.models.base import AudioChunk
from galecore.models.oracle import oracle_for_ref
from galecore.models.stubs import StubVAD
from galecore.schemas import AudioFrame, CallLifecycle, Channel, Signal
from galecore.streaming import Worker, span

RING_S = 20.0


@dataclass
class Ring:
    start: float = 0.0  # media time of samples[0]
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    last_shout: float = 0.0
    last_window: float = 0.0

    @property
    def end(self) -> float:
        return self.start + len(self.samples) / MODEL_SR

    def push(self, f: AudioFrame) -> None:
        x = s16le_to_float(f.pcm_s16le)
        if not len(self.samples):
            self.start = f.ts_start
        self.samples = np.concatenate([self.samples, x])
        extra = len(self.samples) - int(RING_S * MODEL_SR)
        if extra > 0:
            self.samples = self.samples[extra:]
            self.start += extra / MODEL_SR

    def tail(self, seconds: float) -> tuple[float, np.ndarray]:
        n = min(len(self.samples), int(seconds * MODEL_SR))
        return self.end - n / MODEL_SR, self.samples[len(self.samples) - n :]


class AudioStream(Worker):
    service = "audio_worker"

    def __init__(
        self,
        bus: BusClient,
        settings: Settings,
        *,
        shout_hop_s: float = 0.5,
        window_hop_s: float = 1.5,
        overtalk_hop_s: float = 2.0,
        **kw: object,
    ) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.settings = settings
        self.shout_hop_s, self.window_hop_s, self.overtalk_hop_s = (
            shout_hop_s,
            window_hop_s,
            overtalk_hop_s,
        )
        self.vad = StubVAD()
        self.shout = DSPShoutDetector(self.vad, settings.model_dir)
        self.rings: dict[tuple[str, Channel], Ring] = {}
        self.models: dict[str, ModelSet] = {}
        self.last_overtalk: dict[str, float] = {}
        self.onsets: dict[tuple[str, Channel], list[float]] = {}
        for stream in audio_streams():
            self.subscribe(stream, AudioFrame, self.on_frame)

    def models_for(self, call_id: str) -> ModelSet:
        if call_id not in self.models:
            if self.settings.model_mode != "stub":  # pragma: no cover
                raise RealModelsUnavailableError(
                    "real streaming SER/KWS: configure GALE_MODEL_DIR (P3)"
                )
            self.models[call_id] = load_models(
                self.settings, oracle_for_ref(self.clock.refs.get(call_id))
            )
        return self.models[call_id]

    async def _publish(self, sigs: list[Signal], stage: str) -> None:
        for s in sigs:
            await self.bus.publish(SIGNALS, s, idem_key=s.signal_id)
            self.clock.observe_latency(self.service, stage, s.call_id, s.ts_end)

    async def on_frame(self, f: AudioFrame) -> None:
        ring = self.rings.setdefault((f.call_id, f.channel), Ring())
        ring.push(f)
        if ring.end - ring.last_shout >= self.shout_hop_s and ring.end - ring.start >= WIN_S:
            ring.last_shout = ring.end
            await self._shout(f.call_id, f.channel, ring)
        if ring.end - ring.last_window >= self.window_hop_s and ring.end - ring.start >= 3.0:
            ring.last_window = ring.end
            t0, x = ring.tail(3.0)
            chunk = AudioChunk(f.call_id, f.channel, x, offset_s=t0)
            m = self.models_for(f.call_id)
            with span("audio.windows", f.call_id):
                sigs = m.ser.detect(chunk) + m.audio_tox.detect(chunk) + m.kws.detect(chunk)
            await self._publish(sigs, "acoustic_signal")
        if (
            f.channel == Channel.CUSTOMER
            and ring.end - self.last_overtalk.get(f.call_id, 0.0) >= self.overtalk_hop_s
        ):
            self.last_overtalk[f.call_id] = ring.end
            await self._overtalk(f.call_id)

    async def _shout(self, call_id: str, ch: Channel, ring: Ring) -> None:
        regions = self.vad.speech_regions(
            AudioChunk(call_id, ch, ring.samples, offset_s=ring.start)
        )
        t0, x = ring.tail(WIN_S)
        w = Window(
            round(t0, 3),
            round(ring.end, 3),
            round(speech_ratio(t0, ring.end, regions), 3),
            window_features(x, MODEL_SR, f0_track(x, MODEL_SR)),
        )
        sig = self.shout.score_window(call_id, ch, w)
        if sig is not None:
            await self._publish([sig], "dsp_shout")

    async def _overtalk(self, call_id: str) -> None:
        a, c = self.rings.get((call_id, Channel.AGENT)), self.rings.get((call_id, Channel.CUSTOMER))
        if a is None or c is None:
            return
        start, end = max(a.start, c.start), min(a.end, c.end)
        if end - start < 3.0:
            return

        def cut(r: Ring) -> np.ndarray:
            i0 = int((start - r.start) * MODEL_SR)
            return r.samples[i0 : i0 + int((end - start) * MODEL_SR)]

        audio = {Channel.AGENT: cut(a), Channel.CUSTOMER: cut(c)}
        regions = {
            ch: [
                (s + start, e + start)
                for s, e in self.vad.speech_regions(AudioChunk(call_id, ch, x))
            ]
            for ch, x in audio.items()
        }
        # overtalk_signals expects offsets from the start of ``audio``: shift in, then back out
        shifted = {ch: [(s - start, e - start) for s, e in r] for ch, r in regions.items()}
        fresh: list[Signal] = []
        for sig in overtalk_signals(call_id, shifted, audio, MODEL_SR):
            onset = sig.ts_start + start
            seen = self.onsets.setdefault((call_id, sig.channel), [])
            if any(abs(onset - o) < 1.0 for o in seen) or onset > end - 1.0:
                continue  # already published, or too fresh to know the overlap yet
            seen.append(onset)
            fresh.append(
                sig.model_copy(
                    update={
                        "ts_start": round(onset, 3),
                        "ts_end": round(sig.ts_end + start, 3),
                        "signal_id": f"{call_id}:overtalk:{sig.channel.value}:{onset:.1f}",
                    }
                )
            )
        await self._publish(fresh, "overtalk")

    async def on_lifecycle(self, ev: CallLifecycle) -> None:
        if ev.event in ("ended", "hangup"):
            for key in [k for k in self.rings if k[0] == ev.call_id]:
                del self.rings[key]
            self.models.pop(ev.call_id, None)
            self.last_overtalk.pop(ev.call_id, None)
