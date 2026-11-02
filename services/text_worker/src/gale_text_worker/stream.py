"""Streaming text-worker (P4): ``asr.segments`` → ``signals`` (lexicon, text classifier, register).

Partial segments run the **lexicon only** (fast path, docs/01 §1): a gali is flagged while the
utterance is still going. Lexicon signal ids depend on (segment, char offset), and a partial and
its final share the segment id, so the final re-derives the same ids and nothing is published
twice. Context-dependent steps (classifier, register state) wait for the final segment.

Per call: one ``TextWorker`` (context history + register state). A worker that first meets a call
mid-way (restarted after a crash) replays the call's earlier final segments through it, without
publishing, so the register baseline and context are back before the new segment is scored.
"""

from __future__ import annotations

from gale_text_worker.lexicon import LexiconMatcher
from gale_text_worker.register import RuleRegisterDetector
from gale_text_worker.steps import TextWorker, lexicon_signals
from galecore.bangla import normalize
from galecore.bus import ASR_SEGMENTS, SIGNALS, BusClient
from galecore.config import Settings, load_lexicon, load_thresholds
from galecore.models import RealModelsUnavailableError, load_models
from galecore.models.oracle import oracle_for_ref
from galecore.schemas import AsrSegment, CallLifecycle
from galecore.streaming import Worker, span

REPLAY_AFTER_S = 5.0


class TextStream(Worker):
    service = "text_worker"

    def __init__(self, bus: BusClient, settings: Settings, **kw: object) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.settings = settings
        self.matcher = LexiconMatcher(load_lexicon(settings).entries)
        self.thresholds = load_thresholds(settings.config_dir)
        self.calls: dict[str, TextWorker] = {}
        self.replayed = 0
        self.subscribe(ASR_SEGMENTS, AsrSegment, self.on_segment)

    def _new_worker(self, call_id: str) -> TextWorker:
        if self.settings.model_mode != "stub":  # pragma: no cover
            raise RealModelsUnavailableError("real text classifier: configure GALE_MODEL_DIR (P3)")
        clf = load_models(self.settings, oracle_for_ref(self.clock.refs.get(call_id))).text_clf
        return TextWorker(self.matcher, clf, RuleRegisterDetector(), self.thresholds)

    async def on_segment(self, seg: AsrSegment) -> None:
        if not seg.is_final:
            await self._partial(seg)
            return
        worker = self.calls.get(seg.call_id)
        if worker is None:
            worker = self.calls[seg.call_id] = self._new_worker(seg.call_id)
            if seg.ts_start > REPLAY_AFTER_S:  # joined mid-call: rebuild state silently
                earlier = await self.replay(
                    ASR_SEGMENTS, seg.call_id, AsrSegment, before_ts=seg.ts_start
                )
                for old in sorted((s for s in earlier if s.is_final), key=lambda s: s.ts_start):
                    worker.process(old)
                self.replayed += 1
        with span("text.process", seg.call_id):
            sigs = worker.process(seg)
        for s in sigs:
            await self.bus.publish(SIGNALS, s, idem_key=s.signal_id)
            self.clock.observe_latency(self.service, "text_signal", s.call_id, s.ts_end)

    async def _partial(self, seg: AsrSegment) -> None:
        for s in lexicon_signals(self.matcher, seg, normalize(seg.text_redacted)):  # stateless
            await self.bus.publish(SIGNALS, s, idem_key=s.signal_id)
            self.clock.observe_latency(self.service, "lexicon_partial", s.call_id, s.ts_end)

    async def on_lifecycle(self, ev: CallLifecycle) -> None:
        if ev.event in ("ended", "hangup"):
            self.calls.pop(ev.call_id, None)
