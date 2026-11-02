"""Streaming llm-judge (P4): provisional ``events.fused`` → ``llm.decisions`` + grounded LLM
``signals``; at hang-up the whole-call ``call.scores``.

Debounce: at most one window request per call every ``debounce_s`` (docs/01 §2: 1 req / 3 s / call);
all provisional events that arrived meanwhile are judged together. If the judge is offline the
events simply stay provisional (fusion keeps them) and are retried on the next round.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from gale_llm_judge.client import JudgeUnavailableError
from gale_llm_judge.grounding import ground
from gale_llm_judge.windows import utterance_ids, windows_for_events
from galecore.bus import ASR_SEGMENTS, CALL_SCORES, EVENTS_FUSED, LLM_DECISIONS, SIGNALS, BusClient
from galecore.config import Settings
from galecore.models import RealModelsUnavailableError, load_models
from galecore.models.base import LLMJudge
from galecore.models.oracle import oracle_for_ref
from galecore.schemas import (
    AsrSegment,
    CallLifecycle,
    CallScoreMessage,
    EventStatus,
    FusedEvent,
    LlmDecision,
    Signal,
    SignalSource,
)
from galecore.streaming import Worker, span


@dataclass
class CallState:
    segments: dict[str, AsrSegment] = field(default_factory=dict)
    signals: dict[str, Signal] = field(default_factory=dict)
    events: dict[str, FusedEvent] = field(default_factory=dict)
    pending: dict[str, FusedEvent] = field(default_factory=dict)
    judged: set[str] = field(default_factory=set)
    last_request: float = 0.0
    ended_at: float | None = None


class LlmStream(Worker):
    service = "llm_judge"

    def __init__(
        self,
        bus: BusClient,
        settings: Settings,
        *,
        debounce_s: float = 3.0,
        end_grace_s: float = 1.0,
        judge: LLMJudge | None = None,
        **kw: object,
    ) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.settings, self.debounce_s, self.end_grace_s = settings, debounce_s, end_grace_s
        self._judge = judge
        self.calls: dict[str, CallState] = {}
        self.subscribe(EVENTS_FUSED, FusedEvent, self.on_event)
        self.subscribe(ASR_SEGMENTS, AsrSegment, self.on_segment)
        self.subscribe(SIGNALS, Signal, self.on_signal)

    def judge_for(self, call_id: str) -> LLMJudge:
        if self._judge is not None:
            return self._judge
        if self.settings.model_mode != "stub":  # pragma: no cover
            raise RealModelsUnavailableError(
                "real judge: point GALE_LLM_BASE_URL at the vLLM server"
            )
        return load_models(self.settings, oracle_for_ref(self.clock.refs.get(call_id))).llm

    async def on_segment(self, seg: AsrSegment) -> None:
        if seg.is_final:
            self.calls.setdefault(seg.call_id, CallState()).segments[seg.segment_id] = seg

    async def on_signal(self, sig: Signal) -> None:
        if sig.source in (
            SignalSource.DSP_SHOUT,
            SignalSource.SER,
        ):  # acoustic support for the judge
            self.calls.setdefault(sig.call_id, CallState()).signals[sig.signal_id] = sig

    async def on_event(self, ev: FusedEvent) -> None:
        st = self.calls.setdefault(ev.call_id, CallState())
        st.events[ev.event_id] = ev
        if ev.status == EventStatus.PROVISIONAL and ev.event_id not in st.judged:
            st.pending[ev.event_id] = ev

    async def tick(self) -> None:
        now = time.monotonic()
        for call_id, st in list(self.calls.items()):
            if st.pending and now - st.last_request >= self.debounce_s:
                st.last_request = now
                await self._judge_windows(call_id, st)
            if st.ended_at is not None and not st.pending and now - st.ended_at >= self.end_grace_s:
                await self._scorecard(call_id, st)
                del self.calls[call_id]

    async def _judge_windows(self, call_id: str, st: CallState) -> None:
        events = list(st.pending.values())
        segs = sorted(st.segments.values(), key=lambda s: s.ts_start)
        judge = self.judge_for(call_id)
        for w in windows_for_events(call_id, segs, list(st.signals.values()), events):
            try:
                with span("llm.window", call_id):
                    j = judge.judge_window(w.input)
            except JudgeUnavailableError:
                return  # stay pending; retried after the next debounce
            version = getattr(judge, "version", "stub")
            for d in j.provisional_decisions:
                await self.bus.publish(
                    LLM_DECISIONS,
                    LlmDecision(call_id=call_id, decision=d, model_version=version),
                    idem_key=f"{d.event_id}:{d.decision}",
                )
                ev = st.pending.pop(d.event_id, None)
                st.judged.add(d.event_id)
                if ev is not None:
                    self.clock.observe_latency(self.service, "llm_decision", call_id, ev.ts_end)
            for sig in ground(w.input, j, utterance_ids(w.input.segments), version).signals:
                await self.bus.publish(SIGNALS, sig, idem_key=sig.signal_id)
        for eid in [
            e.event_id for e in events
        ]:  # not in any window (no transcript yet): next round
            if eid in st.judged:
                st.pending.pop(eid, None)

    async def _scorecard(self, call_id: str, st: CallState) -> None:
        judge = self.judge_for(call_id)
        segs = sorted(st.segments.values(), key=lambda s: s.ts_start)
        try:
            with span("llm.call", call_id):
                card = judge.judge_call(call_id, segs, list(st.events.values()))
        except JudgeUnavailableError:
            return
        await self.bus.publish(
            CALL_SCORES,
            CallScoreMessage(
                call_id=call_id, scorecard=card, model_version=getattr(judge, "version", "stub")
            ),
            idem_key=f"{call_id}:scorecard",
        )

    async def on_lifecycle(self, ev: CallLifecycle) -> None:
        if ev.event in ("ended", "hangup"):
            self.calls.setdefault(ev.call_id, CallState()).ended_at = time.monotonic()
