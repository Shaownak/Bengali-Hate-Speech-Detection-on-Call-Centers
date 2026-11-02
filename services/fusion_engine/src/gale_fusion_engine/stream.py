"""Streaming fusion-engine (P4): ``signals`` + ``llm.decisions`` → ``events.fused`` and ``alerts``.

Per call, all signals and LLM decisions so far; whenever something new arrives the call is marked
dirty and, after a short coalescing delay, re-fused with the same ``FusionEngine`` as post-call
(``mode="realtime"`` alert thresholds). Only events whose state changed are published, so a
provisional event, its confirmation and its alert are separate messages. Latency per stage
(``provisional_event``, ``confirmed_event``, ``alert``) is the wall-clock time since the media time
of the event's end (docs/01 §5: ≤ 2 s / ≤ 6 s).

A fusion worker that meets a call mid-way (restart) replays the call's signals and decisions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from gale_fusion_engine.fusion import FusionEngine
from gale_fusion_engine.rules import parse_rules
from galecore import metrics as m
from galecore.bus import ALERTS, EVENTS_FUSED, LLM_DECISIONS, SIGNALS, BusClient
from galecore.config import Settings, load_rules, load_thresholds
from galecore.schemas import (
    CallLifecycle,
    EventStatus,
    FusedEvent,
    LlmDecision,
    ProvisionalDecision,
    Signal,
    SignalSource,
)
from galecore.streaming import Worker, span

REPLAY_AFTER_S = 5.0
EVICT_AFTER_S = 120.0  # keep a finished call this long for late LLM decisions


@dataclass
class CallState:
    signals: dict[str, Signal] = field(default_factory=dict)
    decisions: dict[str, ProvisionalDecision] = field(default_factory=dict)
    published: dict[str, tuple[object, ...]] = field(default_factory=dict)
    alerted: set[str] = field(default_factory=set)
    confirmed: set[str] = field(default_factory=set)
    dismissed: set[str] = field(default_factory=set)
    dirty_since: float | None = None
    ended_at: float | None = None


class FusionStream(Worker):
    service = "fusion_engine"

    def __init__(
        self, bus: BusClient, settings: Settings, *, coalesce_s: float = 0.1, **kw: object
    ) -> None:
        super().__init__(bus, **kw)  # type: ignore[arg-type]
        self.engine = FusionEngine(
            parse_rules(load_rules(settings.config_dir)),
            load_thresholds(settings.config_dir),
            mode="realtime",
        )
        self.coalesce_s = coalesce_s
        self.calls: dict[str, CallState] = {}
        self.latencies: list[
            tuple[str, str, str, float]
        ] = []  # (call, event, stage, seconds) for tests/ops
        self.subscribe(SIGNALS, Signal, self.on_signal)
        self.subscribe(LLM_DECISIONS, LlmDecision, self.on_decision)

    async def _state(self, call_id: str, media_ts: float) -> CallState:
        st = self.calls.get(call_id)
        if st is None:
            st = self.calls[call_id] = CallState()
            if media_ts > REPLAY_AFTER_S:  # joined mid-call (restart): rebuild from the bus
                for s in await self.replay(SIGNALS, call_id, Signal):
                    st.signals[s.signal_id] = s
                for d in await self.replay(LLM_DECISIONS, call_id, LlmDecision):
                    st.decisions[d.decision.event_id] = d.decision
        return st

    async def on_signal(self, sig: Signal) -> None:
        m.SIGNALS.labels(sig.source.value, sig.category.value).inc()
        m.SIGNAL_SCORE.labels(sig.source.value).observe(sig.score)
        st = await self._state(sig.call_id, sig.ts_end)
        st.signals[sig.signal_id] = sig
        st.dirty_since = st.dirty_since or time.monotonic()

    async def on_decision(self, d: LlmDecision) -> None:
        st = await self._state(d.call_id, 0.0)
        st.decisions[d.decision.event_id] = d.decision
        st.dirty_since = st.dirty_since or time.monotonic()

    async def tick(self) -> None:
        now = time.monotonic()
        for call_id, st in list(self.calls.items()):
            if st.dirty_since is not None and now - st.dirty_since >= self.coalesce_s:
                st.dirty_since = None
                await self.fuse(call_id, st)
            if st.ended_at is not None and now - st.ended_at > EVICT_AFTER_S:
                del self.calls[call_id]

    def events(self, call_id: str, st: CallState) -> list[FusedEvent]:
        sigs = list(st.signals.values())
        base = [s for s in sigs if s.source != SignalSource.LLM_WINDOW]
        judged = self.engine.apply_judgements(
            self.engine.provisional(call_id, base), st.decisions.values(), base
        )
        llm = self.engine.llm_events(
            call_id, [s for s in sigs if s.source == SignalSource.LLM_WINDOW], judged
        )
        return self.engine.finalize(judged + llm)

    async def fuse(self, call_id: str, st: CallState) -> None:
        with span("fusion.fuse", call_id):
            events = self.events(call_id, st)
        for e in events:
            key = (
                e.status,
                e.severity,
                e.category,
                e.alert,
                round(e.ts_end, 1),
                round(e.probability, 2),
            )
            if st.published.get(e.event_id) == key:
                continue
            first = e.event_id not in st.published
            st.published[e.event_id] = key
            state = hashlib.sha1(repr(key).encode(), usedforsecurity=False).hexdigest()[:12]
            await self.bus.publish(EVENTS_FUSED, e, idem_key=f"{e.event_id}:{state}")
            labels = (e.category.value, e.speaker.value)
            if first:
                m.EVENTS.labels(*labels, "provisional").inc()
                # events only the LLM window finds have no fast path: keep them out of the
                # provisional (≤ 2 s) budget and measure them on their own
                llm_only = e.status == EventStatus.CONFIRMED
                self._lat(call_id, e, "llm_event" if llm_only else "provisional_event")
            if e.status == EventStatus.CONFIRMED and e.event_id not in st.confirmed:
                st.confirmed.add(e.event_id)
                m.EVENTS.labels(*labels, "confirmed").inc()
                self._lat(call_id, e, "confirmed_event")
            if e.status == EventStatus.DISMISSED_BY_LLM and e.event_id not in st.dismissed:
                st.dismissed.add(e.event_id)
                m.EVENTS.labels(*labels, "dismissed").inc()
            if e.alert and e.event_id not in st.alerted:
                st.alerted.add(e.event_id)
                m.ALERTS.labels(*labels).inc()
                await self.bus.publish(ALERTS, e, idem_key=e.event_id)
                self._lat(call_id, e, "alert")

    def _lat(self, call_id: str, e: FusedEvent, stage: str) -> None:
        lag = self.clock.observe_latency(self.service, stage, call_id, e.ts_end)
        if lag is not None:
            self.latencies.append((call_id, e.event_id, stage, lag))

    async def on_lifecycle(self, ev: CallLifecycle) -> None:
        if ev.event in ("ended", "hangup") and ev.call_id in self.calls:
            st = self.calls[ev.call_id]
            st.ended_at = time.monotonic()
            await self.fuse(ev.call_id, st)  # final pass; state kept a while for late LLM decisions
