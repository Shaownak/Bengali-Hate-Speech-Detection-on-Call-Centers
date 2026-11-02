"""Rules simulator: replay stored calls through the active and a candidate rule set (docs/04 §2.8).

Both runs use the same stored signals and post-call thresholds. The diff therefore isolates the
rules change. LLM window decisions aren't stored per event, so a candidate provisional event takes
the decision of the stored event it overlaps (same speaker and category): dismissed stays
dismissed, anything else counts as confirmed. Events with no stored counterpart stay provisional,
as if the judge were offline. Grounded LLM signals (``llm_window``) are replayed as they are.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, Field

from gale_fusion_engine.fusion import FusionEngine
from gale_fusion_engine.rules import FusionRules
from galecore.config import ThresholdsConfig
from galecore.schemas import EventStatus, FusedEvent, ProvisionalDecision, Signal, SignalSource


class AlertKey(BaseModel):
    speaker: str
    category: str
    ts_start: float
    severity: int


class CallDiff(BaseModel):
    call_id: str
    baseline_events: int
    candidate_events: int
    baseline_alerts: int
    candidate_alerts: int
    added_alerts: list[AlertKey] = Field(default_factory=list)
    removed_alerts: list[AlertKey] = Field(default_factory=list)


class SimulationReport(BaseModel):
    calls: int
    baseline_alerts: int
    candidate_alerts: int
    baseline_by_category: dict[str, int]
    candidate_by_category: dict[str, int]
    changed_calls: list[CallDiff]


def _decisions(prov: list[FusedEvent], stored: Iterable[FusedEvent]) -> list[ProvisionalDecision]:
    stored = [e for e in stored if e.status != EventStatus.PROVISIONAL]
    out = []
    for e in prov:
        match = next(
            (
                s
                for s in stored
                if (s.speaker, s.category) == (e.speaker, e.category)
                and s.ts_start <= e.ts_end
                and e.ts_start <= s.ts_end
            ),
            None,
        )
        if match is not None:
            dismissed = match.status == EventStatus.DISMISSED_BY_LLM
            out.append(
                ProvisionalDecision(
                    event_id=e.event_id, decision="dismiss" if dismissed else "confirm"
                )
            )
    return out


def replay(
    engine: FusionEngine, call_id: str, signals: list[Signal], stored: list[FusedEvent]
) -> list[FusedEvent]:
    llm = [s for s in signals if s.source == SignalSource.LLM_WINDOW]
    base = [s for s in signals if s.source != SignalSource.LLM_WINDOW]
    prov = engine.provisional(call_id, base)
    judged = engine.apply_judgements(prov, _decisions(prov, stored), signals)
    return engine.finalize(judged + engine.llm_events(call_id, llm, judged))


def _keys(events: list[FusedEvent]) -> set[tuple[str, str, float, int]]:
    return {
        (e.speaker.value, e.category.value, round(e.ts_start, 1), e.severity)
        for e in events
        if e.alert
    }


def simulate(
    calls: Mapping[str, tuple[list[Signal], list[FusedEvent]]],
    baseline: FusionRules,
    candidate: FusionRules,
    thresholds: ThresholdsConfig,
) -> SimulationReport:
    eb = FusionEngine(baseline, thresholds, mode="postcall")
    ec = FusionEngine(candidate, thresholds, mode="postcall")
    by_b: Counter[str] = Counter()
    by_c: Counter[str] = Counter()
    diffs: list[CallDiff] = []
    tot_b = tot_c = 0
    for call_id, (signals, stored) in calls.items():
        b = replay(eb, call_id, signals, stored)
        c = replay(ec, call_id, signals, stored)
        by_b.update(e.category.value for e in b if e.alert)
        by_c.update(e.category.value for e in c if e.alert)
        kb, kc = _keys(b), _keys(c)
        tot_b += len(kb)
        tot_c += len(kc)
        if kb != kc or len(b) != len(c):
            diffs.append(
                CallDiff(
                    call_id=call_id,
                    baseline_events=len(b),
                    candidate_events=len(c),
                    baseline_alerts=len(kb),
                    candidate_alerts=len(kc),
                    added_alerts=[_ak(k) for k in sorted(kc - kb)],
                    removed_alerts=[_ak(k) for k in sorted(kb - kc)],
                )
            )
    return SimulationReport(
        calls=len(calls),
        baseline_alerts=tot_b,
        candidate_alerts=tot_c,
        baseline_by_category=dict(by_b),
        candidate_by_category=dict(by_c),
        changed_calls=diffs,
    )


def _ak(k: tuple[str, str, float, int]) -> AlertKey:
    return AlertKey(speaker=k[0], category=k[1], ts_start=k[2], severity=k[3])
