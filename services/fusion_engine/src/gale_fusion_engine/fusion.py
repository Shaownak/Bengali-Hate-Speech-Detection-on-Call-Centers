"""Fusion engine v0 (docs/03 §8): calibrated signals → rules → events → hysteresis → alerts.

v0 calibration is the identity (isotonic per source/category arrives with dev data, D-13).
The stacker (§8.3) is P3. Steps, in order:

1. ``candidates``: each rule in ``rules.yaml`` turns signals into candidate events.
2. ``provisional``: candidates of one speaker + category closer than ``merge_gap_s`` merge;
   shouted abuse gets ``shout_plus_abuse``. Status ``provisional``.
3. ``apply_judgements``: LLM window decisions (confirm / dismiss / change …); a dismissal is
   overruled by KWS ≥ ``unless`` (``llm_dismiss``).
4. ``llm_events``: grounded LLM-only findings become events of their own.
5. ``finalize``: escalate severity on repeats, drop below ``log``, alert policy + cool-down.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from gale_fusion_engine.rules import DurationRule, FusionRules
from galecore.config import ThresholdsConfig
from galecore.models.stubs import stable_id
from galecore.schemas import (
    Category,
    Channel,
    EventStatus,
    FusedEvent,
    ProvisionalDecision,
    Signal,
    SignalSource,
    Speaker,
)

Mode = Literal["postcall", "realtime"]
Calibrator = Callable[[Signal], float]
LEVELS = {0: "apni", 1: "tumi", 2: "tui"}
# signals whose time span is word- or window-precise: they decide an event's span when present
PRECISE = {
    SignalSource.LEXICON,
    SignalSource.KWS,
    SignalSource.REGISTER,
    SignalSource.DSP_SHOUT,
    SignalSource.SER,
    SignalSource.VAD_OVERTALK,
}


def identity(signal: Signal) -> float:
    return signal.score


def speaker_of(ch: Channel) -> Speaker | None:
    return {Channel.AGENT: Speaker.AGENT, Channel.CUSTOMER: Speaker.CUSTOMER}.get(ch)  # D-15


def _speaker(ch: Channel) -> Speaker:
    spk = speaker_of(ch)
    if spk is None:
        raise ValueError("mixed-channel signals never reach fusion (D-15)")
    return spk


@dataclass
class Candidate:
    speaker: Speaker
    category: Category
    t0: float
    t1: float
    severity: int
    prob: float
    signal_ids: list[str]
    precise: bool
    rule: str
    members: list[Candidate] = field(default_factory=list)


def _sev(signal: Signal, default: int = 1) -> int:
    return int(min(3, max(1, signal.evidence.features.get("severity", default))))


class FusionEngine:
    def __init__(
        self,
        rules: FusionRules,
        thresholds: ThresholdsConfig,
        mode: Mode = "postcall",
        calibrate: Calibrator = identity,
    ) -> None:
        self.rules, self.thr, self.mode, self.calibrate = rules, thresholds, mode, calibrate

    # --- 1. rules → candidates ---------------------------------------------------------------

    def candidates(self, signals: Iterable[Signal]) -> list[Candidate]:
        sigs = [s for s in signals if not s.shadow and speaker_of(s.channel) is not None]
        r = self.rules
        out: list[Candidate] = []

        def add(s: Signal, cat: Category, sev: int, rule: str) -> None:
            spk = _speaker(s.channel)
            out.append(
                Candidate(
                    spk,
                    cat,
                    s.ts_start,
                    s.ts_end,
                    sev,
                    self.calibrate(s),
                    [s.signal_id],
                    s.source in PRECISE,
                    rule,
                )
            )

        for s in sigs:
            p = self.calibrate(s)
            if s.source == SignalSource.LEXICON and p >= r.lexicon.score_gte:
                add(s, s.category, _sev(s), "lexicon_fast_path")
            elif s.source == SignalSource.KWS and p >= r.kws.score_gte:
                add(s, s.category, r.kws.severity or _sev(s, 2), "kws_fast_path")
            elif (
                s.source == SignalSource.TEXT_CLF
                and p >= r.text_clf.score_gte
                and s.category in r.text_clf.category_in
            ):
                add(s, s.category, _sev(s), "text_clf_event")
            elif s.source == SignalSource.REGISTER:
                spk = _speaker(s.channel)
                level = LEVELS[int(s.evidence.features.get("to_level", 1))]
                sev = r.register_drop.severity_by_level[spk].get(level, 1)
                add(s, Category.DISRESPECT_REGISTER, sev, "register_drop")
            elif s.source == SignalSource.LLM_WINDOW and p >= r.llm_event.score_gte:
                add(s, s.category, _sev(s), "llm_window_event")
        out += self._sustained(
            sigs, SignalSource.DSP_SHOUT, Category.SHOUTING, r.shout, "sustained_shout"
        )
        out += self._sustained(
            sigs, SignalSource.SER, Category.HARSH_TONE, r.harsh_tone, "harsh_tone_ser"
        )
        out += self._overtalk(sigs)
        return out

    def _sustained(
        self, sigs: list[Signal], source: SignalSource, cat: Category, rule: DurationRule, name: str
    ) -> list[Candidate]:
        """Runs of overlapping windows ≥ ``score_gte`` lasting ≥ ``min_duration_s``. The span runs
        between the first and last window centres, widened by half a hop (window ≈ 1.5 s)."""
        out = []
        for ch in (Channel.AGENT, Channel.CUSTOMER):
            hot = sorted(
                (
                    s
                    for s in sigs
                    if s.source == source
                    and s.channel == ch
                    and self.calibrate(s) >= rule.score_gte
                ),
                key=lambda s: s.ts_start,
            )
            run: list[Signal] = []
            for s in [*hot, None]:
                if s is not None and (not run or s.ts_start <= run[-1].ts_end):
                    run.append(s)
                    continue
                if run:
                    centres = [(x.ts_start + x.ts_end) / 2 for x in run]
                    half = (run[0].ts_end - run[0].ts_start) / 6
                    t0, t1 = centres[0] - half, centres[-1] + half
                    if t1 - t0 >= rule.min_duration_s:
                        spk = _speaker(ch)
                        out.append(
                            Candidate(
                                spk,
                                cat,
                                round(t0, 3),
                                round(t1, 3),
                                rule.severity,
                                max(self.calibrate(x) for x in run),
                                [x.signal_id for x in run],
                                True,
                                name,
                            )
                        )
                run = [s] if s is not None else []
        return out

    def _overtalk(self, sigs: list[Signal]) -> list[Candidate]:
        rule = self.rules.overtalk
        out = []
        for ch in (Channel.AGENT, Channel.CUSTOMER):
            xs = sorted(
                (s for s in sigs if s.source == SignalSource.VAD_OVERTALK and s.channel == ch),
                key=lambda s: s.ts_start,
            )
            i = 0
            while i < len(xs):
                group = [x for x in xs[i:] if x.ts_start - xs[i].ts_start <= rule.window_s]
                if len(group) >= rule.count_gte:
                    spk = _speaker(ch)
                    out.append(
                        Candidate(
                            spk,
                            Category.INTERRUPTION_OVERTALK,
                            group[0].ts_start,
                            group[-1].ts_end,
                            rule.severity,
                            min(1.0, 0.5 + 0.1 * len(group)),
                            [x.signal_id for x in group],
                            True,
                            "overtalk",
                        )
                    )
                    i += len(group)
                else:
                    i += 1
        return out

    # --- 2. merge → provisional events ---------------------------------------------------------

    def provisional(self, call_id: str, signals: Iterable[Signal]) -> list[FusedEvent]:
        gap = self.thr.hysteresis.merge_gap_s
        merged: list[Candidate] = []
        for c in sorted(
            self.candidates(signals), key=lambda c: (c.speaker.value, c.category.value, c.t0)
        ):
            last = merged[-1] if merged else None
            if (
                last
                and (last.speaker, last.category) == (c.speaker, c.category)
                and c.t0 - last.t1 < gap
            ):
                last.members.append(c)
                last.t1 = max(last.t1, c.t1)
            else:
                merged.append(
                    replace(c, members=[c])
                )  # the group is a copy: members keep their spans
        events = [self._event(call_id, m) for m in merged]
        return self._shout_plus_abuse(events)

    def _event(self, call_id: str, m: Candidate) -> FusedEvent:
        span_from = [c for c in m.members if c.precise] or m.members
        t0 = min(c.t0 for c in span_from)
        t1 = max(c.t1 for c in span_from)
        ids = sorted({i for c in m.members for i in c.signal_ids})
        return FusedEvent(
            event_id=stable_id(call_id, m.speaker, m.category, round(t0, 1)),
            call_id=call_id,
            speaker=m.speaker,
            category=m.category,
            severity=max(c.severity for c in m.members),
            probability=round(max(c.prob for c in m.members), 4),
            ts_start=round(t0, 3),
            ts_end=round(t1, 3),
            contributing_signals=ids,
        )

    def _shout_plus_abuse(self, events: list[FusedEvent]) -> list[FusedEvent]:
        r = self.rules.shout_abuse
        shouts = [e for e in events if e.category == Category.SHOUTING]
        out = []
        for e in events:
            loud = e.category in r.categories and any(
                s.speaker == e.speaker
                and s.ts_start - r.within_s <= e.ts_end
                and e.ts_start <= s.ts_end + r.within_s
                for s in shouts
            )
            if loud:
                e = e.model_copy(update={"severity": min(3, e.severity + r.severity_delta)})
            out.append(e)
        return out

    # --- 3./4. LLM ----------------------------------------------------------------------------

    def apply_judgements(
        self,
        events: list[FusedEvent],
        decisions: Iterable[ProvisionalDecision],
        signals: Iterable[Signal],
    ) -> list[FusedEvent]:
        by_id = {d.event_id: d for d in decisions}
        kws = [s for s in signals if s.source == SignalSource.KWS]
        out = []
        for e in events:
            d = by_id.get(e.event_id)
            if d is None:
                out.append(e)  # not judged (LLM offline / no window): stays provisional
                continue
            upd: dict[str, object] = {"explanation_bn": d.reason_bn or e.explanation_bn}
            if d.decision == "dismiss":
                sure = any(
                    speaker_of(s.channel) == e.speaker
                    and s.ts_start <= e.ts_end
                    and e.ts_start <= s.ts_end
                    and self.calibrate(s) >= self.rules.llm_dismiss.unless_kws_gte
                    for s in kws
                )
                upd["status"] = EventStatus.CONFIRMED if sure else EventStatus.DISMISSED_BY_LLM
            else:
                upd["status"] = EventStatus.CONFIRMED
                if d.decision == "change_category" and d.new_category is not None:
                    upd["category"] = d.new_category
                if d.decision == "change_severity" and d.new_severity is not None:
                    upd["severity"] = d.new_severity
            out.append(e.model_copy(update=upd))
        return out

    def llm_events(
        self, call_id: str, llm_signals: Iterable[Signal], existing: list[FusedEvent]
    ) -> list[FusedEvent]:
        gap = self.thr.hysteresis.merge_gap_s
        out: list[FusedEvent] = []
        for c in self.candidates(s for s in llm_signals if s.source == SignalSource.LLM_WINDOW):
            dup = any(
                e.speaker == c.speaker
                and e.category == c.category
                and c.t0 - gap < e.ts_end
                and e.ts_start < c.t1 + gap
                for e in [*existing, *out]
            )
            if not dup:
                c = replace(c, members=[c])
                out.append(
                    self._event(call_id, c).model_copy(update={"status": EventStatus.CONFIRMED})
                )
        return out

    # --- 5. hysteresis + alert policy ----------------------------------------------------------

    def finalize(self, events: list[FusedEvent]) -> list[FusedEvent]:
        h = self.thr.hysteresis
        events = sorted(events, key=lambda e: e.ts_start)
        out: list[FusedEvent] = []
        last_alert: dict[tuple[Speaker, Category], float] = {}
        for e in events:
            ct = self.thr.categories[e.category]
            if e.probability < ct.log:
                continue
            live = e.status != EventStatus.DISMISSED_BY_LLM
            repeats = (
                sum(
                    1
                    for p in out
                    if (p.speaker, p.category) == (e.speaker, e.category)
                    and p.status != EventStatus.DISMISSED_BY_LLM
                    and e.ts_start - p.ts_start <= h.escalate_window_s
                )
                + 1
            )
            sev = min(3, e.severity + 1) if live and repeats >= h.escalate_repeats else e.severity
            thr = (ct.alert_postcall if self.mode == "postcall" else ct.alert_realtime).for_speaker(
                e.speaker
            )
            key = (e.speaker, e.category)
            alert = live and e.probability >= thr
            if alert and key in last_alert and e.ts_start - last_alert[key] < h.cooldown_s:
                alert = False  # cool-down: one alert per speaker + category per window
            if alert:
                last_alert[key] = e.ts_start
            out.append(e.model_copy(update={"severity": sev, "alert": alert}))
        return out
