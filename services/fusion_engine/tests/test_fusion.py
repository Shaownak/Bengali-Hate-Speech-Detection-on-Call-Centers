from pathlib import Path

import pytest

from gale_fusion_engine.fusion import FusionEngine
from gale_fusion_engine.rules import parse_rules
from galecore.config import Rule, RulesConfig, load_rules, load_thresholds
from galecore.schemas import (
    Category,
    Channel,
    EventStatus,
    Evidence,
    ProvisionalDecision,
    Signal,
    SignalSource,
    Speaker,
)

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.fixture
def engine() -> FusionEngine:
    return FusionEngine(parse_rules(load_rules(CONFIGS)), load_thresholds(CONFIGS))


def sig(
    source: SignalSource,
    cat: Category,
    t0: float,
    t1: float,
    score: float = 0.95,
    ch: Channel = Channel.CUSTOMER,
    **features: float,
) -> Signal:
    return Signal(
        call_id="c",
        channel=ch,
        signal_id=f"{source}-{cat}-{t0}-{ch}",
        source=source,
        category=cat,
        score=score,
        ts_start=t0,
        ts_end=t1,
        evidence=Evidence(features=features),
        model_version="t",
    )


def shout_run(
    t0: float, n: int, ch: Channel = Channel.CUSTOMER, score: float = 0.95
) -> list[Signal]:
    return [
        sig(SignalSource.DSP_SHOUT, Category.SHOUTING, t0 + 0.5 * k, t0 + 0.5 * k + 1.5, score, ch)
        for k in range(n)
    ]


def test_parse_rules_rejects_unknown_or_missing() -> None:
    cfg = load_rules(CONFIGS)
    with pytest.raises(ValueError, match="missing"):
        parse_rules(RulesConfig(version=2, rules=cfg.rules[:-1]))
    extra = Rule(name="mystery", when={}, then={})
    with pytest.raises(ValueError, match="unknown"):
        parse_rules(RulesConfig(version=2, rules=[*cfg.rules, extra]))


def test_lexicon_text_merge_with_precise_span(engine: FusionEngine) -> None:
    signals = [
        sig(SignalSource.LEXICON, Category.INSULT, 10.2, 10.6, severity=2),
        sig(SignalSource.TEXT_CLF, Category.INSULT, 9.0, 12.0, 0.9, severity=1),
        sig(SignalSource.TEXT_CLF, Category.SLANG_PROFANITY, 9.0, 12.0, 0.4),  # below score_gte
        sig(SignalSource.TEXT_CLF, Category.HARSH_TONE, 9.0, 12.0, 0.99),  # not a text category
    ]
    [e] = engine.provisional("c", signals)
    assert (e.speaker, e.category, e.severity, e.status) == (
        Speaker.CUSTOMER,
        Category.INSULT,
        2,
        EventStatus.PROVISIONAL,
    )
    assert (e.ts_start, e.ts_end) == (10.2, 10.6)  # lexicon decides the span, not the segment
    assert len(e.contributing_signals) == 2


def test_merge_gap_and_separate_events(engine: FusionEngine) -> None:
    a = sig(SignalSource.LEXICON, Category.SLANG_PROFANITY, 10, 10.5, severity=1)
    b = sig(SignalSource.LEXICON, Category.SLANG_PROFANITY, 13, 13.5, severity=3)
    c = sig(SignalSource.LEXICON, Category.SLANG_PROFANITY, 30, 30.5, severity=1)
    evs = engine.provisional("c", [a, b, c])
    assert [(e.ts_start, e.ts_end, e.severity) for e in evs] == [(10, 13.5, 3), (30, 30.5, 1)]


def test_sustained_shout_needs_duration_and_mixed_is_ignored(engine: FusionEngine) -> None:
    assert engine.provisional("c", shout_run(10, 1)) == []  # one window is not "held for 2 s"
    [e] = engine.provisional("c", shout_run(10, 5))
    assert e.category is Category.SHOUTING
    assert e.severity == 2
    assert e.ts_end - e.ts_start >= 2.0
    assert engine.provisional("c", shout_run(10, 5, ch=Channel.MIXED)) == []  # D-15


def test_shout_plus_abuse_and_harsh_tone(engine: FusionEngine) -> None:
    signals = [*shout_run(10, 5), sig(SignalSource.LEXICON, Category.THREAT, 11, 11.4, severity=2)]
    threat = next(e for e in engine.provisional("c", signals) if e.category is Category.THREAT)
    assert threat.severity == 3
    ser = [
        sig(SignalSource.SER, Category.HARSH_TONE, 5 + 0.5 * k, 6.5 + 0.5 * k, 0.9)
        for k in range(3)
    ]
    assert [e.category for e in engine.provisional("c", ser)] == [Category.HARSH_TONE]


def test_register_overtalk_kws_llm_candidates(engine: FusionEngine) -> None:
    signals = [
        sig(
            SignalSource.REGISTER,
            Category.DISRESPECT_REGISTER,
            5,
            5.3,
            0.9,
            Channel.AGENT,
            to_level=2,
        ),
        sig(SignalSource.REGISTER, Category.DISRESPECT_REGISTER, 50, 50.3, 0.8, to_level=1),
        sig(SignalSource.KWS, Category.SLANG_PROFANITY, 70, 70.5, 0.92),
        sig(SignalSource.LLM_WINDOW, Category.SARCASM_MOCKERY, 80, 82, 0.85, severity=1),
        *[
            sig(
                SignalSource.VAD_OVERTALK,
                Category.INTERRUPTION_OVERTALK,
                t,
                t + 1,
                1.0,
                Channel.AGENT,
            )
            for t in (100, 120, 150)
        ],
        *[
            sig(SignalSource.VAD_OVERTALK, Category.INTERRUPTION_OVERTALK, t, t + 1, 1.0)
            for t in (100, 200)
        ],
    ]
    got = {(e.speaker, e.category): e for e in engine.provisional("c", signals)}
    assert got[(Speaker.AGENT, Category.DISRESPECT_REGISTER)].severity == 3
    assert got[(Speaker.CUSTOMER, Category.DISRESPECT_REGISTER)].severity == 1
    assert got[(Speaker.CUSTOMER, Category.SLANG_PROFANITY)].severity == 2
    assert Category.SARCASM_MOCKERY in {c for _, c in got}
    ot = got[(Speaker.AGENT, Category.INTERRUPTION_OVERTALK)]
    assert (ot.ts_start, ot.ts_end) == (100, 151)
    assert (Speaker.CUSTOMER, Category.INTERRUPTION_OVERTALK) not in got  # only 2 interruptions


def test_judgements(engine: FusionEngine) -> None:
    signals = [
        sig(SignalSource.LEXICON, Category.SLANG_PROFANITY, 10, 10.5, severity=2),
        sig(SignalSource.LEXICON, Category.INSULT, 40, 40.5, severity=2),
        sig(SignalSource.LEXICON, Category.THREAT, 70, 70.5, severity=2),
        sig(SignalSource.LEXICON, Category.SLANG_PROFANITY, 100, 100.5, severity=2),
        sig(SignalSource.KWS, Category.SLANG_PROFANITY, 100, 100.5, 0.98),
        sig(SignalSource.LEXICON, Category.INSULT, 130, 130.5, severity=2),
    ]
    evs = engine.provisional("c", signals)
    ids = [e.event_id for e in evs]
    decisions = [
        ProvisionalDecision(event_id=ids[0], decision="dismiss", reason_bn="উদ্ধৃতি"),
        ProvisionalDecision(
            event_id=ids[1], decision="change_category", new_category=Category.SARCASM_MOCKERY
        ),
        ProvisionalDecision(event_id=ids[2], decision="change_severity", new_severity=1),
        ProvisionalDecision(event_id=ids[3], decision="dismiss"),  # overruled: KWS 0.98 ≥ 0.97
    ]
    out = engine.apply_judgements(evs, decisions, signals)
    assert [e.status for e in out] == [
        EventStatus.DISMISSED_BY_LLM,
        EventStatus.CONFIRMED,
        EventStatus.CONFIRMED,
        EventStatus.CONFIRMED,
        EventStatus.PROVISIONAL,
    ]
    assert out[0].explanation_bn == "উদ্ধৃতি"
    assert out[1].category is Category.SARCASM_MOCKERY
    assert out[2].severity == 1


def test_llm_events_dedup(engine: FusionEngine) -> None:
    existing = engine.provisional(
        "c", [sig(SignalSource.LEXICON, Category.INSULT, 10, 10.5, severity=2)]
    )
    llm = [
        sig(SignalSource.LLM_WINDOW, Category.INSULT, 9, 12, 0.9, severity=2),  # same as existing
        sig(SignalSource.LLM_WINDOW, Category.DISMISSIVE_RUDE, 40, 42, 0.9, severity=1),
        sig(SignalSource.LLM_WINDOW, Category.DISMISSIVE_RUDE, 80, 82, 0.5),  # below score_gte
        sig(SignalSource.LEXICON, Category.THREAT, 60, 61),  # not an LLM signal
    ]
    [e] = engine.llm_events("c", llm, existing)
    assert (e.category, e.status) == (Category.DISMISSIVE_RUDE, EventStatus.CONFIRMED)


def test_finalize_escalation_cooldown_and_log(engine: FusionEngine) -> None:
    sigs = [
        sig(
            SignalSource.LEXICON, Category.SLANG_PROFANITY, t, t + 0.4, ch=Channel.AGENT, severity=1
        )
        for t in (10, 20, 29)
    ]
    sigs.append(sig(SignalSource.TEXT_CLF, Category.INSULT, 200, 201, 0.6, severity=1))
    evs = engine.finalize(engine.provisional("c", sigs))
    slang = [e for e in evs if e.category is Category.SLANG_PROFANITY]
    assert [e.severity for e in slang] == [1, 1, 2]  # third repeat within 120 s escalates
    assert [e.alert for e in slang] == [True, False, False]  # 20 s cool-down per speaker+category
    insult = next(e for e in evs if e.category is Category.INSULT)
    assert insult.alert is False  # 0.6 < post-call alert threshold 0.85
    dismissed = engine.apply_judgements(
        engine.provisional("c", sigs[:1]),
        [ProvisionalDecision(event_id=slang[0].event_id, decision="dismiss")],
        sigs,
    )
    assert engine.finalize(dismissed)[0].alert is False
    rt = FusionEngine(engine.rules, engine.thr, mode="realtime")
    assert rt.finalize(engine.provisional("c", sigs[:1]))[0].alert is True
    low = engine.thr.model_copy(deep=True)
    low.categories[Category.SLANG_PROFANITY].log = 0.99
    assert FusionEngine(engine.rules, low).finalize(engine.provisional("c", sigs[:1])) == []
