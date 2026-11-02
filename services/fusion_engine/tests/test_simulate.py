from pathlib import Path

from gale_fusion_engine.fusion import FusionEngine
from gale_fusion_engine.rules import parse_rules
from gale_fusion_engine.simulate import replay, simulate
from galecore.config import RulesConfig, load_rules, load_thresholds
from galecore.schemas import Category, Channel, EventStatus, Signal, SignalSource

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def lex(call: str, t: float, score: float) -> Signal:
    return Signal(
        call_id=call,
        channel=Channel.AGENT,
        signal_id=f"{call}-lex-{t}",
        source=SignalSource.LEXICON,
        category=Category.INSULT,
        score=score,
        ts_start=t,
        ts_end=t + 0.5,
        model_version="t",
    )


def stricter(cfg: RulesConfig, gte: float) -> RulesConfig:
    data = cfg.model_dump()
    rule = next(r for r in data["rules"] if r["name"] == "lexicon_fast_path")
    rule["when"]["score_gte"] = gte
    return RulesConfig.model_validate(data)


def test_simulate_diffs_alerts_between_rule_sets() -> None:
    cfg, thr = load_rules(CONFIGS), load_thresholds(CONFIGS)
    calls = {
        "a": ([lex("a", 5.0, 0.95)], []),
        "b": ([lex("b", 5.0, 0.8)], []),
    }
    same = simulate(calls, parse_rules(cfg), parse_rules(cfg), thr)
    assert same.changed_calls == []
    assert same.baseline_alerts == same.candidate_alerts
    rep = simulate(calls, parse_rules(cfg), parse_rules(stricter(cfg, 0.9)), thr)
    assert rep.calls == 2
    assert (rep.baseline_alerts, rep.candidate_alerts) == (2, 1)
    [diff] = rep.changed_calls  # only the weak hit falls below the stricter threshold
    assert diff.call_id == "b"
    assert (diff.baseline_events, diff.candidate_events) == (1, 0)
    assert [(a.category, a.ts_start) for a in diff.removed_alerts] == [("insult", 5.0)]
    assert diff.added_alerts == []


def test_replay_reuses_stored_llm_dismissals() -> None:
    cfg, thr = load_rules(CONFIGS), load_thresholds(CONFIGS)
    engine = FusionEngine(parse_rules(cfg), thr, mode="postcall")
    signals = [lex("c", 5.0, 0.95)]
    fresh = replay(engine, "c", signals, [])
    assert fresh
    assert {e.status for e in fresh} == {EventStatus.PROVISIONAL}  # judge "offline"
    dismissed = [e.model_copy(update={"status": EventStatus.DISMISSED_BY_LLM}) for e in fresh]
    again = replay(engine, "c", signals, dismissed)
    assert {e.status for e in again} == {EventStatus.DISMISSED_BY_LLM}
    assert not any(e.alert for e in again)
    confirmed = [e.model_copy(update={"status": EventStatus.CONFIRMED}) for e in fresh]
    assert {e.status for e in replay(engine, "c", signals, confirmed)} == {EventStatus.CONFIRMED}
