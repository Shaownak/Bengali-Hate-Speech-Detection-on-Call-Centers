"""Evaluation harness tests: metrics against hand-computed values, slicing, and every component on a
few synthetic fixture calls in stub mode."""

import json
from pathlib import Path

import pytest

from galecore.config import Settings
from ml.eval import metrics as M
from ml.eval.bakeoff import adr_table, run_bakeoff, write_adr
from ml.eval.components import EVALUATORS, Context
from ml.eval.run import FIXTURES, confidence_intervals, evaluate, main
from ml.eval.sets import (
    EvalCall,
    EvalSet,
    duration_bucket,
    load_fixtures,
    load_manifest_split,
    time_of_day,
)
from ml.eval.slices import slice_report

SETTINGS = Settings(env="test", bus_backend="memory")


@pytest.fixture(scope="module")
def small() -> EvalSet:
    es = load_fixtures(FIXTURES)
    keep = {"demo_01", "demo_04", "demo_09"}  # slang+shout, overtalk+insult, context negatives
    return EvalSet("fixtures", [c for c in es.calls if c.call_id in keep])


def test_text_metrics() -> None:
    assert M.wer(["আমি ভাত খাই"], ["আমি ভাত খাই"]) == 0.0
    assert M.wer(["আমি ভাত খাই"], ["আমি খাই"]) == pytest.approx(1 / 3)
    assert M.cer(["আমি"], ["আমি।"]) == 0.0  # punctuation is not scored
    assert M.cer(["ab"], ["ac"]) == 0.5
    assert M.edit_distance("kitten", "sitting") == 3
    match = lambda t: ["g1"] * t.count("<GALI_1>")  # noqa: E731
    assert M.slang_recall(["এক <GALI_1> দুই <GALI_1>"], ["এক <GALI_1> দুই"], match) == 0.5
    assert M.slang_recall(["নেই"], ["নেই"], match) is None
    assert M.script_drift(["আমি hello <PHONE>"]) == pytest.approx(5 / 8)
    assert M.script_drift([""]) == 0.0


def test_matching_and_prf() -> None:
    g = M.Span("c", "agent", "insult", 10.0, 11.0)
    near = M.Span("c", "agent", "insult", 11.5, 12.5)  # IoU 0, centre within gold ± 2 s
    far = M.Span("c", "agent", "insult", 20.0, 21.0)
    other = M.Span("c", "customer", "insult", 10.0, 11.0)
    assert M.docs06_match(near, g)
    assert not M.docs06_match(far, g)
    assert not M.docs06_match(other, g)
    m = M.match_events([near, far, other], [g])
    assert (len(m.tp), len(m.fp), len(m.fn)) == (1, 2, 0)
    assert M.prf(1, 2, 0) == {
        "precision": 0.3333,
        "recall": 1.0,
        "f1": 0.5,
        "tp": 1,
        "fp": 2,
        "fn": 0,
    }
    assert M.prf(0, 0, 0)["f1"] == 0.0
    by = M.prf_by(m, lambda s: s.speaker)
    assert by["customer"]["fp"] == 1
    long = M.Span("c", "agent", "shouting", 0, 10)
    assert M.overlap_match(M.Span("c", "agent", "shouting", 4, 12), long)
    assert not M.overlap_match(M.Span("c", "agent", "shouting", 8, 12), long)


def test_probabilistic_metrics() -> None:
    assert M.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert M.auc([0.5, 0.5], [0, 1]) == 0.5
    assert M.auc([0.5], [1]) is None
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert M.spearman([1, 1, 1], [1, 2, 3]) is None
    cal = M.calibration([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 0])
    assert cal["ece"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert M.calibration([], [])["ece"] is None
    assert M.uar(["a", "a", "b"], ["a", "b", "b"]) == 0.75
    lat = M.latency_summary([0.1] * 19 + [1.0])
    assert lat["p50"] == 0.1
    assert lat["max"] == 1.0
    assert M.latency_summary([])["p95"] is None
    ci = M.bootstrap_ci([1, 2, 3, 4], lambda xs: sum(xs) / len(xs), n=200)
    assert ci is not None
    assert ci == M.bootstrap_ci([1, 2, 3, 4], lambda xs: sum(xs) / len(xs), n=200)  # seeded
    assert ci[0] <= 2.5 <= ci[1]


def test_slices_flag_high_false_alarm_rate() -> None:
    calls = [
        EvalCall(f"c{i}", Path("x.wav"), [], meta={"dialect": "sylhet" if i < 2 else "dhaka"})
        for i in range(8)
    ]
    es = EvalSet("t", calls)
    fps = [
        M.Span("c0", "agent", "insult", 1, 2),
        M.Span("c1", "agent", "insult", 1, 2),
        M.Span("c5", "agent", "insult", 1, 2),
    ]
    rep = slice_report(es, M.Matched([], fps, []))
    assert rep["overall_false_alerts_per_100_calls"] == 37.5
    assert rep["slices"]["dialect"]["sylhet"]["false_alerts_per_100_calls"] == 100.0
    assert [f["value"] for f in rep["flags"] if f["slice"] == "dialect"] == ["sylhet"]


def test_set_helpers() -> None:
    assert [duration_bucket(s) for s in (60, 200, 400, 900)] == ["<2m", "2-5m", "5-10m", ">10m"]
    assert time_of_day("2026-11-02T03:00:00") == "night"
    assert time_of_day("2026-11-02T19:30:00") == "evening"
    assert time_of_day(None) == "unknown"


@pytest.mark.parametrize("component", sorted(EVALUATORS))
def test_every_component_runs_on_fixtures(small: EvalSet, component: str, tmp_path: Path) -> None:
    result, paths = evaluate(component, small, SETTINGS, tmp_path)
    assert result.component == component
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["calls"] == 3
    assert "Synthetic fixtures" in paths["md"].read_text(encoding="utf-8")
    assert paths["html"].read_text(encoding="utf-8").startswith("<!doctype html>")
    m = result.metrics
    if component in ("fusion", "system"):
        assert m["events"]["f1"] == 1.0  # stub models + oracle reproduce the ground truth
        assert m["context_negatives"] == {
            "gold": 2,
            "still_live": 0,
        }  # demo_09 dismissed by the judge
        assert payload["slices"]["slices"]["dialect"]
        assert payload["ci"]["events"] == [1.0, 1.0]
    if component == "system":
        assert m["false_alerts_per_100_calls"] == 0.0
        assert m["call_auc_needs_review"] is None or m["call_auc_needs_review"] >= 0.5
    if component == "asr":
        assert m["cer"] == 0.0  # oracle ASR
        assert m["slang_recall"] == 1.0
    if component == "shout":
        assert m["events"]["recall"] == 1.0
    if component == "llm":
        assert m["decision_accuracy"] == 1.0
        assert m["json_validity"] == 1.0
    if component == "textclf":
        assert m["macro_f1"] is not None


def test_golden_split_needs_a_valid_lock(tmp_path: Path) -> None:
    from ml.data.manifest import ChannelRow, write_manifest
    from ml.data.splits import write_lock

    rows = [
        ChannelRow(
            call_id=f"c{i}",
            channel=ch,
            source="t",
            source_ref="r",
            duration=60,
            sample_rate=8000,  # type: ignore[arg-type]
            split="golden" if i == 0 else "dev",
            agent_id=f"a{i}",
            dialect="dhaka",
            date="2026-10-01T09:00:00",
        )
        for i in range(2)
        for ch in ("agent", "customer")
    ]
    write_manifest(rows, tmp_path / "m.jsonl")
    lock = tmp_path / "golden.lock"
    write_lock(rows, lock)
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "fusion.jsonl").write_text(
        json.dumps(
            {
                "call_id": "c1",
                "events": [
                    {
                        "speaker": "agent",
                        "category": "insult",
                        "start": 1,
                        "end": 2,
                        "severity": 2,
                        "flags": ["reported_speech"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dev = load_manifest_split(tmp_path / "m.jsonl", gold, "dev", tmp_path / "audio", lock)
    assert [c.call_id for c in dev.calls] == ["c1"]
    assert dev.calls[0].events[0].alert is False  # reported speech: should not alert
    assert dev.calls[0].meta["time_of_day"] == "morning"
    assert (
        len(
            load_manifest_split(
                tmp_path / "m.jsonl", gold, "golden", tmp_path / "audio", lock
            ).calls
        )
        == 1
    )
    write_manifest([r.model_copy(update={"split": "golden"}) for r in rows], tmp_path / "m.jsonl")
    with pytest.raises(RuntimeError, match="integrity"):
        load_manifest_split(tmp_path / "m.jsonl", gold, "golden", tmp_path / "audio", lock)


def test_bakeoff_and_adr(small: EvalSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.eval.bakeoff as bk

    monkeypatch.setattr(bk, "load_set", lambda split, **kw: small)
    spec = {
        "component": "fusion",
        "primary": "events.f1",
        "candidates": [
            {"name": "rules_v0"},
            {
                "name": "rules_v0_copy",
                "models": {"fusion": {"backend": "rules_v0", "version": "0.1b"}},
            },
        ],
    }
    rows = run_bakeoff(spec, "fixtures", tmp_path / "out", {})
    assert [r["primary"] for r in rows] == [1.0, 1.0]
    assert sorted(r["rank"] for r in rows) == [1, 2]
    table = adr_table(spec, "fixtures", rows, ["severity_accuracy"])
    assert table.splitlines()[0] == "| candidate | events.f1 | severity_accuracy | rank | report |"
    write_adr(tmp_path / "adr.md", spec, "fixtures", table)
    assert "## Results" in (tmp_path / "adr.md").read_text(encoding="utf-8")


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--component", "shout", "--split", "fixtures", "--out", str(tmp_path)]) == 0
    assert "shout on fixtures (10 calls)" in capsys.readouterr().out
    assert len(list(tmp_path.glob("shout_fixtures_*.md"))) == 1
    ctx = Context(SETTINGS)
    assert confidence_intervals(EVALUATORS["ser"](load_fixtures(FIXTURES), ctx)) == {}
