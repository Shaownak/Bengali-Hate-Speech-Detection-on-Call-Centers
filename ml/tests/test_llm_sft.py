"""LLM-judge SFT windows, metrics and the trainer's tokenisation (Prompt 9). Synthetic rows only:
abusive text is <GALI_n> placeholders (CLAUDE.md rule 2); no torch needed."""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from galecore.schemas import WindowJudgement
from ml.data.open.schema import UnifiedRow
from ml.llm import metrics, sft
from ml.llm.train import encode, end_of_turn, prompt_text


def _row(i: int, labels: dict[str, int], split: str = "train", **kw: Any) -> UnifiedRow:
    toxic = int(any(labels.values()))
    return UnifiedRow(
        uid=f"banglacat:{split}:{i}",
        source="banglacat",
        source_split=None,
        source_id=str(i),
        text_orig=f"<GALI_{i % 20 + 1}> মন্তব্য নম্বর {i}" if toxic else f"সাধারণ মন্তব্য {i}",
        script_orig="bangla",
        text_bn=f"<GALI_{i % 20 + 1}> মন্তব্য নম্বর {i}" if toxic else f"সাধারণ মন্তব্য {i}",
        labels={c: labels.get(c, 0) for c in sft.TEXT_CATEGORIES},
        toxic_any=toxic,
        licence_id="T3",
        commercial_ok=False,
        research_only=True,
        split=split,  # type: ignore[arg-type]
        **kw,
    )


CATS = ["insult", "slang_profanity", "threat", "sexual_harassment", "sarcasm_mockery"]


@pytest.fixture(scope="module")
def built() -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, tuple[list[UnifiedRow], list[UnifiedRow]]] = {}
    for split in ("train", "dev"):
        pos = [_row(i, {CATS[i % len(CATS)]: 1}, split, severity=2) for i in range(40)]
        neg = [_row(100 + i, {}, split) for i in range(20)]
        rows[split] = (pos, neg)
    return sft.build(rows, {"train": 400, "dev": 60}, seed=3)


def test_windows_follow_the_judge_contract(built: dict[str, list[dict[str, Any]]]) -> None:
    train = built["train"]
    assert len(train) == 400
    for x in train:
        sys_, user, asst = (m["content"] for m in x["messages"])
        assert "GAL-E" in sys_
        assert user.startswith("CALL_META: queue=")
        j = WindowJudgement.model_validate_json(asst)
        utts = metrics.transcript(user)
        for e in j.events:  # grounded, speaker-matched evidence (docs/08 §2 post-validation)
            spk, text = utts[e.utterance_id or ""]
            assert spk == e.speaker.value
            assert e.evidence in text
            if e.category.value in ("shouting", "harsh_tone"):
                assert "ACOUSTIC_SIGNALS" in user
                assert e.utterance_id in user.split("ACOUSTIC_SIGNALS")[1].split("TEXT_SIGNALS")[0]
        prov = user.split("PROVISIONAL_EVENTS:")[1].strip().splitlines()
        ids = [line.split(" | ")[0] for line in prov if line]
        assert [d.event_id for d in j.provisional_decisions] == ids  # every one decided
        assert x["meta"]["research_only"] is True
        assert "<PHONE>" not in x["meta"].get("uid", "")


def test_mix_covers_scenarios_and_decisions(built: dict[str, list[dict[str, Any]]]) -> None:
    st = sft.stats(built["train"])
    assert set(st["scenarios"]) == set(sft.SCENARIOS)
    assert {"confirm", "dismiss", "change_severity", "change_category"} <= set(st["decisions"])
    for cat in ("disrespect_register", "dismissive_rude", "shouting", "insult"):
        assert st["event_categories"].get(cat, 0) > 0
    assert 0 < st["empty_targets"] < st["n"] / 2


def test_reported_speech_and_injection_windows(built: dict[str, list[dict[str, Any]]]) -> None:
    for x in built["train"]:
        j = json.loads(x["messages"][2]["content"])
        if x["meta"]["scenario"] == "reported":
            text_events = [e for e in j["events"] if e["category"] in sft.TEXT_CATEGORIES]
            assert all(e["context_flags"] == ["reported_speech"] for e in text_events)
            assert all(e["severity"] == 1 for e in text_events)
            assert all(d["decision"] == "dismiss" for d in j["provisional_decisions"])
        if x["meta"]["scenario"] == "injection":  # the injected line is data: abuse still found
            assert any(e["category"] in sft.TEXT_CATEGORIES for e in j["events"]) or any(
                d["decision"] != "dismiss" for d in j["provisional_decisions"]
            )


def test_build_is_deterministic(built: dict[str, list[dict[str, Any]]]) -> None:
    rows = {
        "dev": (
            [_row(i, {CATS[i % len(CATS)]: 1}, "dev", severity=2) for i in range(40)],
            [_row(100 + i, {}, "dev") for i in range(20)],
        )
    }
    again = sft.build(rows, {"dev": 60}, seed=3)["dev"]
    assert again == built["dev"]


def test_balanced_keeps_rare_categories() -> None:
    rows = [_row(i, {"insult": 1}) for i in range(400)] + [
        _row(500 + i, {"threat": 1}) for i in range(10)
    ]
    kept = sft.balanced(rows, random.Random(0))
    assert sum(r.labels["threat"] == 1 for r in kept) == 10  # the rarest is always kept
    assert sum(r.labels["insult"] == 1 for r in kept) < 300


def test_metrics_perfect_empty_and_ungrounded(built: dict[str, list[dict[str, Any]]]) -> None:
    dev = built["dev"]
    gold = [x["messages"][2]["content"] for x in dev]
    m = metrics.evaluate(dev, gold)
    assert (m["json_validity"], m["event_f1"], m["hallucinated_evidence_rate"]) == (1.0, 1.0, 0.0)
    assert m["decision_accuracy"] == 1.0
    assert set(m["by_scenario"]) <= set(sft.SCENARIOS)
    empty = metrics.evaluate(dev, ['{"events":[],"provisional_decisions":[]}'] * len(dev))
    assert (empty["event_recall"], empty["clean_window_fp_rate"]) == (0.0, 0.0)
    broken = metrics.evaluate(dev, ["not json"] * len(dev))
    assert broken["json_validity"] == 0.0
    # an event whose evidence is not in the utterance is hallucinated
    x = next(x for x in dev if json.loads(x["messages"][2]["content"])["events"])
    j = json.loads(x["messages"][2]["content"])
    j["events"][0]["evidence"] = "এই কথাটা কেউ বলেনি"
    s = metrics.score(json.dumps(j), x["messages"][2]["content"], x["messages"][1]["content"])
    assert s.ungrounded == 1
    assert s.valid
    assert metrics.parse("```json\n" + x["messages"][2]["content"] + "\n```") is not None


class FakeTok:
    """Qwen-style chat template over characters (one id per char)."""

    eos_token = "<|im_end|>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        out = "".join(
            f"<|im_start|>{m['role']}\n"
            + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" else "")
            + f"{m['content']}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + (
                "" if enable_thinking else "<think>\n\n</think>\n\n"
            )
        return out

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}


def test_encode_masks_the_prompt_and_ends_the_turn(built: dict[str, list[dict[str, Any]]]) -> None:
    tok, rec = FakeTok(), built["train"][0]
    assert end_of_turn(tok) == "<|im_end|>"
    ids, n = encode(tok, rec, end_of_turn(tok))
    text = "".join(map(chr, ids))
    p = prompt_text(tok, rec["messages"])
    assert p.endswith("<think>\n\n</think>\n\n")  # thinking off, as at serving time
    assert text[:n] == p
    assert text[n:] == rec["messages"][2]["content"] + "<|im_end|>"


def test_redteam_uses_held_out_injections_only(built: dict[str, list[dict[str, Any]]]) -> None:
    import yaml

    tpl = yaml.safe_load(sft.TEMPLATES.read_text(encoding="utf-8"))
    rows = {
        "test": (
            [_row(i, {CATS[i % len(CATS)]: 1}, "test", severity=2) for i in range(20)],
            [_row(100 + i, {}, "test") for i in range(10)],
        )
    }
    rt = sft.build_redteam(rows, 40, seed=3)
    assert {x["meta"]["scenario"] for x in rt} == {"redteam_injection", "redteam_injection_clean"}
    from galecore.bangla import normalize

    # windows are rendered normalised (as ASR output is): compare normalised phrases
    held = {normalize(h) for h in tpl["redteam_injection"]}
    trained = {normalize(t) for t in tpl["injection"]}
    assert not held & trained
    for x in rt:
        user = x["messages"][1]["content"]
        assert any(h in user for h in held)
        if x["meta"]["scenario"] == "redteam_injection_clean":  # nobody may be framed
            j = json.loads(x["messages"][2]["content"])
            assert not [e for e in j["events"] if e["category"] in sft.TEXT_LIKE]
    for x in built["train"]:  # and the training windows never contain them
        assert not any(h in x["messages"][1]["content"] for h in held)


def test_merge_card_and_calibration_split(tmp_path: Path) -> None:
    from ml.llm.merge import merged_card
    from ml.llm.quantize import calibration_texts

    run = tmp_path / "run"
    run.mkdir()
    (run / "model_card.json").write_text(
        json.dumps({"name": "r1", "best": {"event_f1": 0.5}, "research_only": True}), "utf-8"
    )
    base = tmp_path / "base"
    base.mkdir()
    (base / "REVISION").write_text("Qwen/Qwen3.5-4B@abc\n", encoding="utf-8")
    card = merged_card(run, str(base), tmp_path / "gale-llm-judge")
    assert card["research_only"] is True
    assert card["base_revision"] == "Qwen/Qwen3.5-4B@abc"
    assert card["adapter_run"] == "r1"
    data = tmp_path / "data"
    data.mkdir()
    rec = {"messages": [{"role": "user", "content": "train"}], "meta": {}}
    (data / "sft_window_train.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (data / "sft_window_dev.jsonl").write_text("not read\n", encoding="utf-8")
    assert calibration_texts(5, data=data) == [rec["messages"]]  # train split only


def test_call_scorecards_follow_the_findings() -> None:
    from galecore.schemas import CallScorecard
    from ml.llm.sft_call import build_calls

    rows = {
        "train": (
            [_row(i, {CATS[i % len(CATS)]: 1}, "train", severity=2) for i in range(30)],
            [_row(100 + i, {}, "train") for i in range(15)],
        )
    }
    calls = build_calls(rows, {"train": 60}, seed=5)["train"]
    assert len(calls) == 60
    for x in calls:
        sys_, user, asst = (m["content"] for m in x["messages"])
        assert "scorecard" in sys_
        card = CallScorecard.model_validate_json(asst)
        events = user.split("EVENTS (")[1].splitlines()[1:]
        agent_live = [e for e in events if e.startswith("AGENT") and "| confirmed |" in e]
        assert (card.agent.professionalism == 5) == (not agent_live)
        assert card.escalation.escalated == (card.escalation.who_escalated_first != "none")
        if card.escalation.turning_point_utterance_id:
            assert f"\n{card.escalation.turning_point_utterance_id} |" in user
        assert card.agent.respectful_register_maintained == (
            not any("disrespect_register" in e for e in agent_live)
        )
        assert len(card.coaching_points_bn) <= 5
