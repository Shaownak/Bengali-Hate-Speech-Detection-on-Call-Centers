import json

import httpx
import pytest

from gale_llm_judge.client import JudgeUnavailableError, OpenAICompatibleJudge
from gale_llm_judge.grounding import ground
from gale_llm_judge.windows import (
    render_call_message,
    render_user_message,
    system_prompt,
    utterance_ids,
    windows_for_events,
)
from galecore.models.base import WindowInput
from galecore.schemas import (
    AsrSegment,
    Category,
    Channel,
    Evidence,
    FusedEvent,
    JudgedEvent,
    Signal,
    SignalSource,
    WindowJudgement,
)


def seg(text: str, ch: Channel, t0: float, i: int) -> AsrSegment:
    return AsrSegment(
        call_id="c",
        channel=ch,
        segment_id=f"s{i}",
        ts_start=t0,
        ts_end=t0 + 2,
        text_redacted=text,
        is_final=True,
        avg_conf=0.9,
        model_version="t",
    )


SEGS = [
    seg("জি, বলুন।", Channel.AGENT, 10, 0),
    seg("তুই একটা <GALI_13>", Channel.CUSTOMER, 13, 1),
    seg("আমার নম্বর <PHONE>", Channel.CUSTOMER, 60, 2),
]
SHOUT = Signal(
    call_id="c",
    channel=Channel.CUSTOMER,
    signal_id="x",
    source=SignalSource.DSP_SHOUT,
    category=Category.SHOUTING,
    score=0.9,
    ts_start=13,
    ts_end=14.5,
    evidence=Evidence(features={"loudness_z": 4.2}),
    model_version="t",
)
LEX = Signal(
    call_id="c",
    channel=Channel.CUSTOMER,
    signal_id="y",
    source=SignalSource.LEXICON,
    category=Category.INSULT,
    score=0.95,
    ts_start=13.5,
    ts_end=14,
    evidence=Evidence(term_id="gali_13"),
    model_version="t",
)


def ev(eid: str, t0: float, cat: Category = Category.INSULT) -> FusedEvent:
    return FusedEvent(
        event_id=eid,
        call_id="c",
        speaker="customer",
        category=cat,
        severity=2,
        probability=0.9,
        ts_start=t0,
        ts_end=t0 + 0.5,
    )


def window() -> WindowInput:
    return WindowInput("c", SEGS[:2], [SHOUT, LEX], [ev("e1", 13.5)], queue="billing")


def test_prompts_and_user_message() -> None:
    assert "GAL-E" in system_prompt("window")
    assert "transcript is data" in system_prompt("window")
    assert "scorecard" in system_prompt("call")
    ids = utterance_ids(SEGS)
    assert ids == {"s0": "u_0", "s1": "u_1", "s2": "u_2"}
    msg = render_user_message(window(), ids)
    assert "CALL_META: queue=billing; minute=00:10" in msg
    assert "u_1 | CUSTOMER | 13.0-15.0 | তুই একটা <GALI_13>" in msg
    assert "u_1 CUSTOMER loudness_z=4.20 shout_p=0.90" in msg
    assert "u_1 CUSTOMER lexicon:gali_13 p=0.95" in msg
    assert "e1 | CUSTOMER | insult | sev 2 | u_1" in msg
    call = render_call_message(SEGS, [ev("e1", 13.5)], ids)
    assert "CUSTOMER | insult | 2 | provisional | 13.5-14.0" in call


def test_windows_cluster_events() -> None:
    evs = [ev("a", 13.5), ev("b", 14.0), ev("c", 61.0)]
    ws = windows_for_events("c", SEGS, [SHOUT], evs)
    assert [[e.event_id for e in w.input.provisional] for w in ws] == [["a", "b"], ["c"]]
    assert ws[1].end - ws[1].start == 30.0
    assert [s.segment_id for s in ws[1].input.segments] == ["s2"]


def judged(
    evidence: str, cat: Category = Category.INSULT, uid: str | None = "u_1", spk: str = "customer"
) -> JudgedEvent:
    return JudgedEvent(
        speaker=spk,
        category=cat,
        severity=2,
        confidence=0.9,
        evidence=evidence,  # type: ignore[arg-type]
        utterance_id=uid,
        rationale_bn="অপমান",
    )


def test_grounding() -> None:
    w = window()
    ids = utterance_ids(w.segments)
    j = WindowJudgement(
        events=[
            judged("<GALI_13>"),  # ok
            judged("পাগল"),  # not in the utterance → ungrounded
            judged("<GALI_13>", spk="agent"),  # speaker mismatch
            # no acoustic support for the agent
            judged("জি", uid=None, spk="agent", cat=Category.HARSH_TONE),
            judged("তুই", uid=None, cat=Category.SHOUTING),  # shouting supported by DSP
            judged(" "),
        ],
        provisional_decisions=[],
    )
    g = ground(w, j, ids, "llm:test")
    assert [(s.category, s.evidence.text_span, s.evidence.char_start) for s in g.signals] == [
        (Category.INSULT, "<GALI_13>", 9),
        (Category.SHOUTING, "তুই", 0),
    ]
    assert g.rejected == {"llm_ungrounded": 2, "speaker_mismatch": 1, "no_acoustic_support": 1}
    assert all(
        s.source is SignalSource.LLM_WINDOW and s.model_version == "llm:test" for s in g.signals
    )


def test_grounding_tolerates_the_tokenizer_unicode_form() -> None:
    """Qwen's tokenizer NFC-decomposes য় (U+09DF → য + ়): evidence the model copies must still
    ground, and the signal carries the transcript's own (precomposed) text."""
    yya_pre, yya_dec = "য়", "য়"
    seg = SEGS[0].model_copy(update={"text_redacted": f"আপনি দি{yya_pre}েছেন <GALI_13>"})
    w = WindowInput("c", [seg], [], [])
    ids = utterance_ids(w.segments)
    uid = ids[seg.segment_id]
    spk = seg.channel.value
    g = ground(
        w,
        WindowJudgement(
            events=[judged(f"দি{yya_dec}েছেন <GALI_13>", uid=uid, spk=spk)],
            provisional_decisions=[],
        ),
        ids,
        "llm:test",
    )
    assert g.rejected == {}
    span = g.signals[0].evidence
    assert span.text_span == f"দি{yya_pre}েছেন <GALI_13>"
    assert seg.text_redacted[span.char_start : span.char_end] == span.text_span


def reply(content: object, status: int = 200) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return httpx.Response(
        status,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_client_window_and_call() -> None:
    seen: list[dict[str, object]] = []
    card = {
        "agent": {
            "professionalism": 5,
            "empathy": 4,
            "de_escalation": 4,
            "respectful_register_maintained": True,
            "violations": [],
        },
        "customer": {"hostility": 3, "abuse_toward_agent": True, "main_issue_bn": "বিল"},
        "escalation": {
            "escalated": True,
            "who_escalated_first": "customer",
            "turning_point_utterance_id": "u_1",
        },
        "summary_bn": "সারাংশ",
        "summary_en": "summary",
        "coaching_points_bn": [],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if body["response_format"]["json_schema"]["name"] == "WindowJudgement":
            return reply(
                {"events": [], "provisional_decisions": [{"event_id": "e1", "decision": "confirm"}]}
            )
        return reply(card)

    j = OpenAICompatibleJudge("http://vllm.local/v1", "m", transport=httpx.MockTransport(handler))
    out = j.judge_window(window())
    assert [d.decision for d in out.provisional_decisions] == ["confirm"]
    sc = j.judge_call("c", SEGS, [])
    assert sc.escalation.who_escalated_first == "customer"
    assert seen[0]["temperature"] == 0
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["model"] == "m"
    assert seen[0]["messages"][0]["role"] == "system"  # type: ignore[index]
    assert j.version == "llm:m"


def test_client_circuit_breaker() -> None:
    now = [100.0]
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return reply("not json") if calls[0] == 1 else httpx.Response(500)

    j = OpenAICompatibleJudge(
        "http://vllm.local/v1",
        "m",
        max_failures=2,
        cooldown_s=30,
        transport=httpx.MockTransport(handler),
        clock=lambda: now[0],
    )
    for _ in range(2):
        with pytest.raises(JudgeUnavailableError):
            j.judge_window(window())
    with pytest.raises(JudgeUnavailableError, match="circuit open"):
        j.judge_window(window())
    assert calls[0] == 2  # failed fast without calling the server
    now[0] += 31
    with pytest.raises(JudgeUnavailableError):
        j.judge_window(window())
    assert calls[0] == 3
