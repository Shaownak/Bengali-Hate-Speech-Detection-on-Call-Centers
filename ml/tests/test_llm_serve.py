"""The local judge server (ml.llm.serve) speaks what the production client expects: the real
``OpenAICompatibleJudge`` round-trips through it. Fake generator, no GPU, no text in logs."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gale_llm_judge.client import JudgeUnavailableError, OpenAICompatibleJudge
from galecore.models.base import WindowInput
from galecore.schemas import AsrSegment, Channel
from ml.llm.serve import create_app


def _window() -> WindowInput:
    seg = AsrSegment(
        call_id="c1",
        channel=Channel.CUSTOMER,
        segment_id="c1:0",
        ts_start=1.0,
        ts_end=3.0,
        text_redacted="<GALI_1> বলছি",
        is_final=True,
        avg_conf=0.9,
        model_version="t",
    )
    return WindowInput("c1", [seg], [], [])


GOOD = {
    "events": [
        {
            "speaker": "customer",
            "category": "slang_profanity",
            "severity": 2,
            "confidence": 0.9,
            "evidence": "<GALI_1>",
            "utterance_id": "u_0",
            "rationale_bn": "গ্রাহক গালিগালাজ করেছেন।",
        }
    ],
    "provisional_decisions": [],
}


def _judge(
    answer: str, seen: list[list[dict[str, str]]]
) -> tuple[OpenAICompatibleJudge, TestClient]:
    def generate(messages: list[dict[str, str]], max_new: int) -> tuple[str, int, int]:
        seen.append(messages)
        return answer, 100, 20

    tc = TestClient(create_app(generate, "gale-llm-judge"))

    def forward(req: httpx.Request) -> httpx.Response:
        r = tc.request(req.method, req.url.path, content=req.content, headers=dict(req.headers))
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)

    j = OpenAICompatibleJudge(
        "http://127.0.0.1:8100/v1", "gale-llm-judge", transport=httpx.MockTransport(forward)
    )
    return j, tc


def test_client_round_trips_through_the_local_server() -> None:
    seen: list[list[dict[str, str]]] = []
    judge, tc = _judge(json.dumps(GOOD), seen)
    out = judge.judge_window(_window())
    assert out.events[0].evidence == "<GALI_1>"
    sys_, user = seen[0]
    assert sys_["role"] == "system"
    assert "GAL-E" in sys_["content"]
    assert user["content"].startswith("CALL_META:")
    h = tc.get("/health").json()
    assert (h["requests"], h["invalid"]) == (1, 0)
    assert tc.get("/v1/models").json()["data"][0]["id"] == "gale-llm-judge"


def test_invalid_output_is_counted_and_rejected_by_the_client() -> None:
    judge, tc = _judge('{"events": "nope"}', [])
    with pytest.raises(JudgeUnavailableError):
        judge.judge_window(_window())
    assert tc.get("/health").json()["invalid"] == 1
