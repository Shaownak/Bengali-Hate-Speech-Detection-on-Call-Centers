"""OpenAI-compatible judge client for the self-hosted vLLM server (docs/08 §5).

Structured output via ``response_format: json_schema`` (vLLM guided decoding), temperature 0,
4 s timeout in window mode, and a circuit breaker: after ``max_failures`` consecutive errors the
judge is "offline" for ``cooldown_s`` and calls fail fast, so provisional events stay provisional.
Logs prompt/response hashes, token counts and latency only, never text (CLAUDE.md rule 1).
The base URL must point at our own server: no external AI APIs for call data.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx
from prometheus_client import Counter, Histogram
from pydantic import ValidationError

from gale_llm_judge.windows import (
    render_call_message,
    render_user_message,
    system_prompt,
    utterance_ids,
)
from galecore.logging import get_logger
from galecore.models.base import LLMJudge, WindowInput
from galecore.schemas import AsrSegment, CallScorecard, FusedEvent, WindowJudgement

log = get_logger(__name__)
LATENCY = Histogram("gale_llm_judge_seconds", "LLM judge latency", ["mode"])
CALLS = Counter("gale_llm_judge_calls_total", "LLM judge calls", ["mode", "outcome"])


class JudgeUnavailableError(RuntimeError):
    """The judge is offline (circuit open, timeout, bad response)."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class OpenAICompatibleJudge(LLMJudge):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        window_timeout_s: float = 4.0,
        call_timeout_s: float = 60.0,
        max_failures: int = 3,
        cooldown_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.model = model
        self.window_timeout_s, self.call_timeout_s = window_timeout_s, call_timeout_s
        self.max_failures, self.cooldown_s, self.clock = max_failures, cooldown_s, clock
        self._failures = 0
        self._open_until = 0.0
        self._http = httpx.Client(base_url=base_url.rstrip("/"), transport=transport)

    @property
    def version(self) -> str:
        return f"llm:{self.model}"

    def _post(
        self,
        mode: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        name: str,
        timeout: float,
        max_tokens: int,
    ) -> str:
        if self.clock() < self._open_until:
            CALLS.labels(mode, "circuit_open").inc()
            raise JudgeUnavailableError("circuit open")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # thinking off (docs/03 §7.3): Qwen3.x templates default to a reasoning block, and
            # the judge was fine-tuned on prompts ending in an empty one (vLLM passes this
            # through to the chat template; other templates ignore it)
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        }
        t0 = time.perf_counter()
        try:
            r = self._http.post("/chat/completions", json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            content: str = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            self._fail(mode, type(e).__name__)
            raise JudgeUnavailableError(str(e)) from e
        dt = time.perf_counter() - t0
        LATENCY.labels(mode).observe(dt)
        usage = data.get("usage") or {}
        log.info(
            "llm_judge_call",
            mode=mode,
            prompt_sha=_sha(user),
            response_sha=_sha(content),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=round(dt * 1000, 1),
        )
        return content

    def _fail(self, mode: str, reason: str) -> None:
        self._failures += 1
        CALLS.labels(mode, "error").inc()
        log.warning("llm_judge_error", mode=mode, reason=reason, failures=self._failures)
        if self._failures >= self.max_failures:
            self._open_until = self.clock() + self.cooldown_s
            self._failures = 0

    def _parse[M: (WindowJudgement, CallScorecard)](
        self, mode: str, content: str, model: type[M]
    ) -> M:
        try:
            out = model.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            self._fail(mode, "invalid_json")
            raise JudgeUnavailableError("judge returned invalid JSON") from e
        self._failures = 0
        CALLS.labels(mode, "ok").inc()
        return out

    def judge_window(self, window: WindowInput) -> WindowJudgement:
        user = render_user_message(window, utterance_ids(window.segments))
        content = self._post(
            "window",
            system_prompt("window"),
            user,
            WindowJudgement.model_json_schema(),
            "WindowJudgement",
            self.window_timeout_s,
            600,
        )
        return self._parse("window", content, WindowJudgement)

    def judge_call(
        self, call_id: str, segments: list[AsrSegment], events: list[FusedEvent]
    ) -> CallScorecard:
        user = render_call_message(segments, events, utterance_ids(segments))
        content = self._post(
            "call",
            system_prompt("call"),
            user,
            CallScorecard.model_json_schema(),
            "CallScorecard",
            self.call_timeout_s,
            1200,
        )
        return self._parse("call", content, CallScorecard)
