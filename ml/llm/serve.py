"""Local OpenAI-compatible server for the fine-tuned judge (ADR-021).

vLLM does not run on native Windows, so on the dev PC the judge is served by transformers:
base weights + a run's LoRA adapter, greedy decoding, thinking off. It speaks the subset of
``/v1/chat/completions`` that ``gale_llm_judge.client.OpenAICompatibleJudge`` uses, so the real
pipeline (``GALE_MODEL_MODE=real``, ``make report``, the streaming llm-judge) runs against it
unchanged. On the GPU box vLLM replaces it (``infra/vllm/``).

- No guided decoding: ``response_format`` is accepted, and the output is checked against the
  named schema (WindowJudgement / CallScorecard) only to count ``gale_llm_serve_invalid_total``.
  The client still post-validates and the circuit breaker handles failures.
- One generation at a time (a single 12 GB GPU); requests queue behind a lock.
- Binds 127.0.0.1 only. Logs token counts and latency, never prompt or output text (rule 1).

    .gale/venv-llm/Scripts/python -m ml.llm.serve --base .gale/models/llm/qwen3_5_4b \
        --adapter .gale/models/llm/runs/qwen3_5_4b_open_v1/adapter
    # then: GALE_LLM_WINDOW_TIMEOUT_S=60 GALE_LLM_CALL_TIMEOUT_S=600 make report …
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, ValidationError

from galecore.logging import get_logger
from galecore.schemas import CallScorecard, WindowJudgement

log = get_logger(__name__)
SCHEMAS: dict[str, type[BaseModel]] = {
    "WindowJudgement": WindowJudgement,
    "CallScorecard": CallScorecard,
}

# messages, max_new_tokens → (text, prompt_tokens, completion_tokens)
Generate = Callable[[list[dict[str, str]], int], tuple[str, int, int]]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[Message]
    max_tokens: int = 600
    temperature: float = 0.0
    response_format: dict[str, Any] | None = None


def create_app(generate: Generate, model_name: str) -> FastAPI:
    app = FastAPI(title="gal-e local judge", docs_url=None, redoc_url=None)
    lock = threading.Lock()
    stats = {"requests": 0, "invalid": 0}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_name, **stats}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        msgs = [m.model_dump() for m in req.messages]
        t0 = time.perf_counter()
        with lock:
            text, n_prompt, n_out = generate(msgs, min(req.max_tokens, 2048))
        dt = time.perf_counter() - t0
        stats["requests"] += 1
        schema = ((req.response_format or {}).get("json_schema") or {}).get("name")
        valid = None
        if schema in SCHEMAS:
            try:
                SCHEMAS[schema].model_validate_json(text)
                valid = True
            except ValidationError:
                valid = False
                stats["invalid"] += 1
        log.info(
            "llm_serve_call",
            schema=schema,
            prompt_tokens=n_prompt,
            completion_tokens=n_out,
            latency_ms=round(dt * 1000, 1),
            valid=valid,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length" if n_out >= req.max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_out,
                "total_tokens": n_prompt + n_out,
            },
        }

    return app


def hf_generator(base: str, adapter: str | None) -> Generate:  # pragma: no cover - needs a GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ml.llm.train import prompt_text

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(messages: list[dict[str, str]], max_new: int) -> tuple[str, int, int]:
        ids = tok(prompt_text(tok, messages), add_special_tokens=False, return_tensors="pt")
        ids = ids.to(model.device)
        with torch.no_grad():
            g = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        out = g[0, ids["input_ids"].shape[1] :]
        return tok.decode(out, skip_special_tokens=True), ids["input_ids"].shape[1], len(out)

    return generate


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit: base / merged model)")
    ap.add_argument("--model-name", default="gale-llm-judge")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args(argv)
    if a.adapter and not Path(a.adapter).is_dir():
        ap.error(f"adapter not found: {a.adapter}")
    app = create_app(hf_generator(a.base, a.adapter), a.model_name)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
