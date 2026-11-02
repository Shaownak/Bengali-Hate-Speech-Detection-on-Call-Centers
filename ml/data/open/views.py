"""Per-model training views of ``unified.jsonl`` (docs/09 §7).

- ``textclf``: text + context + partial labels + soft confidences + weight per split.
- ``llm_sft``: open-synthetic call windows in the docs/08 §4 chat format: the open-data text becomes
  the customer's utterance inside a templated exchange, rendered with the same user-message
  format the judge sees in production; the target JSON comes from the mapped labels.
  Tagged ``source=open_synthetic``. Text is redacted before it goes into a prompt (D-01).
"""

from __future__ import annotations

import json
import random
from typing import Any

from gale_llm_judge.windows import render_user_message, system_prompt, utterance_ids
from galecore.models.base import WindowInput
from galecore.redact import redact
from galecore.schemas import AsrSegment, Channel, JudgedEvent, Speaker, WindowJudgement
from ml.data.open.schema import UnifiedRow

JUDGE_CATEGORIES = (
    "slang_profanity",
    "insult",
    "threat",
    "sexual_harassment",
    "hate_identity",
    "sarcasm_mockery",
    "dismissive_rude",
)
SFT_SOURCES = {
    "banglacat",
    "banglamultihate",
    "violens",
    "trac2_bn",
    "banglaharassnet",
    "codemixed_chaos",
}
TARGETS = {
    "individual": "agent",
    "organization": "company",
    "group": "third_party",
    "society": "third_party",
}
AGENT_OPENERS = [
    "আসসালামু আলাইকুম, গ্রাহক সেবায় স্বাগতম। কীভাবে সাহায্য করতে পারি?",
    "জি, বলুন, আমি শুনছি।",
    "আপনার সমস্যাটা একটু বিস্তারিত বলবেন?",
]
AGENT_REPLIES = ["জি, আমি বিষয়টা দেখছি।", "দুঃখিত, একটু অপেক্ষা করুন।", "জি, বুঝতে পেরেছি।"]
LABEL_BN = {
    "slang_profanity": "গালি",
    "insult": "অপমান",
    "threat": "হুমকি",
    "sexual_harassment": "যৌন হয়রানি",
    "hate_identity": "পরিচয়ভিত্তিক বিদ্বেষ",
    "sarcasm_mockery": "ব্যঙ্গ",
    "dismissive_rude": "রূঢ় আচরণ",
}


def textclf_record(r: UnifiedRow) -> dict[str, Any]:
    return {
        "uid": r.uid,
        "text": r.text_bn,
        "context": r.context,
        "labels": r.labels,
        "label_confidence": r.label_confidence,
        "severity": r.severity,
        "toxic_any": r.toxic_any,
        "weight": r.weight,
        "split": r.split,
        "source": r.source,
        "research_only": r.research_only,
        "synthetic": r.synthetic,
    }


def _seg(text: str, ch: Channel, t0: float, i: int, call_id: str) -> AsrSegment:
    words_ = text.split()
    return AsrSegment(
        call_id=call_id,
        channel=ch,
        segment_id=f"{call_id}:{i}",
        ts_start=t0,
        ts_end=t0 + max(1.0, 0.4 * len(words_)),
        text_redacted=text,
        is_final=True,
        avg_conf=0.9,
        model_version="open_synthetic",
    )


def sft_example(r: UnifiedRow, rng: random.Random) -> dict[str, Any] | None:
    """One chat example, or None when the row's judge labels are unknown."""
    known = {c: r.labels.get(c) for c in JUDGE_CATEGORIES if r.labels.get(c) is not None}
    if not known and r.toxic_any != 0:
        return None
    text = redact(r.text_bn).text
    call_id = f"open:{r.uid}"
    t0 = rng.uniform(20, 300)
    segs = [_seg(rng.choice(AGENT_OPENERS), Channel.AGENT, t0, 0, call_id)]
    segs.append(_seg(text, Channel.CUSTOMER, segs[0].ts_end + rng.uniform(0.3, 1.0), 1, call_id))
    segs.append(
        _seg(
            rng.choice(AGENT_REPLIES),
            Channel.AGENT,
            segs[1].ts_end + rng.uniform(0.3, 1.0),
            2,
            call_id,
        )
    )
    window = WindowInput(call_id, segs, [], [])
    ids = utterance_ids(segs)
    sev = r.severity if r.severity else 2
    flags = [
        f
        for f in r.context_flags
        if f in ("banter", "reported_speech", "self_directed", "partial", "quoted_policy")
    ]
    events = [
        JudgedEvent(
            speaker=Speaker.CUSTOMER,
            category=c,
            severity=max(1, min(3, sev)),  # type: ignore[arg-type]
            target=TARGETS.get(r.target or ""),  # type: ignore[arg-type]
            confidence=round(r.label_confidence.get(c, 0.9), 2),
            evidence=text,
            utterance_id=ids[segs[1].segment_id],
            context_flags=flags,  # type: ignore[arg-type]
            rationale_bn=f"গ্রাহকের কথায় {LABEL_BN[c]} আছে।",
        )
        for c, v in known.items()
        if v == 1
    ]
    target = WindowJudgement(events=events, provisional_decisions=[])
    return {
        "messages": [
            {"role": "system", "content": system_prompt("window")},
            {"role": "user", "content": render_user_message(window, ids)},
            {"role": "assistant", "content": target.model_dump_json(exclude_none=True)},
        ],
        "meta": {
            "source": "open_synthetic",
            "dataset": r.source,
            "uid": r.uid,
            "split": r.split,
            "research_only": r.research_only,
            "dialect": r.dialect,
        },
    }


def llm_sft_records(rows: list[UnifiedRow], seed: int = 13) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for r in rows:
        if r.source in SFT_SOURCES and not r.synthetic:
            ex = sft_example(r, rng)
            if ex is not None:
                out.append(ex)
    return out


def dumps(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
