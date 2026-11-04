"""Call-mode SFT: whole-call QA scorecards (docs/08 §1–2 ``CallScorecard``, docs/03 §7.2).

A call joins 2–3 windows from ``ml.llm.sft`` (benign, complaint, abuse, agent behaviour…), and
the gold scorecard is derived from the call's findings by fixed rules, so it is consistent with
what the transcript and event list show:

- agent professionalism = 5 − the agent's violation severities (harsh tone counts 1), empathy
  drops for rude / dismissive / sarcastic / register behaviour, de-escalation is 5 when the agent
  stays clean while the customer is abusive;
- customer hostility = 1 + the customer's worst severity (+1 for repeated abuse); complaints
  alone stay at 1 (docs/08 call rule 5);
- the turning point is the first live (non-flagged) text finding or agent finding.

Rendered with the production ``render_call_message`` and ``call_system.md``. Written by
``python -m ml.llm.sft`` as ``sft_call_{train,dev,test}.jsonl``.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator
from typing import Any

import yaml

from gale_llm_judge.windows import render_call_message, system_prompt, utterance_ids
from galecore.schemas import (
    AgentScore,
    AsrSegment,
    CallScorecard,
    Category,
    Channel,
    CustomerScore,
    Escalation,
    EventStatus,
    FusedEvent,
    Speaker,
)
from ml.data.open.schema import UnifiedRow
from ml.llm.sft import QUEUES, TEMPLATES, TEXT_LIKE, Builder, Finding, _cycle, balanced

CALL_SCENARIOS: dict[str, float] = {  # each call joins 2–3 of these windows
    "open_negative": 0.22,
    "complaint": 0.2,
    "banter_policy": 0.1,
    "customer_abuse": 0.22,
    "agent_behaviour": 0.12,
    "agent_abuse": 0.06,
    "reported": 0.05,
    "injection": 0.03,
}
RUDE = {"dismissive_rude", "sarcasm_mockery", "disrespect_register", "insult"}


def scorecard(
    findings: list[Finding],
    segs: list[AsrSegment],
    uids: dict[str, str],
    queue: str,
    tpl: dict[str, Any],
) -> CallScorecard:
    live = sorted((f for f in findings if not f.flags), key=lambda f: segs[f.seg].ts_start)
    agent = [f for f in live if f.speaker == Speaker.AGENT]
    cust = [f for f in live if f.speaker == Speaker.CUSTOMER]
    cust_text = [f for f in cust if f.category in TEXT_LIKE]
    violations = sorted({f.category for f in agent})
    hostility = 1
    if cust:
        hostility = min(5, 1 + max(f.severity for f in cust) + int(len(cust_text) > 1))
    first = next((f for f in live if f.category in TEXT_LIKE or f.speaker == Speaker.AGENT), None)
    issue_bn, issue_en = tpl["issue_bn"][queue], tpl["issue_en"][queue]
    bn = [f"গ্রাহক {issue_bn} নিয়ে ফোন করেছেন।"]
    en = [f"The customer called about {issue_en}."]
    if cust_text:
        bn.append("গ্রাহক কথোপকথনের সময় অপমানজনক বা আপত্তিকর ভাষা ব্যবহার করেছেন।")
        en.append("The customer used abusive or offensive language during the call.")
    if agent:
        bn.append("এজেন্টের আচরণে পেশাদারিত্বের ঘাটতি ছিল।")
        en.append("The agent's conduct fell short of professional standards.")
    else:
        bn.append("এজেন্ট পুরো কলে ভদ্র ও পেশাদার আচরণ করেছেন।")
        en.append("The agent stayed polite and professional throughout.")
    coaching = [str(tpl["coaching_bn"][c]) for c in violations][:4]
    if cust_text and not agent:
        coaching.append(str(tpl["coaching_bn"]["customer_abuse"]))
    penalty = sum(1 if f.category == "harsh_tone" else f.severity for f in agent)
    return CallScorecard(
        agent=AgentScore(
            professionalism=max(1, 5 - penalty),
            empathy=4 - 2 * int(bool({f.category for f in agent} & RUDE)),
            de_escalation=2 if agent else (5 if cust_text else 4),
            respectful_register_maintained=not any(
                f.category == "disrespect_register" for f in agent
            ),
            violations=violations,
        ),
        customer=CustomerScore(
            hostility=hostility,
            abuse_toward_agent=any(f.target == "agent" for f in cust_text),
            main_issue_bn=issue_bn,
        ),
        escalation=Escalation(
            escalated=bool(agent or cust_text),
            who_escalated_first=first.speaker.value if first else "none",
            turning_point_utterance_id=uids[segs[first.seg].segment_id] if first else None,
        ),
        summary_bn=" ".join(bn),
        summary_en=" ".join(en),
        coaching_points_bn=coaching,
    )


def call_example(
    b: Builder, prow: Iterator[UnifiedRow], nrow: Iterator[UnifiedRow], queue: str
) -> dict[str, Any]:
    r = b.rng
    names, weights = zip(*CALL_SCENARIOS.items(), strict=True)
    drafts = [b.window(r.choices(names, weights)[0], prow, nrow) for _ in range(r.choice([2, 3]))]
    turns: list[tuple[Channel, str]] = []
    findings: list[Finding] = []
    for d in drafts:
        for f in d.findings:
            f.seg += len(turns)
            findings.append(f)
        turns += [(s.channel, s.text_redacted) for s in d.segs]
    call_id = str(uuid.UUID(int=r.getrandbits(128)))
    segs = b._segs(call_id, turns)
    uids = utterance_ids(segs)
    events = [
        FusedEvent(
            event_id=str(uuid.UUID(int=r.getrandbits(128))),
            call_id=call_id,
            speaker=f.speaker,
            category=Category(f.category),
            severity=f.severity,  # type: ignore[arg-type]
            probability=f.confidence,
            ts_start=segs[f.seg].ts_start,
            ts_end=segs[f.seg].ts_end,
            status=EventStatus.DISMISSED_BY_LLM if f.flags else EventStatus.CONFIRMED,
        )
        for f in findings
    ]
    card = scorecard(findings, segs, uids, queue, b.t)
    return {
        "messages": [
            {"role": "system", "content": system_prompt("call")},
            {"role": "user", "content": render_call_message(segs, events, uids)},
            {"role": "assistant", "content": card.model_dump_json()},
        ],
        "meta": {
            "source": "open_synthetic",
            "mode": "call",
            "research_only": True,
            "scenario": "+".join(str(d.meta["scenario"]) for d in drafts),
        },
    }


def build_calls(
    rows: dict[str, tuple[list[UnifiedRow], list[UnifiedRow]]],
    sizes: dict[str, int],
    seed: int = 17,
    templates: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    tpl = templates or yaml.safe_load(TEMPLATES.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for split, n in sizes.items():
        rng = random.Random(f"{seed}:call:{split}")
        pos, neg = rows.get(split, ([], []))
        if not pos or not neg:
            out[split] = []
            continue
        b = Builder(rng, tpl)
        prow, nrow = _cycle(balanced(list(pos), rng), rng), _cycle(list(neg), rng)
        out[split] = [call_example(b, prow, nrow, rng.choice(QUEUES)) for _ in range(n)]
    return out
