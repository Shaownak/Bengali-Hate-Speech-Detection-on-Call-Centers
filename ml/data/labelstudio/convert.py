"""Label Studio exports → annotations → training sets for every model (docs/02 §3.1, Prompt 4).

Annotation records (one per annotator):
- ``T1Segment``: corrected transcript region. Corrected text is normalised and **redacted again**
  (annotators may type a number by mistake; D-01).
- ``T2Span``: event span on one channel with severity, target, flags, confidence, note.
- ``T3Rating``: call-level QA.

Training sets (JSONL, one row per example): ``asr``, ``kws``, ``shout``, ``ser``, ``audio_tox``,
``textclf``, ``llm_sft``, ``fusion``. Golden-split calls are refused (CLAUDE.md rule 3).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from galecore.bangla import normalize
from galecore.redact import redact
from galecore.schemas import AsrSegment, Category, Channel, JudgedEvent, Speaker, WindowJudgement
from ml.data.manifest import ChannelRow

CHANNELS = {0: "agent", 1: "customer"}
ABUSIVE = {
    "slang_profanity",
    "insult",
    "threat",
    "sexual_harassment",
    "hate_identity",
    "sarcasm_mockery",
    "dismissive_rude",
}
TEXT_CATEGORIES = [
    c.value for c in Category if c.value not in ("shouting", "harsh_tone", "interruption_overtalk")
]


class GoldenLeakError(RuntimeError):
    """A training set would include a golden-split call."""


@dataclass
class T1Segment:
    call_id: str
    channel: str
    start: float
    end: float
    text: str  # normalised + redacted
    annotator: str
    contains_slang: bool = False


@dataclass
class T2Span:
    call_id: str
    channel: str
    category: str
    start: float
    end: float
    annotator: str
    severity: int = 1
    target: str | None = None
    flags: list[str] = field(default_factory=list)
    confidence: str = "sure"
    note: str | None = None


@dataclass
class T3Rating:
    call_id: str
    annotator: str
    professionalism: int | None = None
    hostility: int | None = None
    escalated_by_agent: bool | None = None
    de_escalation: int | None = None


def _regions(result: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group result items by region id → {from_name: value}."""
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in result:
        if "id" in item:
            out[item["id"]][item["from_name"]] = item["value"]
    return out


def parse_t1(export: list[dict[str, Any]]) -> list[T1Segment]:
    out = []
    for task in export:
        d = task["data"]
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = str(ann.get("completed_by", "?"))
            slang = any(
                i.get("from_name") == "contains_slang" and i["value"].get("choices")
                for i in ann["result"]
            )
            for reg in _regions(ann["result"]).values():
                tx = reg.get("transcription")
                if not tx:
                    continue
                text = redact(normalize(" ".join(tx.get("text", [])))).text
                out.append(
                    T1Segment(
                        d["call_id"],
                        d["channel"],
                        float(tx["start"]),
                        float(tx["end"]),
                        text,
                        who,
                        slang,
                    )
                )
    return out


def parse_t2(export: list[dict[str, Any]]) -> list[T2Span]:
    out = []
    for task in export:
        cid = task["data"]["call_id"]
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = str(ann.get("completed_by", "?"))
            for reg in _regions(ann["result"]).values():
                cat = reg.get("category")
                if not cat:
                    continue
                note = reg.get("note", {}).get("text", [None])[0]
                out.append(
                    T2Span(
                        cid,
                        CHANNELS[int(cat.get("channel", 0))],
                        cat["labels"][0],
                        float(cat["start"]),
                        float(cat["end"]),
                        who,
                        int(reg.get("severity", {}).get("choices", ["1"])[0]),
                        (reg.get("target", {}).get("choices") or [None])[0],
                        list(reg.get("flags", {}).get("choices", [])),
                        (reg.get("confidence", {}).get("choices") or ["sure"])[0],
                        redact(note).text if note else None,
                    )
                )
    return out


def parse_t3(export: list[dict[str, Any]]) -> list[T3Rating]:
    out = []
    for task in export:
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            r = T3Rating(task["data"]["call_id"], str(ann.get("completed_by", "?")))
            for item in ann["result"]:
                v = item["value"]
                if item["from_name"] in ("professionalism", "hostility", "de_escalation"):
                    setattr(r, item["from_name"], int(v["rating"]))
                elif item["from_name"] == "escalated_by_agent":
                    r.escalated_by_agent = v["choices"][0] == "yes"
            out.append(r)
    return out


# --- training sets ------------------------------------------------------------------------------


def _guard(rows: list[ChannelRow]) -> dict[tuple[str, str], ChannelRow]:
    golden = sorted({r.call_id for r in rows if r.split == "golden"})
    if golden:
        raise GoldenLeakError(f"{len(golden)} golden call(s) passed to a training-set builder")
    return {(r.call_id, r.channel): r for r in rows if not r.dropped}


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def asr_set(segs: list[T1Segment], rows: list[ChannelRow]) -> list[dict[str, Any]]:
    by = _guard(rows)
    return [
        {
            "audio": by[(s.call_id, s.channel)].audio_16k_path,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "split": by[(s.call_id, s.channel)].split,
            "dialect": by[(s.call_id, s.channel)].dialect,
        }
        for s in segs
        if (s.call_id, s.channel) in by
    ]


def kws_set(segs: list[T1Segment], rows: list[ChannelRow], matcher: Any) -> list[dict[str, Any]]:
    """Utterances whose corrected text has a lexicon term (clips are cut around the word later)."""
    by = _guard(rows)
    out = []
    for s in segs:
        if (s.call_id, s.channel) not in by:
            continue
        for hit in matcher.match(s.text):
            out.append(
                {
                    "audio": by[(s.call_id, s.channel)].audio_16k_path,
                    "start": s.start,
                    "end": s.end,
                    "term_id": hit.entry.id,
                    "split": by[(s.call_id, s.channel)].split,
                }
            )
    return out


def _span_windows(
    spans: list[T2Span],
    rows: list[ChannelRow],
    positive: set[str],
    label: str,
    seed: int,
    neg_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    """Spans of ``positive`` categories plus negatives (same speakers and length, no event)."""
    by = _guard(rows)
    rng = random.Random(seed)
    out = []
    events: dict[tuple[str, str], list[T2Span]] = defaultdict(list)
    for s in spans:
        events[(s.call_id, s.channel)].append(s)
    for s in spans:
        key = (s.call_id, s.channel)
        if s.category in positive and key in by:
            out.append(
                {
                    "audio": by[key].audio_16k_path,
                    "start": s.start,
                    "end": s.end,
                    label: 1,
                    "split": by[key].split,
                }
            )
    n_neg = int(len(out) * neg_ratio)
    keys = [k for k in by if by[k].duration > 5]
    for _ in range(n_neg * 5):
        if n_neg <= 0 or not keys:
            break
        key = keys[rng.randrange(len(keys))]
        dur = min(3.0, by[key].duration / 4)
        t0 = rng.uniform(0, by[key].duration - dur)
        if any(_overlap(t0, t0 + dur, e.start, e.end) > 0 for e in events[key]):
            continue
        out.append(
            {
                "audio": by[key].audio_16k_path,
                "start": round(t0, 3),
                "end": round(t0 + dur, 3),
                label: 0,
                "split": by[key].split,
            }
        )
        n_neg -= 1
    return out


def shout_set(spans: list[T2Span], rows: list[ChannelRow], seed: int = 0) -> list[dict[str, Any]]:
    return _span_windows(spans, rows, {"shouting"}, "shouting", seed)


def ser_set(spans: list[T2Span], rows: list[ChannelRow], seed: int = 1) -> list[dict[str, Any]]:
    return _span_windows(spans, rows, {"harsh_tone", "shouting"}, "angry", seed)


def audio_tox_set(
    spans: list[T2Span], rows: list[ChannelRow], seed: int = 2
) -> list[dict[str, Any]]:
    return _span_windows(spans, rows, ABUSIVE, "toxic", seed)


def textclf_set(
    segs: list[T1Segment], spans: list[T2Span], rows: list[ChannelRow]
) -> list[dict[str, Any]]:
    """In-domain stage 3 (docs/03 §6): every utterance with all text categories annotated (T2 covers
    the whole taxonomy, so no masking), severity and target from the overlapping spans."""
    by = _guard(rows)
    out = []
    for s in segs:
        if (s.call_id, s.channel) not in by:
            continue
        hits = [
            e
            for e in spans
            if e.call_id == s.call_id
            and e.channel == s.channel
            and _overlap(s.start, s.end, e.start, e.end) > 0
            and e.category in TEXT_CATEGORIES
        ]
        labels = {c: int(any(e.category == c for e in hits)) for c in TEXT_CATEGORIES}
        out.append(
            {
                "text": s.text,
                "labels": labels,
                "severity": max((e.severity for e in hits), default=0),
                "target": hits[0].target if hits else None,
                "context_flags": sorted({f for e in hits for f in e.flags}),
                "speaker": s.channel,
                "split": by[(s.call_id, s.channel)].split,
                "source": "inhouse_t2",
            }
        )
    return out


def llm_sft_set(
    segs: list[T1Segment], spans: list[T2Span], rows: list[ChannelRow], window_s: float = 30.0
) -> list[dict[str, Any]]:
    """Gold windows (docs/08 §4): ≤ 30 s of corrected transcript ending after each gold span."""
    from gale_llm_judge.windows import render_user_message, system_prompt, utterance_ids
    from galecore.models.base import WindowInput

    by = _guard(rows)
    out = []
    calls = sorted({s.call_id for s in segs})
    for cid in calls:
        call_segs = sorted((s for s in segs if s.call_id == cid), key=lambda s: s.start)
        asr = [
            AsrSegment(
                call_id=cid,
                channel=Channel(s.channel),
                segment_id=f"{cid}:{i}",
                ts_start=s.start,
                ts_end=s.end,
                text_redacted=s.text,
                is_final=True,
                avg_conf=1.0,
                model_version="t1_gold",
            )
            for i, s in enumerate(call_segs)
        ]
        for e in (x for x in spans if x.call_id == cid and x.category in TEXT_CATEGORIES):
            end = e.end + 5
            win = [a for a in asr if a.ts_end > end - window_s and a.ts_start < end]
            if not win or (cid, e.channel) not in by:
                continue
            ids = utterance_ids(win)
            utt = next(
                (
                    a
                    for a in win
                    if a.channel.value == e.channel and a.ts_start <= e.end and e.start <= a.ts_end
                ),
                None,
            )
            if utt is None:
                continue
            gold = [
                x
                for x in spans
                if x.call_id == cid
                and x.category in TEXT_CATEGORIES
                and win[0].ts_start <= x.start
                and x.end <= end
            ]
            events = [
                JudgedEvent(
                    speaker=Speaker(x.channel),
                    category=Category(x.category),
                    severity=x.severity,  # type: ignore[arg-type]
                    target=x.target,
                    confidence=1.0 if x.confidence == "sure" else 0.6,  # type: ignore[arg-type]
                    evidence=utt.text_redacted,
                    utterance_id=ids[utt.segment_id],
                    context_flags=[f for f in x.flags if f != "neutral_register"],  # type: ignore[misc]
                    rationale_bn="মানব অ্যানোটেশন",
                )
                for x in gold
                if x.channel == utt.channel.value
            ]
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt("window")},
                        {
                            "role": "user",
                            "content": render_user_message(WindowInput(cid, win, [], []), ids),
                        },
                        {
                            "role": "assistant",
                            "content": WindowJudgement(
                                events=events, provisional_decisions=[]
                            ).model_dump_json(exclude_none=True),
                        },
                    ],
                    "meta": {
                        "source": "gold",
                        "call_id_hash": cid[:8],
                        "split": by[(cid, e.channel)].split,
                    },
                }
            )
    return out


def fusion_set(spans: list[T2Span], rows: list[ChannelRow]) -> list[dict[str, Any]]:
    """Gold events per call, to train the stacker (docs/03 §8.3) against pipeline signals."""
    by = _guard(rows)
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in spans:
        if (s.call_id, s.channel) in by:
            calls[s.call_id].append(
                {
                    "speaker": s.channel,
                    "category": s.category,
                    "start": s.start,
                    "end": s.end,
                    "severity": s.severity,
                    "flags": s.flags,
                }
            )
    split = {r.call_id: r.split for r in rows}
    return [{"call_id": c, "split": split.get(c), "events": ev} for c, ev in sorted(calls.items())]
