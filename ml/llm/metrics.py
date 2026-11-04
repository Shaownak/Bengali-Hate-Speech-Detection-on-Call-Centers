"""LLM-judge metrics on SFT-format windows (docs/03 §7.4, docs/08 §6).

Pure Python so CI can test it without torch. A prediction is the raw model output text; it is
scored against the gold assistant message and the user message it answered:

- ``json_validity``: the output parses as a ``WindowJudgement`` (target 100 %);
- event P/R/F1 (micro, and per category for macro-F1), matching on (speaker, category,
  utterance_id) like the docs/06 event rule restricted to one window;
- ``decision_accuracy``: provisional decisions with the right verb (and new category/severity);
- ``hallucinated_evidence_rate``: predicted events whose evidence is not a substring of the
  referenced utterance (up to Unicode form, as the production grounding check), or whose speaker
  does not match it (tolerance 0);
- ``severity_accuracy`` on matched events, and the clean-window false-positive rate.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from galecore.bangla import find_canonical
from galecore.schemas import WindowJudgement

_LINE = re.compile(r"^(u_\d+) \| (AGENT|CUSTOMER|MIXED) \| [\d.]+-[\d.]+ \| (.*)$")


def transcript(user_message: str) -> dict[str, tuple[str, str]]:
    """utterance_id → (speaker, text) from a docs/08 §3 user message."""
    out = {}
    for line in user_message.splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = (m.group(2).lower(), m.group(3))
    return out


def parse(text: str) -> WindowJudgement | None:
    """The model output as a judgement; tolerates a ```json fence, nothing else."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    try:
        return WindowJudgement.model_validate_json(s)
    except (ValidationError, ValueError):
        return None


@dataclass
class Score:
    valid: bool
    tp: Counter[str] = field(default_factory=Counter)
    fp: Counter[str] = field(default_factory=Counter)
    fn: Counter[str] = field(default_factory=Counter)
    decisions: int = 0
    decisions_ok: int = 0
    pred_events: int = 0
    ungrounded: int = 0
    sev_pairs: int = 0
    sev_ok: int = 0
    gold_empty: bool = False
    pred_nonempty: bool = False


def score(pred_text: str, gold_text: str, user_message: str) -> Score:
    gold = WindowJudgement.model_validate_json(gold_text)
    pred = parse(pred_text)
    s = Score(valid=pred is not None, gold_empty=not gold.events)
    gold_ev = {(e.speaker.value, e.category.value, e.utterance_id): e for e in gold.events}
    gold_dec = {d.event_id: d for d in gold.provisional_decisions}
    s.decisions = len(gold_dec)
    if pred is None:
        for k in gold_ev:
            s.fn[k[1]] += 1
        return s
    utts = transcript(user_message)
    s.pred_nonempty = bool(pred.events)
    s.pred_events = len(pred.events)
    seen = set()
    for e in pred.events:
        u = utts.get(e.utterance_id or "")
        if u is None or u[0] != e.speaker.value or find_canonical(u[1], e.evidence) is None:
            s.ungrounded += 1
        k = (e.speaker.value, e.category.value, e.utterance_id)
        if k in seen:
            continue
        seen.add(k)
        if k in gold_ev:
            s.tp[k[1]] += 1
            s.sev_pairs += 1
            s.sev_ok += int(e.severity == gold_ev[k].severity)
        else:
            s.fp[k[1]] += 1
    for k in gold_ev.keys() - seen:
        s.fn[k[1]] += 1
    for d in pred.provisional_decisions:
        g = gold_dec.get(d.event_id)
        if g is None or d.decision != g.decision:
            continue
        if d.decision == "change_category" and d.new_category != g.new_category:
            continue
        if d.decision == "change_severity" and d.new_severity != g.new_severity:
            continue
        s.decisions_ok += 1
    return s


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(scores: list[Score]) -> dict[str, Any]:
    n = len(scores)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for s in scores:
        tp += s.tp
        fp += s.fp
        fn += s.fn
    p, r, f = _f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_cat = {}
    for c in sorted(set(tp) | set(fp) | set(fn)):
        cp, cr, cf = _f1(tp[c], fp[c], fn[c])
        per_cat[c] = {
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "f1": round(cf, 4),
            "support": tp[c] + fn[c],
        }
    supported = [v["f1"] for v in per_cat.values() if v["support"]]
    dec = sum(s.decisions for s in scores)
    pred_ev = sum(s.pred_events for s in scores)
    sev = sum(s.sev_pairs for s in scores)
    clean = [s for s in scores if s.gold_empty]
    return {
        "n": n,
        "json_validity": round(sum(s.valid for s in scores) / n, 4) if n else None,
        "event_precision": round(p, 4),
        "event_recall": round(r, 4),
        "event_f1": round(f, 4),
        "event_macro_f1": round(sum(supported) / len(supported), 4) if supported else None,
        "decision_accuracy": round(sum(s.decisions_ok for s in scores) / dec, 4) if dec else None,
        "hallucinated_evidence_rate": (
            round(sum(s.ungrounded for s in scores) / pred_ev, 4) if pred_ev else 0.0
        ),
        "severity_accuracy": round(sum(s.sev_ok for s in scores) / sev, 4) if sev else None,
        "clean_window_fp_rate": (
            round(sum(s.pred_nonempty for s in clean) / len(clean), 4) if clean else None
        ),
        "per_category": per_cat,
    }


def evaluate(records: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    """Overall metrics plus a per-scenario breakdown (injection robustness is one of them)."""
    scores = [
        score(o, r["messages"][2]["content"], r["messages"][1]["content"])
        for r, o in zip(records, outputs, strict=True)
    ]
    out = aggregate(scores)
    by: dict[str, list[Score]] = {}
    for r, s in zip(records, scores, strict=True):
        by.setdefault(r.get("meta", {}).get("scenario", "?"), []).append(s)
    out["by_scenario"] = {
        k: {
            m: v[m]
            for m in ("n", "json_validity", "event_f1", "decision_accuracy", "clean_window_fp_rate")
        }
        for k, v in ((k, aggregate(v)) for k, v in sorted(by.items()))
    }
    return out


def dumps(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
