"""One evaluator per component (docs/06 §2). Each runs the production code path (``gale_postcall``
models and worker logic) on an ``EvalSet`` and returns a JSON-able result.

Components: asr, kws, shout, ser, audiotox, textclf, llm, fusion, system.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf

from gale_audio_worker.shout import DSPShoutDetector
from gale_fusion_engine.fusion import FusionEngine
from gale_fusion_engine.rules import parse_rules
from gale_postcall.models import PipelineModels, build_models
from gale_postcall.pipeline import PostCallResult, run_postcall
from gale_text_worker.lexicon import LexiconMatcher
from galecore.audio import MODEL_SR, resample
from galecore.config import Settings, load_lexicon, load_rules, load_thresholds
from galecore.models import FixtureOracle
from galecore.models.base import AudioChunk, TextContext
from galecore.schemas import Category, Channel, EventStatus, SignalSource
from ml.eval import metrics as M
from ml.eval.sets import EvalCall, EvalSet

TEXT_CATS = [
    c.value for c in Category if c.value not in ("shouting", "harsh_tone", "interruption_overtalk")
]
ABUSIVE = {
    "slang_profanity",
    "insult",
    "threat",
    "sexual_harassment",
    "hate_identity",
    "sarcasm_mockery",
    "dismissive_rude",
}
TARGETS = {
    "agent_sev23_alerts": {"precision": 0.85, "recall": 0.75},
    "false_alerts_per_100_calls": 1.0,
    "call_spearman": 0.6,
    "call_auc": 0.85,
}


@dataclass
class Result:
    component: str
    split: str
    metrics: dict[str, Any]
    matched: M.Matched | None = None
    calibration: dict[str, Any] | None = None
    per_call: list[dict[str, Any]] = field(default_factory=list)  # bootstrap units


class Context:
    """Models per call (stub models key off the fixture oracle) and per-call audio, cached."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._audio: dict[str, dict[Channel, np.ndarray]] = {}

    def models(self, call: EvalCall) -> PipelineModels:
        oracle = (
            FixtureOracle.from_dir(call.oracle_dir)
            if call.oracle_dir and self.settings.model_mode == "stub"
            else None
        )
        return build_models(self.settings, oracle)

    def audio(self, call: EvalCall) -> dict[Channel, np.ndarray]:
        if call.call_id not in self._audio:
            x, sr = sf.read(call.audio, dtype="float32", always_2d=True)
            self._audio[call.call_id] = {
                Channel.AGENT: resample(x[:, 0], sr, MODEL_SR),
                Channel.CUSTOMER: resample(x[:, 1], sr, MODEL_SR),
            }
        return self._audio[call.call_id]


def _span(cid: str, e: Any) -> M.Span:
    return M.Span(
        cid,
        e.speaker.value,
        e.category.value,
        e.ts_start,
        e.ts_end,
        e.probability,
        e.severity,
        e.alert,
    )


# --- system / fusion ----------------------------------------------------------------------------


def _run_calls(es: EvalSet, ctx: Context) -> list[tuple[EvalCall, PostCallResult, float]]:
    out = []
    for c in es.calls:
        t0 = time.perf_counter()
        r = run_postcall(c.audio, ctx.settings, call_id=c.call_id, models=ctx.models(c))
        out.append((c, r, time.perf_counter() - t0))
    return out


def eval_fusion(es: EvalSet, ctx: Context, runs: list[Any] | None = None) -> Result:
    """Event level: live events (not dismissed by the LLM) vs alert-worthy gold; gold that should
    not alert (reported speech, self-directed…) must end up dismissed or absent."""
    runs = runs or _run_calls(es, ctx)
    pred = [
        _span(c.call_id, e)
        for c, r, _ in runs
        for e in r.events
        if e.status != EventStatus.DISMISSED_BY_LLM
    ]
    gold = [g for g in es.gold if g.alert]
    m = M.match_events(pred, gold)
    neg = [g for g in es.gold if not g.alert]
    leaked = M.match_events(pred, neg).tp
    sev_ok = sum(p.severity == g.severity for p, g in m.tp)
    per_call = [
        {
            "tp": sum(g.call_id == c.call_id for _, g in m.tp),
            "fp": sum(p.call_id == c.call_id for p in m.fp),
            "fn": sum(g.call_id == c.call_id for g in m.fn),
        }
        for c in es.calls
    ]
    cal = M.calibration([p.score for p in pred], [int(any(p is q for q, _ in m.tp)) for p in pred])
    metrics = {
        "events": M.prf(len(m.tp), len(m.fp), len(m.fn)),
        "by_category_speaker": M.prf_by(m, lambda s: f"{s.category}/{s.speaker}"),
        "severity_accuracy": round(sev_ok / len(m.tp), 4) if m.tp else None,
        "context_negatives": {"gold": len(neg), "still_live": len(leaked)},
        "ece": cal["ece"],
    }
    return Result("fusion", es.name, metrics, m, cal, per_call)


def eval_system(es: EvalSet, ctx: Context) -> Result:
    runs = _run_calls(es, ctx)
    base = eval_fusion(es, ctx, runs)
    alerts = [_span(c.call_id, e) for c, r, _ in runs for e in r.events if e.alert]
    gold = [g for g in es.gold if g.alert]
    am = M.match_events(alerts, gold)
    agent_23 = M.match_events(
        [a for a in alerts if a.speaker == "agent" and a.severity >= 2],
        [g for g in gold if g.speaker == "agent" and g.severity >= 2],
    )
    cust_abuse = M.match_events(
        [a for a in alerts if a.speaker == "customer" and a.category in ABUSIVE],
        [g for g in gold if g.speaker == "customer" and g.category in ABUSIVE],
    )
    # call level: needs-QA-review label and severity / T3 agreement
    scores, needs, sys_sev, ref = [], [], [], []
    for c, r, _ in runs:
        live = [e for e in r.events if e.status != EventStatus.DISMISSED_BY_LLM]
        scores.append(
            max((e.probability for e in live if e.speaker.value == "agent"), default=0.0)
            + 0.5 * max((e.probability for e in live), default=0.0)
        )
        gold_agent = [g for g in c.events if g.alert and g.speaker == "agent"]
        prof = (c.t3 or {}).get("professionalism")
        needs.append(int(bool(gold_agent) or (prof is not None and prof <= 2)))
        sys_sev.append(sum(e.severity for e in live))
        ref.append(-prof if prof is not None else sum(g.severity for g in c.events if g.alert))
    latency = M.latency_summary([t / max(1e-9, r.duration_s) for _, r, t in runs])
    metrics = {
        **base.metrics,
        "alerts": M.prf(len(am.tp), len(am.fp), len(am.fn)),
        "agent_sev23_alerts": M.prf(len(agent_23.tp), len(agent_23.fp), len(agent_23.fn)),
        "customer_abuse_alerts": M.prf(len(cust_abuse.tp), len(cust_abuse.fp), len(cust_abuse.fn)),
        "false_alerts_per_100_calls": round(100 * len(am.fp) / max(1, len(es.calls)), 3),
        "call_spearman": M.spearman(sys_sev, ref),
        "call_auc_needs_review": M.auc(scores, needs),
        "processing_rtf": latency,
        "stage_p95_s": {
            k: M.latency_summary([r.stages_s.get(k, 0.0) for _, r, _ in runs])["p95"]
            for k in (runs[0][1].stages_s if runs else {})
        },
        "targets": TARGETS,
    }
    return Result("system", es.name, metrics, base.matched, base.calibration, base.per_call)


# --- audio components ---------------------------------------------------------------------------


def eval_shout(es: EvalSet, ctx: Context) -> Result:
    engine = FusionEngine(
        parse_rules(load_rules(ctx.settings.config_dir)), load_thresholds(ctx.settings.config_dir)
    )
    pred: list[M.Span] = []
    for c in es.calls:
        pm = ctx.models(c)
        det = DSPShoutDetector(pm.vad, ctx.settings.model_dir)
        sigs = [
            s for ch, x in ctx.audio(c).items() for s in det.detect(AudioChunk(c.call_id, ch, x))
        ]
        pred += [
            _span(c.call_id, e)
            for e in engine.provisional(c.call_id, sigs)
            if e.category == Category.SHOUTING
        ]
    gold = [g for g in es.gold if g.category == "shouting"]
    m = M.match_events(pred, gold, M.overlap_match)
    return Result(
        "shout", es.name, {"events": M.prf(len(m.tp), len(m.fp), len(m.fn)), "target_f1": 0.85}, m
    )


def _windows(
    es: EvalSet,
    ctx: Context,
    positive: set[str],
    detect: Callable[[PipelineModels], Any],
    win: float = 3.0,
    hop: float = 1.5,
) -> tuple[list[float], list[int]]:
    scores, labels = [], []
    for c in es.calls:
        pm = ctx.models(c)
        for ch, x in ctx.audio(c).items():
            sigs = detect(pm).detect(AudioChunk(c.call_id, ch, x))
            gold = [g for g in c.events if g.speaker == ch.value and g.category in positive]
            speech = [s for s in c.segments if s.channel == ch.value]
            t = 0.0
            while t + win <= len(x) / MODEL_SR:
                if any(s.start < t + win and t < s.end for s in speech):
                    labels.append(int(any(g.start < t + win and t < g.end for g in gold)))
                    scores.append(
                        max(
                            (s.score for s in sigs if s.ts_start < t + win and t < s.ts_end),
                            default=0.0,
                        )
                    )
                t += hop
    return scores, labels


def eval_ser(es: EvalSet, ctx: Context) -> Result:
    s, y = _windows(es, ctx, {"harsh_tone", "shouting"}, lambda pm: pm.ser)
    pred = ["angry" if v >= 0.5 else "neutral" for v in s]
    gold = ["angry" if v else "neutral" for v in y]
    return Result(
        "ser",
        es.name,
        {
            "windows": len(y),
            "angry_auc": M.auc(s, y),
            "uar": M.uar(pred, gold),
            "targets": {"uar": 0.70, "angry_auc": 0.88},
        },
    )


def eval_audiotox(es: EvalSet, ctx: Context) -> Result:
    s, y = _windows(es, ctx, ABUSIVE, lambda pm: pm.audio_tox)
    return Result(
        "audiotox", es.name, {"windows": len(y), "toxic_auc": M.auc(s, y), "positives": sum(y)}
    )


def eval_kws(es: EvalSet, ctx: Context) -> Result:
    matcher = LexiconMatcher(load_lexicon(ctx.settings).entries)
    gold: list[M.Span] = []
    pred: list[M.Span] = []
    hours = 0.0
    for c in es.calls:
        for s in c.segments:
            for w, t0, t1 in s.words:
                for hit in matcher.match(w):
                    gold.append(M.Span(c.call_id, s.channel, hit.entry.id, t0, t1))
        pm = ctx.models(c)
        for ch, x in ctx.audio(c).items():
            hours += len(x) / MODEL_SR / 3600
            for sig in pm.kws.detect(AudioChunk(c.call_id, ch, x)):
                if sig.source == SignalSource.KWS:
                    pred.append(
                        M.Span(
                            c.call_id,
                            ch.value,
                            sig.evidence.term_id or "?",
                            sig.ts_start,
                            sig.ts_end,
                            sig.score,
                        )
                    )
    m = M.match_events(pred, gold)
    per_term = M.prf_by(m, lambda sp: sp.category)
    return Result(
        "kws",
        es.name,
        {
            "occurrences": len(gold),
            "recall": M.prf(len(m.tp), len(m.fp), len(m.fn))["recall"],
            "false_alarms_per_hour": round(len(m.fp) / hours, 3) if hours else None,
            "per_term": per_term,
            "targets": {"recall": 0.90, "fa_per_hour": 1.0},
        },
        m,
    )


def eval_asr(es: EvalSet, ctx: Context) -> Result:
    from gale_asr_worker.steps import transcribe

    matcher = LexiconMatcher(load_lexicon(ctx.settings).entries)
    refs, hyps, audio_s, t_total = [], [], 0.0, 0.0
    for c in es.calls:
        pm = ctx.models(c)
        audio = ctx.audio(c)
        for s in c.segments:
            x = audio[Channel(s.channel)]
            i0, i1 = int(max(0.0, s.start - 0.2) * MODEL_SR), int((s.end + 0.2) * MODEL_SR)
            t0 = time.perf_counter()
            out = transcribe(
                pm.asr,
                AudioChunk(c.call_id, Channel(s.channel), x[i0:i1], offset_s=i0 / MODEL_SR),
                c.call_id,
            )
            t_total += time.perf_counter() - t0
            audio_s += (i1 - i0) / MODEL_SR
            refs.append(s.text)
            hyps.append(" ".join(o.text_redacted for o in out))
    metrics = {
        "segments": len(refs),
        "cer": round(M.cer(refs, hyps), 4),
        "wer": round(M.wer(refs, hyps), 4),
        "slang_recall": M.slang_recall(
            refs, hyps, lambda t: [h.entry.id for h in matcher.match(t)]
        ),
        "script_drift": round(M.script_drift(hyps), 4),
        "rtf": round(t_total / max(1e-9, audio_s), 4),
        "targets": {"cer": 0.15, "wer": 0.30, "script_drift": 0.0},
    }
    return Result(
        "asr",
        es.name,
        metrics,
        per_call=[{"ref": r, "hyp": h} for r, h in zip(refs, hyps, strict=True)],
    )


def eval_textclf(es: EvalSet, ctx: Context) -> Result:
    labels: dict[str, list[int]] = {c: [] for c in TEXT_CATS}
    scores: dict[str, list[float]] = {c: [] for c in TEXT_CATS}
    for c in es.calls:
        clf = ctx.models(c).text_clf
        for s in c.segments:
            out = clf.classify(TextContext(current=s.text, speaker=Channel(s.channel)))
            hits = {
                g.category
                for g in c.events
                if g.speaker == s.channel and g.start < s.end and s.start < g.end
            }
            for cat in TEXT_CATS:
                labels[cat].append(int(cat in hits))
                scores[cat].append(out.scores.get(Category(cat), 0.0))
    per = {}
    for cat in TEXT_CATS:
        y, p = np.array(labels[cat]), np.array(scores[cat]) >= 0.5
        if y.sum():
            per[cat] = M.prf(
                int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
            )
    flat_s = [v for cat in TEXT_CATS for v in scores[cat]]
    flat_y = [v for cat in TEXT_CATS for v in labels[cat]]
    cal = M.calibration(flat_s, flat_y)
    macro = round(float(np.mean([v["f1"] for v in per.values()])), 4) if per else None
    return Result(
        "textclf",
        es.name,
        {
            "per_category": per,
            "macro_f1": macro,
            "ece": cal["ece"],
            "targets": {"macro_f1": 0.70, "slang_insult_threat_f1": 0.80},
        },
        calibration=cal,
    )


def eval_llm(es: EvalSet, ctx: Context) -> Result:
    """Judge decisions on provisional events vs gold alert-worthiness, grounding, latency."""
    runs = _run_calls(es, ctx)
    correct = total = 0
    lat: list[float] = []
    rejected: dict[str, int] = {}
    accepted = windows = 0
    for c, r, _ in runs:
        lat += r.llm_stats.get("window_latency_s", [])
        windows += r.llm_stats.get("windows", 0)
        accepted += r.llm_stats.get("accepted", 0)
        for k, v in r.llm_stats.get("rejected", {}).items():
            rejected[k] = rejected.get(k, 0) + v
        for e in r.events:
            g = next((g for g in c.events if M.docs06_match(_span(c.call_id, e), g)), None)
            if g is None or e.status == EventStatus.PROVISIONAL:
                continue
            total += 1
            correct += (e.status == EventStatus.DISMISSED_BY_LLM) == (not g.alert)
    llm_events = accepted + sum(rejected.values())
    return Result(
        "llm",
        es.name,
        {
            "windows": windows,
            "decision_accuracy": round(correct / total, 4) if total else None,
            "judged_events": total,
            "json_validity": 1.0 if all(r.llm_status == "ok" for _, r, _ in runs) else None,
            "grounding_rate": round(accepted / llm_events, 4) if llm_events else None,
            "rejected": rejected,
            "window_latency_s": M.latency_summary(lat),
            "targets": {"json_validity": 1.0, "grounding_rate": 1.0, "p95_s": 2.5},
        },
    )


EVALUATORS: dict[str, Callable[[EvalSet, Context], Result]] = {
    "asr": eval_asr,
    "kws": eval_kws,
    "shout": eval_shout,
    "ser": eval_ser,
    "audiotox": eval_audiotox,
    "textclf": eval_textclf,
    "llm": eval_llm,
    "fusion": eval_fusion,
    "system": eval_system,
}
