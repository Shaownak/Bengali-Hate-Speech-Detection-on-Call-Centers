"""Weekly active-learning sampler (docs/02 §6; ADR-020).

    python -m gale_api al-sample [--week 2026-W44] [--budget 300] [--push] [--out DIR]

Buckets (shares of the budget from ``configs/pilot.yaml`` → ``sampler.quotas``):

- ``false_positive``: events whose latest review says false positive / wrong category;
- ``false_negative``: abuse QA found that GAL-E missed (``missed_events``);
- ``low_confidence``: unreviewed live events with probability inside ``low_confidence_band``;
- ``disagreement``: signals disagree, e.g. SER/shout say angry but the text says neutral
  (|acoustic − lexical| ≥ ``disagreement_gap``), or the LLM dismissed a strong lexicon/KWS hit;
- ``new_terms``: tokens frequent in flagged spans, rare elsewhere in the same calls, and not in
  the active lexicon (candidate slang for the lexicon team).

Rules: golden calls are never sampled (CLAUDE.md rule 3 — the sampler refuses to run without the
golden manifest outside dev/test); an item already sent in an earlier week is skipped (``ref``);
at most ``per_call_max`` items per call; unused quota flows to the other buckets. The choice is
deterministic per week. Terms and text never reach logs or stdout, only counts.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session

from gale_api import repo
from gale_api.db import (
    AlSample,
    Call,
    EventRow,
    LexiconEntry,
    MissedEvent,
    ReviewRow,
    Segment,
    SignalRow,
)
from gale_api.pilot import week_bounds
from gale_api.pilot.report import FP
from galecore.bangla import TAG, normalize, tokenize
from galecore.labelstudio import LabelStudioClient, event_regions
from galecore.pilot import PilotConfig, SamplerConfig

BUCKETS = ("false_positive", "false_negative", "low_confidence", "disagreement", "new_terms")
ACOUSTIC = frozenset({"dsp_shout", "ser", "audio_tox"})
LEXICAL = frozenset({"lexicon", "kws", "text_clf", "register"})
CHANNEL = {"agent": "agent", "customer": "customer"}


class GoldenManifestMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Item:
    ref: str  # dedupe key across weeks
    bucket: str
    call_id: str
    ts_start: float
    ts_end: float
    speaker: str
    event_id: str | None = None
    category: str | None = None
    priority: float = 0.0  # higher first within a bucket
    reason: dict[str, Any] = field(default_factory=dict)
    term: str | None = None  # new_terms only (internal Label Studio task data, never logged)


# --- selection (pure) ---------------------------------------------------------------------------


def select_items(
    candidates: dict[str, list[Item]],
    cfg: SamplerConfig,
    seed: str,
    seen_refs: set[str] | frozenset[str] = frozenset(),
    golden_calls: set[str] | frozenset[str] = frozenset(),
    budget: int | None = None,
) -> list[Item]:
    total = cfg.budget if budget is None else budget
    # reproducible sampling, not security: the same week always yields the same sample
    rng = random.Random(hashlib.sha256(seed.encode()).hexdigest())  # noqa: S311
    pools: dict[str, list[Item]] = {}
    for b in BUCKETS:
        items = [
            i
            for i in candidates.get(b, [])
            if i.ref not in seen_refs and i.call_id not in golden_calls
        ]
        rng.shuffle(items)  # ties broken deterministically per week
        pools[b] = sorted(items, key=lambda i: -i.priority)
    per_call: Counter[str] = Counter()
    chosen: list[Item] = []
    refs: set[str] = set()

    def take(b: str, n: int) -> int:
        got = 0
        pool = pools[b]
        while pool and got < n:
            it = pool.pop(0)
            if it.ref in refs or per_call[it.call_id] >= cfg.per_call_max:
                continue
            chosen.append(it)
            refs.add(it.ref)
            per_call[it.call_id] += 1
            got += 1
        return got

    weights = {b: cfg.quotas.get(b, 0.0) for b in BUCKETS}
    norm = sum(weights.values()) or 1.0
    for b in BUCKETS:  # first pass: each bucket's share
        take(b, int(total * weights[b] / norm))
    for b in sorted(BUCKETS, key=lambda x: -weights[x]):  # leftovers flow to the other buckets
        if len(chosen) >= total:
            break
        take(b, total - len(chosen))
    return chosen


# --- candidates from the database ---------------------------------------------------------------


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def disagreement(signals: Iterable[tuple[str, float]]) -> float | None:
    """|max acoustic − max lexical| when both families fired, else None."""
    ac = [s for src, s in signals if src in ACOUSTIC]
    lx = [s for src, s in signals if src in LEXICAL]
    if not ac or not lx:
        return None
    return abs(max(ac) - max(lx))


def mine_terms(
    flagged: list[tuple[str, str]],
    background: list[tuple[str, str]],
    lexicon: set[str],
    min_count: int,
    min_calls: int,
) -> list[tuple[str, int, int, str]]:
    """Candidate slang from (call id, normalised text) spans: (term, count, calls, example call).

    A token qualifies when it appears ≥ ``min_count`` times in flagged spans across ≥ ``min_calls``
    calls, at least as often in flagged spans as in the rest of the same calls, and is neither a
    placeholder tag nor a known lexicon term.
    """
    cnt: Counter[str] = Counter()
    calls: dict[str, set[str]] = defaultdict(set)
    for cid, text in flagged:
        for t in tokenize(text):
            w = t.text
            if len(w) > 1 and not TAG.fullmatch(w) and w not in lexicon and not w.isdigit():
                cnt[w] += 1
                calls[w].add(cid)
    bg: Counter[str] = Counter(t.text for _, text in background for t in tokenize(text))
    out = [
        (w, n, len(calls[w]), sorted(calls[w])[0])
        for w, n in cnt.items()
        if n >= min_count and len(calls[w]) >= min_calls and n >= bg.get(w, 0)
    ]
    return sorted(out, key=lambda x: (-x[1], x[0]))


def _lexicon_terms(db: Session) -> set[str]:
    terms: set[str] = set()
    for term, variants, rom in db.execute(
        select(LexiconEntry.term, LexiconEntry.variants, LexiconEntry.romanized).where(
            LexiconEntry.active
        )
    ):
        for t in [term, *(variants or []), *(rom or [])]:
            terms.add(normalize(t))
    return terms


def collect_candidates(db: Session, cfg: PilotConfig, week: str) -> dict[str, list[Item]]:
    sc = cfg.sampler
    start, end = week_bounds(week, cfg.report.timezone)
    in_week = and_(Call.started_at >= start, Call.started_at < end)
    out: dict[str, list[Item]] = {b: [] for b in BUCKETS}
    events = {
        e.id: e
        for e in db.scalars(select(EventRow).join(Call, Call.id == EventRow.call_id).where(in_week))
    }
    latest: dict[str, str] = {}
    for eid, v in db.execute(
        select(ReviewRow.event_id, ReviewRow.verdict)
        .where(ReviewRow.event_id.in_(list(events)), ReviewRow.kind == "verdict")
        .ext(distinct_on(ReviewRow.event_id))
        .order_by(ReviewRow.event_id, ReviewRow.created_at.desc())
    ):
        if v:
            latest[eid] = v

    def item(bucket: str, e: EventRow, priority: float, **reason: Any) -> Item:
        return Item(
            ref=f"{bucket}:{e.id}",
            bucket=bucket,
            call_id=str(e.call_id),
            ts_start=e.ts_start,
            ts_end=e.ts_end,
            speaker=e.speaker,
            event_id=e.id,
            category=e.category,
            priority=priority,
            reason=reason,
        )

    lo, hi = sc.low_confidence_band
    for e in events.values():
        v = latest.get(e.id)
        if v in FP:
            out["false_positive"].append(item("false_positive", e, e.severity, verdict=v))
        elif v is None and e.status != "dismissed_by_llm" and lo <= e.probability <= hi:
            out["low_confidence"].append(
                item("low_confidence", e, -abs(e.probability - 0.5), probability=e.probability)
            )
    # disagreement: signals of the event's speaker overlapping its span
    sigs: dict[str, list[tuple[str, str, float, float, float]]] = defaultdict(list)
    for cid, ch, src, t0, t1, s in db.execute(
        select(
            SignalRow.call_id,
            SignalRow.channel,
            SignalRow.source,
            SignalRow.ts_start,
            SignalRow.ts_end,
            SignalRow.score,
        ).where(
            SignalRow.call_started_at >= start,
            SignalRow.call_started_at < end,
            SignalRow.shadow.is_(False),
        )
    ):
        sigs[str(cid)].append((ch, src, t0, t1, s))
    for e in events.values():
        if e.id in latest:
            continue
        near = [
            (src, s)
            for ch, src, t0, t1, s in sigs.get(str(e.call_id), [])
            if ch == e.speaker and _overlap(t0, t1, e.ts_start - 1, e.ts_end + 1)
        ]
        if e.status == "dismissed_by_llm":
            strong = max((s for src, s in near if src in ("lexicon", "kws")), default=0.0)
            if strong >= 0.8:
                out["disagreement"].append(
                    item("disagreement", e, strong, llm_dismissed_hit=strong)
                )
            continue
        gap = disagreement(near)
        if gap is not None and gap >= sc.disagreement_gap:
            out["disagreement"].append(item("disagreement", e, gap, gap=round(gap, 3)))
    # false negatives QA found in this week's calls
    for me in db.scalars(
        select(MissedEvent).join(Call, Call.id == MissedEvent.call_id).where(in_week)
    ):
        out["false_negative"].append(
            Item(
                ref=f"false_negative:{me.id}",
                bucket="false_negative",
                call_id=str(me.call_id),
                ts_start=me.ts_start,
                ts_end=me.ts_end,
                speaker=me.speaker,
                category=me.category,
                priority=me.severity,
            )
        )
    # new terms: redacted text of true-positive and missed spans vs the rest of those calls
    spans: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for e in events.values():
        if latest.get(e.id) in ("correct", "wrong_severity") or (
            e.id not in latest and e.status == "confirmed"
        ):
            spans[str(e.call_id)].append((e.speaker, e.ts_start, e.ts_end))
    for it in out["false_negative"]:
        spans[it.call_id].append((it.speaker, it.ts_start, it.ts_end))
    if spans:
        flagged: list[tuple[str, str]] = []
        background: list[tuple[str, str]] = []
        where_span: dict[tuple[str, str], tuple[float, float, str]] = {}
        for g in db.scalars(
            select(Segment).where(Segment.call_id.in_([uuid.UUID(c) for c in spans]))
        ):
            gcid = str(g.call_id)
            hit = any(
                ch == g.channel and _overlap(g.ts_start, g.ts_end, t0, t1)
                for ch, t0, t1 in spans[gcid]
            )
            (flagged if hit else background).append((gcid, g.text_norm))
            if hit:
                for t in tokenize(g.text_norm):
                    where_span.setdefault((gcid, t.text), (g.ts_start, g.ts_end, g.channel))
        terms = mine_terms(
            flagged, background, _lexicon_terms(db), sc.new_term_min_count, sc.new_term_min_calls
        )
        for term, n, n_calls, tcid in terms:
            t0, t1, ch = where_span[(tcid, term)]
            digest = hashlib.sha256(term.encode()).hexdigest()[:16]
            out["new_terms"].append(
                Item(
                    ref=f"new_terms:{digest}",
                    bucket="new_terms",
                    call_id=tcid,
                    ts_start=t0,
                    ts_end=t1,
                    speaker=ch,
                    priority=n,
                    reason={"count": n, "calls": n_calls},
                    term=term,
                )
            )
    return out


def golden_call_ids(path: Path) -> set[str]:
    """Call ids of the golden split from a data-engine manifest (JSONL with ``call_id``/``split``);
    both our call UUIDs and PBX ids may appear, so both are returned as given."""
    if not path.exists():
        raise GoldenManifestMissingError(
            f"golden manifest {path} not found: refusing to sample (CLAUDE.md rule 3)"
        )
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") == "golden":
                out.add(str(row["call_id"]))
    return out


def resolve_golden(db: Session, ids: set[str]) -> set[str]:
    """Our call UUIDs for golden ids given either as UUIDs or as external (PBX) call ids."""
    out = {i for i in ids if _is_uuid(i)}
    if ids:
        out |= {
            str(cid)
            for cid in db.scalars(select(Call.id).where(Call.external_call_id.in_(list(ids))))
        }
    return out


def _is_uuid(v: str) -> bool:
    try:
        uuid.UUID(v)
    except ValueError:
        return False
    return True


# --- Label Studio tasks -------------------------------------------------------------------------


def to_task(db: Session, it: Item, week: str, audio_prefix: str) -> dict[str, Any]:
    call = db.get(Call, uuid.UUID(it.call_id))
    audio = (call.recording_uri if call else None) or f"recordings/{it.call_id}.wav"
    near = [
        repo.event_of(e)
        for e in db.scalars(
            select(EventRow).where(
                EventRow.call_id == uuid.UUID(it.call_id),
                EventRow.ts_end >= it.ts_start - 10,
                EventRow.ts_start <= it.ts_end + 10,
            )
        )
    ]
    data: dict[str, Any] = {
        "audio": audio_prefix + audio,
        "call_id": it.call_id,
        "bucket": it.bucket,
        "focus_start": it.ts_start,
        "focus_end": it.ts_end,
        "focus_speaker": it.speaker,
        "event_id": it.event_id,
        "week": week,
    }
    if it.term is not None:
        data["candidate_term"] = it.term
    return {
        "data": data,
        "predictions": [{"model_version": "gale-pipeline", "result": event_regions(near)}]
        if near
        else [],
        "meta": {"ref": it.ref, "reason": it.reason},
    }


@dataclass
class SampleReport:
    week: str
    candidates: dict[str, int]
    selected: dict[str, int]
    golden_excluded: int
    pushed: int = 0
    file: str | None = None


def run_sampler(
    db: Session,
    cfg: PilotConfig,
    week: str,
    golden: set[str],
    *,
    budget: int | None = None,
    push: LabelStudioClient | None = None,
    out_dir: Path | None = None,
) -> SampleReport:
    cands = collect_candidates(db, cfg, week)
    seen = set(db.scalars(select(AlSample.ref)))
    gold = resolve_golden(db, golden)
    excluded = sum(1 for items in cands.values() for i in items if i.call_id in gold)
    chosen = select_items(cands, cfg.sampler, week, seen, gold, budget)
    tasks = [to_task(db, it, week, cfg.sampler.labelstudio.audio_prefix) for it in chosen]
    rep = SampleReport(
        week=week,
        candidates={b: len(v) for b, v in cands.items()},
        selected=dict(Counter(i.bucket for i in chosen)),
        golden_excluded=excluded,
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / f"al_{week}.jsonl"
        f.write_text(
            "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8"
        )
        rep.file = str(f)
    if push is not None and tasks:
        title = cfg.sampler.labelstudio.project
        pid = push.find_project(title)
        if pid is None:
            raise RuntimeError(
                f"Label Studio project {title!r} not found: create it with the T2 labelling "
                "config (`python -m ml.data ls-templates` writes T2.xml); docs/PILOT_PLAYBOOK.md"
            )
        rep.pushed = push.import_tasks(pid, tasks)
    for it in chosen if push is not None or out_dir is not None else []:  # dry run: no record
        db.add(
            AlSample(
                ref=it.ref,
                week=week,
                bucket=it.bucket,
                call_id=uuid.UUID(it.call_id),
                event_id=it.event_id,
                pushed=push is not None,
            )
        )
    return rep


def report_dict(rep: SampleReport) -> dict[str, Any]:
    return asdict(rep)
