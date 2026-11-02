"""Leakage control, conflict resolution and splits (docs/09 §6.5–6.6).

- Duplicate groups (exact or MinHash near-duplicates, across all sources) keep one row. Its labels
  come from a vote weighted by source quality over the members that annotate each label; a tie
  becomes ``None``. If any member is in an official test split, the kept row is test.
- BIDWESH rows (dialect translations) inherit the final split of their BD-SHS parent.
- Official test splits stay test. Everything else is re-split train/dev by a stable hash of the
  uid (``dev_fraction``), so re-runs are reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ml.data.open.dedupe import duplicate_groups
from ml.data.open.schema import LABELS, UnifiedRow

# Weight of a source in duplicate-conflict votes. Papers report agreement unevenly, so default 1.0;
# sources with known weak labelling (single-annotator / distant supervision) get 0.8.
SOURCE_QUALITY: dict[str, float] = {"hsban": 0.8, "romim2020": 0.8, "brac_online_comments": 0.8}
PARENT_SOURCE = {"bidwesh": "bdshs"}


def _vote(values: Sequence[tuple[int | None, float]]) -> int | None:
    score: dict[int, float] = {}
    for v, w in values:
        if v is not None:
            score[v] = score.get(v, 0.0) + w
    if not score:
        return None
    best = max(score.values())
    winners = [v for v, s in score.items() if s == best]
    return winners[0] if len(winners) == 1 else None


def merge_group(rows: list[UnifiedRow]) -> UnifiedRow:
    keep = max(
        rows,
        key=lambda r: (SOURCE_QUALITY.get(r.source, 1.0), r.source_split == "test", -len(r.uid)),
    )
    weights = [SOURCE_QUALITY.get(r.source, 1.0) for r in rows]
    labels = {
        lab: _vote([(r.labels.get(lab), w) for r, w in zip(rows, weights, strict=True)])
        for lab in LABELS
    }
    toxic = _vote([(r.toxic_any, w) for r, w in zip(rows, weights, strict=True)])
    sev = _vote([(r.severity, w) for r, w in zip(rows, weights, strict=True)])
    conf = {
        lab: c for r in rows for lab, c in r.label_confidence.items() if labels.get(lab) is not None
    }
    return keep.model_copy(
        update={
            "labels": labels,
            "toxic_any": toxic,
            "severity": sev,
            "label_confidence": conf,
            "source_split": "test"
            if any(r.source_split == "test" for r in rows)
            else keep.source_split,
            "commercial_ok": all(r.commercial_ok for r in rows),
            "research_only": any(r.research_only for r in rows),
            "context_flags": sorted({f for r in rows for f in r.context_flags}),
        }
    )


def _hash_frac(uid: str) -> float:
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big") / 2**64


def harmonise(
    rows: list[UnifiedRow], dev_fraction: float = 0.1, threshold: float = 0.85
) -> tuple[list[UnifiedRow], dict[str, int]]:
    """Return the de-duplicated rows with final splits, plus counters for the data card."""
    stats = {
        "input": len(rows),
        "duplicate_groups": 0,
        "duplicates_removed": 0,
        "orphan_children": 0,
    }
    # BIDWESH rows are translations of BD-SHS rows: never dedupe a child against its parent
    groups = duplicate_groups([r.text_bn for r in rows], threshold)
    drop: set[int] = set()
    merged: dict[int, UnifiedRow] = {}
    for g in groups:
        members = [rows[i] for i in g]
        if (
            len({m.parent_uid or m.uid for m in members}) == 1
            and len(members) == 2
            and any(m.parent_uid for m in members)
        ):
            continue
        stats["duplicate_groups"] += 1
        stats["duplicates_removed"] += len(g) - 1
        keep = merge_group(members)
        first = min(g)
        merged[first] = keep
        drop.update(i for i in g if i != first)
    out = [merged.get(i, r) for i, r in enumerate(rows) if i not in drop]
    for r in out:
        if r.parent_uid is None:
            r.split = (
                "test"
                if r.source_split == "test"
                else ("dev" if _hash_frac(r.uid) < dev_fraction else "train")
            )
    by_uid = {r.uid: r for r in out}
    for r in out:
        if r.parent_uid is not None:
            parent = by_uid.get(r.parent_uid)
            if parent is None:
                stats["orphan_children"] += 1
                r.split = "dev" if _hash_frac(r.parent_uid) < dev_fraction else "train"
            else:
                r.split = parent.split
    return out, stats


def check_no_leakage(rows: list[UnifiedRow], threshold: float = 0.85) -> list[tuple[str, str]]:
    """(test uid, train/dev uid) pairs that are still near-duplicates. Must be empty."""
    groups = duplicate_groups([r.text_bn for r in rows], threshold)
    bad = []
    for g in groups:
        tests = [rows[i] for i in g if rows[i].split == "test"]
        others = [rows[i] for i in g if rows[i].split != "test"]
        bad += [
            (t.uid, o.uid)
            for t in tests
            for o in others
            if o.parent_uid != t.uid and t.parent_uid != o.uid
        ]
    return bad
