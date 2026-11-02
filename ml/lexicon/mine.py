"""Slang-lexicon candidate mining from the harmonised open data (docs/09 §7, docs/02 §4).

Weighted log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008) between profane /
insulting rows and annotated non-toxic rows, plus PMI with the profane class. Spelling variants are
grouped by the Bangla phonetic key and edit distance ≤ 1. Output: a CSV for **human review**;
nothing enters the lexicon without it.

The CSV holds real abusive words: it is written under ``.gale/`` (git-ignored, CLAUDE.md rule 2) and
terms never go to logs or stdout, only counts.

    python -m ml.lexicon.mine [--unified .gale/data/open/unified.jsonl] [--top 500]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from galecore.bangla import TAG, phonetic_key, tokenize
from galecore.config import REPO_ROOT
from ml.data.open.schema import UnifiedRow

DATA = REPO_ROOT / ".gale" / "data" / "open"
PROFANE = ("slang_profanity", "insult")


@dataclass
class Candidate:
    term: str
    variants: list[str]
    count_profane: int
    count_clean: int
    log_odds_z: float
    pmi: float
    category: str


def _counts(
    rows: list[UnifiedRow],
) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, Counter[str]]]:
    pos: Counter[str] = Counter()
    neg: Counter[str] = Counter()
    allc: Counter[str] = Counter()
    cat: dict[str, Counter[str]] = {}
    for r in rows:
        if r.synthetic:
            continue
        toks = {
            t.text for t in tokenize(r.text_bn) if not TAG.fullmatch(t.text) and len(t.text) > 1
        }
        allc.update(toks)
        hits = [lab for lab in PROFANE if r.labels.get(lab) == 1]
        if hits:
            pos.update(toks)
            for t in toks:
                cat.setdefault(t, Counter()).update(hits)
        elif r.toxic_any == 0:
            neg.update(toks)
    return pos, neg, allc, cat


def log_odds_z(
    pos: Counter[str], neg: Counter[str], prior: Counter[str], alpha0: float = 1000.0
) -> dict[str, float]:
    """z-scores of the log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008)."""
    n1, n2, n0 = sum(pos.values()), sum(neg.values()), sum(prior.values()) or 1
    out = {}
    for w in pos:
        a = alpha0 * prior[w] / n0 + 1e-3
        y1, y2 = pos[w], neg.get(w, 0)
        d = math.log((y1 + a) / (n1 + alpha0 - y1 - a)) - math.log(
            (y2 + a) / (n2 + alpha0 - y2 - a)
        )
        var = 1 / (y1 + a) + 1 / (y2 + a)
        out[w] = d / math.sqrt(var)
    return out


def _close(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(s == t[:i] + t[i + 1 :] for i in range(len(t)))


def mine(
    rows: list[UnifiedRow], top: int = 500, min_count: int = 5, z_min: float = 3.0
) -> list[Candidate]:
    pos, neg, allc, cat = _counts(rows)
    z = log_odds_z(pos, neg, allc)
    n_pos_rows = sum(
        1 for r in rows if not r.synthetic and any(r.labels.get(lab) == 1 for lab in PROFANE)
    )
    n_rows = sum(1 for r in rows if not r.synthetic)
    ranked = sorted((w for w in z if pos[w] >= min_count and z[w] >= z_min), key=lambda w: -z[w])
    out: list[Candidate] = []
    taken: set[str] = set()
    for w in ranked:
        if w in taken:
            continue
        key = phonetic_key(w)
        variants = [
            v
            for v in ranked
            if v != w and v not in taken and (phonetic_key(v) == key or _close(v, w))
        ]
        taken.update([w, *variants])
        pmi = math.log((pos[w] / max(1, n_pos_rows)) / (allc[w] / max(1, n_rows)))
        out.append(
            Candidate(
                w,
                variants[:10],
                pos[w],
                neg.get(w, 0),
                round(z[w], 3),
                round(pmi, 3),
                cat[w].most_common(1)[0][0],
            )
        )
        if len(out) >= top:
            break
    return out


def write_csv(cands: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "candidate_id",
                "term",
                "variants",
                "phonetic_key",
                "count_profane",
                "count_clean",
                "log_odds_z",
                "pmi",
                "suggested_category",
                "review_decision",
                "severity",
                "notes",
            ]
        )
        for i, c in enumerate(cands):
            w.writerow(
                [
                    f"cand_{i:04d}",
                    c.term,
                    "|".join(c.variants),
                    phonetic_key(c.term),
                    c.count_profane,
                    c.count_clean,
                    c.log_odds_z,
                    c.pmi,
                    c.category,
                    "",
                    "",
                    "",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.lexicon.mine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--unified", type=Path, default=DATA / "unified.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "reports" / "lexicon_candidates.csv")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--min-count", type=int, default=5)
    a = ap.parse_args(argv)
    with a.unified.open(encoding="utf-8") as f:
        rows = [UnifiedRow.model_validate(json.loads(line)) for line in f]
    cands = mine(rows, a.top, a.min_count)
    write_csv(cands, a.out)
    by_cat = Counter(c.category for c in cands)
    print(
        f"{len(cands)} candidates ({sum(len(c.variants) for c in cands)} variants) "
        f"for human review: {dict(by_cat)}"
    )
    print(f"wrote {a.out} (contains real terms: keep it in the secure store, never commit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
