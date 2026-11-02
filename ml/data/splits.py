"""Splits by call AND speaker, and the golden-set lock (docs/02 §2.7, §3.2; CLAUDE.md rule 3).

Calls that share an agent or a customer (hash) are connected; each connected component goes to one
split, so no speaker appears in two splits. Components are assigned greedily, largest first, to the
split furthest below its target share within its (dialect, queue) stratum.

Golden lock: ``golden.lock`` holds the sha256 of the sorted golden call ids and speaker ids plus
their counts. ``check_golden`` fails when the lock no longer matches, or when any golden call or
speaker appears in another split. CI runs it on every manifest in the repo (synthetic in CI).

With a small agent pool, strict agent holdout can starve training (Q-02, approved as specced).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ml.data.manifest import ChannelRow

DEFAULT_SHARES = {"train": 0.8, "dev": 0.1, "test": 0.1}


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.p[self.find(a)] = self.find(b)


def speakers(r: ChannelRow) -> list[str]:
    return [
        s
        for s in (
            f"agent:{r.agent_id}" if r.agent_id else None,
            f"customer:{r.customer_hash}" if r.customer_hash else None,
        )
        if s
    ]


def components(rows: list[ChannelRow]) -> list[list[str]]:
    uf = _UF()
    for r in rows:
        uf.find(f"call:{r.call_id}")
        for s in speakers(r):
            uf.union(f"call:{r.call_id}", s)
    groups: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        groups[uf.find(f"call:{r.call_id}")].add(r.call_id)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def assign_splits(
    rows: list[ChannelRow],
    shares: dict[str, float] | None = None,
    fixed: dict[str, str] | None = None,
) -> dict[str, str]:
    """call_id → split. ``fixed`` pins calls (e.g. golden) and pulls their components along."""
    shares = shares or DEFAULT_SHARES
    fixed = fixed or {}
    stratum = {r.call_id: (r.dialect or "unknown", r.queue or "unknown") for r in rows}
    out: dict[str, str] = {}
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: dict.fromkeys(shares, 0))
    for comp in components(rows):
        pinned = {fixed[c] for c in comp if c in fixed}
        if len(pinned) > 1:
            raise ValueError(f"calls pinned to different splits share a speaker: {sorted(pinned)}")
        strat = stratum[comp[0]]
        if pinned:
            split = pinned.pop()
        else:
            total = sum(counts[strat].values()) + len(comp)
            split = min(shares, key=lambda s: (counts[strat][s] + len(comp)) / total - shares[s])
        for c in comp:
            out[c] = split
        if split in counts[strat]:
            counts[strat][split] += len(comp)
    return out


@dataclass(frozen=True)
class GoldenLock:
    calls: int
    speakers: int
    calls_sha256: str
    speakers_sha256: str


def _sha(items: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(items))).encode()).hexdigest()


def golden_lock(rows: list[ChannelRow]) -> GoldenLock:
    gold = [r for r in rows if r.split == "golden"]
    calls = {r.call_id for r in gold}
    spk = {s for r in gold for s in speakers(r)}
    return GoldenLock(len(calls), len(spk), _sha(calls), _sha(spk))


def write_lock(rows: list[ChannelRow], path: Path) -> GoldenLock:
    lock = golden_lock(rows)
    path.write_text(json.dumps(lock.__dict__, indent=2) + "\n", encoding="utf-8")
    return lock


def check_golden(rows: list[ChannelRow], lock_path: Path | None = None) -> list[str]:
    """Problems (empty = OK): golden calls/speakers in other splits, or lock drift."""
    problems = []
    gold = [r for r in rows if r.split == "golden"]
    other = [r for r in rows if r.split not in ("golden", None)]
    g_calls, o_calls = {r.call_id for r in gold}, {r.call_id for r in other}
    g_spk, o_spk = {s for r in gold for s in speakers(r)}, {s for r in other for s in speakers(r)}
    if g_calls & o_calls:
        problems.append(f"{len(g_calls & o_calls)} golden call(s) also in train/dev/test")
    if g_spk & o_spk:
        problems.append(f"{len(g_spk & o_spk)} golden speaker(s) also in train/dev/test")
    if lock_path is not None and lock_path.exists():
        want = json.loads(lock_path.read_text(encoding="utf-8"))
        if want != golden_lock(rows).__dict__:
            problems.append(
                "golden set differs from golden.lock (re-lock only with QA-lead sign-off)"
            )
    return problems
