"""Exact + near-duplicate detection across sources (docs/09 §6.5): MinHash over character
5-grams with LSH banding; candidate pairs are confirmed with the exact Jaccard ≥ threshold."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

SHINGLE = 5
NUM_PERM = 128
BANDS = 32  # 32 bands × 4 rows: candidates from Jaccard ≈ 0.4 up; the exact check decides
_MASK = (1 << 61) - 1


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    t = " ".join(text.split())
    return {t[i : i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class MinHasher:
    def __init__(self, num_perm: int = NUM_PERM, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MASK, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MASK, num_perm, dtype=np.uint64)

    def signature(self, sh: set[str]) -> np.ndarray:
        hs = np.array(
            [
                int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
                & _MASK
                for s in sh
            ],
            dtype=np.uint64,
        )
        # (a·h + b) mod 2^61-1 per permutation, min over shingles (uint64 wrap is fine for hashing)
        return ((np.outer(self.a, hs) + self.b[:, None]) % np.uint64(_MASK)).min(axis=1)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


def duplicate_groups(texts: Sequence[str], threshold: float = 0.85) -> list[list[int]]:
    """Groups (indices, ≥ 2 members) of exact or near-duplicate texts."""
    uf = UnionFind(len(texts))
    exact: dict[str, int] = {}
    sh = [shingles(t) for t in texts]
    for i, t in enumerate(texts):
        key = " ".join(t.split())
        if key in exact:
            uf.union(exact[key], i)
        else:
            exact[key] = i
    mh = MinHasher()
    rows = NUM_PERM // BANDS
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for i in exact.values():
        sig = mh.signature(sh[i])
        for band in range(BANDS):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(i)
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if uf.find(i) != uf.find(j) and jaccard(sh[i], sh[j]) >= threshold:
                    uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        groups[uf.find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]
