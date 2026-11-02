"""Speech-style adaptation (docs/09 §6.7): closing the social-media → phone-call gap.

- ``AsrNoise``: character confusions typical of Bangla ASR (phonetic merges, dropped hasanta,
  split/merged words), optionally fitted from (reference, hypothesis) pairs.
- ``round_trip``: TTS → telephony codec → ASR on a subset (needs real TTS + ASR; hook only here).
- ``rewrite_spoken``: second-person, call-style rewrite by a **self-hosted** LLM
  (OpenAI-compatible);
  outputs are tagged synthetic. Never an external API (CLAUDE.md rule 1).
"""

from __future__ import annotations

import difflib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

import httpx

from galecore.bangla import normalize

DEFAULT_CONFUSIONS: dict[str, list[tuple[str, float]]] = {
    "শ": [("স", 0.5), ("ষ", 0.2)],
    "ষ": [("শ", 0.5), ("স", 0.3)],
    "স": [("শ", 0.3)],
    "য": [("জ", 0.5)],
    "জ": [("য", 0.2)],
    "ণ": [("ন", 0.6)],
    "ন": [("ণ", 0.1)],
    "ী": [("ি", 0.5)],
    "ি": [("ী", 0.2)],
    "ূ": [("ু", 0.5)],
    "ু": [("ূ", 0.2)],
    "ড়": [("র", 0.4)],
    "ঢ়": [("র", 0.4)],
    "র": [("ড়", 0.1)],
    "্": [("", 0.15)],
    "ঁ": [("", 0.5)],
}


class AsrNoise:
    def __init__(
        self,
        confusions: dict[str, list[tuple[str, float]]] | None = None,
        char_rate: float = 0.08,
        word_drop: float = 0.03,
        word_merge: float = 0.03,
    ) -> None:
        self.conf = confusions or DEFAULT_CONFUSIONS
        self.char_rate, self.word_drop, self.word_merge = char_rate, word_drop, word_merge

    @classmethod
    def fit(cls, pairs: Iterable[tuple[str, str]], min_count: int = 3) -> AsrNoise:
        """Character substitutions from aligned (reference, ASR hypothesis) pairs."""
        subs: dict[str, Counter[str]] = defaultdict(Counter)
        totals: Counter[str] = Counter()
        for ref, hyp in pairs:
            ref, hyp = normalize(ref), normalize(hyp)
            totals.update(ref)
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, ref, hyp, autojunk=False
            ).get_opcodes():
                if op == "replace" and i2 - i1 == j2 - j1:
                    for a, b in zip(ref[i1:i2], hyp[j1:j2], strict=True):
                        subs[a][b] += 1
                elif op == "delete":
                    for a in ref[i1:i2]:
                        subs[a][""] += 1
        conf = {
            a: [(b, n / totals[a]) for b, n in c.most_common(3) if n >= min_count]
            for a, c in subs.items()
            if totals[a]
        }
        return cls({a: v for a, v in conf.items() if v} or DEFAULT_CONFUSIONS)

    def __call__(self, text: str, rng: random.Random) -> str:
        words = []
        for w in text.split():
            if rng.random() < self.word_drop:
                continue
            chars = []
            for ch in w:
                opts = self.conf.get(ch)
                if opts and rng.random() < self.char_rate * 4:
                    r, acc = rng.random(), 0.0
                    for rep, p in opts:
                        acc += p
                        if r < acc:
                            ch = rep
                            break
                chars.append(ch)
            words.append("".join(chars))
        out: list[str] = []
        for w in words:
            if out and rng.random() < self.word_merge:
                out[-1] += w
            else:
                out.append(w)
        return " ".join(out)


def round_trip(
    texts: list[str], tts: Callable[[str], object], asr: Callable[[object], str]
) -> list[str]:  # pragma: no cover
    """TTS → telephony codec → ASR (``tts`` returns codec-simulated 8 kHz audio)."""
    return [normalize(asr(tts(t))) for t in texts]


REWRITE_PROMPT = (
    "Rewrite the comment as one sentence a caller would say on a customer-care phone call, in "
    "colloquial Bangla, second person. Keep the meaning, insults, threats and tone exactly; do not "
    "soften or add anything. Output only the sentence."
)


def rewrite_spoken(
    texts: list[str],
    base_url: str,
    model: str,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 30.0,
) -> list[str | None]:
    """Call-style rewrite with our own vLLM server. ``None`` where the server fails."""
    out: list[str | None] = []
    with httpx.Client(base_url=base_url.rstrip("/"), transport=transport, timeout=timeout) as http:
        for t in texts:
            body = {
                "model": model,
                "temperature": 0.3,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": t},
                ],
            }
            try:
                r = http.post("/chat/completions", json=body)
                r.raise_for_status()
                out.append(normalize(r.json()["choices"][0]["message"]["content"]))
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError):
                out.append(None)
    return out
