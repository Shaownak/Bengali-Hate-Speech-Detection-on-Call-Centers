"""Back-transliteration of Banglish (Roman) text into Bangla script (docs/09 §6.3).

``ModelTransliterator`` uses the seq2seq model trained on BanglaTLit by ``ml.translit.train``
(``$GALE_MODEL_DIR/translit``). Until that model exists, ``RuleTransliterator`` gives a rough
phonetic conversion (longest-match over common Banglish spellings) so the pipeline runs end to
end; rows converted by rules are marked so training can down-weight them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from galecore.bangla import normalize


class Transliterator(Protocol):
    name: str

    def __call__(self, texts: list[str]) -> list[str]: ...


# whole-word Banglish → Bangla for frequent words (pronouns matter for register mining)
WORDS = {
    "ami": "আমি",
    "tumi": "তুমি",
    "apni": "আপনি",
    "tui": "তুই",
    "amar": "আমার",
    "tomar": "তোমার",
    "apnar": "আপনার",
    "tor": "তোর",
    "toke": "তোকে",
    "ki": "কি",
    "keno": "কেন",
    "na": "না",
    "nai": "নাই",
    "ar": "আর",
    "ei": "এই",
    "oi": "ওই",
    "bhai": "ভাই",
    "vai": "ভাই",
    "valo": "ভালো",
    "bhalo": "ভালো",
    "kemon": "কেমন",
    "ache": "আছে",
    "ase": "আছে",
    "hobe": "হবে",
    "kotha": "কথা",
    "taka": "টাকা",
    "onek": "অনেক",
    "khub": "খুব",
    "kintu": "কিন্তু",
    "tai": "তাই",
    "ekta": "একটা",
    "kichu": "কিছু",
}
# longest first: consonant clusters, then vowels
CONS = [
    ("ksh", "ক্ষ"),
    ("chh", "ছ"),
    ("kh", "খ"),
    ("gh", "ঘ"),
    ("ng", "ং"),
    ("ch", "চ"),
    ("jh", "ঝ"),
    ("th", "থ"),
    ("dh", "ধ"),
    ("ph", "ফ"),
    ("bh", "ভ"),
    ("sh", "শ"),
    ("k", "ক"),
    ("g", "গ"),
    ("c", "চ"),
    ("j", "জ"),
    ("z", "জ"),
    ("t", "ট"),
    ("d", "ড"),
    ("n", "ন"),
    ("p", "প"),
    ("f", "ফ"),
    ("b", "ব"),
    ("v", "ভ"),
    ("m", "ম"),
    ("y", "য়"),
    ("r", "র"),
    ("l", "ল"),
    ("s", "স"),
    ("h", "হ"),
    ("q", "ক"),
    ("x", "ক্স"),
    ("w", "ও"),
]
VOWEL_SIGN = [
    ("aa", "া"),
    ("ee", "ী"),
    ("oo", "ু"),
    ("oi", "ৈ"),
    ("ou", "ৌ"),
    ("a", "া"),
    ("i", "ি"),
    ("u", "ু"),
    ("e", "ে"),
    ("o", "ো"),
]
VOWEL_FULL = [
    ("aa", "আ"),
    ("ee", "ঈ"),
    ("oo", "উ"),
    ("oi", "ঐ"),
    ("ou", "ঔ"),
    ("a", "আ"),
    ("i", "ই"),
    ("u", "উ"),
    ("e", "এ"),
    ("o", "ও"),
]
HASANTA = "্"
TOKEN = re.compile(
    r"<[A-Z][A-Z0-9_]*>|[A-Za-z]+|[^A-Za-z<]+|<"
)  # placeholder tags stay as they are


def _match(s: str, i: int, table: list[tuple[str, str]]) -> tuple[str, str] | None:
    return next(((r, b) for r, b in table if s.startswith(r, i)), None)


def rule_word(w: str) -> str:
    lw = w.lower()
    if lw in WORDS:
        return WORDS[lw]
    out, i, after_cons = [], 0, False
    while i < len(lw):
        c = _match(lw, i, CONS)
        if c:
            if after_cons:
                out.append(HASANTA)
            out.append(c[1])
            i += len(c[0])
            after_cons = True
            continue
        v = _match(lw, i, VOWEL_SIGN if after_cons else VOWEL_FULL)
        if v:
            # an inherent "o"/"a" after a consonant at word end: Banglish usually writes it out
            if after_cons and v[0] in ("a", "o") and i + 1 == len(lw):
                out.append("া" if v[0] == "a" else "ো")
            elif not (after_cons and v[0] == "o" and i + 1 < len(lw)):
                out.append(v[1])
            i += len(v[0])
            after_cons = False
            continue
        out.append(lw[i])
        i += 1
        after_cons = False
    return "".join(out)


class RuleTransliterator:
    name = "rules-0.1"

    def __call__(self, texts: list[str]) -> list[str]:
        return [
            normalize(
                "".join(
                    rule_word(t) if t[0].isalpha() and t.isascii() and not t.startswith("<") else t
                    for t in TOKEN.findall(x)
                )
            )
            for x in texts
        ]


class ModelTransliterator:  # pragma: no cover - needs torch, transformers and the trained model
    def __init__(self, model_dir: Path, device: str = "cpu", batch_size: int = 32) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = model_dir / "translit"
        self.tok: Any = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model: Any = (
            AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).eval().to(device)
        )
        self.device, self.batch_size = device, batch_size
        self.name = f"seq2seq:{path.name}"

    def __call__(self, texts: list[str]) -> list[str]:
        import torch

        out: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(
                texts[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=160, num_beams=4)
            out += [normalize(t) for t in self.tok.batch_decode(gen, skip_special_tokens=True)]
        return out


def load_transliterator(model_dir: Path | None) -> Transliterator:
    if (
        model_dir is not None and (model_dir / "translit" / "config.json").is_file()
    ):  # pragma: no cover
        return ModelTransliterator(model_dir)
    return RuleTransliterator()
