"""One loader per dataset (docs/09 T1–T15, R1–R4, A1–A4) → ``RawRow`` / ``AudioRow`` / pairs.

Loaders read ``<raw_root>/<key>/`` only; ``download.py`` fills it for datasets with
``download_allowed: true``, everything else is placed by hand. Column names come from the
register, so a format surprise is a config fix. Loaders never log row text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ml.data.open.readers import read_table, resolve_columns
from ml.data.open.registry import Dataset
from ml.data.open.schema import AudioRow, RawRow, Split

MULTI_SEP = re.compile(r"\s*[,;|]\s*")
SPLIT_ALIASES: dict[str, Split] = {
    "train": "train",
    "dev": "dev",
    "val": "dev",
    "valid": "dev",
    "test": "test",
}


class MissingDataError(FileNotFoundError):
    """Raw files for a dataset are not in place."""


def _files(ds: Dataset, root: Path) -> list[tuple[Split | None, Path]]:
    base = root / ds.key
    out: list[tuple[Split | None, Path]] = []
    for split, pattern in ds.files.items():
        found = sorted(p for p in base.glob(pattern) if p.is_file() and not p.name.startswith("."))
        out += [(SPLIT_ALIASES.get(split), p) for p in found]
    if not out:
        how = (
            "python -m ml.data.open download"
            if ds.download_allowed
            else f"place the files from {ds.url}"
        )
        raise MissingDataError(f"{ds.id} {ds.key}: nothing in {base} matching {ds.files}; {how}")
    return out


def _split_multi(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x).strip() for x in json.loads(s)]
        except json.JSONDecodeError:
            pass
    return [x for x in MULTI_SEP.split(s) if x]


def _text_rows(
    ds: Dataset,
    root: Path,
    *,
    fields: tuple[str, ...] = (),
    multi: tuple[str, ...] = (),
    required: tuple[str, ...] = ("text",),
    split_col: str | None = None,
) -> Iterator[RawRow]:
    for split, path in _files(ds, root):
        rows = read_table(path, ds.format if ds.format.startswith("tsv") else "auto")
        if not rows:
            continue
        cols = resolve_columns(ds.columns, list(rows[0].keys()), required, ds.key)
        for i, r in enumerate(rows):
            text = str(r.get(cols["text"]) or "").strip()
            if not text:
                continue
            sid = str(r.get(cols["id"], "") or "") if "id" in cols else ""
            row_split = split
            if split_col and split_col in cols and split is None:
                row_split = SPLIT_ALIASES.get(str(r.get(cols[split_col]) or "").lower().strip())
            got: dict[str, Any] = {}
            for f in fields:
                if f in cols:
                    v = r.get(cols[f])
                    got[f] = (
                        _split_multi(v) if f in multi else (None if v is None else str(v).strip())
                    )
            ctx = _split_multi(r.get(cols["context"])) if "context" in cols else []
            yield RawRow(
                source=ds.key,
                source_id=sid or f"{path.stem}:{i}",
                text=text,
                split=row_split,
                fields=got,
                context=ctx,
                parent_id=str(r.get(cols["parent_id"]))
                if "parent_id" in cols and r.get(cols["parent_id"])
                else None,
                dialect=str(r.get(cols["dialect"]) or "unknown").lower()
                if "dialect" in cols
                else "standard",
                text_bn=str(r.get(cols["text_bn"]))
                if "text_bn" in cols and r.get(cols["text_bn"])
                else None,
            )


# --- text, Bangla script ------------------------------------------------------------------------


def load_banglamultihate(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T1
    yield from _text_rows(
        ds, root, fields=("hate_type", "hate_severity", "to_whom"), required=("text", "hate_type")
    )


def load_bdshs(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T2
    yield from _text_rows(
        ds,
        root,
        fields=("hate", "target", "type"),
        multi=("target", "type"),
        required=("text", "hate"),
    )


def load_banglacat(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T3
    yield from _text_rows(
        ds,
        root,
        fields=("toxic", "types", "context_case"),
        multi=("types",),
        required=("text", "toxic"),
    )


def load_bidwesh(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T4: split comes from the parent
    yield from _text_rows(
        ds,
        root,
        fields=("hate", "type", "target"),
        multi=("type", "target"),
        required=("text", "parent_id", "hate"),
    )


def load_cyberbully_multilabel(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T5
    yield from _text_rows(ds, root, fields=("bully", "sexual", "religious", "threat", "spam"))


def load_violens(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T6
    yield from _text_rows(ds, root, fields=("label",), required=("text", "label"))


def load_trac2_bn(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T7 (Bangla + Roman rows)
    yield from _text_rows(
        ds, root, fields=("aggression", "gendered"), required=("text", "aggression")
    )


def load_karim_bn_hate(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T8: int class labels
    for row in _text_rows(ds, root, fields=("label",), required=("text", "label")):
        v = row.fields.get("label")
        if v is not None and v.isdigit() and ds.label_names:
            row.fields["label"] = ds.label_names[int(v)]
        yield row


def load_romim2020(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T9
    yield from _text_rows(ds, root, fields=("hate",), required=("text", "hate"))


def load_hsban(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T10
    yield from _text_rows(ds, root, fields=("hate",), required=("text", "hate"))


def load_brac_online_comments(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T11
    yield from _text_rows(ds, root, fields=("category",), required=("text", "category"))


def load_banglaharassnet(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T12
    yield from _text_rows(ds, root, fields=("category", "severity"), required=("text", "category"))


def load_bangla_toco(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T13 (with context)
    yield from _text_rows(ds, root, fields=("toxic",), required=("text", "toxic"))


def load_boishommo(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T14
    yield from _text_rows(ds, root, fields=("race", "gender", "religion", "politics"))


def load_hatealert_bengali(ds: Dataset, root: Path) -> Iterator[RawRow]:  # T15 (hydrated tweets)
    yield from _text_rows(ds, root, fields=("label",), required=("text", "label"))


# --- romanised / code-mixed ---------------------------------------------------------------------


def load_banth(
    ds: Dataset, root: Path
) -> Iterator[RawRow]:  # R1 (Roman → back-transliterated in build)
    yield from _text_rows(
        ds, root, fields=("hate", "labels"), multi=("labels",), required=("text", "hate")
    )


def load_codemixed_chaos(
    ds: Dataset, root: Path
) -> Iterator[RawRow]:  # R2: the Bangla version, one 0/1 column per class
    yield from _text_rows(
        ds,
        root,
        fields=CODEMIXED_CLASSES,
        required=("text", "non_toxic"),
    )


CODEMIXED_CLASSES = (
    "vulgar",
    "religious",
    "troll",
    "insult",
    "loathe",
    "threat",
    "race",
    "sexual",
    "political",
    "non_toxic",
)


def load_mixtox_sa(
    ds: Dataset, root: Path
) -> Iterator[RawRow]:  # R3: eval only, drop Hindi rows later
    yield from _text_rows(ds, root, fields=("label", "language"), required=("text", "label"))


def load_banglatlit(ds: Dataset, root: Path) -> Iterator[tuple[Split | None, str, str]]:  # R4
    """(split, roman, bangla) pairs for the back-transliteration model."""
    for split, path in _files(ds, root):
        rows = read_table(path)
        if not rows:
            continue
        cols = resolve_columns(ds.columns, list(rows[0].keys()), ("roman", "bangla"), ds.key)
        for r in rows:
            roman, bangla = (
                str(r.get(cols["roman"]) or "").strip(),
                str(r.get(cols["bangla"]) or "").strip(),
            )
            if roman and bangla:
                yield split, roman, bangla


# --- audio --------------------------------------------------------------------------------------


def load_audio_folders(ds: Dataset, root: Path, research_only: bool) -> Iterator[AudioRow]:  # A1–A4
    """``<key>/<label>/<speaker>_<n>.wav``: the parent folder is the term (BAAD) or class."""
    base = root / ds.key
    for _, path in _files(ds, root):
        rel = path.relative_to(base)
        speaker = path.stem.split("_")[0] if "_" in path.stem else None
        yield AudioRow(
            uid=f"{ds.key}:{rel.as_posix()}",
            source=ds.key,
            path=str(path),
            label=path.parent.name,
            speaker=speaker,
            licence_id=ds.key,
            research_only=research_only,
        )


TEXT_LOADERS: dict[str, Callable[[Dataset, Path], Iterator[RawRow]]] = {
    "banglamultihate": load_banglamultihate,
    "bdshs": load_bdshs,
    "banglacat": load_banglacat,
    "bidwesh": load_bidwesh,
    "cyberbully_multilabel": load_cyberbully_multilabel,
    "violens": load_violens,
    "trac2_bn": load_trac2_bn,
    "karim_bn_hate": load_karim_bn_hate,
    "romim2020": load_romim2020,
    "hsban": load_hsban,
    "brac_online_comments": load_brac_online_comments,
    "banglaharassnet": load_banglaharassnet,
    "bangla_toco": load_bangla_toco,
    "boishommo": load_boishommo,
    "hatealert_bengali": load_hatealert_bengali,
    "banth": load_banth,
    "codemixed_chaos": load_codemixed_chaos,
    "mixtox_sa": load_mixtox_sa,
}
