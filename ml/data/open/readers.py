"""Table readers for the raw dataset formats (csv, tsv, jsonl, json, xlsx, parquet) and column
resolution against the register's ``columns`` candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

csv.field_size_limit(1 << 30)


class ColumnError(ValueError):
    """The raw file does not have the columns the register expects."""


def read_table(path: Path, fmt: str = "auto") -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if fmt == "tsv-unquoted":  # BLP-2025 TSVs contain bare quotes inside comments
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if suffix in (".csv", ".tsv", ".txt"):
        with path.open(encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if suffix == ".tsv" or sample.count("\t") > sample.count(",") else ","
            return list(csv.DictReader(f, delimiter=delim))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data", [])
    if suffix in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook

        ws = load_workbook(path, read_only=True, data_only=True).worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        # some releases put a one-cell title row above the header (BanglaHarassNet)
        while len(rows) > 1 and sum(v is not None for v in rows[0]) < 2:
            rows = rows[1:]
        header = (
            [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(rows[0])]
            if rows
            else []
        )
        return [
            dict(zip(header, r, strict=False)) for r in rows[1:] if any(v is not None for v in r)
        ]
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_table(path).to_pylist())
    raise ValueError(f"unsupported file type: {path.name}")


def resolve_columns(
    spec: dict[str, str | list[str]], header: list[str], required: tuple[str, ...], source: str
) -> dict[str, str]:
    """Logical field → actual column (case-insensitive; candidates tried in order)."""
    lower = {h.lower().strip(): h for h in header}
    out: dict[str, str] = {}
    for name, cands in spec.items():
        for c in [cands] if isinstance(cands, str) else cands:
            if c.lower() in lower:
                out[name] = lower[c.lower()]
                break
    missing = [r for r in required if r not in out]
    if missing:
        raise ColumnError(
            f"{source}: no column for {missing}; file has {header}. "
            "Update `columns` for this dataset in ml/data/datasets.yaml."
        )
    return out
