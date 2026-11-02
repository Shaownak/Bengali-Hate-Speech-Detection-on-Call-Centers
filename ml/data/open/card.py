"""Data card for the harmonised open data (docs/09 §6.9): counts only, never text."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from ml.data.open.registry import Register
from ml.data.open.schema import LABELS, UnifiedRow


def _table(head: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def data_card(rows: list[UnifiedRow], reg: Register, stats: dict[str, Any], run_tag: str) -> str:
    by_src = Counter(r.source for r in rows)
    split_src = Counter((r.source, r.split) for r in rows)
    src_rows = []
    for d in reg.datasets:
        if d.kind != "text":
            continue
        n = by_src.get(d.key, 0)
        status = (
            "used" if n else ("missing" if d.key in stats.get("missing", []) else "not selected")
        )
        src_rows.append(
            [
                d.id,
                d.key,
                n,
                split_src.get((d.key, "train"), 0),
                split_src.get((d.key, "dev"), 0),
                split_src.get((d.key, "test"), 0),
                d.commercial_ok,
                status,
            ]
        )
    lab_rows = []
    for lab in LABELS:
        pos = sum(1 for r in rows if r.labels.get(lab) == 1)
        neg = sum(1 for r in rows if r.labels.get(lab) == 0)
        weak = sum(1 for r in rows if lab in r.label_confidence)
        lab_rows.append([lab, pos, neg, len(rows) - pos - neg, weak])
    scripts = Counter(r.script_orig for r in rows)
    dialects = Counter(r.dialect for r in rows)
    research = sum(r.research_only for r in rows)
    return f"""# Open-data card

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by `python -m ml.data.open build --run-tag {run_tag}`.
Counts only; no text. Labels follow `ml/textclf/label_map.yaml`; "not annotated" means the source does not
label that category (masked in training, never treated as negative).

**Licence:** {research} of {len(rows)} rows come from datasets that are not `commercial_ok: true`, so every model
trained on this build is **research_only** and cannot be promoted (docs/09, ADR-012).

## Sources
{_table(["id", "key", "rows", "train", "dev", "test", "commercial_ok", "status"], src_rows)}

## Labels
{_table(["label", "positive", "negative", "not annotated", "weak (soft)"], lab_rows)}

## Processing
{_table(["step", "count"], [[k, v] for k, v in stats.items() if not isinstance(v, list)])}

Scripts: {dict(scripts)} · dialects: {dict(dialects)} · synthetic (augmented) rows: {sum(r.synthetic for r in rows)}

## Known limitations (docs/09 §9)
Social-media comments, not customer-care dialogue; almost no agent-side behaviour (register, dismissive
rudeness, interruptions); no prosody; dialects limited to BIDWESH; label definitions differ across sources.
"""
