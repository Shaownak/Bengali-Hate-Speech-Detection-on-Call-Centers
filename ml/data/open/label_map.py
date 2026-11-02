"""Apply ``ml/textclf/label_map.yaml`` to a loader's raw fields (docs/09 §5).

Partial labels are the norm: a label the source does not annotate stays ``None`` (no loss);
an annotated label that the row does not hit is 0. Never turn "not annotated" into negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ml.data.open.schema import LABELS, RawRow

LABEL_MAP = Path(__file__).resolve().parents[2] / "textclf" / "label_map.yaml"


@dataclass
class Mapped:
    labels: dict[str, int | None]
    confidence: dict[str, float] = field(default_factory=dict)
    toxic_any: int | None = None
    severity: int | None = None
    severity_delta: int = 0
    target: str | None = None
    context_flags: list[str] = field(default_factory=list)
    weight: float = 1.0
    drop: bool = False


def _norm(v: Any) -> str:
    return str(v).strip().lower()


class LabelMapper:
    def __init__(self, path: Path = LABEL_MAP) -> None:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.weak = float(cfg["weak_confidence"])
        if tuple(cfg["labels"]) != LABELS:
            raise ValueError("label_map.yaml labels differ from the unified schema")
        self.sources: dict[str, dict[str, Any]] = {}
        for key, spec in cfg["sources"].items():
            if "same_as" in spec:
                spec = cfg["sources"][spec["same_as"]]
            unknown = set(spec.get("annotates", [])) - set(LABELS)
            if unknown:
                raise ValueError(f"label_map.yaml {key}: unknown labels {sorted(unknown)}")
            self.sources[key] = spec

    def map(self, row: RawRow) -> Mapped:
        spec = self.sources[row.source]
        annotated = set(spec.get("annotates", []))
        out = Mapped(labels={lab: (0 if lab in annotated else None) for lab in LABELS})
        hits_any = False
        for fname, table in (spec.get("fields") or {}).items():
            vals = row.fields.get(fname)
            values = vals if isinstance(vals, list) else ([] if vals is None else [vals])
            lookup = {_norm(k): v for k, v in table.items()}
            for raw in values:
                for hit in lookup.get(_norm(raw), []):
                    hits_any = True
                    self._apply(out, hit)
        out.toxic_any = self._toxic(spec.get("toxic_any"), row, hits_any)
        if out.toxic_any == 0:  # an annotated non-toxic row is negative for every annotated label
            for lab in annotated:
                out.labels[lab] = 0
        out.severity = self._lookup(spec.get("severity"), row)
        if out.severity is not None:
            out.severity = min(3, int(out.severity) + out.severity_delta)
        elif out.toxic_any == 0:
            out.severity = 0
        tgt = self._lookup(spec.get("target"), row)
        out.target = None if tgt is None else str(tgt)
        flag = self._lookup(spec.get("context_flags"), row)
        if flag:
            out.context_flags.append(str(flag))
        out.confidence = {k: v for k, v in out.confidence.items() if v < 1.0}  # only weak ones
        return out

    def _apply(self, out: Mapped, hit: Any) -> None:
        if isinstance(hit, str):
            hit = {"label": hit}
        if hit.get("drop"):
            out.drop = True
            return
        lab = hit["label"]
        out.labels[lab] = 1
        conf = float(hit.get("conf", 1.0))
        cur = out.confidence.get(lab)
        out.confidence[lab] = conf if cur is None else max(cur, conf)  # a strong hit wins
        out.weight = min(out.weight, float(hit.get("weight", 1.0)))
        out.severity_delta = max(out.severity_delta, int(hit.get("severity_delta", 0)))

    @staticmethod
    def _toxic(spec: dict[str, Any] | None, row: RawRow, hits_any: bool) -> int | None:
        if spec is None:
            return 1 if hits_any else None
        if "constant" in spec:
            return int(spec["constant"])
        if "any_of" in spec:
            pos = {_norm(p) for p in spec["positive"]}
            seen = [row.fields.get(f) for f in spec["any_of"] if row.fields.get(f) is not None]
            return None if not seen else int(any(_norm(v) in pos for v in seen))
        v = row.fields.get(spec["field"])
        if v is None or v == []:
            return None
        values = v if isinstance(v, list) else [v]
        neg = {_norm(n) for n in spec["negative"]}
        return int(any(_norm(x) not in neg for x in values))

    @staticmethod
    def _lookup(spec: dict[str, Any] | None, row: RawRow) -> Any:
        if not spec:
            return None
        v = row.fields.get(spec["field"])
        if isinstance(v, list):
            v = v[0] if v else None
        if v is None:
            return None
        table = {_norm(k): val for k, val in spec["map"].items()}
        return table.get(_norm(v))
