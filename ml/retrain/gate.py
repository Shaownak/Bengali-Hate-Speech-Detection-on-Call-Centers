"""Promotion gate: candidate vs production on the golden set, no gated metric may regress by more
than its tolerance (default 1 pt = 0.01; ``configs/promotion_gate.yaml``).

Both reports must come from the same evaluation set (``set_id``) and split. A metric the
production model has but the candidate lacks fails; a metric neither has (e.g. ``grounding_rate``
with stub models) is reported as ``n/a`` and doesn't count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from galecore.config import REPO_ROOT

Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricSpec:
    path: str
    direction: Direction
    tolerance: float


@dataclass
class Check:
    eval: str
    metric: str
    direction: str
    tolerance: float
    baseline: float | None
    candidate: float | None
    delta: float | None
    status: Literal["ok", "regressed", "missing", "n/a"]


@dataclass
class GateResult:
    passed: bool
    split: str
    set_id: str | None
    checks: list[Check] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.checks if c.status in ("regressed", "missing")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "split": self.split,
            "set_id": self.set_id,
            "regressions": self.regressions,
            "problems": self.problems,
            "checks": [asdict(c) for c in self.checks],
        }


def load_specs(component: str, path: Path | None = None) -> dict[str, list[MetricSpec]]:
    cfg = yaml.safe_load(
        (path or REPO_ROOT / "configs" / "promotion_gate.yaml").read_text(encoding="utf-8")
    )
    default = float(cfg.get("default_tolerance", 0.01))
    comp = cfg["components"].get(component)
    if comp is None:
        raise KeyError(f"no promotion gate for component {component!r}")
    out: dict[str, list[MetricSpec]] = {}
    for ev, metrics in comp["evals"].items():
        specs = []
        for name, v in metrics.items():
            d = v if isinstance(v, dict) else {"direction": v}
            specs.append(MetricSpec(name, d["direction"], float(d.get("tolerance", default))))
        out[ev] = specs
    return out


def _get(d: dict[str, Any], dotted: str) -> float | None:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return float(cur) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def compare(
    specs: dict[str, list[MetricSpec]],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> GateResult:
    """``baseline`` / ``candidate``: eval component → report payload (``ml.eval.report.write``)."""
    problems: list[str] = []
    checks: list[Check] = []
    splits = {r.get("split") for r in [*baseline.values(), *candidate.values()]}
    sets = {r.get("set_id") for r in [*baseline.values(), *candidate.values()]}
    if len(splits) > 1:
        problems.append(f"reports come from different splits: {sorted(map(str, splits))}")
    if len(sets) > 1:
        problems.append("reports come from different evaluation sets (set_id)")
    for ev, metrics in specs.items():
        b, c = baseline.get(ev), candidate.get(ev)
        if b is None or c is None:
            problems.append(f"missing {ev} report for {'baseline' if b is None else 'candidate'}")
            continue
        for m in metrics:
            bv, cv = _get(b["metrics"], m.path), _get(c["metrics"], m.path)
            if bv is None and cv is None:
                status, delta = "n/a", None
            elif cv is None:
                status, delta = "missing", None
            elif bv is None:
                status, delta = "ok", None
            else:
                delta = round(cv - bv, 6)
                worse = -delta if m.direction == "higher" else delta
                status = "regressed" if worse > m.tolerance + 1e-12 else "ok"
            checks.append(Check(ev, m.path, m.direction, m.tolerance, bv, cv, delta, status))
    passed = not problems and all(c.status in ("ok", "n/a") for c in checks)
    split = next(iter(splits)) if len(splits) == 1 else "mixed"
    set_id = next(iter(sets)) if len(sets) == 1 else None
    return GateResult(passed, str(split), set_id, checks, problems)


def to_markdown(component: str, version: str, g: GateResult) -> str:
    head = "PASSED" if g.passed else "FAILED"
    lines = [
        f"# Promotion gate: {component} {version} — {head}",
        "",
        f"Split `{g.split}`, set `{g.set_id}`. Tolerance: a gated metric may not get worse by more "
        "than its tolerance (1 pt = 0.01 by default).",
        "",
        "| eval | metric | better | production | candidate | Δ | tolerance | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in g.checks:
        lines.append(
            f"| {c.eval} | {c.metric} | {c.direction} | {c.baseline} | {c.candidate} | {c.delta} "
            f"| {c.tolerance} | {c.status} |"
        )
    lines += ["", *[f"- problem: {p}" for p in g.problems]]
    return "\n".join(lines) + "\n"
