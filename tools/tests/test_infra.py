"""Static checks on infra/ that run without helm or a cluster (CI also runs `make helm-check`,
promtool and the kind smoke deploy)."""

import re
from pathlib import Path

import pytest
import yaml

from tools import gen_dashboards

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "infra" / "helm" / "gal-e"
METRIC_DEF = re.compile(r'(?:Counter|Histogram|Gauge)\(\s*"(gale_[a-z0-9_]+)"', re.S)
METRIC_USE = re.compile(r"\b(gale_[a-z0-9_]+)\b")
SUFFIXES = ("_bucket", "_sum", "_count", "_total")


def defined_metrics() -> set[str]:
    names: set[str] = set()
    for base in ("packages", "services"):
        for f in (ROOT / base).rglob("*.py"):
            if "tests" in f.parts:
                continue
            names |= set(METRIC_DEF.findall(f.read_text(encoding="utf-8")))
    return names


def base_name(series: str) -> str:
    for s in SUFFIXES:
        if series.endswith(s):
            return series[: -len(s)]
    return series


def used_metrics(text: str) -> set[str]:
    # `gale:...` recording rules are not raw series; label names like gale_role never appear
    return {base_name(m) for m in METRIC_USE.findall(text)}


def test_alert_rules_and_dashboards_reference_real_metrics() -> None:
    defined = {base_name(n) for n in defined_metrics()}
    rules = (CHART / "files" / "prometheus-rules.yaml").read_text(encoding="utf-8")
    groups = yaml.safe_load(rules)["groups"]
    alerts = [r["alert"] for g in groups for r in g["rules"] if "alert" in r]
    # docs/05 §4: bus lag, alert latency, GPU OOM, error rate, volume anomaly, FP rate, ASR drop
    for needed in (
        "GaleBusLagHigh",
        "GaleAlertLatencyHigh",
        "GaleGpuOOM",
        "GaleErrorRateHigh",
        "GaleAlertVolumeAnomaly",
        "GaleFalsePositiveRateHigh",
        "GaleAsrConfidenceDrop",
    ):
        assert needed in alerts
    for g in groups:
        for r in g["rules"]:
            if "alert" in r:
                assert r["labels"]["severity"] in ("critical", "warning")
    missing = used_metrics(rules) - defined
    assert not missing, f"rules use undefined metrics: {missing}"
    for f in (CHART / "files" / "dashboards").glob("*.json"):
        missing = used_metrics(f.read_text(encoding="utf-8")) - defined
        assert not missing, f"{f.name} uses undefined metrics: {missing}"


def test_dashboards_match_generator() -> None:
    for name, d in gen_dashboards.DASHBOARDS.items():
        on_disk = (CHART / "files" / "dashboards" / name).read_text(encoding="utf-8")
        assert on_disk == gen_dashboards.render(d), f"run `make dashboards` ({name})"


@pytest.mark.parametrize(
    "values", ["values.yaml", "values-staging.yaml", "values-prod.yaml", "values-kind.yaml"]
)
def test_values_files(values: str) -> None:
    base = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    v = yaml.safe_load((CHART / values).read_text(encoding="utf-8"))
    assert set(v.get("services", {})) <= set(base["services"])
    for name, svc in v.get("services", {}).items():
        assert int(svc.get("shards", 0)) <= 8, name
    text = (CHART / values).read_text(encoding="utf-8").lower()
    # secrets never live in values (External Secrets / SOPS)
    assert not re.search(r"(password|secret_key|access_key)\s*:\s*\S", text)


def test_images_pinned_by_digest() -> None:
    v = (CHART / "values.yaml").read_text(encoding="utf-8")
    images = re.findall(r"image:\s*(\S+)", v)
    assert images
    for img in images:
        assert "@sha256:" in img, img
    for df in (ROOT / "infra" / "docker").glob("*.Dockerfile"):
        for line in df.read_text(encoding="utf-8").splitlines():
            if line.startswith("FROM "):
                assert "@sha256:" in line, (df.name, line)


def test_templates_are_balanced() -> None:
    for f in (CHART / "templates").glob("*"):
        t = f.read_text(encoding="utf-8")
        assert "\t" not in t, f.name
        assert t.count("{{") == t.count("}}"), f.name


def test_otel_collector_config() -> None:
    cfg = yaml.safe_load((CHART / "files" / "otel-collector.yaml").read_text(encoding="utf-8"))
    assert cfg["service"]["pipelines"]["traces"]["receivers"] == ["otlp"]
    assert "attributes/scrub" in cfg["service"]["pipelines"]["traces"]["processors"]


def _merge(a: dict, b: dict) -> dict:  # type: ignore[type-arg]
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@pytest.mark.parametrize(
    "values", ["values.yaml", "values-staging.yaml", "values-prod.yaml", "values-kind.yaml"]
)
def test_pilot_values(values: str) -> None:
    """The chart's pilot policy is a valid PilotConfig in every environment (ADR-020)."""
    from galecore.pilot import COMPONENTS, PilotConfig

    base = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    v = _merge(base, yaml.safe_load((CHART / values).read_text(encoding="utf-8")) or {})
    cfg = PilotConfig.model_validate(v["pilot"]["config"])
    assert set(v["retrain"]["components"]) <= set(COMPONENTS)
    gate = yaml.safe_load((ROOT / "configs" / "promotion_gate.yaml").read_text(encoding="utf-8"))
    assert set(v["retrain"]["components"]) <= set(gate["components"])
    if values != "values-kind.yaml":  # real clusters start in shadow mode
        assert cfg.alerts.default == "hidden"
        assert cfg.sampler.golden_manifest.startswith("/etc/gale/golden/")


def test_pilot_jobs_never_skip_the_golden_check() -> None:
    for name in ("pilot.yaml", "retrain.yaml"):
        t = (CHART / "templates" / name).read_text(encoding="utf-8")
        assert "--no-golden-manifest" not in t, name
    retrain = (CHART / "templates" / "retrain.yaml").read_text(encoding="utf-8")
    assert "--golden-manifest /etc/gale/golden/calls.jsonl" in retrain
    assert "register-model" in retrain  # candidates are registered as shadow, never activated


def test_trainer_image_built_in_ci() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "infra/docker/trainer.Dockerfile" in ci
