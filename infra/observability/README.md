# Cluster add-ons for GAL-E (docs/05 §2, §4)

The GAL-E chart (`infra/helm/gal-e`) expects these operators and stacks. Install them first, pinned
to the versions below (latest stable when this was written, 2026-11-02; bump deliberately).

| Add-on | Chart (repo) | Version | GAL-E uses |
|---|---|---|---|
| kube-prometheus-stack | `prometheus-community/kube-prometheus-stack` | 92.16.2 | Prometheus, Alertmanager, Grafana (dashboard sidecar), kube-state-metrics: ServiceMonitors, PrometheusRule, `grafana_dashboard` ConfigMaps |
| Loki | `grafana/loki` | 7.5.0 | JSON logs (no transcript text, no PII: CLAUDE.md rule 1) |
| Alloy | `grafana/alloy` | 1.15.0 | log shipping to Loki |
| Tempo | `grafana/tempo` | 1.24.4 | traces from `gale-otel-collector` (trace id = call id) |
| NVIDIA GPU Operator | `nvidia/gpu-operator` (NGC repo) | pin at install | drivers, device plugin, **DCGM exporter** (GPU util/memory/XID metrics) |
| CloudNativePG | `cloudnative-pg/cloudnative-pg` | 0.29.1 (operator 1.30.1) | `Cluster`, `ScheduledBackup` (PITR) |
| KEDA | `kedacore/keda` | 2.21.0 | `ScaledObject` for the post-call worker |
| External Secrets | `external-secrets/external-secrets` | 2.14.0 | `ExternalSecret` → Kubernetes Secrets |
| cert-manager | `jetstack/cert-manager` | 1.21.3 | TLS for the dashboard/API ingress |
| Ingress controller | your choice | — | ingress-nginx was retired upstream in 2026; use a maintained controller and set `ingress.className` / `ingress.annotations` |

Everything stays in-country (ADR-003). In prod the `gale-ml` and `gale-data` namespaces have no
internet egress, so mirror images into the internal registry (`global.imageRegistry`).

## Values that matter for GAL-E

`kube-prometheus-stack`:
```yaml
prometheus:
  prometheusSpec:
    # pick up GAL-E ServiceMonitors / PrometheusRules from the gale-* namespaces
    ruleSelector: { matchLabels: { release: kube-prometheus-stack } }
    serviceMonitorSelector: { matchLabels: { release: kube-prometheus-stack } }
    retention: 30d
grafana:
  sidecar:
    dashboards: { enabled: true, label: grafana_dashboard, searchNamespace: ALL }
  grafana.ini:
    security: { allow_embedding: true }     # /admin/health embeds panels (NEXT_PUBLIC_GRAFANA_URL)
    auth.generic_oauth: { enabled: true }   # Keycloak SSO, realm gale
alertmanager:
  config:
    route:
      receiver: gale-ops
      group_by: [alertname, team]
      routes:
        - { matchers: ['team="gale-ml"'], receiver: gale-ml }
    receivers:
      - name: gale-ops   # set: Slack/email/PagerDuty of the ops on-call
      - name: gale-ml    # set: ML on-call
```

Loki: keep the default JSON parsing; never add a pipeline stage that extracts message bodies.
GAL-E logs carry ids, counts and timings only, and the PII-in-logs lint (`make lint`) enforces it.

Grafana dashboards are generated (`make dashboards`, `tools/gen_dashboards.py`) and shipped as
ConfigMaps by the chart: **GAL-E · Ops**, **GAL-E · ML drift & quality**, **GAL-E · Business**.
