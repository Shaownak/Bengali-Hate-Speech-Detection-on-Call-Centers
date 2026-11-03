"""Prometheus metrics behind the docs/05 §4 dashboards and alert rules (Prompt 14).

Kept in one place so `infra/helm/gal-e/files/prometheus-rules.yaml` and the Grafana dashboards
reference names that exist (a test checks that every `gale_*` series in the rules is defined).
Labels never carry call ids, text or other high-cardinality / personal values.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

# --- bus ---------------------------------------------------------------------------------------
BUS_LAG_SECONDS = Gauge(
    "gale_bus_lag_seconds",
    "Age of the oldest message the consumer group has not yet received (0 = caught up)",
    ["stream", "group"],
)
BUS_PENDING = Gauge(
    "gale_bus_pending_messages",
    "Messages delivered to the group but not yet acknowledged",
    ["stream", "group"],
)

# --- pipeline outputs (ops + business) ---------------------------------------------------------
EVENTS = Counter(
    "gale_events_total",
    "Fused events by lifecycle stage (provisional, confirmed, dismissed)",
    ["category", "speaker", "stage"],
)
ALERTS = Counter("gale_alerts_total", "Alerts raised", ["category", "speaker"])

# --- ML drift ----------------------------------------------------------------------------------
SIGNALS = Counter("gale_signals_total", "Signals received by fusion", ["source", "category"])
SIGNAL_SCORE = Histogram(
    "gale_signal_score", "Signal score distribution (drift)", ["source"], buckets=SCORE_BUCKETS
)
ASR_CONFIDENCE = Histogram(
    "gale_asr_confidence", "Average word confidence of final ASR segments", buckets=SCORE_BUCKETS
)

# --- pilot: candidate models beside the stable ones (ADR-020) ---------------------------------
# outcome: agree | stable_only | candidate_only (an output one side produced and the other didn't)
SHADOW_COMPARE = Counter(
    "gale_shadow_compare_total",
    "Shadow/canary candidate vs stable model outputs on the same calls",
    ["component", "version", "outcome"],
)
CANDIDATE_CALLS = Counter(
    "gale_candidate_calls_total",
    "Calls routed to a candidate model",
    ["component", "version", "role"],
)

# --- human feedback (ML: precision from reviews) -----------------------------------------------
REVIEWS = Counter("gale_reviews_total", "Human reviews", ["kind", "verdict"])

# --- API ---------------------------------------------------------------------------------------
WS_CONNECTIONS = Gauge("gale_ws_connections", "Open dashboard WebSockets", ["endpoint"])
HTTP_REQUESTS = Counter(
    "gale_http_requests_total", "HTTP requests", ["service", "method", "code_class"]
)
HTTP_SECONDS = Histogram(
    "gale_http_request_seconds",
    "HTTP request duration",
    ["service"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
