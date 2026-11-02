"""OpenTelemetry export (docs/05 §4). Off unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (the Helm
chart points it at the in-cluster collector); the standard OTEL_* variables configure the rest."""

from __future__ import annotations

import os

from opentelemetry import trace

_configured = False


def configure_tracing(service: str) -> bool:
    global _configured
    if _configured or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return _configured
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": f"gale-{service}"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    return True
