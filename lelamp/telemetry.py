from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Tracer

_SERVICE_NAME = "lelamp"

# ponytail: console exporter only. Swap for OTLP -> Phoenix/Jaeger once we add
# opentelemetry-exporter-otlp-proto-grpc (needs approval, not in the allowed dep list yet).
def init_telemetry() -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str) -> Tracer:
    return trace.get_tracer(name)
