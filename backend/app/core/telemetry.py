"""OpenTelemetry tracing setup for LLM observability.

Builds a TracerProvider that exports spans to the console and sets
it as the global provider. The instrumented transport opens a span
per LLM round-trip and tags it with GenAI semantic-convention
attributes, this module is what makes those spans go somewhere.

Tracing is off by default and turned on by a settings flag. When
off, this module is never called, no provider is installed, and
the OpenTelemetry API hands out non-recording spans that cost
nothing. The wrapper's span code runs either way without a guard.

Console export is the dev target. The provider is built so adding
an OTLP exporter later is one extra span processor, not a rewrite:
swap or add a processor in _build_provider and nothing else moves.
The GenAI attribute names follow the OpenTelemetry semantic
conventions (gen_ai.system, gen_ai.request.model, gen_ai.usage.*)
so spans are vendor-neutral and export to any OTel backend.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

TRACER_NAME = "learning-system.transport"

# GenAI semantic-convention attribute keys. Named constants so the
# wrapper and any future span site spell them identically. These are
# the OpenTelemetry GenAI convention keys, stable across OTel backends.
ATTR_GEN_AI_SYSTEM = "gen_ai.system"
ATTR_GEN_AI_OPERATION = "gen_ai.operation.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


def _build_provider() -> TracerProvider:
    """Build the tracer provider with its span processors.

    The single processor batches spans to the console exporter.
    To ship spans elsewhere later, add an OTLP processor here and the
    rest of the tracing path is unchanged.
    """
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    return provider


def configure_tracing() -> None:
    """Install the global tracer provider. Idempotent.

    Called from the lifespan when tracing is enabled. OpenTelemetry
    ignores a second set_tracer_provider and logs a warning, so this
    checks first: until a real provider is installed, the global one
    is the default no-op SDK proxy. Re-creating the app (tests) finds
    a real provider already in place and leaves it alone.
    """
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    trace.set_tracer_provider(_build_provider())


def get_tracer() -> trace.Tracer:
    """Return the transport tracer.

    Safe to call whether or not configure_tracing ran. With no
    provider installed, the API returns a tracer that produces
    non-recording spans, so the wrapper needs no on/off branch.
    """
    return trace.get_tracer(TRACER_NAME)
