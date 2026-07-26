"""
OpenTelemetry Distributed Tracing for VCF Analysis Agent

This module provides comprehensive distributed tracing capabilities using OpenTelemetry,
enabling end-to-end observability of requests through the VCF Analysis Agent system.

## Key Features

- **Automatic Instrumentation**: HTTP requests, logging, and asyncio operations
- **Custom Spans**: Tool executions, AI model interactions, VCF processing operations
- **Trace Context Propagation**: Maintains trace context across service boundaries
- **Multiple Export Protocols**: Support for gRPC and HTTP/protobuf to Jaeger
- **Resource Attribution**: Service metadata and version information

## Architecture

The tracing system consists of:
- **TracerProvider**: Central configuration and span management
- **OTLP Exporter**: Sends traces to Jaeger collector
- **BatchSpanProcessor**: Efficient batching and export of spans
- **Auto-Instrumentation**: Automatic tracing of common operations
- **Custom Tracers**: Application-specific span creation

## Usage Examples

### Basic Tracer Setup
```python
from vcf_agent.tracing import init_tracer, setup_auto_instrumentation

# Initialize tracer for your service
tracer = init_tracer("vcf-processor")

# Enable automatic instrumentation
setup_auto_instrumentation()
```

### Creating Custom Spans
```python
from vcf_agent.tracing import init_tracer

tracer = init_tracer("vcf-analysis")

# Basic span creation
with tracer.start_as_current_span("validate_vcf_file") as span:
    span.set_attribute("file.path", "/path/to/file.vcf")
    span.set_attribute("file.size_bytes", 1024000)
    
    # Your validation logic here
    result = validate_file()
    
    span.set_attribute("validation.result", "valid")
    span.set_attribute("variant.count", result.variant_count)

# Nested spans for detailed tracing
with tracer.start_as_current_span("process_vcf_pipeline") as parent_span:
    parent_span.set_attribute("pipeline.version", "1.0")
    
    with tracer.start_as_current_span("parse_header") as header_span:
        header_span.set_attribute("header.sample_count", 10)
        # Parse header logic
    
    with tracer.start_as_current_span("process_variants") as variant_span:
        variant_span.set_attribute("processing.batch_size", 1000)
        # Process variants logic
```

### Error Handling and Status
```python
from opentelemetry.trace import Status, StatusCode

with tracer.start_as_current_span("risky_operation") as span:
    try:
        # Operation that might fail
        result = perform_operation()
        span.set_attribute("operation.result", "success")
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.message", str(e))
        raise
```

### Tool Instrumentation Pattern
```python
def my_analysis_tool(input_data: str) -> str:
    \"\"\"Example of instrumenting an agent tool.\"\"\"
    tracer = init_tracer("vcf-tools")
    
    with tracer.start_as_current_span("tool.my_analysis") as span:
        span.set_attribute("tool.name", "my_analysis")
        span.set_attribute("tool.version", "1.0")
        span.set_attribute("input.size", len(input_data))
        
        try:
            result = process_data(input_data)
            span.set_attribute("output.size", len(result))
            span.set_attribute("tool.status", "success")
            return result
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("tool.status", "error")
            span.set_attribute("error.type", type(e).__name__)
            raise
```

## Environment Variables

### Required Configuration
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`: Jaeger collector endpoint
  - gRPC: `http://localhost:4317` (default)
  - HTTP: `http://localhost:4318/v1/traces`

### Optional Configuration
- `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`: Export protocol (`grpc` or `http/protobuf`)
- `OTEL_EXPORTER_OTLP_TRACES_HEADERS`: Additional headers for authentication
- `OTEL_PYTHON_LOG_LEVEL`: Set to `debug` for detailed OTel logging
- `OTEL_SERVICE_NAME`: Override service name (defaults to function parameter)

## Auto-Instrumentation

The module automatically instruments:
- **HTTP Requests**: All outbound HTTP calls via `requests` library
- **Logging**: Correlation between logs and traces
- **Asyncio**: Async/await operations and task scheduling

Auto-instrumentation is enabled once per application lifecycle and provides
zero-code tracing for common operations.

## Best Practices

### Span Naming
- Use descriptive, hierarchical names: `service.operation.sub_operation`
- Include the operation type: `tool.validate_vcf`, `api.get_variant`
- Be consistent across similar operations

### Attributes
- Use semantic conventions where possible
- Include relevant context: file paths, IDs, counts, durations
- Avoid high-cardinality values (user IDs, timestamps)
- Use structured attribute names: `file.path`, `variant.count`

### Error Handling
- Always set span status for errors
- Include error type and message as attributes
- Don't let tracing exceptions break application flow

### Performance
- Spans have minimal overhead (~microseconds)
- Batch processing optimizes export performance
- Use sampling in high-throughput scenarios

## Integration with Metrics

Tracing integrates seamlessly with the metrics system:
- Trace context is automatically added to structured logs
- Metrics can be correlated with specific traces
- Both systems share service metadata and configuration

## Troubleshooting

### No Traces in Jaeger
1. Check Jaeger collector is running on the configured endpoint
2. Verify network connectivity from agent to Jaeger
3. Enable debug logging: `OTEL_PYTHON_LOG_LEVEL=debug`
4. Check for export errors in application logs

### Missing Spans
1. Ensure `init_tracer()` is called before creating spans
2. Verify span creation is within an active trace context
3. Check that spans are properly closed (use context managers)

### Performance Issues
1. Adjust batch processor settings for your workload
2. Consider sampling for high-volume applications
3. Monitor export queue depth and processing times

## Example: Complete Service Setup
```python
from vcf_agent.tracing import init_tracer, setup_auto_instrumentation

def main():
    # Initialize tracing for the service
    tracer = init_tracer("vcf-analysis-service")
    setup_auto_instrumentation()
    
    # Your application code with automatic tracing
    with tracer.start_as_current_span("main_operation"):
        # All HTTP requests, logs, and async operations
        # will be automatically traced
        process_vcf_files()

if __name__ == "__main__":
    main()
```

For more information, see the observability documentation at:
docs/source/monitoring_with_prometheus.md
"""

import os
import logging
import importlib.metadata  # Added for versioning
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider  # Explicit SDK type
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCOTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPOTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Import all resource components on one line
from opentelemetry.sdk.resources import (
    Resource, SERVICE_NAME, TELEMETRY_SDK_LANGUAGE, TELEMETRY_SDK_NAME, TELEMETRY_SDK_VERSION
)
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor

# Get the VCF Agent version for telemetry
try:
    from vcf_agent import __version__  # Assuming __version__ is in vcf_agent/__init__.py
except ImportError:
    from . import __version__  # Fallback for relative import if tracing.py is inside vcf_agent package

# Set up debug logging if requested
_otel_python_log_level_env = os.getenv('OTEL_PYTHON_LOG_LEVEL', '').lower()
if _otel_python_log_level_env == 'debug':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # force=True will remove and re-add handlers to the root logger
    )
    print(f"[TRACING] Root logger configured to DEBUG because OTEL_PYTHON_LOG_LEVEL is set to "
          f"'{_otel_python_log_level_env}'")


def init_tracer(service_name: str) -> trace.Tracer:
    """Initialize OpenTelemetry tracing with OTLP export to Jaeger."""
    
    # Create resource with service information
    resource = Resource.create({
        SERVICE_NAME: service_name,
        TELEMETRY_SDK_LANGUAGE: "python",
        TELEMETRY_SDK_NAME: "opentelemetry",
        TELEMETRY_SDK_VERSION: importlib.metadata.version('opentelemetry-api'),  # Corrected SDK version
        "service.version": __version__
    })

    provider = SdkTracerProvider(resource=resource)

    # Determine the exporter endpoint and protocol
    exporter_endpoint = os.getenv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT')
    _otel_protocol = os.getenv('OTEL_EXPORTER_OTLP_TRACES_PROTOCOL', 'grpc').lower()
    
    if not exporter_endpoint:
        if _otel_protocol == 'grpc':
            exporter_endpoint = "http://localhost:4317"  # Default for gRPC
            print(f"[TRACING] WARNING: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set. "
                  f"Defaulting to {exporter_endpoint} for gRPC protocol.")
        elif _otel_protocol in ['http/protobuf', 'http']:
            exporter_endpoint = "http://localhost:4318"  # Default for HTTP/protobuf
            print(f"[TRACING] WARNING: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set. "
                  f"Defaulting to {exporter_endpoint} for http/protobuf protocol.")
        else:
            print(f"[TRACING] WARNING: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set and protocol is "
                  f"'{_otel_protocol}'. Exporter might not work.")
    
    # Create the appropriate exporter
    if _otel_protocol == 'grpc':
        exporter = GRPCOTLPSpanExporter(endpoint=exporter_endpoint)
    elif _otel_protocol in ['http/protobuf', 'http']:
        exporter = HTTPOTLPSpanExporter(endpoint=exporter_endpoint)
    else:
        # Default to gRPC if protocol is unrecognized
        exporter = GRPCOTLPSpanExporter(endpoint=exporter_endpoint)
        print(f"[TRACING] WARNING: Unrecognized protocol '{_otel_protocol}', defaulting to gRPC.")
    
    # Create span processor with custom settings
    processor = BatchSpanProcessor(
        exporter,
        max_queue_size=2048,  # Default 2048
        schedule_delay_millis=200,  # Default 5000
        export_timeout_millis=10000  # Default 30000, but let's try a bit longer for flush
    )

    # This call is guarded by Once() inside OTel
    provider.add_span_processor(processor)

    # Check if a TracerProvider was already set
    current_provider = trace.get_tracer_provider()
    if hasattr(current_provider, 'add_span_processor'):
        trace.set_tracer_provider(provider)
    else:
        print(f"OpenTelemetry TracerProvider was already SET. init_tracer for '{service_name}' will use "
              f"existing provider. Type: {type(current_provider)}")

    tracer = trace.get_tracer(
        instrumenting_module_name=f"vcf_agent.{service_name.replace('-', '_')}",
        instrumenting_library_version=__version__  # Corrected keyword
    )
    
    print(f"[TRACING] Tracer initialized for service '{service_name}' with endpoint '{exporter_endpoint}' "
          f"using protocol '{_otel_protocol}'")
    return tracer


def setup_auto_instrumentation():
    """Set up automatic instrumentation for common libraries."""
    current_provider = trace.get_tracer_provider()
    
    # Check if we have an actual TracerProvider (not the default no-op one)
    if not hasattr(current_provider, 'add_span_processor'):
        print("Warning: SDK TracerProvider not set before setup_auto_instrumentation. "
              "Initializing with default service name.")
        init_tracer("vcf-agent-default")

    # Set up automatic instrumentation
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()
    AsyncioInstrumentor().instrument()

    print("[TRACING] Auto-instrumentation enabled for requests, logging, and asyncio")


# Example usage (commented out for production)
# if __name__ == "__main__":
#     tracer = init_tracer("example-service")
#     setup_auto_instrumentation()
#
#     with tracer.start_as_current_span("example_operation") as span:
#         span.set_attribute("example.attribute", "value")
#         print("Example span created")
#
#     print("Tracing initialized and example span created.") 
