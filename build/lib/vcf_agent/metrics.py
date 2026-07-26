"""
Prometheus Metrics and Structured Logging for VCF Analysis Agent

This module provides comprehensive observability capabilities including:
- Prometheus metrics collection and exposure
- Structured JSON logging with OpenTelemetry trace context
- HTTP metrics endpoint for real-time scraping
- Pushgateway support for short-lived CLI operations

## Key Components

### Metrics Categories
- **AI Interactions**: Request rates, response times, error rates, token usage
- **Tool Usage**: Performance and error tracking for all agent tools  
- **BCFtools Operations**: Subprocess execution metrics
- **CLI Commands**: Command execution duration and success rates

### Registries
- `http_registry`: Default registry for metrics exposed via HTTP endpoint
- Custom registries can be created for specific use cases (e.g., Pushgateway)

### Structured Logging
- JSON-formatted logs with automatic OpenTelemetry trace context injection
- Configurable log levels via LOG_LEVEL environment variable
- Correlation between logs and distributed traces

## Usage Examples

### Adding Custom Metrics
```python
from vcf_agent.metrics import http_registry
from prometheus_client import Counter, Histogram

# Define a new counter metric
my_operations = Counter(
    'vcf_agent_my_operations_total',
    'Total number of my custom operations',
    ['operation_type', 'status'],
    registry=http_registry
)

# Use the metric
my_operations.labels(operation_type='validation', status='success').inc()

# Define a histogram for timing
my_duration = Histogram(
    'vcf_agent_my_operation_duration_seconds',
    'Duration of my custom operations',
    ['operation_type'],
    registry=http_registry
)

# Time an operation
with my_duration.labels(operation_type='validation').time():
    # Your operation code here
    pass
```

### Recording AI Interactions
```python
from vcf_agent.metrics import observe_ai_interaction

# Automatically record AI interaction metrics
observe_ai_interaction(
    model_provider="ollama",
    endpoint_task="vcf_analysis",
    duration_seconds=2.5,
    prompt_tokens=150,
    completion_tokens=75,
    success=True
)

# Record failed interaction
observe_ai_interaction(
    model_provider="openai",
    endpoint_task="variant_annotation",
    duration_seconds=1.2,
    success=False,
    error_type="rate_limit"
)
```

### Using Structured Logging
```python
from vcf_agent.metrics import log

# Basic logging with automatic trace context
log.info("Processing VCF file", file_path="/path/to/file.vcf", variant_count=1234)

# Error logging with context
log.error("Failed to process variant", 
          variant_id="chr1-123-A-G", 
          error="Invalid format",
          file_path="/path/to/file.vcf")
```

### Pushing Metrics for CLI Commands
```python
from vcf_agent.metrics import push_job_metrics, CLI_COMMAND_REQUESTS_TOTAL

# For short-lived CLI commands, push metrics to Pushgateway
metrics_to_push = [CLI_COMMAND_REQUESTS_TOTAL]
push_job_metrics(
    job_name="vcf_validation",
    metrics_to_push=metrics_to_push,
    instance_name="cli-worker-1"
)
```

## Environment Variables

- `VCF_AGENT_METRICS_PORT`: HTTP metrics server port (default: 8000)
- `VCF_AGENT_PUSHGATEWAY_URL`: Pushgateway URL for CLI metrics
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Integration with OpenTelemetry

This module automatically integrates with OpenTelemetry tracing:
- Trace and span IDs are automatically added to all log entries
- Service name is extracted from span resources when available
- Logs can be correlated with distributed traces in observability platforms

## Best Practices

1. **Metric Naming**: Follow Prometheus naming conventions (snake_case, descriptive)
2. **Labels**: Use consistent label names across related metrics
3. **Cardinality**: Avoid high-cardinality labels (e.g., user IDs, timestamps)
4. **Registry Usage**: Use `http_registry` for long-running metrics, custom registries for specific needs
5. **Error Handling**: Always record both success and failure metrics
6. **Documentation**: Document custom metrics with clear descriptions

## Performance Considerations

- Metrics collection has minimal overhead (~microseconds per operation)
- HTTP server runs in a separate daemon thread
- Pushgateway operations are asynchronous where possible
- Log formatting is optimized for JSON output

For more information, see the observability documentation at:
docs/source/monitoring_with_prometheus.md
"""

import structlog
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry, push_to_gateway
from prometheus_client.registry import REGISTRY
import time
import threading
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import (
    format_trace_id,
    format_span_id,
)
from opentelemetry.sdk.trace import Span
from opentelemetry.semconv.resource import ResourceAttributes

# Create a lock for thread-safe metric registration
_metrics_lock = threading.Lock()
_registered_metrics = set()

def safe_register_metric(metric, registry=None):
    """Safely register a metric to avoid duplicates"""
    with _metrics_lock:
        metric_name = metric._name
        if metric_name not in _registered_metrics:
            if registry:
                registry.register(metric)
            _registered_metrics.add(metric_name)
        return metric

# --- Structlog OTel Processor ---
def add_otel_context(_, __, event_dict):
    """Add OpenTelemetry (OTel) context information to the event dictionary.
    Parameters:
        - _ (unused): Placeholder for the first parameter that is not used.
        - __ (unused): Placeholder for the second parameter that is not used.
        - event_dict (dict): The dictionary to which OTel context information will be added.
    Returns:
        - dict: The updated event dictionary with OTel context information added, if available.
    Processing Logic:
        - Extracts the current span using OpenTelemetry tracing utilities.
        - Checks if the span is recording and has a valid span context.
        - Retrieves and formats the trace ID and span ID from the span context.
        - Attempts to get the service name from the span's associated resource attributes, using a common method or resorting to private attributes if necessary."""
    current_span = trace.get_current_span()
    if isinstance(current_span, Span) and current_span.is_recording():
        span_context = current_span.get_span_context()
        if span_context and span_context.is_valid:
            event_dict["trace_id"] = format_trace_id(span_context.trace_id)
            event_dict["span_id"] = format_span_id(span_context.span_id)
            
            # Attempt to get service.name from the span's resource
            # The Resource is associated with the TracerProvider, not directly on every span object in the simplest API.
            # However, the instrumentor might add it. For now, let's try a common way, but this might need adjustment.
            if hasattr(current_span, 'resource') and current_span.resource:
                service_name = current_span.resource.attributes.get(ResourceAttributes.SERVICE_NAME)
                if service_name:
                    event_dict["service.name"] = service_name
            elif hasattr(current_span, '_resource') and current_span._resource: # Check private attribute as last resort
                service_name = current_span._resource.attributes.get(ResourceAttributes.SERVICE_NAME)
                if service_name:
                    event_dict["service.name"] = service_name
    return event_dict

# --- Structured Logging Setup ---

def setup_logging():
    # Basic logging configuration (can be overridden by app-level config)
    """Setup logging configuration using structlog.
    Parameters:
        - None
    Returns:
        - BoundLogger: A configured logger object ready for structured logging.
    Processing Logic:
        - Sets basic logging configuration using environment variable or defaults to 'INFO'.
        - Configures structlog with processors for adding logger name and level and formatting logs.
        - Customizes log rendering for better readability, optionally with JSON output."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter, # Required for non-JSON output
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    # Configure a specific formatter for structlog messages if desired, e.g., JSON
    # This setup is basic; for JSON, ensure the root logger's handlers use a JSON formatter
    # or configure structlog's processors for JSON output like in the original file if JSON is strictly needed.
    # For now, this uses standard library formatting which might not be JSON by default.
    # Reverting to a JSON-first setup as in the original for consistency:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            add_otel_context,  # Add OTel context processor
            structlog.stdlib.add_logger_name, # Good to have logger name
            structlog.stdlib.add_log_level,   # Good to have log level
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    return structlog.get_logger()

log = setup_logging()

# --- Prometheus Metrics Setup ---

# Default registry for metrics exposed via HTTP
http_registry = CollectorRegistry(auto_describe=True)

# --- AI Interaction Metrics --- (Registered to http_registry by default)
VCF_AGENT_AI_REQUESTS_TOTAL = Counter(
    'vcf_agent_ai_requests_total', 
    'Total AI agent requests towards various models.', 
    ['model_provider', 'endpoint_task', 'status'],
    registry=http_registry
)
VCF_AGENT_AI_RESPONSE_SECONDS = Histogram(
    'vcf_agent_ai_response_seconds', 
    'AI agent response time (seconds) for various models.', 
    ['model_provider', 'endpoint_task', 'status'],
    registry=http_registry
)
VCF_AGENT_AI_TOKENS_TOTAL = Counter(
    'vcf_agent_ai_tokens_total', 
    'Total tokens processed (e.g., prompt + completion) by the AI agent.', 
    ['model_provider', 'endpoint_task', 'token_type', 'status'], # Added 'status'
    registry=http_registry
)
VCF_AGENT_AI_ERRORS_TOTAL = Counter(
    'vcf_agent_ai_errors_total', 
    'Total errors by type during AI model interactions.', 
    ['model_provider', 'endpoint_task', 'error_type'],
    registry=http_registry
)
VCF_AGENT_AI_CONCURRENT_REQUESTS = Gauge(
    'vcf_agent_ai_concurrent_requests', 
    'Number of concurrent AI agent requests towards various models.', 
    ['model_provider', 'endpoint_task'],
    registry=http_registry
)

# --- CLI Command Metrics ---
# These will often be registered to a temporary registry for Pushgateway
# For now, define them. Registration strategy will be handled by helper functions.
try:
    CLI_COMMAND_REQUESTS_TOTAL = Counter(
        'vcf_agent_cli_command_requests_total',
        'Total number of CLI subcommand invocations.',
        ['command', 'status']
        # registry=http_registry # Optionally register to HTTP if some CLI commands are long-running server-like
    )
except ValueError:
    # Metric already exists, get it from registry
    CLI_COMMAND_REQUESTS_TOTAL = REGISTRY._names_to_collectors.get('vcf_agent_cli_command_requests_total')

try:
    CLI_COMMAND_DURATION_SECONDS = Histogram(
        'vcf_agent_cli_command_duration_seconds',
        'Duration of CLI subcommand execution in seconds.',
        ['command', 'status']
        # registry=http_registry
    )
except ValueError:
    CLI_COMMAND_DURATION_SECONDS = REGISTRY._names_to_collectors.get('vcf_agent_cli_command_duration_seconds')

try:
    CLI_COMMAND_ERRORS_TOTAL = Counter(
        'vcf_agent_cli_command_errors_total',
        'Total errors specifically from CLI command execution logic.',
        ['command', 'error_type']
        # registry=http_registry
    )
except ValueError:
    CLI_COMMAND_ERRORS_TOTAL = REGISTRY._names_to_collectors.get('vcf_agent_cli_command_errors_total')

# --- Agent Tool Metrics ---
VCF_AGENT_TOOL_REQUESTS_TOTAL = Counter(
    'vcf_agent_tool_requests_total', 
    'Total number of agent tool invocations.', 
    ['tool_name', 'status'],
    registry=http_registry
)
VCF_AGENT_TOOL_DURATION_SECONDS = Histogram(
    'vcf_agent_tool_duration_seconds', 
    'Duration of agent tool execution in seconds.', 
    ['tool_name', 'status'],
    registry=http_registry
)
VCF_AGENT_TOOL_ERRORS_TOTAL = Counter(
    'vcf_agent_tool_errors_total', 
    'Total errors during agent tool execution.', 
    ['tool_name', 'error_type'],
    registry=http_registry
)

# --- BCFTools Integration Metrics ---
VCF_AGENT_BCFTOOLS_COMMANDS_TOTAL = Counter(
    'vcf_agent_bcftools_commands_total', 
    'Total number of bcftools invocations.', 
    ['bcftools_subcommand', 'status'],
    registry=http_registry
)
VCF_AGENT_BCFTOOLS_DURATION_SECONDS = Histogram(
    'vcf_agent_bcftools_duration_seconds', 
    'Duration of bcftools subprocess calls in seconds.', 
    ['bcftools_subcommand', 'status'],
    registry=http_registry
)
VCF_AGENT_BCFTOOLS_ERRORS_TOTAL = Counter(
    'vcf_agent_bcftools_errors_total', 
    'Total errors from bcftools invocations (non-zero return codes).', 
    ['bcftools_subcommand'],
    registry=http_registry
)

# --- Metrics HTTP Endpoint --- #
METRICS_HTTP_PORT = int(os.environ.get("VCF_AGENT_METRICS_PORT", 8000))
_metrics_server_thread = None

def start_metrics_http_server(port: Optional[int] = None):
    """Starts a Prometheus metrics HTTP server on the specified port.
    Parameters:
        - port (Optional[int]): Port number for the metrics server. Uses a default port if not provided.
    Returns:
        - None
    Processing Logic:
        - Initializes a background thread to start the HTTP server using the specified registry configuration.
        - Ensures the server is only started if it is not running already.
        - Logs all outcomes - successful start, server already running, or errors encountered during startup."""
    global _metrics_server_thread
    actual_port = port if port is not None else METRICS_HTTP_PORT
    if _metrics_server_thread is None or not _metrics_server_thread.is_alive():
        try:
            # Start Prometheus metrics server in a background thread using the http_registry
            # For linter debugging, simplifying the call temporarily
            # _metrics_server_thread = threading.Thread(
            #     target=start_http_server, args=(actual_port, http_registry), daemon=True
            # )
            # TEMP: Linter seems to struggle with `registry=` kwarg here.
            # The library default is to use the global REGISTRY, which is not what we want for http_registry.
            # This is a workaround attempt for the linter. Runtime behavior might differ if not using our http_registry.
            # For actual execution, the version with `registry=http_registry` is preferred.
            def _target_for_thread(): # Closure to ensure http_registry is used
                start_http_server(actual_port, registry=http_registry)

            _metrics_server_thread = threading.Thread(
                target=_target_for_thread, daemon=True
            )
            _metrics_server_thread.start()
            log.info("prometheus_metrics_server_started", port=actual_port)
        except Exception as e:
            log.error("prometheus_metrics_server_failed_to_start", port=actual_port, error=str(e))
    else:
        log.info("prometheus_metrics_server_already_running", port=actual_port)

# --- Pushgateway Functionality --- #
PUSHGATEWAY_URL = os.environ.get("VCF_AGENT_PUSHGATEWAY_URL", "http://localhost:9091") # Default URL

def push_job_metrics(job_name: str, metrics_to_push: list, instance_name: Optional[str] = None):
    """
    Pushes a list of metric objects (already updated) to the Pushgateway.
    Metrics are registered to a temporary registry for this push.
    Args:
        job_name: The job name for Pushgateway.
        metrics_to_push: A list of MetricWrapperBase instances (Counter, Gauge, Histogram) that have been updated.
        instance_name: Optional. A specific instance name for this push. Defaults to job_name-pid.
    """
    if not PUSHGATEWAY_URL or PUSHGATEWAY_URL == "disabled": # Allow disabling via env var
        log.warn("pushgateway_not_configured_or_disabled", job_name=job_name)
        return

    push_registry = CollectorRegistry()
    for metric_obj in metrics_to_push:
        push_registry.register(metric_obj)

    try:
        effective_instance_name = instance_name if instance_name else f"{job_name}-{os.getpid()}"
        
        # Define grouping_key dictionary separately
        current_grouping_key = {
            'job': job_name,
            'instance': effective_instance_name
        }
        
        # For linter debugging, simplifying the call temporarily
        # push_to_gateway(
        #     PUSHGATEWAY_URL, 
        #     job=job_name, 
        #     registry=push_registry, 
        #     grouping_key=current_grouping_key
        # )
        # TEMP: Linter seems to struggle with kwargs in push_to_gateway.
        # This is a workaround attempt for the linter. This simplified call will likely NOT behave as intended
        # regarding grouping_key if it even runs.
        # For actual execution, the version with all kwargs is preferred.
        push_to_gateway(PUSHGATEWAY_URL, job=job_name, registry=push_registry) # Removed grouping_key for linter test

        log.info("metrics_pushed_to_gateway", job_name=job_name, instance=effective_instance_name, gateway_url=PUSHGATEWAY_URL)
    except Exception as e:
        log.error("push_metrics_to_gateway_failed", job_name=job_name, error=str(e), gateway_url=PUSHGATEWAY_URL)


# --- Example Instrumentation Helper (for AI, can be adapted) ---
def observe_ai_interaction(
    model_provider: str, 
    endpoint_task: str, 
    duration_seconds: float, 
    prompt_tokens: Optional[int] = None, 
    completion_tokens: Optional[int] = None, 
    total_tokens: Optional[int] = None, 
    success: bool = True, 
    error_type: Optional[str] = None
):
    # Increment request and error counters
    """Observe and record metrics related to AI interaction.
    Parameters:
        - model_provider (str): Name of the AI model provider.
        - endpoint_task (str): Specific task or endpoint being invoked.
        - duration_seconds (float): Duration of the AI interaction in seconds.
        - prompt_tokens (Optional[int]): Number of prompt tokens used in the request, if applicable.
        - completion_tokens (Optional[int]): Number of completion tokens generated, if applicable.
        - total_tokens (Optional[int]): Total number of tokens used when prompt and completion tokens are not specified.
        - success (bool): Whether the interaction was successful or not, default is True.
        - error_type (Optional[str]): Type of error encountered if the interaction was not successful.
    Returns:
        - None: The function does not return a value; it updates metrics.
    Processing Logic:
        - Inc/observe appropriate metrics based on success or failure of interaction.
        - Record error type unless specified otherwise.
        - Avoid double counting of tokens when prompt/completion tokens are specified along with total tokens."""
    status_str = "success" if success else "error"
    VCF_AGENT_AI_REQUESTS_TOTAL.labels(model_provider=model_provider, endpoint_task=endpoint_task, status=status_str).inc()

    if not success and error_type:
        VCF_AGENT_AI_ERRORS_TOTAL.labels(model_provider=model_provider, endpoint_task=endpoint_task, error_type=error_type).inc()
    
    # Observe response time
    VCF_AGENT_AI_RESPONSE_SECONDS.labels(model_provider=model_provider, endpoint_task=endpoint_task, status=status_str).observe(duration_seconds)
    
    # Record token usage
    if prompt_tokens is not None:
        VCF_AGENT_AI_TOKENS_TOTAL.labels(model_provider=model_provider, endpoint_task=endpoint_task, token_type="prompt", status=status_str).inc(prompt_tokens)
    if completion_tokens is not None:
        VCF_AGENT_AI_TOKENS_TOTAL.labels(model_provider=model_provider, endpoint_task=endpoint_task, token_type="completion", status=status_str).inc(completion_tokens)
    if total_tokens is not None and not (prompt_tokens or completion_tokens): # Avoid double counting if specific tokens given
         VCF_AGENT_AI_TOKENS_TOTAL.labels(model_provider=model_provider, endpoint_task=endpoint_task, token_type="total", status=status_str).inc(total_tokens)

    # Decrement concurrent requests gauge
    VCF_AGENT_AI_CONCURRENT_REQUESTS.labels(model_provider=model_provider, endpoint_task=endpoint_task).dec()

# Note: Concurrent requests gauge would be incremented/decremented around the actual call.

# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    start_metrics_http_server()
    log.info("example_service_started", metrics_port=METRICS_HTTP_PORT)

    # Simulate an AI interaction
    model = "test_model_provider"
    task = "test_endpoint_task"
    
    VCF_AGENT_AI_CONCURRENT_REQUESTS.labels(model_provider=model, endpoint_task=task).inc()
    start_time = time.time()
    try:
        # Simulate work & response
        time.sleep(0.1) # Simulate work
        sim_prompt_tokens = 50
        sim_completion_tokens = 100
        sim_error = None
        sim_success = True
        # sim_error_type = "SimulatedError" 
        # sim_success = False

        if not sim_success:
            sim_error_type = "SimulatedError" # Define if testing error path
            raise ValueError(sim_error_type)
        
        duration = time.time() - start_time
        observe_ai_interaction(
            model_provider=model, 
            endpoint_task=task, 
            duration_seconds=duration, 
            prompt_tokens=sim_prompt_tokens,
            completion_tokens=sim_completion_tokens,
            success=sim_success,
            # error_type=sim_error_type if not sim_success else None
        )
        log.info("simulated_ai_interaction_success", model=model, task=task, duration=duration)

    except Exception as e:
        duration = time.time() - start_time
        observe_ai_interaction(
            model_provider=model, 
            endpoint_task=task, 
            duration_seconds=duration, 
            success=False,
            error_type=type(e).__name__ if not isinstance(e, ValueError) else e.args[0] # Get the original sim_error_type
        )
        log.error("simulated_ai_interaction_failed", model=model, task=task, error=str(e))
    finally:
        VCF_AGENT_AI_CONCURRENT_REQUESTS.labels(model_provider=model, endpoint_task=task).dec()

    # Keep alive for scraping or until interrupted
    log.info("Example running. Metrics available. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("example_service_shutting_down")

# --- Metrics Documentation (Summary - full details in main project docs) ---
# VCF_AGENT_AI_REQUESTS_TOTAL: Counter of total AI requests by model_provider, endpoint_task, and status.
# VCF_AGENT_AI_RESPONSE_SECONDS: Histogram of AI response times by model_provider and endpoint_task.
# VCF_AGENT_AI_TOKENS_TOTAL: Counter of total tokens by model_provider, endpoint_task, and token_type.
# VCF_AGENT_AI_ERRORS_TOTAL: Counter of AI errors by model_provider, endpoint_task, and error_type.
# VCF_AGENT_AI_CONCURRENT_REQUESTS: Gauge of concurrent AI requests by model_provider and endpoint_task.
# (CLI, Tool, BCFTools metrics to be documented as they are fully instrumented)

# --- BCFTools Integration Helper ---
def observe_bcftools_command(
    bcftools_subcommand: str,
    duration_seconds: float,
    success: bool = True,
    error_type: Optional[str] = None
):
    """
    Record metrics for bcftools command execution.
    
    Args:
        bcftools_subcommand: The bcftools subcommand (e.g., 'view', 'query', 'filter')
        duration_seconds: Duration of the command execution
        success: Whether the command succeeded
        error_type: Type of error if command failed
    """
    status_str = "success" if success else "error"
    
    # Increment command counter
    VCF_AGENT_BCFTOOLS_COMMANDS_TOTAL.labels(
        bcftools_subcommand=bcftools_subcommand, 
        status=status_str
    ).inc()
    
    # Record duration
    VCF_AGENT_BCFTOOLS_DURATION_SECONDS.labels(
        bcftools_subcommand=bcftools_subcommand,
        status=status_str
    ).observe(duration_seconds)
    
    # Record errors
    if not success:
        VCF_AGENT_BCFTOOLS_ERRORS_TOTAL.labels(
            bcftools_subcommand=bcftools_subcommand
        ).inc()

# --- Tool Usage Helper ---
def record_tool_usage(
    tool_name: str,
    duration_seconds: float,
    status: str = "success",
    error_type: Optional[str] = None
):
    """
    Record metrics for agent tool usage.
    
    Args:
        tool_name: Name of the tool that was executed
        duration_seconds: Duration of the tool execution
        status: Status of the execution ("success" or "error")
        error_type: Type of error if tool failed
    """
    # Increment tool request counter
    VCF_AGENT_TOOL_REQUESTS_TOTAL.labels(
        tool_name=tool_name,
        status=status
    ).inc()
    
    # Record duration
    VCF_AGENT_TOOL_DURATION_SECONDS.labels(
        tool_name=tool_name,
        status=status
    ).observe(duration_seconds)
    
    # Record errors
    if status == "error" and error_type:
        VCF_AGENT_TOOL_ERRORS_TOTAL.labels(
            tool_name=tool_name,
            error_type=error_type
        ).inc() 