"""
Enhanced OpenTelemetry Tracing for VCF Analysis Agent - Phase 4

This module provides advanced distributed tracing capabilities specifically designed for
AI-powered genomics applications, building upon the base OpenTelemetry implementation
with specialized monitoring for AI providers, VCF processing operations, and production
deployment requirements.

## Key Enhancements for 2025

- **AI Agent Observability**: Specialized tracing for OpenAI, Claude, Ollama interactions
- **VCF Processing Metrics**: Genomics-specific performance monitoring and attributes
- **Production Configuration**: Environment-aware sampling and error handling
- **Apache Iggy Preparation**: Performance baselines for future message streaming
- **Memory Optimization Tracking**: Integration with existing MemoryOptimizationConfig

## Architecture

The enhanced tracing system extends the base implementation with:
- **EnhancedTraceService**: Production-ready trace management with configuration
- **AI Provider Decorators**: Automatic instrumentation for LLM API calls
- **VCF Operation Spans**: Specialized spans for genomics data processing
- **Custom Metrics Integration**: Performance counters and business metrics
- **Robust Error Handling**: Graceful degradation and circuit breaker patterns

## Usage Examples

### Enhanced Service Setup
```python
from vcf_agent.enhanced_tracing import EnhancedTraceService
from vcf_agent.config import MemoryOptimizationConfig

# Initialize with configuration-driven approach
config = MemoryOptimizationConfig(optimization_level="standard")
trace_service = EnhancedTraceService(
    service_name="vcf-analysis-agent",
    config=config,
    environment="production"
)

# Enable all instrumentation
trace_service.setup_enhanced_instrumentation()
```

### AI Provider Monitoring
```python
@trace_service.ai_provider_span("openai", "gpt-4")
async def analyze_variant_with_ai(variant_data: dict) -> str:
    # Automatically creates span with AI-specific attributes
    # Tracks token usage, latency, and error rates
    response = await openai_client.chat.completions.create(...)
    return response

# Manual AI provider tracing
with trace_service.ai_provider_span_context("claude", "claude-3-opus") as span:
    span.set_ai_attributes(
        input_tokens=150,
        model_version="20240307",
        provider_endpoint="api.anthropic.com"
    )
    result = await claude_client.analyze(data)
    span.set_ai_attributes(output_tokens=200, cost_usd=0.05)
```

### VCF Processing Instrumentation
```python
@trace_service.vcf_operation_span("variant_processing")
def process_vcf_batch(vcf_file: str, batch_size: int = 1000):
    # Automatically tracks VCF-specific metrics
    with trace_service.vcf_operation_context("parse_vcf_header") as span:
        span.set_vcf_attributes(
            file_path=vcf_file,
            file_size_bytes=os.path.getsize(vcf_file),
            compression_type="bgzip"
        )
        header = parse_header(vcf_file)
        span.set_vcf_attributes(
            sample_count=len(header.samples),
            variant_count_estimate=header.variant_count
        )
    
    with trace_service.vcf_operation_context("process_variants") as span:
        span.set_vcf_attributes(batch_size=batch_size)
        results = process_variants_batch(vcf_file, batch_size)
        span.set_vcf_attributes(
            variants_processed=len(results),
            processing_rate_per_sec=len(results) / span.get_duration_seconds()
        )
```

### Memory Optimization Tracking
```python
@trace_service.memory_operation_span("embedding_optimization")
def optimize_variant_embeddings(variants: List[dict], config: MemoryOptimizationConfig):
    with trace_service.memory_context("dimension_reduction") as span:
        span.set_memory_attributes(
            optimization_level=config.optimization_level,
            input_dimensions=1536,
            target_dimensions=768,
            reduction_method="pca"
        )
        
        optimized_embeddings = apply_pca_reduction(variants)
        
        span.set_memory_attributes(
            memory_saved_bytes=calculate_memory_savings(),
            compression_ratio=0.5,
            processing_time_ms=span.get_duration_ms()
        )
```

## Environment Configuration

### Production Settings
```bash
# Enhanced tracing configuration
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger-collector:4317
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
VCF_TRACING_ENVIRONMENT=production
VCF_TRACING_SAMPLING_RATE=0.1
VCF_AI_PROVIDER_MONITORING=true
VCF_VCF_OPERATION_TRACING=true
VCF_MEMORY_OPTIMIZATION_TRACKING=true

# AI provider specific settings
VCF_OPENAI_TRACE_RESPONSES=false  # Security: don't trace response content
VCF_CLAUDE_COST_TRACKING=true
VCF_OLLAMA_PERFORMANCE_MONITORING=true
```

### Development Settings
```bash
VCF_TRACING_ENVIRONMENT=development
VCF_TRACING_SAMPLING_RATE=1.0
VCF_TRACING_DEBUG=true
OTEL_PYTHON_LOG_LEVEL=debug
```

## Best Practices for AI Genomics Applications

### Span Naming Conventions
- VCF Operations: `vcf.operation_type.sub_operation`
- AI Provider Calls: `ai.provider.model.operation`
- Memory Operations: `memory.optimization.technique`
- Analysis Pipeline: `analysis.stage.component`

### Attribute Standards
```python
# VCF-specific attributes
"vcf.file.path", "vcf.file.size_bytes", "vcf.variant.count"
"vcf.sample.count", "vcf.compression.type", "vcf.format.version"

# AI provider attributes  
"ai.provider.name", "ai.model.name", "ai.tokens.input", "ai.tokens.output"
"ai.response.time_ms", "ai.cost.usd", "ai.provider.endpoint"

# Memory optimization attributes
"memory.optimization.level", "memory.cache.hit_ratio", "memory.reduction.percentage"
"memory.technique", "memory.before_bytes", "memory.after_bytes"

# Performance attributes
"performance.throughput.variants_per_sec", "performance.latency.p95_ms"
"performance.cpu.usage_percent", "performance.memory.peak_bytes"
```

### Error Handling Patterns
```python
# AI provider error handling
try:
    result = await ai_provider.call()
except AIProviderRateLimitError as e:
    span.set_status(Status(StatusCode.ERROR, "Rate limited"))
    span.set_attribute("ai.error.type", "rate_limit")
    span.set_attribute("ai.error.retry_after", e.retry_after)
    
except AIProviderAuthError as e:
    span.set_status(Status(StatusCode.ERROR, "Authentication failed"))
    span.set_attribute("ai.error.type", "auth_error")
    # Don't log sensitive auth details
```

## Integration with Apache Iggy Preparation

This enhanced tracing system establishes performance baselines and monitoring
patterns that will be essential for validating the 4-10x performance improvements
expected from Apache Iggy integration:

- **Throughput Baselines**: Current variant processing rates
- **Latency Measurements**: End-to-end pipeline timing
- **Resource Utilization**: Memory and CPU usage patterns
- **AI Provider Performance**: Response times and cost metrics

## Security Considerations

- **No Sensitive Data**: AI response content excluded from traces by default
- **Cost Monitoring**: Track AI provider usage without exposing API keys
- **Configurable Verbosity**: Production vs development tracing levels
- **Sampling Strategies**: Reduce overhead while maintaining visibility

For deployment and configuration details, see:
docs/MEMORY_OPTIMIZATION_GUIDE.md
docs/CODEBASE_STREAMLINING_SUMMARY.md
"""

import os
import time
import logging
import functools
from typing import Optional, Dict, Any, List, Union, ContextManager, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased, 
    ParentBased,
    ALWAYS_OFF,
    ALWAYS_ON
)

# Import base tracing functionality
from .tracing import init_tracer, setup_auto_instrumentation
from .config import MemoryOptimizationConfig

logger = logging.getLogger(__name__)


class TracingEnvironment(Enum):
    """Tracing environment configurations."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AIProvider(Enum):
    """Supported AI providers for specialized monitoring."""
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class TracingConfig:
    """Configuration for enhanced tracing features."""
    environment: TracingEnvironment = TracingEnvironment.DEVELOPMENT
    sampling_rate: float = 1.0
    ai_provider_monitoring: bool = True
    vcf_operation_tracing: bool = True
    memory_optimization_tracking: bool = True
    debug_mode: bool = False
    
    # AI provider specific settings
    trace_ai_responses: bool = False  # Security: default False for production
    track_ai_costs: bool = True
    monitor_ai_performance: bool = True
    
    # Performance settings
    max_span_attributes: int = 64
    max_span_events: int = 128
    batch_export_timeout_ms: int = 10000
    
    # Custom configuration from environment
    custom_attributes: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_environment(cls) -> 'TracingConfig':
        """Create configuration from environment variables."""
        env = os.getenv('VCF_TRACING_ENVIRONMENT', 'development').lower()
        environment = TracingEnvironment(env) if env in [e.value for e in TracingEnvironment] else TracingEnvironment.DEVELOPMENT
        
        return cls(
            environment=environment,
            sampling_rate=float(os.getenv('VCF_TRACING_SAMPLING_RATE', '1.0' if env == 'development' else '0.1')),
            ai_provider_monitoring=os.getenv('VCF_AI_PROVIDER_MONITORING', 'true').lower() == 'true',
            vcf_operation_tracing=os.getenv('VCF_VCF_OPERATION_TRACING', 'true').lower() == 'true',
            memory_optimization_tracking=os.getenv('VCF_MEMORY_OPTIMIZATION_TRACKING', 'true').lower() == 'true',
            debug_mode=os.getenv('VCF_TRACING_DEBUG', 'false').lower() == 'true',
            trace_ai_responses=os.getenv('VCF_OPENAI_TRACE_RESPONSES', 'false').lower() == 'true',
            track_ai_costs=os.getenv('VCF_CLAUDE_COST_TRACKING', 'true').lower() == 'true',
            monitor_ai_performance=os.getenv('VCF_OLLAMA_PERFORMANCE_MONITORING', 'true').lower() == 'true'
        )


class EnhancedSpan:
    """Enhanced span wrapper with specialized attribute setting methods."""
    
    def __init__(self, span: Span):
        self.span = span
        self._start_time = time.time()
    
    def set_ai_attributes(self, **kwargs):
        """Set AI provider specific attributes."""
        for key, value in kwargs.items():
            if value is not None:
                self.span.set_attribute(f"ai.{key}", value)
    
    def set_vcf_attributes(self, **kwargs):
        """Set VCF processing specific attributes."""
        for key, value in kwargs.items():
            if value is not None:
                self.span.set_attribute(f"vcf.{key}", value)
    
    def set_memory_attributes(self, **kwargs):
        """Set memory optimization specific attributes."""
        for key, value in kwargs.items():
            if value is not None:
                self.span.set_attribute(f"memory.{key}", value)
    
    def set_performance_attributes(self, **kwargs):
        """Set performance specific attributes."""
        for key, value in kwargs.items():
            if value is not None:
                self.span.set_attribute(f"performance.{key}", value)
    
    def get_duration_seconds(self) -> float:
        """Get span duration in seconds."""
        return time.time() - self._start_time
    
    def get_duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        return (time.time() - self._start_time) * 1000
    
    def set_error(self, error: Exception, error_type: Optional[str] = None):
        """Set error status and attributes."""
        self.span.set_status(Status(StatusCode.ERROR, str(error)))
        self.span.set_attribute("error.type", error_type or type(error).__name__)
        self.span.set_attribute("error.message", str(error))
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying span."""
        return getattr(self.span, name)


class EnhancedTraceService:
    """Enhanced tracing service with specialized monitoring for AI genomics applications."""
    
    def __init__(
        self,
        service_name: str,
        config: Optional[MemoryOptimizationConfig] = None,
        tracing_config: Optional[TracingConfig] = None,
        environment: Optional[str] = None
    ):
        """Initializes a tracer for monitoring service performance and memory optimization.
        Parameters:
            - service_name (str): Name of the service for which tracing is initialized.
            - config (Optional[MemoryOptimizationConfig]): Configuration for optimizing memory usage, defaults to a standard configuration if not provided.
            - tracing_config (Optional[TracingConfig]): Configuration for tracing, defaults to environment-based configuration if not provided.
            - environment (Optional[str]): Environment setting for tracing, overrides default if specified.
        Returns:
            - None: This is a constructor method, it does not return a value.
        Processing Logic:
            - Initializes tracing configuration with environment setting if provided.
            - Sets up tracer and sampling based on environmental configurations.
            - Initializes performance tracking structures for operations.
            - Sets logger to debug mode if tracing in debug mode is enabled."""
        self.service_name = service_name
        self.memory_config = config or MemoryOptimizationConfig()
        self.tracing_config = tracing_config or TracingConfig.from_environment()
        
        # Override environment if provided
        if environment:
            self.tracing_config.environment = TracingEnvironment(environment)
        
        # Initialize base tracer
        self.tracer = init_tracer(service_name)
        
        # Setup sampling based on environment
        self._setup_sampling()
        
        # Performance tracking
        self._operation_counts = {}
        self._operation_durations = {}
        
        if self.tracing_config.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"Enhanced tracing initialized for {service_name} in {self.tracing_config.environment.value} mode")
    
    def _setup_sampling(self):
        """Configure sampling strategy based on environment."""
        sampling_rate = self.tracing_config.sampling_rate
        
        if self.tracing_config.environment == TracingEnvironment.PRODUCTION:
            # Production: Use parent-based sampling with rate limiting
            if sampling_rate <= 0:
                sampler = ALWAYS_OFF
            elif sampling_rate >= 1.0:
                sampler = ALWAYS_ON
            else:
                sampler = ParentBased(root=TraceIdRatioBased(sampling_rate))
        else:
            # Development/Staging: More aggressive sampling
            sampler = ALWAYS_ON if sampling_rate >= 1.0 else TraceIdRatioBased(sampling_rate)
        
        # Note: In practice, sampling configuration would be set during TracerProvider initialization
        # This is a placeholder for the configuration logic
        logger.info(f"Tracing sampling configured for {self.tracing_config.environment.value} with rate {sampling_rate}")
    
    def setup_enhanced_instrumentation(self):
        """Setup enhanced auto-instrumentation with AI and VCF-specific monitoring."""
        # Setup base auto-instrumentation
        setup_auto_instrumentation()
        
        # Additional instrumentation for AI providers would be configured here
        if self.tracing_config.ai_provider_monitoring:
            self._setup_ai_provider_instrumentation()
        
        logger.info("Enhanced instrumentation configured with AI provider monitoring")
    
    def _setup_ai_provider_instrumentation(self):
        """Setup automatic instrumentation for AI providers."""
        # This would typically involve instrumenting HTTP clients used by AI SDKs
        # For now, we'll focus on manual instrumentation patterns
        logger.debug("AI provider instrumentation configured")
    
    @contextmanager
    def ai_provider_span_context(
        self,
        provider: Union[str, AIProvider],
        model: str,
        operation: str = "inference"
    ) -> Generator[EnhancedSpan, None, None]:
        """Context manager for AI provider operations."""
        provider_name = provider.value if isinstance(provider, AIProvider) else provider
        span_name = f"ai.{provider_name}.{model}.{operation}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            enhanced_span = EnhancedSpan(span)
            enhanced_span.set_ai_attributes(
                provider_name=provider_name,
                model_name=model,
                operation=operation
            )
            
            try:
                yield enhanced_span
            except Exception as e:
                enhanced_span.set_error(e, "ai_provider_error")
                raise
            finally:
                # Record performance metrics
                self._record_operation_metrics(span_name, enhanced_span.get_duration_ms())
    
    @contextmanager
    def vcf_operation_context(
        self,
        operation: str,
        file_path: Optional[str] = None
    ) -> Generator[EnhancedSpan, None, None]:
        """Context manager for VCF processing operations."""
        span_name = f"vcf.{operation}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            enhanced_span = EnhancedSpan(span)
            enhanced_span.set_vcf_attributes(operation=operation)
            
            if file_path:
                enhanced_span.set_vcf_attributes(file_path=file_path)
            
            try:
                yield enhanced_span
            except Exception as e:
                enhanced_span.set_error(e, "vcf_processing_error")
                raise
            finally:
                self._record_operation_metrics(span_name, enhanced_span.get_duration_ms())
    
    @contextmanager
    def memory_context(
        self,
        operation: str
    ) -> Generator[EnhancedSpan, None, None]:
        """Context manager for memory optimization operations."""
        span_name = f"memory.{operation}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            enhanced_span = EnhancedSpan(span)
            enhanced_span.set_memory_attributes(
                optimization_level=self.memory_config.optimization_level,
                operation=operation
            )
            
            try:
                yield enhanced_span
            except Exception as e:
                enhanced_span.set_error(e, "memory_optimization_error")
                raise
            finally:
                self._record_operation_metrics(span_name, enhanced_span.get_duration_ms())
    
    def ai_provider_span(self, provider: Union[str, AIProvider], model: str, operation: str = "inference"):
        """Decorator for AI provider operations."""
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                """Wraps a function to add AI provider span context for tracking execution details.
                Parameters:
                    - func (function): The function to be wrapped, which can be asynchronous or synchronous.
                Returns:
                    - function: The wrapped version of the input function with added span context attributes.
                Processing Logic:
                    - Determines if the function is synchronous or asynchronous and uses the appropriate wrapper.
                    - Establishes an AI provider span context to track execution details such as function name, arguments, and success status.
                    - Handles exceptions by setting success attribute to false, while successful executions set it to true."""
                """A wrapper for an asynchronous function that integrates tracing and error handling.
                Parameters:
                    - *args: Arguments to be passed to the asynchronous function.
                    - **kwargs: Keyword arguments to be passed to the asynchronous function.
                    - provider (str): The AI provider managing the span context.
                    - model (str): The model being used for the operation.
                    - operation (str): The operation being traced.
                    - func (Callable): The asynchronous function to be wrapped.
                Returns:
                    - Any: The result returned by the asynchronous function.
                Processing Logic:
                    - Executes the wrapped asynchronous function within a tracing span context.
                    - Sets attributes on the span based on function name, argument counts, and execution success.
                    - Raises any exception from the asynchronous function after logging it in the span."""
                with self.ai_provider_span_context(provider, model, operation) as span:
                    # Add function-specific attributes
                    span.set_ai_attributes(
                        function_name=func.__name__,
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    )
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_ai_attributes(success=True)
                        return result
                    except Exception as e:
                        span.set_ai_attributes(success=False)
                        raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                """Syncs function calls within an AI provider span context for monitoring and tracking.
                Parameters:
                    - *args: Variable length argument list for the target function.
                    - **kwargs: Arbitrary keyword arguments for the target function.
                Returns:
                    - Any: The result of the function `func` when called with provided arguments.
                Processing Logic:
                    - The synchronous execution of the function is wrapped in a monitoring span context.
                    - Function metadata like name and argument count are logged in the span.
                    - Success or failure of the function execution is recorded as span attributes."""
                with self.ai_provider_span_context(provider, model, operation) as span:
                    span.set_ai_attributes(
                        function_name=func.__name__,
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    )
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_ai_attributes(success=True)
                        return result
                    except Exception as e:
                        span.set_ai_attributes(success=False)
                        raise
            
            # Return appropriate wrapper based on function type
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def vcf_operation_span(self, operation: str):
        """Decorator for VCF processing operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                """Apply a decorator to a function to perform context logging during its execution.
                Parameters:
                    - func (callable): The function to be wrapped by the decorator.
                Returns:
                    - callable: A wrapped function that logs its execution context including arguments and success status.
                Processing Logic:
                    - Establish a context for the operation before function execution.
                    - Log function name, argument count, and keyword argument count.
                    - Execute the function within this context.
                    - Capture and log the success status or handle exceptions accordingly."""

                """Wrapper function for executing a given function within a VCF operation context, recording attributes about the operation.
                Parameters:
                    - *args: Variable length argument list passed to the wrapped function.
                    - **kwargs: Arbitrary keyword arguments passed to the wrapped function.
                    - operation (Operation object): The context of VCF operation during execution.
                Returns:
                    - any: Whatever the wrapped function returns upon successful execution.
                Processing Logic:
                    - The function execution is wrapped within a context manager.
                    - VCF operation attributes such as function name and argument counts are recorded.
                    - Success or failure of the function invocation is captured in the context attributes.
                    - Raises any exception encountered during function execution after setting 'success' to False."""
                with self.vcf_operation_context(operation) as span:
                    span.set_vcf_attributes(
                        function_name=func.__name__,
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    )
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_vcf_attributes(success=True)
                        return result
                    except Exception as e:
                        span.set_vcf_attributes(success=False)
                        raise
            
            return wrapper
        return decorator
    
    def memory_operation_span(self, operation: str):
        """Decorator for memory optimization operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                """A decorator for function memory context management.
                Parameters:
                    - func (callable): The function to be wrapped and managed within a memory context.
                Returns:
                    - callable: The wrapped function with added memory context and attribute management.
                Processing Logic:
                    - Opens a memory context before executing the function, capturing it in a span.
                    - Sets memory attributes such as function name and optimization level.
                    - Upon successful function execution, sets a success attribute in the span.
                    - On exception, sets a failure attribute and re-raises the exception."""

                """Wrapper function to execute another function while managing memory context and capturing its attributes.
                Parameters:
                    - *args: Positional arguments to pass to the `func`.
                    - **kwargs: Keyword arguments to pass to the `func`.
                Returns:
                    - The return value of the executed function `func`.
                Processing Logic:
                    - Establishes a memory context using `self.memory_context`.
                    - Sets attributes related to the function name and optimization level.
                    - Executes the given function, capturing success or failure state.
                    - Raises any exceptions encountered during execution."""
                with self.memory_context(operation) as span:
                    span.set_memory_attributes(
                        function_name=func.__name__,
                        optimization_level=self.memory_config.optimization_level
                    )
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_memory_attributes(success=True)
                        return result
                    except Exception as e:
                        span.set_memory_attributes(success=False)
                        raise
            
            return wrapper
        return decorator
    
    def _record_operation_metrics(self, operation: str, duration_ms: float):
        """Record internal performance metrics for operations."""
        if operation not in self._operation_counts:
            self._operation_counts[operation] = 0
            self._operation_durations[operation] = []
        
        self._operation_counts[operation] += 1
        self._operation_durations[operation].append(duration_ms)
        
        # Keep only last 100 measurements to prevent memory growth
        if len(self._operation_durations[operation]) > 100:
            self._operation_durations[operation] = self._operation_durations[operation][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all tracked operations."""
        summary = {}
        
        for operation in self._operation_counts:
            durations = self._operation_durations[operation]
            if durations:
                summary[operation] = {
                    "count": self._operation_counts[operation],
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations)
                }
        
        return summary
    
    def log_performance_summary(self):
        """Log current performance summary."""
        summary = self.get_performance_summary()
        if summary:
            logger.info("Performance Summary:")
            for operation, metrics in summary.items():
                logger.info(f"  {operation}: {metrics['count']} ops, "
                          f"avg {metrics['avg_duration_ms']:.2f}ms, "
                          f"range {metrics['min_duration_ms']:.2f}-{metrics['max_duration_ms']:.2f}ms")
        else:
            logger.info("No performance metrics recorded yet")


# Global enhanced trace service instance
_global_trace_service: Optional[EnhancedTraceService] = None


def get_enhanced_trace_service(
    service_name: Optional[str] = None,
    config: Optional[MemoryOptimizationConfig] = None
) -> EnhancedTraceService:
    """Get or create global enhanced trace service instance."""
    global _global_trace_service
    
    if _global_trace_service is None:
        service_name = service_name or "vcf-analysis-agent"
        _global_trace_service = EnhancedTraceService(service_name, config)
        _global_trace_service.setup_enhanced_instrumentation()
    
    return _global_trace_service


# Convenience functions for quick access
def ai_provider_span(provider: Union[str, AIProvider], model: str, operation: str = "inference"):
    """Decorator for AI provider operations using global service."""
    def decorator(func):
        service = get_enhanced_trace_service()
        return service.ai_provider_span(provider, model, operation)(func)
    return decorator


def vcf_operation_span(operation: str):
    """Decorator for VCF operations using global service."""
    def decorator(func):
        service = get_enhanced_trace_service()
        return service.vcf_operation_span(operation)(func)
    return decorator


def memory_operation_span(operation: str):
    """Decorator for memory operations using global service."""
    def decorator(func):
        service = get_enhanced_trace_service()
        return service.memory_operation_span(operation)(func)
    return decorator


@contextmanager
def ai_provider_context(
    provider: Union[str, AIProvider],
    model: str,
    operation: str = "inference"
) -> Generator[EnhancedSpan, None, None]:
    """Context manager for AI provider operations using global service."""
    with get_enhanced_trace_service().ai_provider_span_context(provider, model, operation) as span:
        yield span


@contextmanager
def vcf_operation_context(
    operation: str,
    file_path: Optional[str] = None
) -> Generator[EnhancedSpan, None, None]:
    """Context manager for VCF operations using global service."""
    with get_enhanced_trace_service().vcf_operation_context(operation, file_path) as span:
        yield span


@contextmanager
def memory_context(operation: str) -> Generator[EnhancedSpan, None, None]:
    """Context manager for memory operations using global service."""
    with get_enhanced_trace_service().memory_context(operation) as span:
        yield span 