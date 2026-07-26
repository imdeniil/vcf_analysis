"""
Phase 5 Monitoring and Observability
====================================

Comprehensive monitoring system for the hybrid Apache Iggy + Kafka streaming
architecture with OpenTelemetry integration, metrics collection, and health monitoring.

Features:
- Real-time performance metrics collection
- OpenTelemetry tracing and metrics integration
- Health monitoring and alerting
- Performance bottleneck detection
- Dashboard-ready metrics export
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from enum import Enum
from collections import deque
import threading

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import structlog

from .config import Phase5Config, get_config

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for VCF processing."""
    variants_processed: int = 0
    variants_failed: int = 0
    processing_time_total: float = 0.0
    average_latency_ms: float = 0.0
    throughput_variants_per_sec: float = 0.0
    compression_ratio: float = 1.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PlatformHealthStatus:
    """Health status for streaming platforms."""
    platform_name: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: float = 0.0
    error_count: int = 0
    connection_count: int = 0
    status_message: str = ""


class PrometheusMetrics:
    """Prometheus metrics collector for Phase 5."""
    
    def __init__(self):
        # VCF Processing Metrics
        """Initializes various metrics for monitoring VCF processing and system performance.
        Parameters:
            None
        Returns:
            None
        Processing Logic:
            - Sets up counters, histograms, and gauges to track VCF processing, platform health, 
        system resource usage, message metrics, errors, and performance targets.
            - Metrics are categorized into VCF processing, platform health, system resources, 
        message properties, error tracking, and performance monitoring.
            - Specific labels are applied to track dimensions such as platform, chromosome, status, 
        component, error type, severity, compression, and others crucial for detailed monitoring."""
        self.variants_processed_total = Counter(
            'vcf_variants_processed_total',
            'Total number of VCF variants processed',
            ['platform', 'chromosome', 'status']
        )
        
        self.variants_processing_duration = Histogram(
            'vcf_variants_processing_duration_seconds',
            'Time spent processing VCF variants',
            ['platform', 'chromosome'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.variants_batch_size = Histogram(
            'vcf_variants_batch_size',
            'Size of VCF variant batches processed',
            ['platform'],
            buckets=[100, 500, 1000, 2000, 5000, 10000]
        )
        
        # Platform Health Metrics
        self.platform_health_status = Gauge(
            'platform_health_status',
            'Health status of streaming platforms (1=healthy, 0=unhealthy)',
            ['platform']
        )
        
        self.platform_response_time = Histogram(
            'platform_response_time_seconds',
            'Response time for platform health checks',
            ['platform'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.platform_connections = Gauge(
            'platform_connections_active',
            'Number of active connections per platform',
            ['platform']
        )
        
        # System Resource Metrics
        self.memory_usage_bytes = Gauge(
            'vcf_agent_memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        self.cpu_usage_percent = Gauge(
            'vcf_agent_cpu_usage_percent',
            'CPU usage percentage',
            ['component']
        )
        
        # Message Metrics
        self.message_size_bytes = Histogram(
            'vcf_message_size_bytes',
            'Size of VCF messages in bytes',
            ['platform', 'compression_enabled'],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
        )
        
        self.compression_ratio = Histogram(
            'vcf_message_compression_ratio',
            'Compression ratio for VCF messages',
            ['platform'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'vcf_agent_errors_total',
            'Total number of errors',
            ['component', 'error_type', 'severity']
        )
        
        # Performance Targets
        self.performance_target_throughput = Gauge(
            'vcf_agent_target_throughput_variants_per_sec',
            'Target throughput for VCF processing'
        )
        
        self.performance_actual_throughput = Gauge(
            'vcf_agent_actual_throughput_variants_per_sec',
            'Actual throughput for VCF processing'
        )
        
        # System Information
        self.system_info = Info(
            'vcf_agent_system_info',
            'System information for VCF agent'
        )


class CircuitBreakerState(Enum):
    """Circuit breaker states for platform health management."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure state - requests blocked
    HALF_OPEN = "half_open"  # Recovery testing


@dataclass
class HealthMetrics:
    """Health metrics for platform monitoring."""
    platform: str
    timestamp: float
    latency_ms: float
    error_rate: float
    throughput_per_sec: float
    is_healthy: bool
    connection_status: bool
    error_count: int = 0
    success_count: int = 0
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        if not self.connection_status:
            return 0.0
        
        # Weight different factors
        latency_score = max(0, 1.0 - (self.latency_ms / 1000))  # 1s = 0 score
        error_score = max(0, 1.0 - self.error_rate)
        throughput_score = min(1.0, self.throughput_per_sec / 100)  # 100 ops/sec = full score
        
        return (latency_score * 0.4 + error_score * 0.4 + throughput_score * 0.2)


class CircuitBreaker:
    """
    Circuit breaker implementation for platform health management.
    
    Implements the classic circuit breaker pattern with configurable thresholds
    and recovery logic to prevent cascade failures in the hybrid architecture.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 half_open_max_calls: int = 3):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before transitioning to half-open
            half_open_max_calls: Max calls allowed in half-open state for testing
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        
        self._lock = threading.Lock()
        
        # Metrics
        self.state_changes = meter.create_counter(
            "circuit_breaker_state_changes_total",
            description="Total circuit breaker state changes"
        )
    
    def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit state."""
        with self._lock:
            current_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                return True
            
            elif self.state == CircuitBreakerState.OPEN:
                if current_time - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    self.state_changes.add(1, {"from": "open", "to": "half_open"})
                    logger.info("Circuit breaker transitioning to HALF_OPEN for recovery testing")
                    return True
                return False
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self):
        """Record successful operation."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                # If we've had enough successful calls, close the circuit
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.state_changes.add(1, {"from": "half_open", "to": "closed"})
                    logger.info("Circuit breaker closed - platform recovered")
            
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)  # Decay failures
    
    def record_failure(self):
        """Record failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.state_changes.add(1, {"from": "closed", "to": "open"})
                    logger.warning(f"Circuit breaker opened - {self.failure_count} failures detected")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.state_changes.add(1, {"from": "half_open", "to": "open"})
                logger.warning("Circuit breaker reopened during recovery testing")
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state


class IggyMetricsCollector:
    """
    Metrics collector for Apache Iggy platform monitoring.
    
    Tracks Iggy-specific performance metrics, health status, and provides
    data for intelligent routing decisions in the hybrid architecture.
    """
    
    def __init__(self):
        """Initializes an object with monitoring attributes for Apache Iggy operations.
        Parameters:
            - None
        Returns:
            - None
        Processing Logic:
            - Initializes a deque to maintain a history of up to 100 metrics records.
            - Sets up a CircuitBreaker with specific failure and recovery parameters to handle operation failures.
            - Creates and assigns performance metrics for latency, throughput, and health using a meter object."""
        self.is_running = False
        self.metrics_history: deque = deque(maxlen=100)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,  # Iggy is primary - be sensitive to issues
            recovery_timeout=30,
            half_open_max_calls=2
        )
        
        # Performance metrics
        self.latency_histogram = meter.create_histogram(
            "iggy_operation_latency_seconds",
            description="Apache Iggy operation latency"
        )
        self.throughput_gauge = meter.create_gauge(
            "iggy_throughput_variants_per_sec",
            description="Apache Iggy throughput in variants per second"
        )
        self.health_gauge = meter.create_gauge(
            "iggy_health_score",
            description="Apache Iggy health score (0-1)"
        )
    
    async def start(self):
        """Start metrics collection."""
        self.is_running = True
        logger.info("Iggy metrics collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.is_running = False
        logger.info("Iggy metrics collector stopped")
    
    def record_operation(self, latency_ms: float, success: bool, throughput: float = 0):
        """Record an Iggy operation for monitoring."""
        if success:
            self.circuit_breaker.record_success()
        else:
            self.circuit_breaker.record_failure()
        
        # Update metrics
        self.latency_histogram.record(latency_ms / 1000, {"platform": "iggy"})
        self.throughput_gauge.set(throughput, {"platform": "iggy"})
        
        # Store for history
        metrics = HealthMetrics(
            platform="iggy",
            timestamp=time.time(),
            latency_ms=latency_ms,
            error_rate=0.0 if success else 1.0,
            throughput_per_sec=throughput,
            is_healthy=success,
            connection_status=success
        )
        
        self.metrics_history.append(metrics)
        self.health_gauge.set(metrics.get_health_score(), {"platform": "iggy"})
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health status."""
        if not self.metrics_history:
            return HealthMetrics(
                platform="iggy",
                timestamp=time.time(),
                latency_ms=0,
                error_rate=1.0,
                throughput_per_sec=0,
                is_healthy=False,
                connection_status=False
            )
        
        return self.metrics_history[-1]
    
    def can_handle_request(self) -> bool:
        """Check if Iggy can handle requests based on circuit breaker."""
        return self.circuit_breaker.can_execute()


class KafkaMetricsCollector:
    """
    Metrics collector for Apache Kafka platform monitoring.
    
    Tracks Kafka-specific performance metrics, health status, and provides
    fallback readiness data for intelligent routing decisions.
    """
    
    def __init__(self):
        """Initializes the Kafka monitoring and circuit breaking mechanism.
        Parameters:
            None
        Returns:
            None
        Processing Logic:
            - Sets up a circuit breaker with specified thresholds for monitoring Kafka operations.
            - Initializes performance metrics using histograms and gauges for latency, throughput, health, and consumer lag.
            - The metrics history is stored in a deque with a maximum length of 100 for efficient tracking."""
        self.is_running = False
        self.metrics_history: deque = deque(maxlen=100)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,  # Kafka is fallback - more tolerant
            recovery_timeout=60,
            half_open_max_calls=3
        )
        
        # Performance metrics
        self.latency_histogram = meter.create_histogram(
            "kafka_operation_latency_seconds",
            description="Apache Kafka operation latency"
        )
        self.throughput_gauge = meter.create_gauge(
            "kafka_throughput_variants_per_sec",
            description="Apache Kafka throughput in variants per second"
        )
        self.health_gauge = meter.create_gauge(
            "kafka_health_score",
            description="Apache Kafka health score (0-1)"
        )
        self.consumer_lag_gauge = meter.create_gauge(
            "kafka_consumer_lag",
            description="Kafka consumer lag"
        )
    
    async def start(self):
        """Start metrics collection."""
        self.is_running = True
        logger.info("Kafka metrics collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.is_running = False
        logger.info("Kafka metrics collector stopped")
    
    def record_operation(self, latency_ms: float, success: bool, throughput: float = 0):
        """Record a Kafka operation for monitoring."""
        if success:
            self.circuit_breaker.record_success()
        else:
            self.circuit_breaker.record_failure()
        
        # Update metrics
        self.latency_histogram.record(latency_ms / 1000, {"platform": "kafka"})
        self.throughput_gauge.set(throughput, {"platform": "kafka"})
        
        # Store for history
        metrics = HealthMetrics(
            platform="kafka",
            timestamp=time.time(),
            latency_ms=latency_ms,
            error_rate=0.0 if success else 1.0,
            throughput_per_sec=throughput,
            is_healthy=success,
            connection_status=success
        )
        
        self.metrics_history.append(metrics)
        self.health_gauge.set(metrics.get_health_score(), {"platform": "kafka"})
    
    def record_consumer_lag(self, lag: int):
        """Record Kafka consumer lag for monitoring."""
        self.consumer_lag_gauge.set(lag, {"platform": "kafka"})
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health status."""
        if not self.metrics_history:
            return HealthMetrics(
                platform="kafka",
                timestamp=time.time(),
                latency_ms=0,
                error_rate=1.0,
                throughput_per_sec=0,
                is_healthy=False,
                connection_status=False
            )
        
        return self.metrics_history[-1]
    
    def can_handle_request(self) -> bool:
        """Check if Kafka can handle requests based on circuit breaker."""
        return self.circuit_breaker.can_execute()


class PerformanceMonitor:
    """
    Comprehensive performance monitor for the hybrid streaming architecture.
    
    Coordinates health monitoring across both Apache Iggy and Kafka platforms,
    implements circuit breaker patterns, and provides intelligent routing
    decisions based on real-time performance metrics.
    
    Features:
    - Dual-platform health monitoring with circuit breakers
    - Configurable failover thresholds per environment
    - Real-time performance tracking and alerting
    - Integration with Prometheus and OpenTelemetry
    """
    
    def __init__(self, config_thresholds: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.
        
        Args:
            config_thresholds: Configuration thresholds for different environments
        """
        self.iggy_collector = IggyMetricsCollector()
        self.kafka_collector = KafkaMetricsCollector()
        
        # Configuration thresholds
        self.thresholds = config_thresholds or {
            "max_latency_ms": 5000,      # 5s max latency before failover
            "max_error_rate": 0.05,      # 5% max error rate
            "min_throughput": 10,        # 10 variants/sec minimum
            "health_check_interval": 30   # 30s between health checks
        }
        
        self.is_running = False
        self.last_health_check = 0
        
        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Alerting
        self.alert_counter = meter.create_counter(
            "performance_alerts_total",
            description="Total performance alerts triggered"
        )
    
    async def start(self):
        """Start performance monitoring."""
        if self.is_running:
            return
        
        logger.info("Starting performance monitor")
        
        await self.iggy_collector.start()
        await self.kafka_collector.start()
        
        self.is_running = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitor started")
    
    async def stop(self):
        """Stop performance monitoring."""
        if not self.is_running:
            return
        
        logger.info("Stopping performance monitor")
        
        self.is_running = False
        
        # Stop monitoring task
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.iggy_collector.stop()
        await self.kafka_collector.stop()
        
        logger.info("Performance monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.thresholds["health_check_interval"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Back off on errors
    
    async def _perform_health_check(self):
        """Perform comprehensive health check across platforms."""
        current_time = time.time()
        
        # Get current health from both platforms
        iggy_health = self.iggy_collector.get_current_health()
        kafka_health = self.kafka_collector.get_current_health()
        
        # Check thresholds and trigger alerts
        if iggy_health.latency_ms > self.thresholds["max_latency_ms"]:
            self.alert_counter.add(1, {"platform": "iggy", "type": "latency"})
            logger.warning(f"Iggy latency threshold exceeded: {iggy_health.latency_ms}ms")
        
        if iggy_health.error_rate > self.thresholds["max_error_rate"]:
            self.alert_counter.add(1, {"platform": "iggy", "type": "error_rate"})
            logger.warning(f"Iggy error rate threshold exceeded: {iggy_health.error_rate}")
        
        if kafka_health.latency_ms > self.thresholds["max_latency_ms"] * 2:  # More tolerant for fallback
            self.alert_counter.add(1, {"platform": "kafka", "type": "latency"})
            logger.warning(f"Kafka latency threshold exceeded: {kafka_health.latency_ms}ms")
        
        self.last_health_check = current_time
    
    def record_iggy_operation(self, latency_ms: float, success: bool, throughput: float = 0):
        """Record Iggy operation metrics."""
        self.iggy_collector.record_operation(latency_ms, success, throughput)
    
    def record_kafka_operation(self, latency_ms: float, success: bool, throughput: float = 0):
        """Record Kafka operation metrics."""
        self.kafka_collector.record_operation(latency_ms, success, throughput)
    
    def get_recommended_platform(self) -> str:
        """Get recommended platform based on current health metrics."""
        iggy_can_handle = self.iggy_collector.can_handle_request()
        kafka_can_handle = self.kafka_collector.can_handle_request()
        
        if iggy_can_handle:
            iggy_health = self.iggy_collector.get_current_health()
            if iggy_health.get_health_score() > 0.7:  # Good health threshold
                return "iggy"
        
        if kafka_can_handle:
            kafka_health = self.kafka_collector.get_current_health()
            if kafka_health.get_health_score() > 0.5:  # Lower threshold for fallback
                return "kafka"
        
        # If both are unhealthy, prefer Iggy for performance when it recovers
        return "iggy"
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status."""
        iggy_health = self.iggy_collector.get_current_health()
        kafka_health = self.kafka_collector.get_current_health()
        
        return {
            "recommended_platform": self.get_recommended_platform(),
            "iggy": {
                "health_score": iggy_health.get_health_score(),
                "can_handle_requests": self.iggy_collector.can_handle_request(),
                "circuit_breaker_state": self.iggy_collector.circuit_breaker.get_state().value,
                "latency_ms": iggy_health.latency_ms,
                "error_rate": iggy_health.error_rate,
                "throughput": iggy_health.throughput_per_sec
            },
            "kafka": {
                "health_score": kafka_health.get_health_score(),
                "can_handle_requests": self.kafka_collector.can_handle_request(),
                "circuit_breaker_state": self.kafka_collector.circuit_breaker.get_state().value,
                "latency_ms": kafka_health.latency_ms,
                "error_rate": kafka_health.error_rate,
                "throughput": kafka_health.throughput_per_sec
            },
            "thresholds": self.thresholds,
            "last_health_check": self.last_health_check
        }


# Utility functions
def create_performance_metrics(environment: str = "development") -> PerformanceMonitor:
    """Create performance monitor with environment-specific thresholds."""
    if environment == "production":
        thresholds = {
            "max_latency_ms": 1000,    # Stricter for production
            "max_error_rate": 0.01,    # 1% max error rate
            "min_throughput": 100,     # Higher throughput requirement
            "health_check_interval": 15 # More frequent checks
        }
    elif environment == "staging":
        thresholds = {
            "max_latency_ms": 2000,
            "max_error_rate": 0.03,
            "min_throughput": 50,
            "health_check_interval": 20
        }
    else:  # development
        thresholds = {
            "max_latency_ms": 5000,
            "max_error_rate": 0.10,
            "min_throughput": 10,
            "health_check_interval": 60
        }
    
    return PerformanceMonitor(thresholds) 