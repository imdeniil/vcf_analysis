"""
Phase 5 Configuration Management
===============================

Production-ready configuration system for the hybrid Apache Iggy + Kafka
streaming architecture with comprehensive validation and environment support.

Features:
- Environment-specific configurations (dev, staging, production)
- Comprehensive validation and error handling
- Performance tuning parameters
- Security configuration (TLS, encryption)
- Monitoring and observability settings
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TransportProtocol(Enum):
    """Supported transport protocols."""
    QUIC = "quic"
    TCP = "tcp"
    HTTP = "http"


@dataclass
class IggyConfig:
    """
    Apache Iggy streaming platform configuration.
    
    Optimized for ultra-low latency genomic data streaming.
    """
    # Connection settings
    host: str = "localhost"
    tcp_port: int = 8090
    http_port: int = 3000
    quic_port: int = 8080
    
    # Transport configuration
    transport_protocol: TransportProtocol = TransportProtocol.QUIC
    connection_timeout: int = 30
    request_timeout: int = 10
    
    # Performance settings
    max_connections: int = 1000
    connection_pool_size: int = 100
    keep_alive_interval: int = 30
    
    # Stream configuration
    stream_name: str = "vcf-variants-stream"
    topic_name: str = "vcf-variants"
    partition_count: int = 24  # One per chromosome + extras
    
    # Message settings
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    batch_size: int = 1000
    compression_enabled: bool = True
    compression_level: int = 3
    
    # Security settings
    tls_enabled: bool = True
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    
    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    
    # Performance optimization
    zero_copy_enabled: bool = True
    message_deduplication: bool = True
    auto_commit: bool = True
    
    # Monitoring
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    sampling_rate: float = 0.1  # 10% sampling for production
    
    def get_connection_url(self) -> str:
        """Get the connection URL based on transport protocol."""
        if self.transport_protocol == TransportProtocol.QUIC:
            return f"quic://{self.host}:{self.quic_port}"
        elif self.transport_protocol == TransportProtocol.TCP:
            return f"tcp://{self.host}:{self.tcp_port}"
        elif self.transport_protocol == TransportProtocol.HTTP:
            protocol = "https" if self.tls_enabled else "http"
            return f"{protocol}://{self.host}:{self.http_port}"
        else:
            raise ValueError(f"Unsupported transport protocol: {self.transport_protocol}")


@dataclass
class KafkaConfig:
    """
    Apache Kafka fallback platform configuration.
    
    Provides reliable fallback for the Iggy primary platform.
    """
    # Broker settings
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    
    # Topic configuration
    topic_name: str = "vcf-variants-fallback"
    partition_count: int = 24
    replication_factor: int = 3
    
    # Producer settings
    acks: str = "all"  # Wait for all replicas
    retries: int = 2147483647  # Infinite retries
    max_in_flight_requests: int = 5
    enable_idempotence: bool = True
    compression_type: str = "zstd"
    
    # Consumer settings
    group_id: str = "vcf-analysis-agent"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    
    # Performance settings
    batch_size: int = 16384
    linger_ms: int = 5
    buffer_memory: int = 33554432  # 32MB
    max_poll_records: int = 500
    
    # Security settings
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # SASL settings
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    
    # Monitoring
    metrics_enabled: bool = True
    jmx_port: Optional[int] = 9999


@dataclass
class PerformanceConfig:
    """Performance and scaling configuration."""
    # Throughput targets
    target_throughput_variants_per_sec: int = 5000
    max_throughput_variants_per_sec: int = 10000
    
    # Resource limits
    max_memory_usage_mb: int = 2048
    max_cpu_percentage: int = 80
    
    # Scaling parameters
    auto_scaling_enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.3
    
    # Batching settings
    batch_size: int = 1000
    batch_timeout_ms: int = 100
    max_batch_size: int = 5000
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_s: int = 30
    half_open_max_calls: int = 3


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    # OpenTelemetry settings
    otel_enabled: bool = True
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "vcf-agent-phase5"
    
    # Tracing configuration
    tracing_enabled: bool = True
    trace_sampling_rate: float = 0.1
    trace_export_timeout_s: int = 30
    
    # Metrics configuration
    metrics_enabled: bool = True
    metrics_port: int = 8080
    metrics_path: str = "/metrics"
    
    # Prometheus settings
    prometheus_enabled: bool = True
    prometheus_pushgateway_url: Optional[str] = None
    
    # Logging configuration
    log_level: str = "INFO"
    structured_logging: bool = True
    log_format: str = "json"
    
    # Health check settings
    health_check_enabled: bool = True
    health_check_interval_s: int = 30
    
    # Performance monitoring
    collect_performance_metrics: bool = True
    performance_metric_interval_s: int = 10


@dataclass
class Phase5Config:
    """
    Master configuration for Phase 5 hybrid streaming architecture.
    
    Combines Iggy, Kafka, performance, and monitoring configurations
    with environment-specific overrides.
    """
    # Environment configuration
    environment: Environment = Environment.DEVELOPMENT
    
    # Platform configurations
    iggy: IggyConfig = field(default_factory=IggyConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    
    # System configurations
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Hybrid architecture settings
    primary_platform: str = "iggy"  # "iggy" or "kafka"
    fallback_enabled: bool = True
    failover_threshold_error_rate: float = 0.05  # 5% error rate triggers failover
    failover_threshold_latency_ms: int = 100
    
    # Data processing settings
    vcf_processing_enabled: bool = True
    variant_validation_enabled: bool = True
    duplicate_detection_enabled: bool = True
    
    # Storage integration
    lancedb_enabled: bool = True
    kuzu_enabled: bool = True
    
    def __post_init__(self):
        """Apply environment-specific configurations."""
        self._apply_environment_overrides()
        self._validate_configuration()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == Environment.PRODUCTION:
            # Production optimizations
            self.iggy.sampling_rate = 0.01  # 1% sampling
            self.iggy.max_connections = 5000
            self.iggy.tls_enabled = True
            self.iggy.encryption_enabled = True
            
            self.kafka.replication_factor = 3
            self.kafka.acks = "all"
            self.kafka.security_protocol = "SSL"
            
            self.performance.max_instances = 50
            self.performance.auto_scaling_enabled = True
            
            self.monitoring.trace_sampling_rate = 0.01
            
        elif self.environment == Environment.STAGING:
            # Staging optimizations
            self.iggy.sampling_rate = 0.1  # 10% sampling
            self.iggy.max_connections = 1000
            
            self.kafka.replication_factor = 2
            
            self.performance.max_instances = 10
            
        elif self.environment == Environment.DEVELOPMENT:
            # Development optimizations
            self.iggy.sampling_rate = 1.0  # 100% sampling
            self.iggy.tls_enabled = False
            self.iggy.encryption_enabled = False
            
            self.kafka.replication_factor = 1
            self.kafka.security_protocol = "PLAINTEXT"
            
            self.performance.auto_scaling_enabled = False
            self.performance.max_instances = 1
            
            self.monitoring.trace_sampling_rate = 1.0
    
    def _validate_configuration(self):
        """Validate configuration consistency and requirements."""
        errors = []
        
        # Validate port conflicts
        if self.iggy.tcp_port == self.iggy.http_port:
            errors.append("Iggy TCP and HTTP ports cannot be the same")
        
        # Validate performance settings
        if self.performance.min_instances > self.performance.max_instances:
            errors.append("min_instances cannot be greater than max_instances")
        
        # Validate thresholds
        if self.performance.scale_up_threshold <= self.performance.scale_down_threshold:
            errors.append("scale_up_threshold must be greater than scale_down_threshold")
        
        # Validate security in production
        if self.environment == Environment.PRODUCTION:
            if not self.iggy.tls_enabled:
                errors.append("TLS must be enabled in production")
            if not self.iggy.encryption_enabled:
                errors.append("Encryption must be enabled in production")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @classmethod
    def from_environment(cls, env: str = None) -> 'Phase5Config':
        """
        Create configuration from environment variables.
        
        Args:
            env: Environment name (development, staging, production)
            
        Returns:
            Configured Phase5Config instance
        """
        if env is None:
            env = os.getenv('VCF_AGENT_ENV', 'development')
        
        environment = Environment(env.lower())
        config = cls(environment=environment)
        
        # Override with environment variables
        config._load_from_env_vars()
        
        return config
    
    def _load_from_env_vars(self):
        """Load configuration overrides from environment variables."""
        # Iggy overrides
        if os.getenv('IGGY_HOST'):
            self.iggy.host = os.getenv('IGGY_HOST')
        if os.getenv('IGGY_QUIC_PORT'):
            self.iggy.quic_port = int(os.getenv('IGGY_QUIC_PORT'))
        if os.getenv('IGGY_STREAM_NAME'):
            self.iggy.stream_name = os.getenv('IGGY_STREAM_NAME')
        
        # Kafka overrides
        if os.getenv('KAFKA_BOOTSTRAP_SERVERS'):
            self.kafka.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS').split(',')
        if os.getenv('KAFKA_TOPIC_NAME'):
            self.kafka.topic_name = os.getenv('KAFKA_TOPIC_NAME')
        
        # Monitoring overrides
        if os.getenv('OTEL_ENDPOINT'):
            self.monitoring.otel_endpoint = os.getenv('OTEL_ENDPOINT')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def get_iggy_client_config(self) -> Dict[str, Any]:
        """Get Iggy client configuration dictionary."""
        return {
            'host': self.iggy.host,
            'port': self.iggy.quic_port if self.iggy.transport_protocol == TransportProtocol.QUIC else self.iggy.tcp_port,
            'transport': self.iggy.transport_protocol.value,
            'tls_enabled': self.iggy.tls_enabled,
            'encryption_enabled': self.iggy.encryption_enabled,
            'max_connections': self.iggy.max_connections,
            'connection_timeout': self.iggy.connection_timeout,
            'username': self.iggy.username,
            'password': self.iggy.password,
            'api_key': self.iggy.api_key
        }
    
    def get_kafka_producer_config(self) -> Dict[str, Any]:
        """Get Kafka producer configuration dictionary."""
        return {
            'bootstrap_servers': self.kafka.bootstrap_servers,
            'acks': self.kafka.acks,
            'retries': self.kafka.retries,
            'max_in_flight_requests_per_connection': self.kafka.max_in_flight_requests,
            'enable_idempotence': self.kafka.enable_idempotence,
            'compression_type': self.kafka.compression_type,
            'batch_size': self.kafka.batch_size,
            'linger_ms': self.kafka.linger_ms,
            'buffer_memory': self.kafka.buffer_memory,
            'security_protocol': self.kafka.security_protocol
        }
    
    def get_kafka_consumer_config(self) -> Dict[str, Any]:
        """Get Kafka consumer configuration dictionary."""
        return {
            'bootstrap_servers': self.kafka.bootstrap_servers,
            'group_id': self.kafka.group_id,
            'auto_offset_reset': self.kafka.auto_offset_reset,
            'enable_auto_commit': self.kafka.enable_auto_commit,
            'session_timeout_ms': self.kafka.session_timeout_ms,
            'heartbeat_interval_ms': self.kafka.heartbeat_interval_ms,
            'max_poll_records': self.kafka.max_poll_records,
            'security_protocol': self.kafka.security_protocol
        }


# Configuration factory functions
def create_development_config() -> Phase5Config:
    """Create development environment configuration."""
    return Phase5Config.from_environment('development')


def create_staging_config() -> Phase5Config:
    """Create staging environment configuration."""
    return Phase5Config.from_environment('staging')


def create_production_config() -> Phase5Config:
    """Create production environment configuration."""
    return Phase5Config.from_environment('production')


# Global configuration instance
_global_config: Optional[Phase5Config] = None


def get_config() -> Phase5Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Phase5Config.from_environment()
    return _global_config


def set_config(config: Phase5Config):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config 