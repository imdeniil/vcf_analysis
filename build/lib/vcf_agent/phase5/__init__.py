"""
Phase 5: Hybrid Distributed Streaming Architecture
==================================================

This module implements the hybrid Apache Iggy + Kafka streaming architecture
for ultra-high-performance genomic variant processing.

Key Components:
- IggyVCFProcessor: Primary streaming platform using Apache Iggy
- KafkaVCFProcessor: Fallback streaming platform using Apache Kafka
- StreamingCoordinator: Dual-platform message routing and failover
- VCFVariantMessage: Optimized VCF variant serialization
- PerformanceMonitor: Health monitoring and metrics collection

Performance Targets:
- 10-50x throughput improvement (1,000-5,000 variants/sec)
- <1ms latency with QUIC transport
- 99.99% availability with automatic failover
- Millions of messages per second capability

Author: VCF Analysis Agent Phase 5 Team
Version: 0.5.0
License: Apache 2.0
"""

__version__ = "0.5.0"
__author__ = "VCF Analysis Agent Phase 5 Team"

# Core streaming components
from .iggy_processor import IggyVCFProcessor
from .kafka_processor import KafkaVCFProcessor
from .streaming_coordinator import StreamingCoordinator
from .vcf_message import VCFVariantMessage, VCFMessageSerializer
from .monitoring import PerformanceMonitor, IggyMetricsCollector
from .config import (
    Phase5Config, IggyConfig, KafkaConfig, Environment,
    create_development_config, create_staging_config, create_production_config
)

__all__ = [
    # Main processors
    "IggyVCFProcessor",
    "KafkaVCFProcessor", 
    "StreamingCoordinator",
    
    # Message handling
    "VCFVariantMessage",
    "VCFMessageSerializer",
    
    # Monitoring
    "PerformanceMonitor",
    "IggyMetricsCollector",
    
    # Configuration
    "Phase5Config",
    "IggyConfig", 
    "KafkaConfig",
    "Environment",
    "create_development_config",
    "create_staging_config", 
    "create_production_config"
] 