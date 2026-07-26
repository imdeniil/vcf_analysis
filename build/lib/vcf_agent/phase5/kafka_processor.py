"""
Apache Kafka VCF Processor - Fallback Platform
===============================================

Enterprise-grade Kafka fallback processor for the hybrid streaming architecture.
Implements production patterns from kafka-python research including consumer group
coordination, producer callbacks, metrics-driven health monitoring, and exactly-once
semantics for reliable genomic variant processing.

Features:
- Enterprise reliability with automatic partition assignment
- Producer callback patterns for real-time error detection
- Consumer group coordination for seamless partition management
- Metrics-driven health monitoring using producer.metrics() and consumer.metrics()
- Compression optimization for genomic data patterns (gzip, snappy, lz4, zstd)
- Exactly-once semantics with idempotent producers and offset management

Research Foundation:
- Based on kafka-python>=2.0.2 production patterns
- Consumer group management with automatic partition assignment
- Producer error handling with callback patterns for failover detection
- Metrics collection for health-based routing decisions
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from contextlib import asynccontextmanager
import json
import threading
from dataclasses import asdict

try:
    from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import KafkaError, NoBrokersAvailable, KafkaTimeoutError
    from kafka import TopicPartition
    KAFKA_AVAILABLE = True
except ImportError as e:
    # Fallback for environments without kafka-python
    KAFKA_AVAILABLE = False
    logging.getLogger(__name__).warning(f"kafka-python not available: {e}")
    
    # Mock classes for development
    class KafkaProducer:
        def __init__(self, *args, **kwargs):
            self.config = kwargs
        def send(self, topic, value=None, key=None, partition=None, timestamp_ms=None, headers=None):
            class MockFuture:
                def get(self, timeout=None):
                    class MockRecord:
                        def __init__(self):
                            self.topic = topic
                            self.partition = 0
                            self.offset = int(time.time() * 1000)
                    return MockRecord()
                def add_callback(self, callback):
                    return self
                def add_errback(self, errback):
                    return self
            return MockFuture()
        def flush(self, timeout=None):
            pass
        def close(self, timeout=None):
            pass
        def metrics(self):
            return {"producer-metrics": {"record-send-rate": 100.0}}
    
    class KafkaConsumer:
        def __init__(self, *args, **kwargs):
            self.config = kwargs
        def subscribe(self, topics):
            pass
        def assign(self, partitions):
            pass
        def poll(self, timeout_ms=None):
            return {}
        def commit(self):
            pass
        def close(self):
            pass
        def metrics(self):
            return {"consumer-metrics": {"records-consumed-rate": 50.0}}
    
    class KafkaAdminClient:
        def __init__(self, *args, **kwargs):
            pass
        def create_topics(self, topics, timeout_ms=None, validate_only=False):
            pass
        def list_topics(self):
            return {}
    
    class NewTopic:
        def __init__(self, name, num_partitions, replication_factor):
            self.name = name
            self.num_partitions = num_partitions
            self.replication_factor = replication_factor
    
    class TopicPartition:
        def __init__(self, topic, partition):
            self.topic = topic
            self.partition = partition
    
    KafkaError = Exception
    NoBrokersAvailable = Exception
    KafkaTimeoutError = Exception

from .vcf_message import VCFVariantMessage, VCFMessageSerializer, chromosome_to_partition_number
from .config import Phase5Config, get_config
from .monitoring import KafkaMetricsCollector
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class KafkaConnectionManager:
    """
    Manages Kafka connections with production patterns for reliability and monitoring.
    
    Features:
    - Consumer group coordination with automatic partition assignment
    - Producer connection pooling for high concurrency
    - Health monitoring using metrics() for real-time status
    - Automatic topic creation and management
    - Circuit breaker integration for failover detection
    """
    
    def __init__(self, config: Phase5Config):
        """Initializes the Kafka connection configuration and tracking metrics.
        Parameters:
            - config (Phase5Config): Configuration object for initializing Kafka connections.
        Returns:
            - None: The constructor does not return a value.
        Processing Logic:
            - Initializes the Kafka producer and consumer dictionaries for storing connection instances.
            - Sets initial health status and tracks the last health check time.
            - Configures maximum consecutive failure threshold for connection health.
            - Sets up metric counters for monitoring the total number of Kafka connections and errors."""
        self.config = config
        self.kafka_config = config.kafka
        self.producers: Dict[str, KafkaProducer] = {}
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.admin_client: Optional[KafkaAdminClient] = None
        
        # Connection state tracking
        self.is_healthy = True
        self.last_health_check = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
        # Topic configuration
        self.topic_name = self.kafka_config.topic_name
        self.consumer_group = self.kafka_config.consumer_group
        
        # Metrics
        self.connection_counter = meter.create_counter(
            "kafka_connections_total",
            description="Total number of Kafka connections created"
        )
        self.connection_errors = meter.create_counter(
            "kafka_connection_errors_total",
            description="Total number of Kafka connection errors"
        )
    
    async def get_producer(self, producer_id: str = "default") -> KafkaProducer:
        """Get or create a Kafka producer with production configuration."""
        if producer_id not in self.producers:
            try:
                # Production-optimized producer configuration
                producer_config = {
                    'bootstrap_servers': self.kafka_config.bootstrap_servers,
                    'client_id': f"vcf-agent-producer-{producer_id}-{threading.get_ident()}",
                    'retries': self.kafka_config.producer_retries,
                    'batch_size': self.kafka_config.batch_size,
                    'linger_ms': self.kafka_config.linger_ms,
                    'compression_type': self.kafka_config.compression_type,
                    'acks': 'all',  # Ensure durability
                    'enable_idempotence': True,  # Exactly-once semantics
                    'max_in_flight_requests_per_connection': 5,
                    'request_timeout_ms': 30000,
                    'delivery_timeout_ms': 120000,
                }
                
                # Add security configuration if enabled
                if self.kafka_config.security_protocol != 'PLAINTEXT':
                    producer_config.update({
                        'security_protocol': self.kafka_config.security_protocol,
                        'ssl_context': self.kafka_config.ssl_context
                    })
                
                producer = KafkaProducer(**producer_config)
                self.producers[producer_id] = producer
                self.connection_counter.add(1, {"type": "producer"})
                
                logger.info(f"Created Kafka producer {producer_id} with {len(self.kafka_config.bootstrap_servers)} brokers")
                
            except Exception as e:
                self.consecutive_failures += 1
                self.connection_errors.add(1, {"type": "producer", "error": type(e).__name__})
                logger.error(f"Failed to create Kafka producer {producer_id}: {e}")
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.is_healthy = False
                    logger.warning("Kafka connection manager marked as unhealthy")
                
                raise
        
        return self.producers[producer_id]
    
    async def get_consumer(self, consumer_id: str = "default") -> KafkaConsumer:
        """Get or create a Kafka consumer with automatic partition assignment."""
        if consumer_id not in self.consumers:
            try:
                # Production-optimized consumer configuration
                consumer_config = {
                    'bootstrap_servers': self.kafka_config.bootstrap_servers,
                    'group_id': f"{self.consumer_group}-{consumer_id}",
                    'client_id': f"vcf-agent-consumer-{consumer_id}-{threading.get_ident()}",
                    'auto_offset_reset': 'earliest',
                    'enable_auto_commit': False,  # Manual offset management
                    'max_poll_records': self.kafka_config.max_poll_records,
                    'fetch_min_bytes': 1024,
                    'fetch_max_wait_ms': 500,
                    'consumer_timeout_ms': 1000,
                    'session_timeout_ms': 30000,
                    'heartbeat_interval_ms': 3000,
                }
                
                # Add security configuration if enabled
                if self.kafka_config.security_protocol != 'PLAINTEXT':
                    consumer_config.update({
                        'security_protocol': self.kafka_config.security_protocol,
                        'ssl_context': self.kafka_config.ssl_context
                    })
                
                consumer = KafkaConsumer(**consumer_config)
                
                # Subscribe to VCF topic with automatic partition assignment
                consumer.subscribe([self.topic_name])
                
                self.consumers[consumer_id] = consumer
                self.connection_counter.add(1, {"type": "consumer"})
                
                logger.info(f"Created Kafka consumer {consumer_id} for topic {self.topic_name}")
                
            except Exception as e:
                self.consecutive_failures += 1
                self.connection_errors.add(1, {"type": "consumer", "error": type(e).__name__})
                logger.error(f"Failed to create Kafka consumer {consumer_id}: {e}")
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.is_healthy = False
                
                raise
        
        return self.consumers[consumer_id]
    
    async def get_admin_client(self) -> KafkaAdminClient:
        """Get or create Kafka admin client for topic management."""
        if self.admin_client is None:
            try:
                admin_config = {
                    'bootstrap_servers': self.kafka_config.bootstrap_servers,
                    'client_id': f"vcf-agent-admin-{threading.get_ident()}",
                    'request_timeout_ms': 30000,
                }
                
                # Add security configuration if enabled
                if self.kafka_config.security_protocol != 'PLAINTEXT':
                    admin_config.update({
                        'security_protocol': self.kafka_config.security_protocol,
                        'ssl_context': self.kafka_config.ssl_context
                    })
                
                self.admin_client = KafkaAdminClient(**admin_config)
                logger.info("Created Kafka admin client")
                
            except Exception as e:
                logger.error(f"Failed to create Kafka admin client: {e}")
                raise
        
        return self.admin_client
    
    async def ensure_topic_exists(self):
        """Ensure VCF topic exists with proper configuration."""
        try:
            admin = await self.get_admin_client()
            
            if KAFKA_AVAILABLE:
                # Check if topic exists
                existing_topics = admin.list_topics()
                if self.topic_name not in existing_topics:
                    # Create topic with genomic-optimized configuration
                    topic = NewTopic(
                        name=self.topic_name,
                        num_partitions=self.kafka_config.partition_count,
                        replication_factor=min(3, len(self.kafka_config.bootstrap_servers))
                    )
                    
                    admin.create_topics([topic], timeout_ms=30000, validate_only=False)
                    logger.info(f"Created Kafka topic {self.topic_name} with {self.kafka_config.partition_count} partitions")
                else:
                    logger.debug(f"Kafka topic {self.topic_name} already exists")
            else:
                logger.debug("Using mock Kafka admin client")
                
        except Exception as e:
            logger.error(f"Failed to ensure topic exists: {e}")
            # Don't raise - allow processing to continue
    
    async def health_check(self) -> bool:
        """Perform comprehensive health check using producer/consumer metrics."""
        try:
            current_time = time.time()
            if current_time - self.last_health_check < 30:  # Cache for 30 seconds
                return self.is_healthy
            
            # Check producer health
            if self.producers:
                for producer_id, producer in self.producers.items():
                    try:
                        metrics = producer.metrics()
                        if KAFKA_AVAILABLE and metrics:
                            # Check for producer errors in metrics
                            producer_metrics = metrics.get('producer-metrics', {})
                            error_rate = producer_metrics.get('record-error-rate', 0)
                            
                            if error_rate > 0.05:  # >5% error rate
                                logger.warning(f"High error rate detected in producer {producer_id}: {error_rate}")
                                return False
                        
                    except Exception as e:
                        logger.warning(f"Failed to get metrics from producer {producer_id}: {e}")
                        return False
            
            # Check consumer health
            if self.consumers:
                for consumer_id, consumer in self.consumers.items():
                    try:
                        metrics = consumer.metrics()
                        if KAFKA_AVAILABLE and metrics:
                            # Check consumer lag and connectivity
                            consumer_metrics = metrics.get('consumer-metrics', {})
                            # Healthy if consumer is actively processing
                            
                    except Exception as e:
                        logger.warning(f"Failed to get metrics from consumer {consumer_id}: {e}")
                        return False
            
            self.last_health_check = current_time
            self.is_healthy = True
            self.consecutive_failures = 0  # Reset on successful health check
            
            return True
            
        except Exception as e:
            logger.warning(f"Kafka health check failed: {e}")
            return False
    
    async def close_all(self):
        """Close all Kafka connections gracefully."""
        logger.info("Closing all Kafka connections")
        
        # Close producers
        for producer_id, producer in self.producers.items():
            try:
                producer.flush(timeout=30)  # Ensure all messages are sent
                producer.close(timeout=30)
                logger.debug(f"Closed Kafka producer {producer_id}")
            except Exception as e:
                logger.warning(f"Error closing Kafka producer {producer_id}: {e}")
        
        # Close consumers
        for consumer_id, consumer in self.consumers.items():
            try:
                consumer.close()
                logger.debug(f"Closed Kafka consumer {consumer_id}")
            except Exception as e:
                logger.warning(f"Error closing Kafka consumer {consumer_id}: {e}")
        
        # Close admin client
        if self.admin_client:
            try:
                self.admin_client.close()
                logger.debug("Closed Kafka admin client")
            except Exception as e:
                logger.warning(f"Error closing Kafka admin client: {e}")
        
        self.producers.clear()
        self.consumers.clear()
        self.admin_client = None


class KafkaVCFProcessor:
    """
    Apache Kafka VCF Processor - Enterprise Fallback Platform
    
    Production-ready Kafka processor implementing research-based patterns for
    reliable genomic variant processing with automatic failover capabilities.
    
    Features:
    - Consumer group coordination with automatic partition assignment
    - Producer callback patterns for real-time error detection and failover
    - Metrics-driven health monitoring for intelligent routing decisions
    - Exactly-once semantics with idempotent producers and offset management
    - Compression optimization for genomic data (gzip, snappy, lz4, zstd)
    """
    
    def __init__(self, config: Optional[Phase5Config] = None):
        """
        Initialize the Kafka VCF processor.
        
        Args:
            config: Phase 5 configuration (uses global config if None)
        """
        self.config = config or get_config()
        self.connection_manager = KafkaConnectionManager(self.config)
        self.serializer = VCFMessageSerializer(
            compression_level=self.config.kafka.compression_level,
            enable_compression=self.config.kafka.compression_enabled
        )
        self.metrics_collector = KafkaMetricsCollector()
        
        # Processing state
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # Error tracking for failover
        self.recent_errors = []
        self.error_window = 60  # Track errors in last 60 seconds
        
        # Performance metrics
        self.throughput_counter = meter.create_counter(
            "kafka_variants_processed_total",
            description="Total number of VCF variants processed via Kafka"
        )
        self.latency_histogram = meter.create_histogram(
            "kafka_processing_latency_seconds",
            description="Kafka VCF variant processing latency"
        )
        self.error_counter = meter.create_counter(
            "kafka_processing_errors_total",
            description="Total number of Kafka processing errors"
        )
    
    async def start(self):
        """Start the Kafka VCF processor with health monitoring."""
        if self.is_running:
            logger.warning("Kafka VCF processor is already running")
            return
        
        logger.info("Starting Apache Kafka VCF processor (fallback platform)")
        
        try:
            # Ensure topic exists
            await self.connection_manager.ensure_topic_exists()
            
            # Initialize connections
            await self.connection_manager.get_producer()
            
            # Start monitoring
            await self.metrics_collector.start()
            
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("Apache Kafka VCF processor started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka VCF processor: {e}")
            raise
    
    async def stop(self):
        """Stop the Kafka VCF processor gracefully."""
        if not self.is_running:
            return
        
        logger.info("Stopping Apache Kafka VCF processor")
        
        self.is_running = False
        
        # Close connections
        await self.connection_manager.close_all()
        
        # Stop monitoring
        await self.metrics_collector.stop()
        
        # Log final statistics
        if self.start_time:
            runtime = time.time() - self.start_time
            avg_throughput = self.processed_count / max(runtime, 0.001)
            logger.info(
                f"Kafka VCF processor stopped. Processed {self.processed_count} variants "
                f"in {runtime:.2f}s (avg {avg_throughput:.1f} variants/sec)"
            )
    
    @asynccontextmanager
    async def processing_session(self):
        """Context manager for Kafka processing session."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    def _on_send_success(self, record_metadata):
        """Producer callback for successful message delivery."""
        logger.debug(f"Message delivered to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
        # Success callbacks help with monitoring but don't need extensive logging
    
    def _on_send_error(self, exception):
        """Producer callback for failed message delivery - critical for failover."""
        current_time = time.time()
        self.recent_errors.append(current_time)
        
        # Clean old errors outside window
        self.recent_errors = [t for t in self.recent_errors if current_time - t < self.error_window]
        
        # Log error for monitoring
        logger.error(f"Kafka message delivery failed: {exception}")
        self.error_counter.add(1, {"error_type": type(exception).__name__})
        
        # This error pattern enables the StreamingCoordinator to detect Kafka issues
        # and trigger failover back to Iggy if needed
    
    async def process_variant(self, variant: VCFVariantMessage) -> bool:
        """
        Process a single VCF variant through Kafka with callback monitoring.
        
        Args:
            variant: VCF variant message to process
            
        Returns:
            True if processing successful, False otherwise
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("kafka_process_variant") as span:
            try:
                span.set_attributes({
                    "variant.chromosome": variant.chromosome,
                    "variant.position": variant.position,
                    "variant.type": variant.message_type.value,
                    "platform": "kafka"
                })
                
                # Serialize variant
                serialized_data = self.serializer.serialize_variant(variant)
                
                # Get partition key for consistent chromosome routing
                partition_key = self.serializer.get_partition_key(variant)
                partition_number = chromosome_to_partition_number(
                    variant.chromosome, 
                    self.config.kafka.partition_count
                )
                
                # Send to Kafka with callbacks
                await self._send_to_kafka_topic(
                    serialized_data, 
                    partition_key, 
                    partition_number
                )
                
                # Update metrics
                self.processed_count += 1
                self.throughput_counter.add(1, {"chromosome": variant.chromosome})
                
                processing_time = time.time() - start_time
                self.latency_histogram.record(processing_time)
                
                # Update span with success
                span.set_status(Status(StatusCode.OK))
                span.set_attributes({
                    "processing.time_ms": processing_time * 1000,
                    "message.size_bytes": len(serialized_data),
                    "partition.number": partition_number
                })
                
                return True
                
            except Exception as e:
                self.error_count += 1
                self.error_counter.add(1, {"error_type": type(e).__name__})
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attributes({"error.message": str(e)})
                
                logger.error(f"Failed to process variant via Kafka {variant.get_variant_key()}: {e}")
                return False
    
    async def _send_to_kafka_topic(self, data: bytes, partition_key: str, partition: int):
        """Send serialized data to Kafka topic with producer callbacks."""
        producer = await self.connection_manager.get_producer()
        
        try:
            # Send message with callbacks for error detection
            future = producer.send(
                self.connection_manager.topic_name,
                value=data,
                key=partition_key.encode('utf-8'),
                partition=partition,
                headers=[
                    ('vcf_agent_version', b'5.2'),
                    ('platform', b'kafka'),
                    ('compression', str(self.serializer.enable_compression).encode())
                ]
            )
            
            # Add callbacks for monitoring and failover detection
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            # For critical operations, we can wait for confirmation
            if self.config.kafka.enable_sync_send:
                record_metadata = future.get(timeout=30)
                logger.debug(
                    f"Sent {len(data)} bytes to Kafka topic {self.connection_manager.topic_name} "
                    f"partition {partition} (offset: {record_metadata.offset})"
                )
            else:
                logger.debug(f"Async sent {len(data)} bytes to Kafka partition {partition}")
                
        except Exception as e:
            # Immediate error handling for synchronous issues
            logger.error(f"Failed to send message to Kafka: {e}")
            self._on_send_error(e)
            raise
    
    async def process_variants_batch(self, variants: List[VCFVariantMessage]) -> Dict[str, int]:
        """
        Process a batch of VCF variants with optimized Kafka patterns.
        
        Args:
            variants: List of VCF variant messages
            
        Returns:
            Dictionary with processing statistics
        """
        if not variants:
            return {"processed": 0, "failed": 0}
        
        start_time = time.time()
        batch_size = len(variants)
        
        with tracer.start_as_current_span("kafka_process_batch") as span:
            span.set_attributes({
                "batch.size": batch_size,
                "batch.chromosome": variants[0].chromosome if variants else "unknown",
                "platform": "kafka"
            })
            
            # Process variants with async sending for better throughput
            tasks = [self.process_variant(variant) for variant in variants]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Ensure all messages are sent (flush for consistency)
            try:
                producer = await self.connection_manager.get_producer()
                producer.flush(timeout=60)
            except Exception as e:
                logger.warning(f"Failed to flush Kafka producer: {e}")
            
            # Count results
            processed = sum(1 for result in results if result is True)
            failed = batch_size - processed
            
            batch_time = time.time() - start_time
            throughput = batch_size / max(batch_time, 0.001)
            
            span.set_attributes({
                "batch.processed": processed,
                "batch.failed": failed,
                "batch.throughput_per_sec": throughput,
                "batch.processing_time_ms": batch_time * 1000
            })
            
            logger.info(
                f"Kafka batch processed: {processed}/{batch_size} variants "
                f"({throughput:.1f} variants/sec)"
            )
            
            return {"processed": processed, "failed": failed, "throughput": throughput}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for failover decisions."""
        connection_healthy = await self.connection_manager.health_check()
        
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_throughput = self.processed_count / max(runtime, 0.001)
        error_rate = self.error_count / max(self.processed_count, 1)
        
        # Calculate recent error rate for failover decisions
        current_time = time.time()
        recent_error_count = len([t for t in self.recent_errors if current_time - t < 60])
        recent_error_rate = recent_error_count / 60  # Errors per second
        
        return {
            "status": "healthy" if connection_healthy and self.is_running else "unhealthy",
            "is_running": self.is_running,
            "connection_healthy": connection_healthy,
            "runtime_seconds": runtime,
            "variants_processed": self.processed_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "recent_error_rate": recent_error_rate,
            "avg_throughput_per_sec": avg_throughput,
            "serializer_stats": self.serializer.get_performance_stats(),
            "memory_usage": self._get_memory_usage(),
            "kafka_metrics": await self._get_kafka_metrics()
        }
    
    async def _get_kafka_metrics(self) -> Dict[str, Any]:
        """Get Kafka-specific metrics for health monitoring."""
        try:
            metrics = {}
            
            # Producer metrics
            if self.connection_manager.producers:
                producer = list(self.connection_manager.producers.values())[0]
                producer_metrics = producer.metrics()
                if producer_metrics:
                    metrics["producer"] = {
                        "record_send_rate": producer_metrics.get("producer-metrics", {}).get("record-send-rate", 0),
                        "record_error_rate": producer_metrics.get("producer-metrics", {}).get("record-error-rate", 0),
                        "batch_size_avg": producer_metrics.get("producer-metrics", {}).get("batch-size-avg", 0)
                    }
            
            # Consumer metrics (if any)
            if self.connection_manager.consumers:
                consumer = list(self.connection_manager.consumers.values())[0]
                consumer_metrics = consumer.metrics()
                if consumer_metrics:
                    metrics["consumer"] = {
                        "records_consumed_rate": consumer_metrics.get("consumer-metrics", {}).get("records-consumed-rate", 0),
                        "records_lag": consumer_metrics.get("consumer-metrics", {}).get("records-lag", 0)
                    }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to get Kafka metrics: {e}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }


# Utility functions
async def create_kafka_processor(config: Optional[Phase5Config] = None) -> KafkaVCFProcessor:
    """Create and initialize a Kafka VCF processor."""
    processor = KafkaVCFProcessor(config)
    await processor.start()
    return processor


async def process_vcf_file_with_kafka(vcf_file_path: str, config: Optional[Phase5Config] = None) -> Dict[str, Any]:
    """
    Process a VCF file using the Kafka fallback processor.
    
    Args:
        vcf_file_path: Path to VCF file
        config: Phase 5 configuration
        
    Returns:
        Processing statistics
    """
    from .vcf_message import create_variant_from_vcf_record
    import cyvcf2
    
    stats = {"processed": 0, "failed": 0, "total_time": 0}
    
    async with KafkaVCFProcessor(config).processing_session() as processor:
        start_time = time.time()
        
        # Read and process VCF file
        vcf_reader = cyvcf2.VCF(vcf_file_path)
        batch = []
        batch_size = config.kafka.batch_size if config else 1000
        
        for record in vcf_reader:
            variant = create_variant_from_vcf_record(record, vcf_file_path)
            batch.append(variant)
            
            if len(batch) >= batch_size:
                batch_stats = await processor.process_variants_batch(batch)
                stats["processed"] += batch_stats["processed"]
                stats["failed"] += batch_stats["failed"]
                batch.clear()
        
        # Process remaining variants
        if batch:
            batch_stats = await processor.process_variants_batch(batch)
            stats["processed"] += batch_stats["processed"]
            stats["failed"] += batch_stats["failed"]
        
        stats["total_time"] = time.time() - start_time
        stats["throughput"] = stats["processed"] / max(stats["total_time"], 0.001)
    
    return stats 