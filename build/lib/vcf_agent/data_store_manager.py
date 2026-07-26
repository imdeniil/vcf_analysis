"""
Unified Data Store Manager for VCF Analysis Agent

This module provides a unified interface for managing both LanceDB (vector database)
and Kuzu (graph database) as specified in DECISION-001. It handles data synchronization,
performance optimization, and provides a single API for all data operations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import json

# Import our database integrations
from .lancedb_integration import (
    VCFVariant, VariantEmbeddingService, get_db as get_lancedb,
    get_or_create_vcf_table, batch_add_vcf_variants, hybrid_search_variants,
    search_similar_variants, get_variant_statistics as get_lancedb_stats
)
from .graph_integration import (
    get_kuzu_db_connection, create_enhanced_schema, batch_add_genomic_data,
    find_sample_variants, find_variant_genes, find_similar_samples,
    get_genomic_statistics as get_kuzu_stats
)
from .config import SessionConfig
from .optimizations import get_optimizer, OptimizationConfig
from .memory_optimization import (
    memory_optimized_batch_add_vcf_variants,
    Phase1MemoryOptimizer,
    PHASE1_CONFIG
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DataStoreConfig:
    """Configuration for the unified data store manager."""
    lancedb_path: str = "./lancedb"
    kuzu_path: str = "./kuzu_db"
    lancedb_table_name: str = "vcf_variants"
    enable_sync: bool = True
    sync_batch_size: int = 1000
    performance_monitoring: bool = True
    max_workers: int = 4

@dataclass
class PerformanceMetrics:
    """Performance metrics for data store operations."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    records_processed: int
    success: bool
    error_message: Optional[str] = None
    
    @property
    def records_per_second(self) -> float:
        """Calculate records processed per second."""
        if self.duration > 0:
            return self.records_processed / self.duration
        return 0.0

class UnifiedDataStoreManager:
    """
    Unified manager for LanceDB and Kuzu databases.
    
    Provides a single interface for all data operations while maintaining
    data consistency between vector and graph databases.
    """
    
    def __init__(self, config: Optional[DataStoreConfig] = None, session_config: Optional[SessionConfig] = None):
        """
        Initialize the unified data store manager.
        
        Args:
            config: Data store configuration.
            session_config: Session configuration for AI models.
        """
        self.config = config or DataStoreConfig()
        self.session_config = session_config or SessionConfig()
        
        # Initialize Phase 1 Memory Optimization
        self.phase1_optimizer = Phase1MemoryOptimizer()
        logger.info(f"Phase 1 Memory Optimization enabled: {PHASE1_CONFIG}")
        
        # Update config with Phase 1 optimized batch size
        if self.config.sync_batch_size > PHASE1_CONFIG["batch_size"]:
            logger.info(f"Reducing batch size from {self.config.sync_batch_size} to {PHASE1_CONFIG['batch_size']} for Phase 1 optimization")
            self.config.sync_batch_size = PHASE1_CONFIG["batch_size"]
        
        # Initialize performance optimizer
        optimization_config = OptimizationConfig(
            enable_embedding_cache=True,
            enable_query_batching=True,
            enable_async_processing=True,
            max_concurrent_tasks=PHASE1_CONFIG["max_workers"]  # Use Phase 1 optimized worker count
        )
        self.optimizer = get_optimizer(optimization_config)
        
        # Database connections
        self.lancedb = None
        self.lance_table = None
        self.kuzu_conn = None
        
        # Services
        self.embedding_service = VariantEmbeddingService(self.session_config)
        # Optimize embedding service
        self.embedding_service = self.optimizer.optimize_embedding_service(self.embedding_service)
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.metrics_lock = threading.Lock()
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self) -> None:
        """Initialize database connections and schemas."""
        try:
            logger.info("Initializing unified data store connections...")
            
            # Initialize LanceDB
            self.lancedb = get_lancedb(self.config.lancedb_path)
            self.lance_table = get_or_create_vcf_table(self.lancedb, self.config.lancedb_table_name)
            logger.info(f"LanceDB initialized: {self.config.lancedb_path}")
            
            # Initialize Kuzu
            self.kuzu_conn = get_kuzu_db_connection(self.config.kuzu_path)
            create_enhanced_schema(self.kuzu_conn)
            logger.info(f"Kuzu initialized: {self.config.kuzu_path}")
            
            logger.info("Unified data store manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data store manager: {e}")
            raise
    
    def _record_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        if self.config.performance_monitoring:
            with self.metrics_lock:
                self.performance_metrics.append(metrics)
                # Keep only last 1000 metrics to prevent memory issues
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
    
    def add_sample_with_variants(
        self,
        sample_data: Dict[str, Any],
        variants_data: List[Dict[str, Any]],
        genes_data: Optional[List[Dict[str, Any]]] = None,
        analysis_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Add a sample with its variants to both databases with full synchronization.
        
        Args:
            sample_data: Sample information dictionary.
            variants_data: List of variant dictionaries.
            genes_data: Optional list of gene dictionaries.
            analysis_data: Optional list of analysis dictionaries.
            
        Returns:
            Dictionary with operation results and performance metrics.
        """
        start_time = time.time()
        logger.info(f"Adding sample {sample_data.get('id', 'unknown')} with {len(variants_data)} variants")
        
        try:
            results = {
                "sample_id": sample_data.get('id'),
                "lancedb_results": {},
                "kuzu_results": {},
                "sync_status": "success",
                "performance": {}
            }
            
            # Prepare data for both databases
            lance_variants = []
            kuzu_samples = [sample_data]
            kuzu_variants = []
            kuzu_genes = genes_data or []
            kuzu_relationships = []
            
            # Process each variant
            for variant_data in variants_data:
                # Prepare LanceDB variant record
                variant_data['sample_id'] = sample_data['id']
                lance_record = self._prepare_lance_variant(variant_data)
                lance_variants.append(lance_record)
                
                # Prepare Kuzu variant record
                kuzu_variant = self._prepare_kuzu_variant(variant_data)
                kuzu_variants.append(kuzu_variant)
                
                # Create HasVariant relationship
                relationship = {
                    "type": "HasVariant",
                    "sample_id": sample_data['id'],
                    "variant_id": variant_data['id'],
                    "properties": {
                        "genotype": variant_data.get('genotype', '0/1'),
                        "quality": variant_data.get('quality_score', 0.0),
                        "depth": variant_data.get('depth', 0),
                        "allele_depth": variant_data.get('allele_depth', '')
                    }
                }
                kuzu_relationships.append(relationship)
                
                # Create LocatedIn relationships if gene information is available
                if 'gene_symbol' in variant_data and genes_data:
                    for gene_data in genes_data:
                        if gene_data['symbol'] == variant_data['gene_symbol']:
                            gene_relationship = {
                                "type": "LocatedIn",
                                "variant_id": variant_data['id'],
                                "gene_id": gene_data['id'],
                                "properties": {
                                    "impact": variant_data.get('impact', ''),
                                    "consequence": variant_data.get('consequence', ''),
                                    "amino_acid_change": variant_data.get('amino_acid_change', ''),
                                    "codon_change": variant_data.get('codon_change', '')
                                }
                            }
                            kuzu_relationships.append(gene_relationship)
            
            # Execute database operations in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit LanceDB operation with Phase 1 memory optimization
                lance_future = executor.submit(
                    memory_optimized_batch_add_vcf_variants,  # Use Phase 1 optimized function
                    self.lance_table,
                    lance_variants,
                    self.embedding_service,
                    PHASE1_CONFIG["batch_size"],  # Use Phase 1 optimized batch size
                    PHASE1_CONFIG["max_workers"]  # Use Phase 1 optimized worker count
                )
                
                # Submit Kuzu operation
                kuzu_future = executor.submit(
                    batch_add_genomic_data,
                    self.kuzu_conn,
                    kuzu_samples,
                    kuzu_variants,
                    kuzu_genes,
                    kuzu_relationships,
                    self.config.sync_batch_size
                )
                
                # Collect results
                results["lancedb_results"]["variants_added"] = lance_future.result()
                results["kuzu_results"] = kuzu_future.result()
            
            # Record performance metrics
            end_time = time.time()
            duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                operation="add_sample_with_variants",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                records_processed=len(variants_data),
                success=True
            )
            self._record_performance(metrics)
            
            results["performance"] = {
                "duration_seconds": duration,
                "variants_per_second": len(variants_data) / duration if duration > 0 else 0,
                "total_variants": len(variants_data)
            }
            
            logger.info(f"Successfully added sample with {len(variants_data)} variants in {duration:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error adding sample with variants: {e}")
            
            # Record failed performance metrics
            end_time = time.time()
            metrics = PerformanceMetrics(
                operation="add_sample_with_variants",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                records_processed=0,
                success=False,
                error_message=str(e)
            )
            self._record_performance(metrics)
            
            raise
    
    def search_variants(
        self,
        query: str,
        search_type: str = "hybrid",
        metadata_filter: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search variants using multiple search strategies.
        
        Args:
            query: Search query text.
            search_type: Type of search ("hybrid", "vector", "graph").
            metadata_filter: Optional metadata filter for hybrid search.
            limit: Maximum number of results.
            similarity_threshold: Minimum similarity score.
            
        Returns:
            Dictionary with search results and metadata.
        """
        start_time = time.time()
        logger.info(f"Searching variants: '{query}' (type: {search_type})")
        
        try:
            results = {
                "query": query,
                "search_type": search_type,
                "results": [],
                "metadata": {},
                "performance": {}
            }
            
            if search_type in ["hybrid", "vector"]:
                # Perform vector search using LanceDB
                lance_results = hybrid_search_variants(
                    self.lance_table,
                    query,
                    self.embedding_service,
                    metadata_filter,
                    limit,
                    similarity_threshold
                )
                
                # Convert to list of dictionaries
                if not lance_results.empty:
                    results["results"] = lance_results.to_dict('records')
                    results["metadata"]["lance_results_count"] = len(lance_results)
            
            if search_type == "graph":
                # Perform graph-based search using Kuzu
                # This would involve more complex Cypher queries
                # For now, we'll use a simple approach
                kuzu_results = []
                results["results"] = kuzu_results
                results["metadata"]["kuzu_results_count"] = len(kuzu_results)
            
            # Record performance
            end_time = time.time()
            duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                operation="search_variants",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                records_processed=len(results["results"]),
                success=True
            )
            self._record_performance(metrics)
            
            results["performance"] = {
                "duration_seconds": duration,
                "results_count": len(results["results"])
            }
            
            logger.info(f"Search completed: {len(results['results'])} results in {duration:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error searching variants: {e}")
            
            # Record failed performance metrics
            end_time = time.time()
            metrics = PerformanceMetrics(
                operation="search_variants",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                records_processed=0,
                success=False,
                error_message=str(e)
            )
            self._record_performance(metrics)
            
            raise
    
    def get_sample_analysis(self, sample_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis for a sample using both databases.
        
        Args:
            sample_id: ID of the sample to analyze.
            
        Returns:
            Dictionary with comprehensive sample analysis.
        """
        start_time = time.time()
        logger.info(f"Getting comprehensive analysis for sample: {sample_id}")
        
        try:
            analysis = {
                "sample_id": sample_id,
                "variants": [],
                "genes": [],
                "similar_samples": [],
                "statistics": {},
                "performance": {}
            }
            
            # Get variants from Kuzu with relationship details
            variants = find_sample_variants(self.kuzu_conn, sample_id, limit=1000)
            analysis["variants"] = variants
            
            # Optimize gene lookups using batch queries
            variant_ids = [variant['variant_id'] for variant in variants]
            if variant_ids:
                # Use optimized batch gene queries
                gene_results = self.optimizer.optimize_gene_queries(self.kuzu_conn, variant_ids)
                
                # Collect unique genes
                unique_genes = set()
                for variant_id, genes in gene_results.items():
                    for gene in genes:
                        unique_genes.add(gene['symbol'])
                
                analysis["genes"] = list(unique_genes)
            else:
                analysis["genes"] = []
            
            # Find similar samples
            similar_samples = find_similar_samples(self.kuzu_conn, sample_id, min_shared_variants=5)
            analysis["similar_samples"] = similar_samples
            
            # Get statistics
            analysis["statistics"] = {
                "total_variants": len(variants),
                "unique_genes": len(analysis["genes"]),
                "similar_samples_count": len(similar_samples)
            }
            
            # Record performance
            end_time = time.time()
            duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                operation="get_sample_analysis",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                records_processed=len(variants),
                success=True
            )
            self._record_performance(metrics)
            
            analysis["performance"] = {
                "duration_seconds": duration,
                "variants_analyzed": len(variants),
                "optimization_stats": self.optimizer.get_optimization_stats()
            }
            
            logger.info(f"Sample analysis completed: {len(variants)} variants, {len(analysis['genes'])} genes in {duration:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting sample analysis: {e}")
            
            # Record failed performance metrics
            end_time = time.time()
            metrics = PerformanceMetrics(
                operation="get_sample_analysis",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                records_processed=0,
                success=False,
                error_message=str(e)
            )
            self._record_performance(metrics)
            
            raise
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from both databases.
        
        Returns:
            Dictionary with statistics from both LanceDB and Kuzu.
        """
        try:
            logger.info("Generating comprehensive statistics from both databases")
            
            stats = {
                "lancedb_stats": get_lancedb_stats(self.lance_table),
                "kuzu_stats": get_kuzu_stats(self.kuzu_conn),
                "performance_stats": self._get_performance_statistics(),
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info("Comprehensive statistics generated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating comprehensive statistics: {e}")
            raise
    
    def _get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics from recorded metrics."""
        with self.metrics_lock:
            if not self.performance_metrics:
                return {"message": "No performance metrics available"}
            
            # Group metrics by operation
            operations = {}
            for metric in self.performance_metrics:
                if metric.operation not in operations:
                    operations[metric.operation] = []
                operations[metric.operation].append(metric)
            
            # Calculate statistics for each operation
            stats = {}
            for operation, metrics in operations.items():
                successful_metrics = [m for m in metrics if m.success]
                
                if successful_metrics:
                    durations = [m.duration for m in successful_metrics]
                    records_per_sec = [m.records_per_second for m in successful_metrics]
                    
                    stats[operation] = {
                        "total_operations": len(metrics),
                        "successful_operations": len(successful_metrics),
                        "success_rate": len(successful_metrics) / len(metrics),
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "avg_records_per_second": sum(records_per_sec) / len(records_per_sec) if records_per_sec else 0
                    }
                else:
                    stats[operation] = {
                        "total_operations": len(metrics),
                        "successful_operations": 0,
                        "success_rate": 0.0
                    }
            
            return stats
    
    def _prepare_lance_variant(self, variant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variant data for LanceDB insertion."""
        # This would use the existing create_vcf_variant_record function
        # but we'll create a simplified version here
        description = self.embedding_service.generate_variant_description(variant_data)
        embedding = self.embedding_service.generate_embedding_sync(description)
        
        now = datetime.now()
        return {
            "variant_id": variant_data['id'],
            "chromosome": variant_data.get('chr', variant_data.get('chromosome', '')),
            "position": variant_data.get('pos', variant_data.get('position', 0)),
            "reference": variant_data.get('ref', variant_data.get('reference', '')),
            "alternate": variant_data.get('alt', variant_data.get('alternate', '')),
            "variant_description": description,
            "variant_vector": embedding,
            "analysis_summary": variant_data.get('analysis_summary', ''),
            "sample_id": variant_data.get('sample_id', ''),
            "quality_score": variant_data.get('quality_score'),
            "filter_status": variant_data.get('filter_status'),
            "genotype": variant_data.get('genotype'),
            "allele_frequency": variant_data.get('allele_frequency'),
            "clinical_significance": variant_data.get('clinical_significance'),
            "gene_symbol": variant_data.get('gene_symbol'),
            "consequence": variant_data.get('consequence'),
            "created_at": now,
            "updated_at": now
        }
    
    def _prepare_kuzu_variant(self, variant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variant data for Kuzu insertion."""
        return {
            "id": variant_data['id'],
            "chr": variant_data.get('chr', variant_data.get('chromosome', '')),
            "pos": variant_data.get('pos', variant_data.get('position', 0)),
            "ref": variant_data.get('ref', variant_data.get('reference', '')),
            "alt": variant_data.get('alt', variant_data.get('alternate', '')),
            "variant_type": variant_data.get('variant_type', 'SNV'),
            "quality_score": variant_data.get('quality_score', 0.0),
            "filter_status": variant_data.get('filter_status', 'UNKNOWN'),
            "allele_frequency": variant_data.get('allele_frequency', 0.0)
        }
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        logger.info("Closing unified data store manager")
        
        # Close connections if they exist
        if self.kuzu_conn:
            # Kuzu connections are typically closed automatically
            self.kuzu_conn = None
        
        if self.lancedb:
            # LanceDB connections are typically closed automatically
            self.lancedb = None
            self.lance_table = None
        
        logger.info("Unified data store manager closed successfully")

# Convenience function for creating a manager instance
def create_data_store_manager(
    lancedb_path: str = "./lancedb",
    kuzu_path: str = "./kuzu_db",
    session_config: Optional[SessionConfig] = None
) -> UnifiedDataStoreManager:
    """
    Create a unified data store manager with the specified configuration.
    
    Args:
        lancedb_path: Path to LanceDB database.
        kuzu_path: Path to Kuzu database.
        session_config: Session configuration for AI models.
        
    Returns:
        Configured UnifiedDataStoreManager instance.
    """
    config = DataStoreConfig(
        lancedb_path=lancedb_path,
        kuzu_path=kuzu_path
    )
    
    return UnifiedDataStoreManager(config, session_config) 