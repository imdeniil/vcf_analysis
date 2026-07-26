"""
Performance Optimizations for VCF Analysis Agent

This module contains specific optimizations identified through performance analysis:
1. Embedding caching and batch generation
2. Database query optimization
3. Memory usage optimization
4. Async processing improvements
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import hashlib

# Use msgpack for faster serialization, fallback to json
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    import json
    HAS_MSGPACK = False

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_embedding_cache: bool = True
    embedding_cache_size: int = 10000
    enable_query_batching: bool = True
    batch_query_size: int = 50
    enable_async_processing: bool = True
    max_concurrent_tasks: int = 10
    enable_memory_optimization: bool = True

class EmbeddingCache:
    """Thread-safe LRU cache for embeddings with optimized persistence."""
    
    def __init__(self, max_size: int = 10000, persist_file: Optional[str] = None, max_cache_file_size_mb: int = 50):
        """Initialize a cache object with specific constraints and optional persistence.
        Parameters:
            - max_size (int): Maximum number of items the cache can hold. Default is 10000.
            - persist_file (Optional[str]): File path for persistence option. If provided, cache can be saved to or loaded from this file.
            - max_cache_file_size_mb (int): Maximum size of the cache file in megabytes. Default is 50.
        Returns:
            - None: The constructor initializes attributes and loads cache if persist_file is specified.
        Processing Logic:
            - Initializes a thread-safe cache and tracks access order for eviction purposes.
            - Sets up attributes for managing cache size and persistence options.
            - Tries to load cache data from persist_file if the file is provided."""
        self.max_size = max_size
        self.persist_file = persist_file
        self.max_cache_file_size_mb = max_cache_file_size_mb
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._save_counter = 0
        self._load_cache()
    
    def _generate_key(self, text: str, model: str = "default") -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from persistence file with optimized format."""
        if not self.persist_file:
            return
            
        try:
            with open(self.persist_file, 'rb') as f:
                if HAS_MSGPACK:
                    data = msgpack.unpack(f, raw=False)
                else:
                    import json
                    f.seek(0)
                    data = json.load(f)
                
                self._cache = data.get('cache', {})
                self._access_order = data.get('access_order', [])
                
            logger.info(f"Loaded {len(self._cache)} embeddings from cache using {'msgpack' if HAS_MSGPACK else 'json'}")
        except (FileNotFoundError, Exception) as e:
            if not isinstance(e, FileNotFoundError):
                logger.warning(f"Failed to load cache: {e}")
            logger.info("Starting with fresh cache")
    
    def _save_cache(self):
        """Save cache to persistence file with optimized format and size limits."""
        if not self.persist_file:
            return
            
        # Only save every 10 operations to reduce I/O
        self._save_counter += 1
        if self._save_counter % 10 != 0:
            return
            
        try:
            # Check if cache would be too large
            estimated_size = len(self._cache) * 1536 * 4 / 1024 / 1024  # Rough estimate in MB
            if estimated_size > self.max_cache_file_size_mb:
                # Trim cache to 80% of max size
                target_size = int(self.max_size * 0.8)
                while len(self._cache) > target_size and self._access_order:
                    lru_key = self._access_order.pop(0)
                    self._cache.pop(lru_key, None)
                logger.info(f"Trimmed cache to {len(self._cache)} entries to stay under size limit")
            
            data = {
                'cache': self._cache,
                'access_order': self._access_order
            }
            
            with open(self.persist_file, 'wb') as f:
                if HAS_MSGPACK:
                    msgpack.pack(data, f)
                else:
                    import json
                    # For JSON, we need text mode
                    f.close()
                    with open(self.persist_file, 'w') as f_text:
                        json.dump(data, f_text)
                        
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)
    
    def get(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._generate_key(text, model)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        
        return None
    
    def put(self, text: str, embedding: List[float], model: str = "default"):
        """Put embedding in cache."""
        key = self._generate_key(text, model)
        
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                self._access_order.remove(key)
            
            # Add to cache
            self._cache[key] = embedding
            self._access_order.append(key)
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                self._evict_lru()
            
            # Periodically save cache
            if len(self._cache) % 100 == 0:
                self._save_cache()
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._save_cache()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size
            }

# Phase 2 Memory Recovery Integration
try:
    from .phase2_memory_optimization import (
        phase2_embedding_recovery,
        MemoryAwareEmbeddingCache,
        PHASE2_CONFIG
    )
    PHASE2_AVAILABLE = True
    logger.info("Phase 2 Memory Recovery integration available")
except ImportError as e:
    PHASE2_AVAILABLE = False
    logger.info(f"Phase 2 Memory Recovery not available: {e}")

class OptimizedEmbeddingService:
    """
    Optimized embedding service with caching and batching.
    
    Phase 2 Enhancement: Integrated with Enhanced Embedding Recovery system
    for improved memory management and long-term stability.
    """
    
    def __init__(self, base_service, config: OptimizationConfig):
        """Initialize the OptimizedEmbeddingService with caching capabilities.
        Parameters:
            - base_service: The base service to be optimized with embedding caching.
            - config (OptimizationConfig): Configuration object for the embedding optimization settings.
        Returns:
            - None
        Processing Logic:
            - Initializes with MemoryAwareEmbeddingCache if Phase 2 is available and caching is enabled.
            - Registers the cache with the Phase 2 recovery system if applicable.
            - Falls back to an original EmbeddingCache if Phase 2 is not available but caching is enabled.
            - Sets up internal statistics tracking for cache usage and embedding requests."""
        self.base_service = base_service
        self.config = config
        
        # Phase 2: Use MemoryAwareEmbeddingCache if available
        if PHASE2_AVAILABLE and config.enable_embedding_cache:
            self.cache = MemoryAwareEmbeddingCache(
                max_size_mb=PHASE2_CONFIG["embedding_cache_max_mb"],
                max_entries=config.embedding_cache_size
            )
            # Register with Phase 2 recovery system
            phase2_embedding_recovery.register_embedding_cache(self.cache)
            logger.info("OptimizedEmbeddingService using Phase 2 MemoryAwareEmbeddingCache")
        elif config.enable_embedding_cache:
            # Fallback to original cache if Phase 2 not available
            self.cache = EmbeddingCache(
                max_size=config.embedding_cache_size,
                persist_file=".embedding_cache.json" if config.enable_embedding_cache else None
            )
            logger.info("OptimizedEmbeddingService using original EmbeddingCache")
        else:
            self.cache = None
        
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_requests": 0,
            "total_embeddings": 0,
            "phase2_cleanups": 0
        }
    
    def __getattr__(self, name):
        """Delegate any missing methods to the base service."""
        return getattr(self.base_service, name)
    
    def generate_variant_description(self, variant_data: dict) -> str:
        """Generate variant description using base service."""
        return self.base_service.generate_variant_description(variant_data)
    
    def generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding with caching and Phase 2 memory recovery."""
        
        # Phase 2: Track embedding operation
        if PHASE2_AVAILABLE:
            phase2_embedding_recovery.track_embedding_operation("optimized_sync_embedding")
        
        # Check cache first
        if self.cache:
            # Get model name safely
            model_name = getattr(self.base_service, 'model', 'default')
            
            if PHASE2_AVAILABLE and hasattr(self.cache, 'get'):
                # Phase 2 MemoryAwareEmbeddingCache
                cache_key = f"{model_name}:{text}"
                cached = self.cache.get(cache_key)
            else:
                # Original EmbeddingCache
                cached = self.cache.get(text, model_name)
            
            if cached:
                self._stats["cache_hits"] += 1
                return cached
            self._stats["cache_misses"] += 1
        
        # Generate embedding
        embedding = self.base_service.generate_embedding_sync(text)
        
        # Cache result
        if self.cache and embedding:
            model_name = getattr(self.base_service, 'model', 'default')
            
            if PHASE2_AVAILABLE and hasattr(self.cache, 'put'):
                # Phase 2 MemoryAwareEmbeddingCache
                cache_key = f"{model_name}:{text}"
                self.cache.put(cache_key, embedding)
            else:
                # Original EmbeddingCache
                self.cache.put(text, embedding, model_name)
        
        # Phase 2: Check memory threshold after caching
        if PHASE2_AVAILABLE and phase2_embedding_recovery.check_memory_threshold():
            logger.info("Memory threshold exceeded in OptimizedEmbeddingService, triggering cleanup")
            phase2_embedding_recovery.force_embedding_cleanup()
            self._stats["phase2_cleanups"] += 1
        
        self._stats["total_embeddings"] += 1
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch with caching and Phase 2 memory recovery."""
        
        # Phase 2: Track batch embedding operation
        if PHASE2_AVAILABLE:
            phase2_embedding_recovery.track_embedding_operation(f"optimized_batch_embedding_{len(texts)}")
        
        if not self.config.enable_embedding_cache:
            # Fallback to individual generation
            return [self.base_service.generate_embedding_sync(text) for text in texts]
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Get model name safely
        model_name = getattr(self.base_service, 'model', 'default')
        
        # Check cache for all texts
        for i, text in enumerate(texts):
            if PHASE2_AVAILABLE and hasattr(self.cache, 'get'):
                # Phase 2 MemoryAwareEmbeddingCache
                cache_key = f"{model_name}:{text}"
                cached = self.cache.get(cache_key)
            else:
                # Original EmbeddingCache
                cached = self.cache.get(text, model_name)
            
            if cached:
                results.append(cached)
                self._stats["cache_hits"] += 1
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._stats["cache_misses"] += 1
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            if hasattr(self.base_service, 'generate_embeddings_batch'):
                # Use batch API if available
                uncached_embeddings = self.base_service.generate_embeddings_batch(uncached_texts)
            else:
                # Fallback to individual generation
                uncached_embeddings = [
                    self.base_service.generate_embedding_sync(text) 
                    for text in uncached_texts
                ]
            
            # Fill in results and cache
            for i, embedding in enumerate(uncached_embeddings):
                idx = uncached_indices[i]
                results[idx] = embedding
                if embedding:
                    if PHASE2_AVAILABLE and hasattr(self.cache, 'put'):
                        # Phase 2 MemoryAwareEmbeddingCache
                        cache_key = f"{model_name}:{uncached_texts[i]}"
                        self.cache.put(cache_key, embedding)
                    else:
                        # Original EmbeddingCache
                        self.cache.put(uncached_texts[i], embedding, model_name)
        
        # Phase 2: Check memory threshold after batch processing
        if PHASE2_AVAILABLE and phase2_embedding_recovery.check_memory_threshold():
            logger.info("Memory threshold exceeded after batch embedding, triggering cleanup")
            phase2_embedding_recovery.force_embedding_cleanup()
            self._stats["phase2_cleanups"] += 1
        
        self._stats["batch_requests"] += 1
        self._stats["total_embeddings"] += len(texts)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics including Phase 2 metrics."""
        stats = self._stats.copy()
        
        if self.cache:
            if PHASE2_AVAILABLE and hasattr(self.cache, 'get_cache_stats'):
                # Phase 2 MemoryAwareEmbeddingCache stats
                stats["cache_stats"] = self.cache.get_cache_stats()
                stats["phase2_recovery"] = phase2_embedding_recovery.get_recovery_stats()
            else:
                # Original EmbeddingCache stats
                stats["cache_stats"] = self.cache.stats()
            
            if stats["cache_hits"] + stats["cache_misses"] > 0:
                stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        
        return stats

class QueryBatcher:
    """Batch database queries for better performance."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.batch_size = config.batch_query_size
    
    def batch_gene_queries(self, kuzu_conn, variant_ids: List[str]) -> Dict[str, List[Dict]]:
        """Batch gene lookup queries."""
        if not self.config.enable_query_batching or len(variant_ids) <= 1:
            # Fallback to individual queries
            from .graph_integration import find_variant_genes
            return {
                variant_id: find_variant_genes(kuzu_conn, variant_id)
                for variant_id in variant_ids
            }
        
        # Batch query using IN clause
        results = {}
        
        for i in range(0, len(variant_ids), self.batch_size):
            batch_ids = variant_ids[i:i + self.batch_size]
            
            # Create parameterized query for batch
            query = """
            MATCH (v:Variant)-[r:LocatedIn]->(g:Gene)
            WHERE v.id IN $variant_ids
            RETURN v.id as variant_id, g.id as gene_id, g.symbol, g.name, g.chromosome,
                   r.impact, r.consequence, r.amino_acid_change, r.codon_change
            """
            
            try:
                batch_results = kuzu_conn.execute(query, {"variant_ids": batch_ids})
                
                # Group results by variant_id
                for row in batch_results:
                    variant_id = row['variant_id']
                    if variant_id not in results:
                        results[variant_id] = []
                    
                    results[variant_id].append({
                        'gene_id': row['gene_id'],
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'chromosome': row['chromosome'],
                        'impact': row['impact'],
                        'consequence': row['consequence'],
                        'amino_acid_change': row['amino_acid_change'],
                        'codon_change': row['codon_change']
                    })
                
            except Exception as e:
                logger.warning(f"Batch query failed, falling back to individual queries: {e}")
                # Fallback to individual queries
                from .graph_integration import find_variant_genes
                for variant_id in batch_ids:
                    results[variant_id] = find_variant_genes(kuzu_conn, variant_id)
        
        # Ensure all variant_ids have entries (even if empty)
        for variant_id in variant_ids:
            if variant_id not in results:
                results[variant_id] = []
        
        return results

class AsyncProcessor:
    """Async processing for I/O bound operations."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.max_concurrent = config.max_concurrent_tasks
    
    async def process_variants_async(self, variants_data: List[Dict], processor_func) -> List[Any]:
        """Process variants asynchronously."""
        if not self.config.enable_async_processing:
            return [processor_func(variant) for variant in variants_data]
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(variant):
            async with semaphore:
                # Run in thread pool for CPU-bound work
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, processor_func, variant)
        
        tasks = [process_with_semaphore(variant) for variant in variants_data]
        return await asyncio.gather(*tasks)
    
    def process_variants_threaded(self, variants_data: List[Dict], processor_func) -> List[Any]:
        """Process variants using thread pool."""
        if not self.config.enable_async_processing or len(variants_data) <= 10:
            return [processor_func(variant) for variant in variants_data]
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [executor.submit(processor_func, variant) for variant in variants_data]
            return [future.result() for future in as_completed(futures)]

class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    @staticmethod
    def optimize_dataframe_memory(df):
        """Optimize pandas DataFrame memory usage."""
        if not hasattr(df, 'memory_usage'):
            return df
        
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        return df
    
    @staticmethod
    def batch_process_large_data(data: List[Any], batch_size: int, processor_func):
        """Process large datasets in batches to manage memory."""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
        
        return results

class PerformanceOptimizer:
    """Main optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initializes an optimization system with configurable components.
        Parameters:
            - config (OptimizationConfig, optional): Configuration settings for optimization. Defaults to a new OptimizationConfig instance if not provided.
        Returns:
            - None: This function does not return a value.
        Processing Logic:
            - Initializes `embedding_cache` to None.
            - Sets up `query_batcher`, `async_processor`, and `memory_optimizer` using the provided or default configuration.
            - Initializes a statistics dictionary to track applied optimizations, time saved, and memory saved."""
        self.config = config or OptimizationConfig()
        self.embedding_cache = None
        self.query_batcher = QueryBatcher(self.config)
        self.async_processor = AsyncProcessor(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self._stats = {
            "optimizations_applied": 0,
            "time_saved_seconds": 0.0,
            "memory_saved_mb": 0.0
        }
    
    def optimize_embedding_service(self, embedding_service):
        """Wrap embedding service with optimizations."""
        if not self.config.enable_embedding_cache:
            return embedding_service
        
        optimized = OptimizedEmbeddingService(embedding_service, self.config)
        self._stats["optimizations_applied"] += 1
        return optimized
    
    def optimize_variant_processing(self, variants_data: List[Dict], processor_func) -> List[Any]:
        """Optimize variant processing with async/threading."""
        start_time = time.time()
        
        if self.config.enable_async_processing and len(variants_data) > 10:
            try:
                # Try async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    self.async_processor.process_variants_async(variants_data, processor_func)
                )
                loop.close()
            except Exception as e:
                logger.warning(f"Async processing failed, falling back to threaded: {e}")
                results = self.async_processor.process_variants_threaded(variants_data, processor_func)
        else:
            results = [processor_func(variant) for variant in variants_data]
        
        duration = time.time() - start_time
        self._stats["time_saved_seconds"] += max(0, len(variants_data) * 0.01 - duration)
        self._stats["optimizations_applied"] += 1
        
        return results
    
    def optimize_gene_queries(self, kuzu_conn, variant_ids: List[str]) -> Dict[str, List[Dict]]:
        """Optimize gene lookup queries with batching."""
        start_time = time.time()
        
        results = self.query_batcher.batch_gene_queries(kuzu_conn, variant_ids)
        
        duration = time.time() - start_time
        # Estimate time saved vs individual queries
        estimated_individual_time = len(variant_ids) * 0.05  # 50ms per query
        self._stats["time_saved_seconds"] += max(0, estimated_individual_time - duration)
        self._stats["optimizations_applied"] += 1
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        return self._stats.copy()

# Global optimizer instance
_global_optimizer = None

def get_optimizer(config: OptimizationConfig = None) -> PerformanceOptimizer:
    """Get global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer

def reset_optimizer():
    """Reset global optimizer instance."""
    global _global_optimizer
    _global_optimizer = None 