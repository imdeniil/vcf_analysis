# NOTE: If you see linter errors for lancedb imports, ensure your editor is using the correct virtual environment. The import paths are correct for LanceDB >=0.22.1.
# This module's logging strategy aims to align with the Auditable Security Framework,
# focusing on capturing key operations, parameters (with masking for sensitive data),
# and success/failure status.
from lancedb.pydantic import LanceModel, Vector
import lancedb
from typing import List, Optional, Annotated, Type, cast, Literal, Dict, Tuple, Any
import logging # Added for error logging
from lancedb.table import LanceTable # Ensure this import is present and correct
import threading # Added for locking
import re # Added for SQL literal masking
import pandas as pd # Ensure pandas is imported for DataFrame operations
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import time
from openai import OpenAI  # OpenAI-compatible client for embeddings (LM Studio / etc.)

# Import Kuzu integration functions
from . import graph_integration
# Import API clients for embedding generation
from .api_clients import OpenAIClient, CredentialManager
from .config import SessionConfig, MemoryOptimizationConfig

# Enhanced tracing imports - Phase 4.2 integration
from .enhanced_tracing import (
    vcf_operation_span,
    vcf_operation_context,
    memory_operation_span,
    get_enhanced_trace_service
)

# Add import for sqlparse if available
try:
    import sqlparse  # type: ignore
except ImportError:
    sqlparse = None

# Optional imports for advanced optimizations
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__) # Added for error logging

# Global lock for LanceDB write operations to ensure thread safety with local file-based DB
# This lock serializes write-related operations (add, update, delete, create_index)
# to prevent potential race conditions or corruption when accessing the underlying LanceDB files
# from multiple threads within a single VCF Agent process.
# For multi-process or distributed access, a more robust distributed locking mechanism or a
# database designed for high concurrency (like LanceDB Cloud) would be necessary.
lancedb_write_lock = threading.Lock()

# Configurable list of sensitive column names
SENSITIVE_COLUMNS = [
    'patient_id', 'ssn', 'email', 'user_id', 'phone', 'dob', 'address', 'name', 'mrn', 'genotype', 'sample_id'
]
# Configurable regex patterns for sensitive data
SENSITIVE_PATTERNS = [
    re.compile(r'[\w\.-]+@[\w\.-]+'),  # Email addresses
    re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'),  # SSN-like patterns
    re.compile(r'\b\d{10,}\b'),  # Long numbers (IDs, phone numbers)
    re.compile(r'\b(?:[A-Z][a-z]+ ){1,2}[A-Z][a-z]+\b'),  # Names (simple heuristic)
]

def mask_sensitive_sql(filter_sql: str, sensitive_columns=None, patterns=None) -> str:
    """
    Mask sensitive data in SQL filter strings for logging.

    Args:
        filter_sql (str): The SQL filter string to mask.
        sensitive_columns (list, optional): List of sensitive column names to mask. Defaults to SENSITIVE_COLUMNS.
        patterns (list, optional): List of regex patterns to mask. Defaults to SENSITIVE_PATTERNS.

    Returns:
        str: The masked SQL string, or '[MASKED_SQL]' if ambiguous/unparseable.

    Example:
        >>> mask_sensitive_sql("email = 'john.doe@example.com' AND ssn = '123-45-6789'")
        "email = '[MASKED]' AND ssn = '[MASKED]'"
    """
    if not filter_sql or not isinstance(filter_sql, str):
        return '[MASKED_SQL]'
    sensitive_columns = sensitive_columns or SENSITIVE_COLUMNS
    patterns = patterns or SENSITIVE_PATTERNS

    # Try to parse with sqlparse if available
    if sqlparse:
        try:
            parsed = sqlparse.parse(filter_sql)
            if not parsed:
                raise ValueError('Empty parse result')
            tokens = list(parsed[0].flatten())
            masked_tokens = []
            last_col = None
            for t in tokens:
                # Mask string/numeric literals
                if t.ttype in sqlparse.tokens.Literal.String.Single or t.ttype in sqlparse.tokens.Literal.Number:
                    masked_tokens.append("'[MASKED]'")
                # Mask values after sensitive column names
                elif last_col and t.value.strip() in ('=', '==', '!=', '<>', 'LIKE'):
                    masked_tokens.append(t.value)
                    last_col = last_col  # Keep for next token
                elif last_col:
                    masked_tokens.append("'[MASKED]'")
                    last_col = None
                elif t.ttype is sqlparse.tokens.Name and t.value.lower() in sensitive_columns:
                    masked_tokens.append(t.value)
                    last_col = t.value.lower()
                else:
                    masked_tokens.append(t.value)
                    last_col = None
            return ''.join(masked_tokens)
        except Exception:
            pass  # Fallback to regex masking

    # Regex-based masking for common sensitive patterns
    masked = filter_sql
    for pat in patterns:
        masked = pat.sub('[MASKED]', masked)

    # Regex masking for sensitive columns: mask value after sensitive_column = ...
    for col in sensitive_columns:
        # Handles: col = 'value', col = value, col=123, col = "value"
        # Use non-greedy match for value, allow optional whitespace, single/double quotes, or unquoted
        masked = re.sub(rf'({col}\s*=\s*)([\'\"]?)[^\s\'\"]+\2', rf'\1[MASKED]', masked, flags=re.IGNORECASE)

    # If the result is still too revealing (e.g., contains long numbers or emails), redact
    if any(pat.search(masked) for pat in patterns):
        return '[MASKED_SQL]'
    return masked

# Enhanced VCF Variant model based on DECISION-001 specifications
class VCFVariant(LanceModel):
    """
    Enhanced Pydantic model representing a VCF variant record in LanceDB.
    Based on DECISION-001 specifications for comprehensive genomic variant storage.

    Attributes:
        variant_id (str): Unique identifier for the variant (e.g., 'chr1-123456-A-G').
        chromosome (str): Chromosome identifier (e.g., '1', '2', 'X', 'Y', 'MT').
        position (int): Genomic position (1-based coordinate).
        reference (str): Reference allele sequence.
        alternate (str): Alternate allele sequence.
        variant_description (str): Human-readable description for embedding generation.
        variant_vector: Annotated[List[float], Vector(1536)]  # Original dimension for backward compatibility
        analysis_summary (str): AI-generated analysis summary of the variant.
        sample_id (str): Identifier for the sample containing this variant.
        quality_score (Optional[float]): Variant quality score from VCF.
        filter_status (Optional[str]): Filter status from VCF (PASS, FAIL, etc.).
        genotype (Optional[str]): Genotype information (e.g., '0/1', '1/1').
        allele_frequency (Optional[float]): Allele frequency if available.
        clinical_significance (Optional[str]): Clinical significance annotation.
        gene_symbol (Optional[str]): Associated gene symbol if available.
        consequence (Optional[str]): Variant consequence (e.g., 'missense_variant').
        created_at (datetime): Timestamp when record was created.
        updated_at (datetime): Timestamp when record was last updated.
    """
    variant_id: str
    chromosome: str
    position: int
    reference: str
    alternate: str
    variant_description: str
    variant_vector: Annotated[List[float], Vector(1536)]  # Original dimension for backward compatibility
    analysis_summary: str
    sample_id: str
    quality_score: Optional[float] = None
    filter_status: Optional[str] = None
    genotype: Optional[str] = None
    allele_frequency: Optional[float] = None
    clinical_significance: Optional[str] = None
    gene_symbol: Optional[str] = None
    consequence: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class VCFVariantPhase3(LanceModel):
    """
    Phase 3 optimized Pydantic model representing a VCF variant record in LanceDB.
    Uses 768-dimensional embeddings for 50% memory reduction.
    
    This model is used when Phase 3 embedding optimization is enabled.
    """
    variant_id: str
    chromosome: str
    position: int
    reference: str
    alternate: str
    variant_description: str
    variant_vector: Annotated[List[float], Vector(768)]  # Phase 3 optimized dimensions (50% reduction)
    analysis_summary: str
    sample_id: str
    quality_score: Optional[float] = None
    filter_status: Optional[str] = None
    genotype: Optional[str] = None
    allele_frequency: Optional[float] = None
    clinical_significance: Optional[str] = None
    gene_symbol: Optional[str] = None
    consequence: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# Legacy Variant model for backward compatibility
class Variant(LanceModel):
    """
    Legacy Pydantic model for backward compatibility.
    Use VCFVariant or VCFVariantPhase3 for new implementations.
    """
    variant_id: str
    embedding: Annotated[List[float], Vector(1024)]  # type: ignore[valid-type]
    chrom: str
    pos: int
    ref: str
    alt: str
    clinical_significance: Optional[str] = None

# Embedding service for generating variant embeddings
class MemoryAwareEmbeddingCache:
    """
    Memory-aware caching system for embeddings with automatic cleanup.
    Integrated directly into the embedding service for production use.
    """
    def __init__(self, max_size_mb: int = 40, max_entries: int = 1000):
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.cache = {}
        self.access_times = {}
        self.memory_usage_mb = 0.0
        
    def _estimate_memory_usage(self, embedding: List[float]) -> float:
        """Estimate memory usage of an embedding in MB."""
        return len(embedding) * 4 / 1024 / 1024  # 4 bytes per float
    
    def _cleanup_cache(self):
        """Remove least recently used entries to free memory."""
        if not self.cache:
            return
            
        # Sort by access time and remove oldest entries
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        entries_to_remove = len(sorted_items) // 4  # Remove 25% of entries
        
        for key, _ in sorted_items[:entries_to_remove]:
            if key in self.cache:
                embedding_size = self._estimate_memory_usage(self.cache[key])
                del self.cache[key]
                del self.access_times[key]
                self.memory_usage_mb -= embedding_size
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get cached embedding with access time tracking."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, embedding: List[float]):
        """Cache embedding with memory management."""
        embedding_size = self._estimate_memory_usage(embedding)
        
        # Check if we need to cleanup before adding
        if (self.memory_usage_mb + embedding_size > self.max_size_mb or 
            len(self.cache) >= self.max_entries):
            self._cleanup_cache()
        
        self.cache[key] = embedding
        self.access_times[key] = time.time()
        self.memory_usage_mb += embedding_size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_entries": len(self.cache),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "max_size_mb": self.max_size_mb,
            "max_entries": self.max_entries,
            "memory_utilization_percent": round((self.memory_usage_mb / self.max_size_mb) * 100, 1)
        }

class DimensionReducer:
    """
    PCA-based dimension reduction for embeddings.
    Integrated directly into the embedding service.
    """
    def __init__(self, target_dimensions: int = 768, original_dimensions: int = 1536):
        """Initialize an object for dimensionality reduction using PCA.
        Parameters:
            - target_dimensions (int): Desired number of dimensions for the reduced dataset. Defaults to 768.
            - original_dimensions (int): Dimensionality of the source embedding model.
              Defaults to 1536 (OpenAI text-embedding-3-small); set to 1024 when
              serving a local model such as bge-m3 via EMBEDDING_DIM.
        Returns:
            - None: The function does not return any value. It sets up internal state for future processing.
        Processing Logic:
            - Sets up initial dimensions and checks for availability of scikit-learn.
            - Initializes model attributes such as PCA model and scaler as None.
            - Flags is_trained to False, indicating model needs training.
            - Prepares to store training embeddings and specifies a minimum sample size for reliable PCA application.
            - Issues a warning when scikit-learn is unavailable, showing dimension reduction feature will not work."""
        self.target_dimensions = target_dimensions
        self.original_dimensions = original_dimensions
        self.pca_model = None
        self.scaler = None
        self.is_trained = False
        self.training_embeddings = []
        self.training_samples_needed = 1000  # Minimum samples for reliable PCA
        
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available, dimension reduction disabled")
    
    def can_reduce_dimensions(self) -> bool:
        """Check if dimension reduction is available."""
        return HAS_SKLEARN and self.is_trained
    
    def add_training_embedding(self, embedding: List[float]) -> bool:
        """Add embedding to training dataset. Returns True when ready to train."""
        if not HAS_SKLEARN or len(embedding) != self.original_dimensions:
            return False
            
        self.training_embeddings.append(embedding)
        
        # Train when we have enough samples
        if len(self.training_embeddings) >= self.training_samples_needed and not self.is_trained:
            self._train_pca_model()
            return True
        return False
    
    def _train_pca_model(self):
        """Train PCA model on collected embeddings."""
        if not HAS_SKLEARN or not self.training_embeddings:
            return
            
        try:
            training_data = np.array(self.training_embeddings)
            
            # Standardize the data
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(training_data)
            
            # Train PCA
            self.pca_model = PCA(n_components=self.target_dimensions)
            self.pca_model.fit(scaled_data)
            
            self.is_trained = True
            
            # Calculate variance retention
            variance_retained = np.sum(self.pca_model.explained_variance_ratio_)
            logger.info(f"PCA model trained: {self.original_dimensions}→{self.target_dimensions} dimensions, {variance_retained:.1%} variance retained")
            
            # Clear training data to free memory
            self.training_embeddings.clear()
            
        except Exception as e:
            logger.error(f"PCA training failed: {e}")
    
    def reduce_embedding(self, embedding: List[float]) -> List[float]:
        """Reduce embedding dimensions using trained PCA model."""
        if not self.can_reduce_dimensions() or len(embedding) != self.original_dimensions:
            return embedding
            
        try:
            # Apply same preprocessing and reduction
            scaled_embedding = self.scaler.transform([embedding])
            reduced_embedding = self.pca_model.transform(scaled_embedding)
            return reduced_embedding[0].tolist()
        except Exception as e:
            logger.error(f"Dimension reduction failed: {e}")
            return embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dimension reduction statistics."""
        stats = {
            "is_trained": self.is_trained,
            "target_dimensions": self.target_dimensions,
            "original_dimensions": self.original_dimensions,
            "training_samples_collected": len(self.training_embeddings),
            "training_samples_needed": self.training_samples_needed
        }
        
        if self.is_trained and self.pca_model is not None:
            stats["variance_retained"] = float(np.sum(self.pca_model.explained_variance_ratio_))
            stats["memory_reduction_percent"] = (1 - self.target_dimensions / self.original_dimensions) * 100
        
        return stats

class VariantEmbeddingService:
    """
    Production-ready embedding service with integrated memory optimizations.
    
    Features:
    - Memory-aware caching with automatic cleanup
    - PCA dimension reduction for 50% memory savings
    - Streaming batch processing for large datasets
    - Configurable optimization levels
    - Comprehensive performance monitoring
    """
    
    def __init__(self, session_config: Optional[SessionConfig] = None):
        """
        Initialize the embedding service with integrated optimizations.
        
        Args:
            session_config: Session configuration including memory optimization settings.
        """
        self.session_config = session_config or SessionConfig()
        self.memory_config = self.session_config.memory_optimization
        self.openai_client = None
        
        # Initialize memory-aware caching
        if self.memory_config.caching_strategy == "memory_aware":
            self.embedding_cache = MemoryAwareEmbeddingCache(
                max_size_mb=self.memory_config.cache_max_size_mb,
                max_entries=self.memory_config.cache_max_entries
            )
        else:
            self.embedding_cache = {}  # Simple dictionary cache
        
        # Embeddings are INDEPENDENT from the chat LLM provider: a local
        # OpenAI-compatible server (LM Studio / Ollama / vLLM) serves the
        # embedding model (default: bge-m3, 1024-dim) regardless of which LLM
        # (Z.AI GLM, OpenAI, ...) powers the agent.
        # NOTE: embedding_dim must be resolved BEFORE creating the DimensionReducer
        # so PCA is trained against the correct input dimensionality.
        self.openai_client = None
        self.embedding_client = None  # OpenAI-compatible client (any provider)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3")
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

        # Initialize dimension reduction.
        # PCA reduces embeddings to target_dimensions (default 768). The reducer
        # must know the *real* input dimensionality (original_dimensions) so it
        # accepts training vectors that actually come from the served model
        # (bge-m3=1024, OpenAI text-embedding-3-small=1536, ...).
        self.dimension_reducer = None
        if self.memory_config.dimension_reduction_enabled and HAS_SKLEARN:
            # Guard: PCA target must be strictly smaller than the input dim.
            if self.memory_config.target_dimensions < self.embedding_dim:
                self.dimension_reducer = DimensionReducer(
                    target_dimensions=self.memory_config.target_dimensions,
                    original_dimensions=self.embedding_dim,
                )
            else:
                logger.info(
                    f"PCA reduction disabled: target ({self.memory_config.target_dimensions}) "
                    f">= embedding dim ({self.embedding_dim}); nothing to reduce."
                )

        provider = self.session_config.model_provider
        if provider == "openai":
            # Original path: OpenAI cloud embeddings.
            try:
                credential_manager = CredentialManager(self.session_config.credentials_file)
                self.openai_client = OpenAIClient(credential_manager)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client for embeddings: {e}")
        elif provider in ("zai", "ollama", "cerebras"):
            # Local/embedgins endpoint via OpenAI-compatible API (LM Studio, etc.).
            base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
            api_key = os.getenv("EMBEDDING_API_KEY", "lm-studio")
            try:
                self.embedding_client = OpenAI(base_url=base_url, api_key=api_key)
                logger.info(f"Embedding client initialized: {base_url} model={self.embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding client ({base_url}): {e}")
        
        # Performance tracking
        self.generation_stats = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "dimension_reductions": 0,
            "batch_generations": 0
        }
        
        logger.info(f"VariantEmbeddingService initialized: optimization_level={self.memory_config.optimization_level}, "
                   f"dimensions={self.get_embedding_dimensions()}, caching={self.memory_config.caching_strategy}")

    def generate_variant_description(self, variant_data: Dict) -> str:
        """
        Generate a human-readable description for a variant.
        
        Args:
            variant_data: Dictionary containing variant information.
            
        Returns:
            Human-readable variant description for embedding.
        """
        chrom = variant_data.get('chromosome', variant_data.get('chrom', 'unknown'))
        pos = variant_data.get('position', variant_data.get('pos', 0))
        ref = variant_data.get('reference', variant_data.get('ref', ''))
        alt = variant_data.get('alternate', variant_data.get('alt', ''))
        gene = variant_data.get('gene_symbol', '')
        consequence = variant_data.get('consequence', '')
        clinical_sig = variant_data.get('clinical_significance', '')
        
        description_parts = [
            f"Variant at chromosome {chrom} position {pos}",
            f"reference allele {ref} to alternate allele {alt}"
        ]
        
        if gene:
            description_parts.append(f"in gene {gene}")
        
        if consequence:
            description_parts.append(f"with consequence {consequence}")
            
        if clinical_sig:
            description_parts.append(f"clinical significance {clinical_sig}")
        
        return " ".join(description_parts)
    
    def _check_memory_threshold(self) -> bool:
        """Check if memory usage is above threshold and cleanup if needed."""
        if not HAS_PSUTIL or not self.memory_config.memory_management_enabled:
            return False
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.memory_config.memory_cleanup_threshold_mb:
                logger.info(f"Memory threshold exceeded ({memory_mb:.1f}MB), triggering cleanup")
                if hasattr(self.embedding_cache, '_cleanup_cache'):
                    self.embedding_cache._cleanup_cache()
                return True
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")
        return False
    
    def generate_embedding_sync(self, text: str) -> List[float]:
        """
        Generate embedding for the given text with integrated optimizations.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            List of float values representing the embedding vector.
        """
        # Check cache first
        cache_key = f"{self.session_config.model_provider}:{text}"
        
        if hasattr(self.embedding_cache, 'get'):
            # Memory-aware cache
            cached_embedding = self.embedding_cache.get(cache_key)
            if cached_embedding:
                self.generation_stats["cache_hits"] += 1
                return cached_embedding
        elif isinstance(self.embedding_cache, dict):
            # Simple cache
            if cache_key in self.embedding_cache:
                self.generation_stats["cache_hits"] += 1
                return self.embedding_cache[cache_key]
        
        try:
            # Generate original embedding
            provider = self.session_config.model_provider
            original_embedding = None
            if provider == "openai" and self.openai_client:
                response = self.openai_client.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                original_embedding = response.data[0].embedding
            elif provider in ("zai", "ollama", "cerebras") and self.embedding_client is not None:
                # Real embeddings via OpenAI-compatible local server (LM Studio, etc.)
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                original_embedding = response.data[0].embedding

            # Fallback ONLY when no embedding client is available (e.g. server down,
            # dev environment without a local model). Produces a deterministic-shape
            # random vector so the pipeline does not crash, but similarity search on
            # such vectors is meaningless and a warning is logged.
            if original_embedding is None:
                dim = self.embedding_dim
                logger.warning(
                    f"No embedding client for provider '{provider}' "
                    f"(is the embedding server running at EMBEDDING_BASE_URL?). "
                    f"Falling back to a {dim}-dim random vector — similarity search "
                    f"will NOT be meaningful."
                )
                original_embedding = np.random.normal(0, 1, dim).tolist()
            
            # Add to dimension reducer training if applicable
            if self.dimension_reducer and not self.dimension_reducer.is_trained:
                self.dimension_reducer.add_training_embedding(original_embedding)
            
            # Apply dimension reduction if enabled and trained
            final_embedding = original_embedding
            if (self.memory_config.dimension_reduction_enabled and 
                self.dimension_reducer and 
                self.dimension_reducer.can_reduce_dimensions()):
                try:
                    final_embedding = self.dimension_reducer.reduce_embedding(original_embedding)
                    self.generation_stats["dimension_reductions"] += 1
                except Exception as e:
                    logger.warning(f"Dimension reduction failed, using original embedding: {e}")
            
            # Cache result
            if hasattr(self.embedding_cache, 'put'):
                self.embedding_cache.put(cache_key, final_embedding)
            elif isinstance(self.embedding_cache, dict):
                self.embedding_cache[cache_key] = final_embedding
            
            self.generation_stats["embeddings_generated"] += 1
            
            # Check memory threshold after generation
            self._check_memory_threshold()
            
            return final_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return appropriate dimension fallback
            fallback_dimensions = self.get_embedding_dimensions()
            return np.random.normal(0, 1, fallback_dimensions).tolist()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text (async version)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embedding_sync, text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with streaming optimization.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        
        self.generation_stats["batch_generations"] += 1
        batch_size = self.memory_config.streaming_batch_size
        all_embeddings = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_embedding_sync(text)
                batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
            
            # Memory cleanup between batches if needed
            if self.memory_config.memory_management_enabled:
                self._check_memory_threshold()
        
        logger.info(f"Generated {len(all_embeddings)} embeddings using streaming batch processing (batch_size: {batch_size})")
        return all_embeddings
    
    def get_embedding_dimensions(self) -> int:
        """Get the current embedding dimensions based on configuration."""
        return self.memory_config.get_embedding_dimensions()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "optimization_level": self.memory_config.optimization_level,
            "embedding_dimensions": self.get_embedding_dimensions(),
            "generation_stats": self.generation_stats.copy(),
            "memory_management_enabled": self.memory_config.memory_management_enabled,
            "dimension_reduction_enabled": self.memory_config.dimension_reduction_enabled
        }
        
        # Add cache statistics
        if hasattr(self.embedding_cache, 'get_cache_stats'):
            stats["cache_stats"] = self.embedding_cache.get_cache_stats()
        elif isinstance(self.embedding_cache, dict):
            stats["cache_stats"] = {"cache_entries": len(self.embedding_cache), "type": "simple"}
        
        # Add dimension reduction statistics
        if self.dimension_reducer:
            stats["dimension_reduction_stats"] = self.dimension_reducer.get_stats()
        
        # Calculate efficiency metrics
        total_operations = self.generation_stats["embeddings_generated"] + self.generation_stats["cache_hits"]
        if total_operations > 0:
            stats["cache_hit_rate"] = round((self.generation_stats["cache_hits"] / total_operations) * 100, 1)
        
        return stats

# LanceDB connection (local path for now)
def get_db(db_path: str = "./lancedb"):
    """
    Connects to a LanceDB database at the given path.

    Args:
        db_path (str): Path to the LanceDB directory. Defaults to './lancedb'.

    Returns:
        lancedb.DBConnection: LanceDB database connection object.

    Raises:
        Exception: If connection fails.

    Example:
        >>> db = get_db("./lancedb")
    """
    logger.info(f"Attempting to connect to LanceDB at {db_path}...")
    try:
        db = lancedb.connect(db_path)
        logger.info(f"Successfully connected to LanceDB at {db_path}.")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB at {db_path}: {e}")
        raise

def get_or_create_table(db, table_name: str = "variants"):
    """
    Opens an existing LanceDB table or creates a new one if it doesn't exist.

    Args:
        db (lancedb.DBConnection): LanceDB database connection object.
        table_name (str): Name of the table to open or create. Defaults to 'variants'.

    Returns:
        LanceTable: The opened or newly created LanceDB table.

    Raises:
        Exception: If table creation or opening fails.

    Example:
        >>> db = get_db()
        >>> table = get_or_create_table(db, "variants")
    """
    logger.info(f"Attempting to get or create table '{table_name}'...")
    try:
        if table_name in db.table_names():
            table = db.open_table(table_name)
            logger.info(f"Opened existing table: '{table_name}'.")
        else:
            table = db.create_table(table_name, schema=Variant, mode="overwrite")
            logger.info(f"Created new table: '{table_name}' with schema {Variant.__name__}.") # Log schema name
        return table
    except Exception as e:
        logger.error(f"Failed to get or create table '{table_name}': {e}")
        raise

def add_variants(table: LanceTable, variants: List[dict]):
    """
    Adds a batch of variant records to the specified LanceDB table.

    Args:
        table (LanceTable): The LanceDB table object.
        variants (List[dict]): List of variant dictionaries matching the Variant schema.

    Returns:
        None

    Raises:
        Exception: If the add operation fails.

    Example:
        >>> add_variants(table, [{"variant_id": "chr1-123-A-G", ...}])
    """
    num_variants = len(variants)
    logger.info(f"Attempting to add {num_variants} variants to table '{table.name}'...")
    if not variants:
        logger.warning(f"add_variants called for table '{table.name}' with an empty list. No data will be added.")
        return

    try:
        with lancedb_write_lock:
            table.add(variants)
        logger.info(f"Successfully added {num_variants} variants to table '{table.name}'.")
    except Exception as e:
        logger.error(f"Failed to add batch of {num_variants} variants to table '{table.name}': {e}")
        # Depending on requirements, you might want to:
        # 1. Raise the exception to let the caller handle it.
        # 2. Implement partial success/retry logic if applicable.
        # 3. Log problematic records if possible.
        raise

def search_by_embedding(table: LanceTable, query_embedding, limit: int = 10, filter_sql: Optional[str] = None) -> pd.DataFrame:
    """
    Searches the LanceDB table by a query embedding, with optional limit and SQL filter.
    Enriches results with context from the Kuzu graph database.

    Args:
        table (LanceTable): The LanceDB table object.
        query_embedding (List[float] or np.ndarray): Embedding vector to search by.
        limit (int): Maximum number of results to return. Defaults to 10.
        filter_sql (Optional[str]): Optional SQL-like filter string for metadata columns.

    Returns:
        pd.DataFrame: DataFrame of search results, enriched with Kuzu context in 'kuzu_observed_samples'.

    Raises:
        Exception: If the search or enrichment fails.

    Example:
        >>> df = search_by_embedding(table, [0.1]*1024, limit=5)
    """
    embedding_summary = f"shape: {len(query_embedding)}x{len(query_embedding[0])}" if isinstance(query_embedding, list) and query_embedding and isinstance(query_embedding[0], list) else f"length: {len(query_embedding)}" if isinstance(query_embedding, list) else "opaque_embedding"
    masked_filter_sql = mask_sensitive_sql(filter_sql or "") # Mask SQL literals for logging
    logger.info(f"Attempting to search table '{table.name}' with embedding ({embedding_summary}), limit: {limit}, filter_sql: '{masked_filter_sql}'.")
    try:
        q = table.search(query_embedding, vector_column_name='embedding').limit(limit)
        if filter_sql:
            q = q.where(filter_sql)
        results_df = q.to_pandas()
        logger.info(f"Search in table '{table.name}' completed. Found {len(results_df)} results for limit: {limit}, filter_sql: '{masked_filter_sql}'.")

        # Always add the 'kuzu_observed_samples' column, even if empty
        results_df['kuzu_observed_samples'] = [[] for _ in range(len(results_df))]

        # Enrich with Kuzu graph context if results are found
        if not results_df.empty and 'variant_id' in results_df.columns:
            variant_ids_to_enrich = results_df['variant_id'].tolist()
            if variant_ids_to_enrich:
                try:
                    logger.info(f"Fetching Kuzu context for {len(variant_ids_to_enrich)} variant IDs.")
                    kuzu_conn = graph_integration.get_managed_kuzu_connection()
                    if kuzu_conn:
                        variant_contexts = graph_integration.get_variant_context(kuzu_conn, variant_ids_to_enrich)
                        if isinstance(variant_contexts, list):
                            from typing import Dict, List, Tuple
                            context_map: Dict[str, List[Tuple[str, str]]] = {}
                            for row in variant_contexts:
                                v_id = row['variant_id']
                                tup: Tuple[str, str] = (row['sample_id'], row['zygosity'])
                                if v_id not in context_map:
                                    context_map[v_id] = []
                                context_map[v_id].append(tup)
                            variant_contexts = context_map
                        results_df['kuzu_observed_samples'] = results_df['variant_id'].apply(
                            lambda vid: variant_contexts.get(vid, [])
                        )
                        logger.info("Successfully enriched LanceDB results with Kuzu context.")
                    else:
                        logger.warning("Kuzu connection not available, skipping enrichment.")
                        # Already set to empty lists above
                except Exception as kuzu_e:
                    logger.error(f"Error during Kuzu context enrichment: {kuzu_e}")
                    import pandas as pd
                    results_df['kuzu_observed_samples'] = [pd.NA] * len(results_df)
            # else: already set to empty lists above
        elif 'variant_id' not in results_df.columns:
            logger.warning("'variant_id' column not found in LanceDB results, cannot enrich with Kuzu context.")
            # Already set to empty lists above
        # else: results_df is empty, already set to empty lists above

        return results_df
    except Exception as e:
        logger.error(f"Search by embedding failed in table '{table.name}' (limit: {limit}, filter_sql: '{masked_filter_sql}'): {e}")
        raise

def update_variant(table: LanceTable, variant_id: str, updates: dict):
    """
    Updates a specific variant record identified by variant_id.

    Args:
        table (LanceTable): The LanceDB table object.
        variant_id (str): The ID of the variant to update.
        updates (dict): Dictionary of fields and new values to update.

    Returns:
        None

    Raises:
        Exception: If the update operation fails.

    Example:
        >>> update_variant(table, "chr1-123-A-G", {"clinical_significance": "Benign"})
    """
    update_keys = list(updates.keys()) if updates else []
    logger.info(f"Attempting to update variant '{variant_id}' in table '{table.name}' with updates for keys: {update_keys}.")

    if not updates:
        logger.warning(f"update_variant called for variant '{variant_id}' in table '{table.name}' with no updates. Skipping.")
        return

    try:
        with lancedb_write_lock:
            # Ensure variant_id is properly quoted if it's a string type in the schema.
            # LanceDB's SQL-like .where() clauses expect string literals to be single-quoted.
            where_clause = f"variant_id = '{variant_id}'"
            table.update(where=where_clause, values=updates)
        logger.info(f"Successfully updated variant '{variant_id}' in table '{table.name}' with keys: {update_keys}.")
    except Exception as e:
        # Avoid logging the full 'updates' dict here directly in case of sensitive values.
        logger.error(f"Failed to update variant '{variant_id}' in table '{table.name}' with update keys {update_keys}: {e}")
        raise

def delete_variants(table: LanceTable, filter_sql: str):
    """
    Deletes variant records from the table based on a SQL filter expression.

    Args:
        table (LanceTable): The LanceDB table object.
        filter_sql (str): SQL-like filter string to identify records for deletion.

    Returns:
        None

    Raises:
        ValueError: If filter_sql is empty.
        Exception: If the delete operation fails.

    Example:
        >>> delete_variants(table, "variant_id = 'chr1-123-A-G'")
    """
    masked_filter_sql = mask_sensitive_sql(filter_sql or "") # Mask SQL literals for logging
    logger.info(f"Attempting to delete variants from table '{table.name}' matching filter: '{masked_filter_sql}'.")
    # TODO: Security Review: filter_sql is logged directly. If filter_sql could contain sensitive PII
    # (e.g., filtering on a direct identifier that is PII), it should be masked or summarized
    # in production logs according to the data handling policies. -> Addressed by masking.
    if not filter_sql: # Original filter_sql used for logic, masked_filter_sql for logging.
        logger.error(f"delete_variants called for table '{table.name}' with an empty filter_sql. No records will be deleted.")
        # Potentially raise an error or return early depending on desired behavior
        raise ValueError("filter_sql cannot be empty for delete_variants operation.")

    try:
        with lancedb_write_lock:
            table.delete(filter_sql)
        logger.info(f"Successfully deleted variants from table '{table.name}' matching filter: '{masked_filter_sql}'.")
    except Exception as e:
        logger.error(f"Failed to delete variants from table '{table.name}' using filter '{masked_filter_sql}': {e}")
        raise

LanceScalarIndexType = Literal['BTREE', 'BITMAP', 'LABEL_LIST'] # Define Literal for index types

def create_scalar_index(table: LanceTable, column_name: str, index_type: Optional[LanceScalarIndexType] = None, replace: bool = False, **kwargs):
    """
    Creates a scalar index on a specified column to improve filter performance.

    Args:
        table (LanceTable): The LanceDB table object.
        column_name (str): Name of the scalar column to index (e.g., 'chrom', 'pos').
        index_type (Optional[str]): Type of index ('BTREE', 'BITMAP', 'LABEL_LIST'). Defaults to 'BTREE'.
        replace (bool): If True, replaces an existing index. Defaults to False.
        **kwargs: Additional keyword arguments for index creation.

    Returns:
        None

    Raises:
        Exception: If index creation fails.

    Example:
        >>> create_scalar_index(table, "chrom", index_type="BTREE")
    """
    kwargs_summary = list(kwargs.keys()) # Log only keys of kwargs
    # Log the index_type that will actually be used.
    actual_index_type_to_log = index_type if index_type is not None else 'BTREE'
    logger.info(f"Attempting to create scalar index on column '{column_name}' in table '{table.name}' with type '{actual_index_type_to_log}', replace={replace}, kwargs_keys={kwargs_summary}.")
    try:
        with lancedb_write_lock:
            if index_type is not None:
                table.create_scalar_index(column_name, index_type=index_type, replace=replace, **kwargs)
            else:
                # Explicitly use 'BTREE' which is a valid Literal member
                table.create_scalar_index(column_name, index_type='BTREE', replace=replace, **kwargs)
        
        logger.info(f"Successfully created/updated scalar index on column '{column_name}' in table '{table.name}' using type '{actual_index_type_to_log}', replace={replace}.")
    except Exception as e:
        logger.error(f"Failed to create scalar index on column '{column_name}' in table '{table.name}' (type: {actual_index_type_to_log}, replace: {replace}): {e}")
        raise

def filter_variants_by_metadata(table: LanceTable, filter_sql: str, select_columns: Optional[List[str]] = None, limit: Optional[int] = None):
    """
    Filters variants in the table based on a SQL metadata filter.

    Args:
        table (LanceTable): The LanceDB table object.
        filter_sql (str): SQL-like filter string to apply to metadata columns.
        select_columns (Optional[List[str]]): List of columns to return. If None, all columns are returned.
        limit (Optional[int]): Maximum number of records to return.

    Returns:
        pd.DataFrame: DataFrame of filtered results.

    Raises:
        ValueError: If filter_sql is empty.
        Exception: If the filter operation fails.

    Example:
        >>> df = filter_variants_by_metadata(table, "chrom = '1' AND pos > 1000")
    """
    masked_filter_sql = mask_sensitive_sql(filter_sql or "") # Mask SQL literals for logging
    logger.info(f"Attempting to filter variants from table '{table.name}' using filter: '{masked_filter_sql}', select_columns: {select_columns}, limit: {limit}.")
    if not filter_sql: # Original filter_sql used for logic, masked_filter_sql for logging.
        logger.error(f"filter_variants_by_metadata called for table '{table.name}' with an empty filter_sql. This would return all rows (if not for safety checks).")
        raise ValueError("filter_sql cannot be empty for metadata filtering operations.")

    # Enhanced check for potentially unsafe SQL patterns
    unsafe_patterns = [
        "'1'='1'", " or 1=1", # Trivial true conditions
        ";",                   # Statement separator
        "--",                  # Comment
        "drop ", "delete ", "insert ", "update ", "alter " # DML/DDL keywords typically not expected in a simple filter
    ]
    filter_sql_lower = filter_sql.lower()
    if any(pattern in filter_sql_lower for pattern in unsafe_patterns):
        logger.error(f"Potentially unsafe SQL filter detected in table '{table.name}': '{masked_filter_sql}'. Rejecting query.")
        raise ValueError(f"Unsafe SQL filter detected: Query rejected. Filter was: {masked_filter_sql}")

    try:
        # logger.debug(f"Table object in filter_variants_by_metadata: type={{type(table)}}, dir={{dir(table)}}") # DEBUG LINE # Kept for potential future debugging
        # Hypothesis: Start with search(None) or search() for pure scalar queries
        # If search() requires a vector, this will fail. 
        # If search(None) works, or search() alone initiates a full scan, this is the path.
        # Try calling search() without any arguments, assuming it initiates a full table query context
        query_builder = table.search() 
        
        if select_columns:
            query_builder = query_builder.select(select_columns)
        
        query_builder = query_builder.where(filter_sql)

        if limit is not None:
            query_builder = query_builder.limit(limit)
        
        results_df = query_builder.to_pandas()
        logger.info(f"Successfully filtered variants from table '{table.name}' using filter '{masked_filter_sql}'. Found {len(results_df)} records (select_columns: {select_columns}, limit: {limit}).")
        return results_df
    except Exception as e:
        logger.error(f"Failed to filter variants from table '{table.name}' using filter '{masked_filter_sql}' (select_columns: {select_columns}, limit: {limit}): {e}")
        raise

# Enhanced VCF Variant operations based on DECISION-001 specifications

def get_or_create_vcf_table(db, table_name: str = "vcf_variants", use_optimized_model: bool = None):
    """
    Opens an existing VCF variants LanceDB table or creates a new one if it doesn't exist.
    Uses the enhanced VCFVariant schema or VCFVariantPhase3 for optimized storage.

    Args:
        db (lancedb.DBConnection): LanceDB database connection object.
        table_name (str): Name of the table to open or create. Defaults to 'vcf_variants'.
        use_optimized_model (bool): Whether to use optimized model. If None, uses default configuration.

    Returns:
        LanceTable: The opened or newly created LanceDB table with appropriate schema.

    Raises:
        Exception: If table creation or opening fails.

    Example:
        >>> db = get_db()
        >>> table = get_or_create_vcf_table(db, "vcf_variants")
    """
    logger.info(f"Attempting to get or create VCF table '{table_name}'...")
    
    # Use default optimization configuration if not specified
    if use_optimized_model is None:
        default_config = MemoryOptimizationConfig()
        use_optimized_model = default_config.is_optimized_model_enabled()
    
    # Select appropriate model
    schema_model = VCFVariantPhase3 if use_optimized_model else VCFVariant
    schema_name = "VCFVariantPhase3" if use_optimized_model else "VCFVariant"
    
    try:
        if table_name in db.table_names():
            table = db.open_table(table_name)
            logger.info(f"Opened existing VCF table: '{table_name}'.")
        else:
            table = db.create_table(table_name, schema=schema_model, mode="overwrite")
            logger.info(f"Created new VCF table: '{table_name}' with schema {schema_name} (optimized: {use_optimized_model}).")
        return table
    except Exception as e:
        logger.error(f"Failed to get or create VCF table '{table_name}': {e}")
        raise

def create_vcf_variant_record(
    variant_data: Dict,
    embedding_service: Optional[VariantEmbeddingService] = None,
    analysis_summary: str = ""
) -> Dict:
    """
    Create a VCFVariant record with embedding generation and integrated optimizations.
    
    Args:
        variant_data: Dictionary containing variant information.
        embedding_service: Service for generating embeddings.
        analysis_summary: AI-generated analysis summary.
        
    Returns:
        Dictionary ready for insertion into LanceDB with appropriate embedding dimensions.
    """
    if embedding_service is None:
        embedding_service = VariantEmbeddingService()
    
    # Generate variant description
    description = embedding_service.generate_variant_description(variant_data)
    
    # Generate embedding with integrated optimizations
    embedding = embedding_service.generate_embedding_sync(description)
    
    # Validate embedding dimensions
    expected_dimensions = embedding_service.get_embedding_dimensions()
    if len(embedding) != expected_dimensions:
        logger.warning(f"Embedding dimension mismatch: expected {expected_dimensions}, got {len(embedding)}")
    
    # Create timestamp
    now = datetime.now()
    
    # Build the record
    record = {
        "variant_id": variant_data.get("variant_id", f"{variant_data.get('chromosome', 'unknown')}-{variant_data.get('position', 0)}-{variant_data.get('reference', '')}-{variant_data.get('alternate', '')}"),
        "chromosome": variant_data.get("chromosome", variant_data.get("chrom", "")),
        "position": variant_data.get("position", variant_data.get("pos", 0)),
        "reference": variant_data.get("reference", variant_data.get("ref", "")),
        "alternate": variant_data.get("alternate", variant_data.get("alt", "")),
        "variant_description": description,
        "variant_vector": embedding,
        "analysis_summary": analysis_summary,
        "sample_id": variant_data.get("sample_id", ""),
        "quality_score": variant_data.get("quality_score", variant_data.get("qual")),
        "filter_status": variant_data.get("filter_status", variant_data.get("filter")),
        "genotype": variant_data.get("genotype", variant_data.get("gt")),
        "allele_frequency": variant_data.get("allele_frequency", variant_data.get("af")),
        "clinical_significance": variant_data.get("clinical_significance"),
        "gene_symbol": variant_data.get("gene_symbol"),
        "consequence": variant_data.get("consequence"),
        "created_at": now,
        "updated_at": now
    }
    
    logger.debug(f"Created VCF variant record with {len(embedding)}-dimensional embedding")
    return record

def batch_add_vcf_variants(
    table: LanceTable,
    variants_data: List[Dict],
    embedding_service: Optional[VariantEmbeddingService] = None,
    batch_size: int = 1000,
    max_workers: int = 4
) -> int:
    """
    Add a batch of VCF variant records to the specified LanceDB table with optimized performance.
    Target: >10,000 variants/second as per DECISION-001.

    Args:
        table (LanceTable): The LanceDB table object.
        variants_data (List[Dict]): List of variant dictionaries.
        embedding_service: Service for generating embeddings.
        batch_size: Number of variants to process in each batch.
        max_workers: Number of worker threads for parallel processing.

    Returns:
        int: Number of variants successfully added.

    Raises:
        Exception: If batch insertion fails.
    """
    if not variants_data:
        logger.warning("No variants provided for batch insertion.")
        return 0
    
    if embedding_service is None:
        embedding_service = VariantEmbeddingService()
    
    logger.info(f"Starting batch insertion of {len(variants_data)} variants with batch_size={batch_size}, max_workers={max_workers}")
    
    def process_batch(batch_data: List[Dict]) -> List[Dict]:
        """Process a batch of variants with embeddings and optimizations."""
        processed_records = []
        
        # Use streaming batch processing if available
        try:
            # Extract descriptions
            descriptions = []
            for variant_data in batch_data:
                description = embedding_service.generate_variant_description(variant_data)
                descriptions.append(description)
            
            # Generate embeddings in batch
            embeddings = embedding_service.generate_embeddings_batch(descriptions)
            
            # Create records with pre-generated embeddings
            for i, variant_data in enumerate(batch_data):
                try:
                    # Build record manually to use pre-generated embedding
                    now = datetime.now()
                    record = {
                        "variant_id": variant_data.get("variant_id", f"{variant_data.get('chromosome', 'unknown')}-{variant_data.get('position', 0)}-{variant_data.get('reference', '')}-{variant_data.get('alternate', '')}"),
                        "chromosome": variant_data.get("chromosome", variant_data.get("chrom", "")),
                        "position": variant_data.get("position", variant_data.get("pos", 0)),
                        "reference": variant_data.get("reference", variant_data.get("ref", "")),
                        "alternate": variant_data.get("alternate", variant_data.get("alt", "")),
                        "variant_description": descriptions[i],
                        "variant_vector": embeddings[i],
                        "analysis_summary": "",
                        "sample_id": variant_data.get("sample_id", ""),
                        "quality_score": variant_data.get("quality_score", variant_data.get("qual")),
                        "filter_status": variant_data.get("filter_status", variant_data.get("filter")),
                        "genotype": variant_data.get("genotype", variant_data.get("gt")),
                        "allele_frequency": variant_data.get("allele_frequency", variant_data.get("af")),
                        "clinical_significance": variant_data.get("clinical_significance"),
                        "gene_symbol": variant_data.get("gene_symbol"),
                        "consequence": variant_data.get("consequence"),
                        "created_at": now,
                        "updated_at": now
                    }
                    processed_records.append(record)
                except Exception as e:
                    logger.error(f"Failed to process variant {variant_data.get('variant_id', 'unknown')}: {e}")
                    continue
                    
            logger.debug(f"Processed batch of {len(processed_records)} variants using optimized processing")
            return processed_records
            
        except Exception as e:
            logger.warning(f"Batch processing failed, falling back to individual processing: {e}")
            
            # Fallback to individual processing
            for variant_data in batch_data:
                try:
                    record = create_vcf_variant_record(variant_data, embedding_service, "")
                    processed_records.append(record)
                except Exception as e:
                    logger.error(f"Failed to process variant {variant_data.get('variant_id', 'unknown')}: {e}")
                    continue
            return processed_records
    
    total_added = 0
    
    try:
        with lancedb_write_lock:
            # Process variants in batches with parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Split data into batches
                batches = [variants_data[i:i + batch_size] for i in range(0, len(variants_data), batch_size)]
                
                # Process batches in parallel
                future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
                
                for future in future_to_batch:
                    try:
                        processed_records = future.result()
                        if processed_records:
                            # Add to LanceDB
                            table.add(processed_records)
                            total_added += len(processed_records)
                            logger.info(f"Added batch of {len(processed_records)} variants. Total: {total_added}")
                    except Exception as e:
                        logger.error(f"Failed to add batch: {e}")
                        continue
        
        logger.info(f"Successfully added {total_added} variants to table with optimized processing")
        return total_added
        
    except Exception as e:
        logger.error(f"Batch insertion failed: {e}")
        raise

def hybrid_search_variants(
    table: LanceTable,
    query_text: str,
    embedding_service: Optional[VariantEmbeddingService] = None,
    metadata_filter: Optional[str] = None,
    limit: int = 10,
    similarity_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Perform hybrid search combining vector similarity and metadata filtering.
    Target: <100ms for similarity queries as per DECISION-001.

    Args:
        table (LanceTable): The LanceDB table object.
        query_text (str): Text query for semantic similarity search.
        embedding_service: Service for generating query embeddings.
        metadata_filter (Optional[str]): SQL filter for metadata (e.g., "chromosome = '1'").
        limit (int): Maximum number of results to return.
        similarity_threshold (float): Minimum similarity score (0-1).

    Returns:
        pd.DataFrame: Search results with similarity scores.

    Example:
        >>> results = hybrid_search_variants(
        ...     table, 
        ...     "pathogenic variant in BRCA1 gene",
        ...     metadata_filter="chromosome = '17' AND gene_symbol = 'BRCA1'",
        ...     limit=5
        ... )
    """
    if embedding_service is None:
        embedding_service = VariantEmbeddingService()
    
    logger.info(f"Performing hybrid search for: '{query_text}' with filter: {mask_sensitive_sql(metadata_filter or 'None')}")
    
    try:
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding_sync(query_text)
        
        # Perform vector search with optional metadata filter
        search_query = table.search(query_embedding, vector_column_name="variant_vector").limit(limit)
        
        if metadata_filter:
            search_query = search_query.where(metadata_filter)
        
        # Execute search
        results = search_query.to_pandas()
        
        # Filter by similarity threshold if needed
        if similarity_threshold > 0 and '_distance' in results.columns:
            # Convert distance to similarity (assuming cosine distance)
            results['similarity'] = 1 - results['_distance']
            results = results[results['similarity'] >= similarity_threshold]
        
        logger.info(f"Hybrid search returned {len(results)} results.")
        return results
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise

def search_similar_variants(
    table: LanceTable,
    reference_variant_id: str,
    limit: int = 10,
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Find variants similar to a reference variant using vector similarity.

    Args:
        table (LanceTable): The LanceDB table object.
        reference_variant_id (str): ID of the reference variant.
        limit (int): Maximum number of similar variants to return.
        include_metadata (bool): Whether to include metadata in results.

    Returns:
        pd.DataFrame: Similar variants with similarity scores.
    """
    try:
        # Get the reference variant
        ref_results = table.search().where(f"variant_id = '{reference_variant_id}'").limit(1).to_pandas()
        
        if ref_results.empty:
            logger.warning(f"Reference variant {reference_variant_id} not found.")
            return pd.DataFrame()
        
        # Get the reference embedding
        ref_embedding = ref_results.iloc[0]['variant_vector']
        
        # Search for similar variants (excluding the reference itself)
        search_query = table.search(ref_embedding, vector_column_name="variant_vector").limit(limit + 1)
        results = search_query.to_pandas()
        
        # Remove the reference variant from results
        results = results[results['variant_id'] != reference_variant_id]
        
        if not include_metadata:
            # Return only essential columns
            essential_cols = ['variant_id', 'chromosome', 'position', 'reference', 'alternate', '_distance']
            results = results[[col for col in essential_cols if col in results.columns]]
        
        logger.info(f"Found {len(results)} similar variants to {reference_variant_id}.")
        return results
        
    except Exception as e:
        logger.error(f"Similar variant search failed: {e}")
        raise

def get_variant_statistics(table: LanceTable) -> Dict:
    """
    Get comprehensive statistics about the variants in the table.

    Args:
        table (LanceTable): The LanceDB table object.

    Returns:
        Dict: Statistics including counts, distributions, and performance metrics.
    """
    try:
        # Get basic counts
        total_count = table.count_rows()
        
        # Get sample data for analysis
        sample_data = table.search().limit(1000).to_pandas()
        
        stats = {
            "total_variants": total_count,
            "sample_size": len(sample_data),
            "chromosomes": sample_data['chromosome'].value_counts().to_dict() if not sample_data.empty else {},
            "filter_status": sample_data['filter_status'].value_counts().to_dict() if not sample_data.empty else {},
            "clinical_significance": sample_data['clinical_significance'].value_counts().to_dict() if not sample_data.empty else {},
            "has_gene_symbol": sample_data['gene_symbol'].notna().sum() if not sample_data.empty else 0,
            "has_consequence": sample_data['consequence'].notna().sum() if not sample_data.empty else 0,
            "quality_score_stats": sample_data['quality_score'].describe().to_dict() if not sample_data.empty and 'quality_score' in sample_data.columns else {}
        }
        
        logger.info(f"Generated statistics for {total_count} variants.")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to generate variant statistics: {e}")
        return {"error": str(e)}

__all__ = [
    # Core functions
    'mask_sensitive_sql',
    'get_db',
    'get_or_create_table',
    'add_variants',
    'search_by_embedding',
    'update_variant',
    'delete_variants',
    'create_scalar_index',
    'filter_variants_by_metadata',
    
    # Enhanced VCF-specific functions (DECISION-001)
    'get_or_create_vcf_table',
    'create_vcf_variant_record',
    'batch_add_vcf_variants',
    'hybrid_search_variants',
    'search_similar_variants',
    'get_variant_statistics',
    
    # Models
    'Variant',  # Legacy model
    'VCFVariant',  # Enhanced model
    
    # Services
    'VariantEmbeddingService',
    
    # Types
    'LanceScalarIndexType',
] 