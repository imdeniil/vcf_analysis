"""
VCF Ingestion Pipeline

This module provides a comprehensive VCF file ingestion pipeline that processes
VCF files and populates both LanceDB (vector database) and Kuzu (graph database)
simultaneously with memory-efficient streaming and batch processing.

Key Features:
- Memory-efficient streaming of large VCF files
- Batch processing for optimal performance
- Dual database ingestion (LanceDB + Kuzu)
- Comprehensive validation and error handling
- Progress tracking and resumable ingestion
- Embedding generation for variant sequences
"""

import os
import sys
import time
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterator, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# VCF processing
from cyvcf2 import VCF

# Database integrations
from .lancedb_integration import get_db, get_or_create_table, add_variants, Variant
from .graph_integration import get_kuzu_db_connection, create_schema
from .vcf_utils import populate_kuzu_from_vcf

# Progress tracking
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.current = 0
        
        def update(self, n=1):
            self.current += n
            if self.total > 0:
                print(f"Progress: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)")
        
        def close(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

logger = logging.getLogger(__name__)

@dataclass
class IngestionConfig:
    """Configuration for VCF ingestion pipeline."""
    vcf_file: str
    lancedb_path: str = "./lancedb"
    kuzu_path: str = "./kuzu_db"
    table_name: str = "variants"
    batch_size: int = 1000
    validate_only: bool = False
    resume_from: Optional[str] = None  # Format: "CHROM:POS"
    sample_name_override: Optional[str] = None
    embedding_dim: int = 1024
    max_memory_mb: int = 4096
    # Incremental / nightly-ingest controls.
    # checkpoint_path: where to persist the last-processed CHROM:POS so a run can
    # be paused (Ctrl+C) and later resumed automatically. If None, no checkpoint
    # file is written (original behaviour).
    checkpoint_path: Optional[str] = None
    # max_runtime_minutes: stop cleanly after this many minutes (writes the
    # checkpoint first). Lets you run ingest during a bounded night window and
    # free the machine for the day. None = no time limit.
    max_runtime_minutes: Optional[float] = None

@dataclass
class IngestionResult:
    """Result of VCF ingestion operation."""
    variants_processed: int = 0
    lancedb_records: int = 0
    kuzu_variants: int = 0
    kuzu_samples: int = 0
    kuzu_links: int = 0
    errors: List[str] = None
    duration_seconds: float = 0.0
    message: str = "completed"
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class VCFValidator:
    """Validates VCF files for ingestion readiness."""
    
    @staticmethod
    def validate_file(vcf_path: str) -> Tuple[bool, List[str]]:
        """
        Validate VCF file format and accessibility.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check file existence
        if not os.path.exists(vcf_path):
            errors.append(f"VCF file not found: {vcf_path}")
            return False, errors
        
        # Check file readability
        if not os.access(vcf_path, os.R_OK):
            errors.append(f"VCF file not readable: {vcf_path}")
            return False, errors
        
        # Check file format
        try:
            vcf = VCF(vcf_path)
            # Try to read first variant
            first_variant = next(iter(vcf), None)
            if first_variant is None:
                errors.append("VCF file contains no variants")
                return False, errors
                
            # Check for required fields
            if not hasattr(first_variant, 'CHROM') or not first_variant.CHROM:
                errors.append("VCF file missing CHROM field")
            if not hasattr(first_variant, 'POS') or not first_variant.POS:
                errors.append("VCF file missing POS field")
            if not hasattr(first_variant, 'REF') or not first_variant.REF:
                errors.append("VCF file missing REF field")
            if not hasattr(first_variant, 'ALT') or not first_variant.ALT:
                errors.append("VCF file missing ALT field")
            
            vcf.close()
                
        except Exception as e:
            errors.append(f"Invalid VCF format: {str(e)}")
            return False, errors
        
        return len(errors) == 0, errors

class VCFStreamer:
    """Memory-efficient VCF file streaming processor."""
    
    def __init__(self, vcf_path: str, batch_size: int = 1000, resume_from: Optional[str] = None):
        """Initializes an instance with paths and settings for processing VCF files.
        Parameters:
            - vcf_path (str): Path to the VCF file to be processed.
            - batch_size (int): Number of records to process at a time, defaults to 1000.
            - resume_from (Optional[str]): Position to resume processing from, in format 'chrom:pos'.
        Returns:
            - None
        Processing Logic:
            - Parses the 'resume_from' parameter to extract the chromosome and position if provided.
            - Logs a warning if the 'resume_from' format is invalid.
            - Sets the current processing position to None initially."""
        self.vcf_path = vcf_path
        self.batch_size = batch_size
        self.resume_from = resume_from
        self.current_position = None
        
        # Parse resume position if provided
        self.resume_chrom = None
        self.resume_pos = None
        if resume_from:
            try:
                parts = resume_from.split(':')
                if len(parts) == 2:
                    self.resume_chrom = parts[0]
                    self.resume_pos = int(parts[1])
            except ValueError:
                logger.warning(f"Invalid resume position format: {resume_from}")
    
    def count_variants(self) -> int:
        """Count total variants in VCF file for progress tracking."""
        try:
            vcf = VCF(self.vcf_path)
            count = 0
            for _ in vcf:
                count += 1
            vcf.close()
            return count
        except Exception as e:
            logger.warning(f"Could not count variants: {e}")
            return 0
    
    def stream_variants(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Stream VCF variants in batches.
        
        Yields:
            List of variant dictionaries for each batch
        """
        vcf = VCF(self.vcf_path)
        try:
            batch = []
            skipping = self.resume_from is not None
            
            for variant in vcf:
                # Handle resume logic
                if skipping:
                    if (variant.CHROM == self.resume_chrom and 
                        variant.POS >= self.resume_pos):
                        skipping = False
                    else:
                        continue
                
                # Parse variant data
                variant_data = self._parse_variant(variant, vcf.samples)
                batch.append(variant_data)
                
                # Update current position for checkpointing
                self.current_position = f"{variant.CHROM}:{variant.POS}"
                
                # Yield batch when full
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            
            # Yield remaining variants
            if batch:
                yield batch
        finally:
            vcf.close()
    
    def _parse_variant(self, variant, samples: List[str]) -> Dict[str, Any]:
        """Parse a single variant record into a dictionary."""
        # Generate variant ID
        alt_allele = variant.ALT[0] if variant.ALT else "."
        variant_id = f"{variant.CHROM}-{variant.POS}-{variant.REF}-{alt_allele}"
        
        # Extract basic variant information
        variant_data = {
            'variant_id': variant_id,
            'chrom': str(variant.CHROM),
            'pos': int(variant.POS),
            'ref': str(variant.REF),
            'alt': str(alt_allele),
            'rs_id': variant.ID if variant.ID else None,
            'qual': float(variant.QUAL) if variant.QUAL is not None else None,
            'filter': variant.FILTER if variant.FILTER else "PASS",
            'info': dict(variant.INFO) if hasattr(variant, 'INFO') else {},
            'samples': samples,
            'genotypes': []
        }
        
        # Extract genotype information
        if hasattr(variant, 'genotypes') and variant.genotypes:
            for i, gt in enumerate(variant.genotypes):
                if i < len(samples):
                    genotype_data = {
                        'sample_id': samples[i],
                        'gt': gt[:2] if len(gt) >= 2 else [0, 0],  # [allele1, allele2]
                        'phased': gt[2] if len(gt) > 2 else False
                    }
                    variant_data['genotypes'].append(genotype_data)
        
        return variant_data

class EmbeddingGenerator:
    """Generates embeddings for variant sequences."""
    
    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim
    
    def generate_embedding(self, variant_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding vector for a variant.
        
        For now, this creates a simple hash-based embedding.
        In production, this could use a pre-trained genomics model.
        """
        # Create sequence context
        sequence_context = f"{variant_data['ref']}>{variant_data['alt']} at {variant_data['chrom']}:{variant_data['pos']}"
        
        # Simple hash-based embedding (replace with real model in production)
        hash_value = hash(sequence_context)
        np.random.seed(abs(hash_value) % (2**32))
        embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

class VCFIngestionPipeline:
    """Main VCF ingestion pipeline orchestrator."""
    
    def __init__(self, config: IngestionConfig):
        """Initialize the ingestion process with configuration settings.
        Parameters:
            - config (IngestionConfig): The configuration settings for the ingestion process which includes file paths, batch size, and embedding dimensions.
        Returns:
            - None
        Processing Logic:
            - Initializes the VCFStreamer with the specified file, batch size, and resume point from the configuration.
            - Sets up an EmbeddingGenerator with the specified embedding dimensions from the configuration.
            - Establishes initial null state for database connections which will be setup later as required.
            - Initializes the progress tracking mechanism and prepares results tracking."""
        self.config = config
        self.validator = VCFValidator()
        self.streamer = VCFStreamer(
            config.vcf_file,
            config.batch_size,
            config.resume_from
        )
        self.embedding_generator = EmbeddingGenerator(config.embedding_dim)

        # Real embedding service (bge-m3 via LM Studio / OpenAI-compatible server).
        # Falls back to EmbeddingGenerator only if the service has no client.
        try:
            from .lancedb_integration import VariantEmbeddingService
            from .config import SessionConfig
            self.embedding_service = VariantEmbeddingService(SessionConfig())
            if self.embedding_service.embedding_client is None and self.embedding_service.openai_client is None:
                logger.warning(
                    "No embedding client configured (EMBEDDING_BASE_URL unreachable?). "
                    "Falling back to hash-based EmbeddingGenerator — similarity search "
                    "will NOT be meaningful."
                )
                self.embedding_service = None
            else:
                logger.info(
                    f"Using real embedding service: model={self.embedding_service.embedding_model}, "
                    f"dim={self.embedding_service.embedding_dim}"
                )
        except Exception as e:
            logger.warning(f"Could not init VariantEmbeddingService ({e}); using hash-based fallback")
            self.embedding_service = None

        # Database connections (initialized in setup)
        self.lancedb_conn = None
        self.lancedb_table = None
        self.kuzu_conn = None
        
        # Progress tracking
        self.progress_bar = None
        self.total_variants = 0
        
        # Results tracking
        self.result = IngestionResult()
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete VCF ingestion pipeline.
        
        Returns:
            Dictionary with ingestion results and statistics
        """
        start_time = time.time()
        
        try:
            # Phase 1: Validation
            if not self._validate_inputs():
                raise ValueError("Input validation failed")
            
            if self.config.validate_only:
                logger.info("Validation completed successfully. Skipping ingestion.")
                return self.result.__dict__
            
            # Phase 2: Setup databases
            self._setup_databases()
            
            # Phase 3: Count variants for progress tracking
            self.total_variants = self.streamer.count_variants()
            logger.info(f"Processing {self.total_variants} variants from {self.config.vcf_file}")
            
            # Phase 4: Process variants in batches
            self._process_variants()
            
            # Phase 5: Finalize
            self._finalize_ingestion()
            
            self.result.duration_seconds = time.time() - start_time
            logger.info(f"VCF ingestion completed in {self.result.duration_seconds:.2f} seconds")
            
            return self.result.__dict__
            
        except Exception as e:
            self.result.errors.append(str(e))
            self.result.duration_seconds = time.time() - start_time
            logger.error(f"VCF ingestion failed: {e}")
            logger.error(traceback.format_exc())
            raise
        
        finally:
            self._cleanup()
    
    def _validate_inputs(self) -> bool:
        """Validate all inputs before processing."""
        logger.info("Validating VCF file...")
        
        is_valid, errors = self.validator.validate_file(self.config.vcf_file)
        if not is_valid:
            for error in errors:
                self.result.errors.append(error)
                logger.error(error)
            return False
        
        # Validate database paths
        try:
            os.makedirs(os.path.dirname(self.config.lancedb_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.kuzu_path), exist_ok=True)
        except Exception as e:
            error_msg = f"Cannot create database directories: {e}"
            self.result.errors.append(error_msg)
            logger.error(error_msg)
            return False
        
        logger.info("✅ Input validation passed")
        return True
    
    def _setup_databases(self):
        """Initialize database connections and schemas."""
        logger.info("Setting up database connections...")
        
        try:
            # Setup LanceDB
            self.lancedb_conn = get_db(self.config.lancedb_path)
            self.lancedb_table = get_or_create_table(self.lancedb_conn, self.config.table_name)
            logger.info(f"✅ LanceDB connected: {self.config.lancedb_path}")
            
            # Setup Kuzu
            self.kuzu_conn = get_kuzu_db_connection(self.config.kuzu_path)
            create_schema(self.kuzu_conn)
            logger.info(f"✅ Kuzu connected: {self.config.kuzu_path}")
            
        except Exception as e:
            error_msg = f"Database setup failed: {e}"
            self.result.errors.append(error_msg)
            logger.error(error_msg)
            raise
    
    def _process_variants(self):
        """Process variants in batches, with optional checkpointing and time limit.

        Supports incremental / nightly ingestion:
        - On start, if --resume-from was not given but a checkpoint file exists,
          resume from the saved CHROM:POS automatically.
        - After every processed batch, write the current CHROM:POS to the
          checkpoint file (so a Ctrl+C or a time-limit stop can be resumed).
        - If max_runtime_minutes is set, stop cleanly once the budget is spent,
          after writing the checkpoint.
        - KeyboardInterrupt (Ctrl+C) writes the checkpoint before re-raising.
        """
        logger.info("Processing variants...")

        # Auto-resume from checkpoint when --resume-from was not explicitly given.
        if self.config.resume_from is None and self.config.checkpoint_path:
            saved = self._read_checkpoint()
            if saved:
                logger.info(f"Auto-resuming from checkpoint: {saved}")
                self.config.resume_from = saved
                # Rebuild the streamer so it picks up the new resume point.
                self.streamer = VCFStreamer(
                    self.config.vcf_file, self.config.batch_size, self.config.resume_from
                )

        # Initialize progress bar
        self.progress_bar = tqdm(
            total=self.total_variants,
            desc="Processing variants",
            unit="variants"
        )

        start_time = time.time()
        time_limit_reached = False
        try:
            for batch in self.streamer.stream_variants():
                self._process_batch(batch)
                self.progress_bar.update(len(batch))

                # Persist progress so the run can be paused/resumed.
                if self.config.checkpoint_path and self.streamer.current_position:
                    self._write_checkpoint(self.streamer.current_position)

                # Time-budget check (for bounded nightly runs).
                if self.config.max_runtime_minutes is not None:
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= self.config.max_runtime_minutes:
                        logger.info(
                            f"Reached max_runtime_minutes={self.config.max_runtime_minutes} "
                            f"(elapsed {elapsed_min:.1f}min). Stopping cleanly; resume with the same command."
                        )
                        time_limit_reached = True
                        break
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C). Checkpoint written; resume with the same command.")
            if self.config.checkpoint_path and self.streamer.current_position:
                self._write_checkpoint(self.streamer.current_position)
            raise
        finally:
            if self.progress_bar:
                self.progress_bar.close()

        if time_limit_reached:
            self.result.message = "stopped_after_time_budget"

    def _write_checkpoint(self, position: str) -> None:
        """Persist the last-processed CHROM:POS to the checkpoint file."""
        try:
            with open(self.config.checkpoint_path, "w", encoding="utf-8") as f:
                f.write(position)
        except Exception as e:
            logger.warning(f"Could not write checkpoint {self.config.checkpoint_path}: {e}")

    def _read_checkpoint(self) -> Optional[str]:
        """Read the last-processed CHROM:POS from the checkpoint file, if any."""
        try:
            with open(self.config.checkpoint_path, "r", encoding="utf-8") as f:
                pos = f.read().strip()
            return pos or None
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Could not read checkpoint {self.config.checkpoint_path}: {e}")
            return None
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a single batch of variants."""
        try:
            # Prepare LanceDB records
            lancedb_records = []

            # Generate embeddings. Prefer the real bge-m3 service with TRUE batched
            # HTTP requests (one round-trip per chunk) — this is ~10-100x faster than
            # per-variant calls. Fall back to the hash-based EmbeddingGenerator only
            # when no embedding server is configured.
            embeddings: List[List[float]] = []
            if self.embedding_service is not None:
                descriptions = [
                    self.embedding_service.generate_variant_description(v) for v in batch
                ]
                embeddings = self.embedding_service.generate_embeddings_batch(descriptions)
            else:
                for variant_data in batch:
                    emb = self.embedding_generator.generate_embedding(variant_data)
                    embeddings.append(emb.tolist())

            for i, variant_data in enumerate(batch):
                # Create LanceDB record
                lancedb_record = {
                    'variant_id': variant_data['variant_id'],
                    'embedding': embeddings[i],
                    'chrom': variant_data['chrom'],
                    'pos': variant_data['pos'],
                    'ref': variant_data['ref'],
                    'alt': variant_data['alt'],
                    'clinical_significance': None  # Could be populated from INFO field
                }
                lancedb_records.append(lancedb_record)

            # Batch insert to LanceDB
            if lancedb_records:
                add_variants(self.lancedb_table, lancedb_records)
                self.result.lancedb_records += len(lancedb_records)
            
            # Process for Kuzu (reuse existing logic)
            # Create temporary VCF-like data for Kuzu processing
            kuzu_result = self._process_kuzu_batch(batch)
            self.result.kuzu_variants += kuzu_result.get('variants', 0)
            self.result.kuzu_samples += kuzu_result.get('samples', 0)
            self.result.kuzu_links += kuzu_result.get('links', 0)
            
            self.result.variants_processed += len(batch)
            
        except Exception as e:
            error_msg = f"Batch processing failed: {e}"
            self.result.errors.append(error_msg)
            logger.error(error_msg)
            raise
    
    def _process_kuzu_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process batch for Kuzu graph database."""
        from .graph_integration import add_variant, add_sample, link_variant_to_sample
        
        counts = {'variants': 0, 'samples': 0, 'links': 0}
        processed_samples = set()
        
        try:
            for variant_data in batch:
                # Add variant to Kuzu
                kuzu_variant_data = {
                    'variant_id': variant_data['variant_id'],
                    'chrom': variant_data['chrom'],
                    'pos': variant_data['pos'],
                    'ref': variant_data['ref'],
                    'alt': variant_data['alt'],
                    'rs_id': variant_data.get('rs_id', '')
                }
                
                try:
                    add_variant(self.kuzu_conn, kuzu_variant_data)
                    counts['variants'] += 1
                except Exception as e:
                    logger.debug(f"Variant {variant_data['variant_id']} may already exist: {e}")
                
                # Process samples and genotypes
                for genotype in variant_data.get('genotypes', []):
                    sample_id = genotype['sample_id']
                    
                    # Override sample name if specified
                    if self.config.sample_name_override:
                        sample_id = self.config.sample_name_override
                    
                    # Add sample if not already processed
                    if sample_id not in processed_samples:
                        try:
                            add_sample(self.kuzu_conn, {'sample_id': sample_id})
                            counts['samples'] += 1
                            processed_samples.add(sample_id)
                        except Exception as e:
                            logger.debug(f"Sample {sample_id} may already exist: {e}")
                    
                    # Determine zygosity and create link
                    gt = genotype['gt']
                    if any(allele > 0 for allele in gt):  # Has alternate allele
                        if gt[0] == gt[1] and gt[0] > 0:
                            zygosity = "HOM_ALT"
                        elif gt[0] != gt[1]:
                            zygosity = "HET"
                        else:
                            zygosity = "HOM_REF"
                            continue  # Skip HOM_REF
                        
                        try:
                            link_variant_to_sample(
                                self.kuzu_conn,
                                sample_id,
                                variant_data['variant_id'],
                                {'zygosity': zygosity}
                            )
                            counts['links'] += 1
                        except Exception as e:
                            logger.debug(f"Link may already exist: {e}")
            
            return counts
            
        except Exception as e:
            logger.error(f"Kuzu batch processing failed: {e}")
            return counts
    
    def _finalize_ingestion(self):
        """Finalize the ingestion process."""
        logger.info("Finalizing ingestion...")
        
        # Could add index creation, optimization, etc.
        # For now, just log completion
        logger.info("✅ Ingestion finalized")
    
    def _cleanup(self):
        """Clean up resources, including releasing the Kuzu database lock.

        Kuzu takes an exclusive file lock on the database file; if the connection
        is not closed, the next ingest run (e.g. a resume the following night)
        fails with "Could not set lock on file". This is critical for the
        pause/resume nightly workflow.
        """
        if self.progress_bar:
            self.progress_bar.close()

        # Release the Kuzu connection + its file lock.
        # Kuzu locks the database file at the Database (C) level, so we must
        # close BOTH the Connection and the underlying Database; closing only
        # the Connection leaves the file lock in place and blocks the next run.
        if self.kuzu_conn is not None:
            try:
                self.kuzu_conn.close()
            except Exception as e:
                logger.warning(f"Error closing Kuzu connection: {e}")
            kuzu_db = getattr(self.kuzu_conn, "_kuzu_db", None)
            if kuzu_db is not None:
                try:
                    kuzu_db.close()
                except Exception as e:
                    logger.warning(f"Error closing Kuzu database: {e}")
            self.kuzu_conn = None

        logger.info("Cleanup completed") 