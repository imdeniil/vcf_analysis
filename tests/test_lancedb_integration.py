"""
Test suite for enhanced LanceDB integration with VCF variants.
Tests the DECISION-001 implementation including VCFVariant model, embedding service, and batch operations.
"""

import pytest
import tempfile
import shutil
import os
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd

from src.vcf_agent.lancedb_integration import (
    VCFVariant,
    VariantEmbeddingService,
    get_db,
    get_or_create_vcf_table,
    create_vcf_variant_record,
    batch_add_vcf_variants,
    hybrid_search_variants,
    search_similar_variants,
    get_variant_statistics
)
from src.vcf_agent.config import SessionConfig


class TestVCFVariantModel:
    """Test the VCFVariant Pydantic model."""
    
    def test_vcf_variant_creation(self):
        """Test creating a VCFVariant instance."""
        now = datetime.now()
        variant = VCFVariant(
            variant_id="chr1-123456-A-G",
            chromosome="1",
            position=123456,
            reference="A",
            alternate="G",
            variant_description="Test variant",
            variant_vector=[0.1] * 1536,  # 1536-dimensional vector
            analysis_summary="Test analysis",
            sample_id="sample_001",
            created_at=now,
            updated_at=now
        )
        
        assert variant.variant_id == "chr1-123456-A-G"
        assert variant.chromosome == "1"
        assert variant.position == 123456
        assert len(variant.variant_vector) == 1536
    
    def test_vcf_variant_optional_fields(self):
        """Test VCFVariant with optional fields."""
        now = datetime.now()
        variant = VCFVariant(
            variant_id="chr1-123456-A-G",
            chromosome="1",
            position=123456,
            reference="A",
            alternate="G",
            variant_description="Test variant",
            variant_vector=[0.1] * 1536,
            analysis_summary="Test analysis",
            sample_id="sample_001",
            quality_score=99.5,
            filter_status="PASS",
            genotype="0/1",
            allele_frequency=0.25,
            clinical_significance="Pathogenic",
            gene_symbol="BRCA1",
            consequence="missense_variant",
            created_at=now,
            updated_at=now
        )
        
        assert variant.quality_score == 99.5
        assert variant.filter_status == "PASS"
        assert variant.clinical_significance == "Pathogenic"
        assert variant.gene_symbol == "BRCA1"


class TestVariantEmbeddingService:
    """Test the VariantEmbeddingService."""
    
    def test_embedding_service_initialization(self):
        """Test embedding service initialization."""
        service = VariantEmbeddingService()
        assert service.session_config is not None
        assert service.embedding_cache == {}
    
    def test_generate_variant_description(self):
        """Test variant description generation."""
        service = VariantEmbeddingService()
        variant_data = {
            "chromosome": "1",
            "position": 123456,
            "reference": "A",
            "alternate": "G",
            "gene_symbol": "BRCA1",
            "consequence": "missense_variant",
            "clinical_significance": "Pathogenic"
        }
        
        description = service.generate_variant_description(variant_data)
        
        assert "chromosome 1" in description
        assert "position 123456" in description
        assert "reference allele A" in description
        assert "alternate allele G" in description
        assert "gene BRCA1" in description
        assert "missense_variant" in description
        assert "Pathogenic" in description
    
    def test_generate_variant_description_minimal(self):
        """Test variant description with minimal data."""
        service = VariantEmbeddingService()
        variant_data = {
            "chromosome": "X",
            "position": 789,
            "reference": "T",
            "alternate": "C"
        }
        
        description = service.generate_variant_description(variant_data)
        
        assert "chromosome X" in description
        assert "position 789" in description
        assert "reference allele T" in description
        assert "alternate allele C" in description
    
    def test_generate_embedding_fallback_when_no_client(self, monkeypatch):
        """When no embedding server is reachable, fall back to a random vector
        whose size matches the configured embedding dimension (bge-m3 = 1024)."""
        service = VariantEmbeddingService()
        # Force the fallback path: no embedding client available.
        service.embedding_client = None
        service.openai_client = None
        # bge-m3 default dimension
        service.embedding_dim = 1024

        embedding = service.generate_embedding_sync("test text")

        # Random fallback vector must match the configured dim, not a hardcoded 1536.
        assert len(embedding) == 1024
        assert all(isinstance(v, float) for v in embedding)


class TestLanceDBOperations:
    """Test LanceDB database operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_get_db_connection(self, temp_db_path):
        """Test database connection."""
        db = get_db(temp_db_path)
        assert db is not None
    
    def test_get_or_create_vcf_table(self, temp_db_path):
        """Test VCF table creation."""
        db = get_db(temp_db_path)
        table = get_or_create_vcf_table(db, "test_variants")
        
        assert table is not None
        assert "test_variants" in db.table_names()
    
    @patch('src.vcf_agent.lancedb_integration.VariantEmbeddingService')
    def test_create_vcf_variant_record(self, mock_embedding_service):
        """Test VCF variant record creation."""
        # Mock the embedding service
        mock_service = Mock()
        mock_service.generate_variant_description.return_value = "Test description"
        mock_service.generate_embedding_sync.return_value = [0.1] * 1536
        mock_embedding_service.return_value = mock_service
        
        variant_data = {
            "chromosome": "1",
            "position": 123456,
            "reference": "A",
            "alternate": "G",
            "sample_id": "sample_001",
            "quality_score": 99.5,
            "filter_status": "PASS"
        }
        
        record = create_vcf_variant_record(variant_data, mock_service, "Test analysis")
        
        assert record["variant_id"] == "1-123456-A-G"
        assert record["chromosome"] == "1"
        assert record["position"] == 123456
        assert record["reference"] == "A"
        assert record["alternate"] == "G"
        assert record["variant_description"] == "Test description"
        assert len(record["variant_vector"]) == 1536
        assert record["analysis_summary"] == "Test analysis"
        assert record["sample_id"] == "sample_001"
        assert record["quality_score"] == 99.5
        assert record["filter_status"] == "PASS"
        assert isinstance(record["created_at"], datetime)
        assert isinstance(record["updated_at"], datetime)


class TestBatchOperations:
    """Test batch operations for performance."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.vcf_agent.lancedb_integration.VariantEmbeddingService')
    def test_batch_add_vcf_variants_empty(self, mock_embedding_service, temp_db_path):
        """Test batch addition with empty data."""
        db = get_db(temp_db_path)
        table = get_or_create_vcf_table(db, "test_variants")
        
        result = batch_add_vcf_variants(table, [])
        assert result == 0
    
    @patch('src.vcf_agent.lancedb_integration.VariantEmbeddingService')
    def test_batch_add_vcf_variants_small_batch(self, mock_embedding_service, temp_db_path):
        """Test batch addition with small dataset."""
        # Mock the embedding service
        mock_service = Mock()
        mock_service.generate_variant_description.return_value = "Test description"
        mock_service.generate_embedding_sync.return_value = [0.1] * 1536
        mock_embedding_service.return_value = mock_service
        
        db = get_db(temp_db_path)
        table = get_or_create_vcf_table(db, "test_variants")
        
        variants_data = [
            {
                "chromosome": "1",
                "position": 123456,
                "reference": "A",
                "alternate": "G",
                "sample_id": "sample_001"
            },
            {
                "chromosome": "2",
                "position": 789012,
                "reference": "T",
                "alternate": "C",
                "sample_id": "sample_002"
            }
        ]
        
        result = batch_add_vcf_variants(table, variants_data, mock_service, batch_size=1, max_workers=1)
        assert result == 2


class TestSearchOperations:
    """Test search and query operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.vcf_agent.lancedb_integration.VariantEmbeddingService')
    def test_hybrid_search_variants(self, mock_embedding_service, temp_db_path):
        """Test hybrid search functionality."""
        # Mock the embedding service
        mock_service = Mock()
        mock_service.generate_embedding_sync.return_value = [0.1] * 1536
        mock_embedding_service.return_value = mock_service
        
        db = get_db(temp_db_path)
        table = get_or_create_vcf_table(db, "test_variants")
        
        # Add some test data first
        test_variants = [
            {
                "chromosome": "1",
                "position": 123456,
                "reference": "A",
                "alternate": "G",
                "sample_id": "sample_001",
                "gene_symbol": "BRCA1"
            }
        ]
        
        batch_add_vcf_variants(table, test_variants, mock_service)
        
        # Test search
        results = hybrid_search_variants(
            table,
            "pathogenic variant in BRCA1",
            mock_service,
            metadata_filter="chromosome = '1'",
            limit=5
        )
        
        assert isinstance(results, pd.DataFrame)


class TestStatistics:
    """Test statistics and analytics functions."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_get_variant_statistics_empty_table(self, temp_db_path):
        """Test statistics on empty table."""
        db = get_db(temp_db_path)
        table = get_or_create_vcf_table(db, "test_variants")
        
        stats = get_variant_statistics(table)
        
        assert stats["total_variants"] == 0
        assert stats["sample_size"] == 0
        assert stats["chromosomes"] == {}


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.vcf_agent.lancedb_integration.VariantEmbeddingService')
    def test_complete_workflow(self, mock_embedding_service, temp_db_path):
        """Test complete workflow from data ingestion to search."""
        # Mock the embedding service
        mock_service = Mock()
        mock_service.generate_variant_description.return_value = "Test variant description"
        mock_service.generate_embedding_sync.return_value = [0.1] * 1536
        mock_embedding_service.return_value = mock_service
        
        # 1. Create database and table
        db = get_db(temp_db_path)
        table = get_or_create_vcf_table(db, "integration_test")
        
        # 2. Prepare test data
        variants_data = [
            {
                "chromosome": "1",
                "position": 123456,
                "reference": "A",
                "alternate": "G",
                "sample_id": "sample_001",
                "quality_score": 99.5,
                "filter_status": "PASS",
                "gene_symbol": "BRCA1",
                "clinical_significance": "Pathogenic"
            },
            {
                "chromosome": "17",
                "position": 789012,
                "reference": "T",
                "alternate": "C",
                "sample_id": "sample_002",
                "quality_score": 95.0,
                "filter_status": "PASS",
                "gene_symbol": "TP53",
                "clinical_significance": "Likely pathogenic"
            }
        ]
        
        # 3. Batch insert variants
        inserted_count = batch_add_vcf_variants(table, variants_data, mock_service)
        assert inserted_count == 2
        
        # 4. Get statistics
        stats = get_variant_statistics(table)
        assert stats["total_variants"] == 2
        
        # 5. Test search
        search_results = hybrid_search_variants(
            table,
            "pathogenic variant",
            mock_service,
            limit=10
        )
        assert isinstance(search_results, pd.DataFrame)
        
        print("✅ Complete LanceDB integration workflow test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 