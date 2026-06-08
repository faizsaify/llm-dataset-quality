"""
Tests for AI Training Data Factory.
"""

import pytest
from ai_training_data_factory.models import Document, Batch
from ai_training_data_factory.cleaning.normalizer import TextNormalizer
from ai_training_data_factory.quality.metrics import QualityMetrics
from ai_training_data_factory.deduplication.exact_dedup import ExactDeduplicator


class TestDocument:
    """Tests for Document model."""
    
    def test_document_creation(self):
        doc = Document(
            content="Test content",
            url="https://example.com",
            title="Test"
        )
        assert doc.content == "Test content"
        assert doc.url == "https://example.com"
        assert doc.doc_id is not None
        assert doc.content_hash is not None
    
    def test_document_to_dict(self):
        doc = Document(content="Test", title="Test Doc")
        d = doc.to_dict()
        assert d["content"] == "Test"
        assert d["title"] == "Test Doc"
    
    def test_document_from_dict(self):
        data = {
            "content": "Test",
            "title": "Test Doc",
            "doc_id": "123",
            "url": None
        }
        doc = Document.from_dict(data)
        assert doc.content == "Test"
        assert doc.doc_id == "123"


class TestBatch:
    """Tests for Batch model."""
    
    def test_batch_creation(self):
        docs = [Document(content=f"Doc {i}") for i in range(5)]
        batch = Batch(documents=docs)
        assert len(batch) == 5
    
    def test_batch_filter_valid(self):
        docs = [
            Document(content="Valid", is_duplicate=False, passed_quality_filter=True),
            Document(content="Duplicate", is_duplicate=True, passed_quality_filter=True),
            Document(content="Low quality", is_duplicate=False, passed_quality_filter=False),
        ]
        batch = Batch(documents=docs)
        valid = batch.filter_valid()
        assert len(valid) == 1


class TestTextNormalizer:
    """Tests for TextNormalizer."""
    
    def test_normalize_unicode(self):
        normalizer = TextNormalizer()
        text = "café"
        result = normalizer.normalize_unicode(text)
        assert "é" in result or "e" in result  # Depends on normalization form
    
    def test_normalize_whitespace(self):
        normalizer = TextNormalizer()
        text = "Hello    world\n\n\n\nTest"
        result = normalizer.normalize_whitespace(text)
        assert "    " not in result
        assert "\n\n\n" not in result
    
    def test_remove_control_characters(self):
        normalizer = TextNormalizer()
        text = "Hello\x00World\x1f"
        result = normalizer.remove_control_characters(text)
        assert "\x00" not in result
        assert "\x1f" not in result


class TestQualityMetrics:
    """Tests for QualityMetrics."""
    
    def test_count_words(self):
        metrics = QualityMetrics()
        text = "The quick brown fox jumps"
        assert metrics.count_words(text) == 5
    
    def test_count_sentences(self):
        metrics = QualityMetrics()
        text = "First sentence. Second sentence! Third sentence?"
        assert metrics.count_sentences(text) == 3
    
    def test_compute_readability(self):
        metrics = QualityMetrics()
        # Simple text should have high readability
        simple_text = "The cat sat on the mat. It was a sunny day."
        score = metrics.compute_readability(simple_text)
        assert 0 <= score <= 100
    
    def test_compute_information_density(self):
        metrics = QualityMetrics()
        text = "Machine learning algorithms process data to make predictions."
        density = metrics.compute_information_density(text)
        assert 0 <= density <= 1


class TestExactDeduplicator:
    """Tests for ExactDeduplicator."""
    
    def test_compute_hash(self):
        dedup = ExactDeduplicator()
        hash1 = dedup.compute_hash("Test content")
        hash2 = dedup.compute_hash("Test content")
        assert hash1 == hash2
    
    def test_is_duplicate(self):
        dedup = ExactDeduplicator()
        assert not dedup.is_duplicate("First document")
        assert not dedup.is_duplicate("Second document")
        assert dedup.is_duplicate("First document")  # Duplicate
    
    def test_different_content_different_hash(self):
        dedup = ExactDeduplicator()
        hash1 = dedup.compute_hash("Content A")
        hash2 = dedup.compute_hash("Content B")
        assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
