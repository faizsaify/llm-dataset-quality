"""
Quality filtering for documents.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models import Document
from ..config import QualityConfig
from .metrics import QualityMetrics
from .toxicity import ToxicityDetector
from .pii import PIIDetector

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Filter documents based on quality metrics.
    
    Evaluates:
    - Readability (Flesch-Kincaid, etc.)
    - Grammar quality
    - Information density
    - Toxicity
    - PII presence
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        
        # Initialize components
        self.metrics = QualityMetrics()
        
        self.toxicity_detector = ToxicityDetector() if self.config.check_toxicity else None
        self.pii_detector = PIIDetector() if self.config.check_pii else None
        
        # Statistics
        self.total_processed = 0
        self.total_filtered = 0
    
    def filter(self, document: Document) -> Document:
        """
        Evaluate and filter a document based on quality.
        
        Args:
            document: Document to filter
            
        Returns:
            Document with quality scores and filter decision
        """
        if not document.content:
            document.passed_quality_filter = False
            return document
        
        self.total_processed += 1
        scores: Dict[str, float] = {}
        filter_reasons: List[str] = []
        
        # Word count check
        word_count = self.metrics.count_words(document.content)
        if word_count < self.config.min_word_count:
            filter_reasons.append(f"word_count:{word_count}<{self.config.min_word_count}")
        if word_count > self.config.max_word_count:
            filter_reasons.append(f"word_count:{word_count}>{self.config.max_word_count}")
        
        # Readability check
        if self.config.check_readability:
            readability = self.metrics.compute_readability(document.content)
            document.readability_score = readability
            scores['readability'] = readability
            
            if readability < self.config.min_readability_score:
                filter_reasons.append(f"readability:{readability:.1f}<{self.config.min_readability_score}")
        
        # Grammar check (simplified)
        if self.config.check_grammar:
            grammar_score = self.metrics.compute_grammar_score(document.content)
            scores['grammar'] = grammar_score
        
        # Toxicity check
        if self.toxicity_detector:
            toxicity_score = self.toxicity_detector.detect(document.content)
            document.toxicity_score = toxicity_score
            scores['toxicity'] = toxicity_score
            
            if toxicity_score > self.config.max_toxicity_score:
                filter_reasons.append(f"toxicity:{toxicity_score:.2f}>{self.config.max_toxicity_score}")
        
        # PII check
        if self.pii_detector:
            has_pii, pii_types = self.pii_detector.detect(document.content)
            scores['has_pii'] = 1.0 if has_pii else 0.0
            
            if has_pii:
                filter_reasons.append(f"pii_detected:{','.join(pii_types)}")
        
        # Compute overall quality score
        quality_score = self._compute_overall_score(scores)
        document.quality_score = quality_score
        
        if quality_score < self.config.min_quality_score:
            filter_reasons.append(f"quality:{quality_score:.2f}<{self.config.min_quality_score}")
        
        # Final filter decision
        document.passed_quality_filter = len(filter_reasons) == 0
        
        if not document.passed_quality_filter:
            self.total_filtered += 1
            logger.debug(f"Document {document.doc_id} filtered: {filter_reasons}")
        
        # Store scores in metadata
        document.metadata['quality_scores'] = scores
        document.metadata['filter_reasons'] = filter_reasons
        
        return document
    
    def filter_batch(self, documents: List[Document]) -> List[Document]:
        """Filter multiple documents."""
        return [self.filter(doc) for doc in documents]
    
    def _compute_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Compute overall quality score from individual scores.
        
        Args:
            scores: Dictionary of individual scores
            
        Returns:
            Overall quality score (0-1)
        """
        if not scores:
            return 0.5
        
        weights = {
            'readability': 0.3,
            'grammar': 0.2,
            'toxicity': 0.3,
            'has_pii': 0.2,
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for metric, weight in weights.items():
            if metric in scores:
                value = scores[metric]
                
                # Normalize scores to 0-1 range
                if metric == 'readability':
                    value = min(100, max(0, value)) / 100
                elif metric == 'toxicity':
                    value = 1 - value  # Invert (lower toxicity = better)
                elif metric == 'has_pii':
                    value = 1 - value  # Invert (no PII = better)
                
                weighted_sum += value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            'total_processed': self.total_processed,
            'total_filtered': self.total_filtered,
            'pass_rate': (self.total_processed - self.total_filtered) / max(1, self.total_processed)
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_processed = 0
        self.total_filtered = 0
