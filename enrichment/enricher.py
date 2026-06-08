"""
Metadata enrichment for documents.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from ..models import Document
from ..config import EnrichmentConfig
from ..quality.metrics import QualityMetrics

logger = logging.getLogger(__name__)


class MetadataEnricher:
    """
    Enrich documents with automatically generated metadata.
    
    Extracts:
    - Language
    - Domain/Source
    - Topics
    - Quality scores
    - Readability metrics
    - Content statistics
    """
    
    def __init__(self, config: Optional[EnrichmentConfig] = None):
        self.config = config or EnrichmentConfig()
        self.metrics = QualityMetrics()
        
        # Initialize language detector
        self._langdetect_available = False
        try:
            from langdetect import detect, detect_langs
            self._langdetect_available = True
        except ImportError:
            logger.warning("langdetect not installed. Language detection disabled.")
        
        # Topic classifier
        self._topic_classifier = None
        if self.config.extract_topics:
            self._init_topic_classifier()
    
    def _init_topic_classifier(self):
        """Initialize topic classifier."""
        if self.config.topic_model == "zero-shot":
            try:
                from transformers import pipeline
                self._topic_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info("Zero-shot topic classifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize topic classifier: {e}")
    
    def enrich(self, document: Document) -> Document:
        """
        Enrich a document with metadata.
        
        Args:
            document: Document to enrich
            
        Returns:
            Enriched document
        """
        if not document.content:
            return document
        
        metadata = document.metadata or {}
        
        # Extract language
        if self.config.extract_language and self._langdetect_available:
            lang, confidence = self._detect_language(document.content)
            document.language = lang
            document.language_confidence = confidence
            metadata['language'] = lang
            metadata['language_confidence'] = confidence
        
        # Extract domain from URL
        if self.config.extract_domain and document.url:
            domain = self._extract_domain(document.url)
            metadata['domain'] = domain
            metadata['source_type'] = self._classify_source(domain)
        
        # Extract topics
        if self.config.extract_topics and self._topic_classifier:
            topics = self._extract_topics(document.content)
            metadata['topics'] = topics
            if topics:
                metadata['primary_topic'] = topics[0]
        
        # Calculate quality score
        if self.config.calculate_quality_score:
            quality_score = self._compute_quality_score(document)
            document.quality_score = quality_score
            metadata['quality_score'] = quality_score
        
        # Calculate readability
        if self.config.calculate_readability:
            readability = self.metrics.compute_readability(document.content)
            document.readability_score = readability
            metadata['readability'] = readability
            metadata['readability_level'] = self._get_readability_level(readability)
        
        # Content statistics
        metadata['word_count'] = self.metrics.count_words(document.content)
        metadata['sentence_count'] = self.metrics.count_sentences(document.content)
        metadata['character_count'] = len(document.content)
        
        # Information density
        metadata['information_density'] = self.metrics.compute_information_density(document.content)
        
        # Repetition score
        metadata['repetition_score'] = self.metrics.compute_repetition_score(document.content)
        
        # Update document
        document.metadata = metadata
        document.processed_at = datetime.now()
        
        return document
    
    def enrich_batch(self, documents: List[Document]) -> List[Document]:
        """Enrich multiple documents."""
        return [self.enrich(doc) for doc in documents]
    
    def _detect_language(self, text: str) -> tuple:
        """Detect language of text."""
        try:
            from langdetect import detect_langs
            
            sample = text[:5000]
            results = detect_langs(sample)
            
            if results:
                return results[0].lang, results[0].prob
            
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        return None, 0.0
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
            
        except Exception:
            return ""
    
    def _classify_source(self, domain: str) -> str:
        """Classify source type from domain."""
        domain_lower = domain.lower()
        
        # Academic/Research
        if any(x in domain_lower for x in ['arxiv', 'springer', 'ieee', 'acm', 'nature', 'science']):
            return 'academic'
        
        # Wikipedia/Encyclopedia
        if 'wikipedia' in domain_lower or 'wiki' in domain_lower:
            return 'encyclopedia'
        
        # Documentation
        if any(x in domain_lower for x in ['docs.', 'documentation', 'readthedocs']):
            return 'documentation'
        
        # Forums/Q&A
        if any(x in domain_lower for x in ['stackoverflow', 'reddit', 'quora', 'forum']):
            return 'forum'
        
        # Blogs
        if any(x in domain_lower for x in ['blog', 'medium.com', 'dev.to', 'hashnode']):
            return 'blog'
        
        # GitHub
        if 'github' in domain_lower:
            return 'code_repository'
        
        # News
        if any(x in domain_lower for x in ['news', 'bbc', 'cnn', 'nytimes', 'guardian']):
            return 'news'
        
        return 'general'
    
    def _extract_topics(self, text: str, candidate_labels: Optional[List[str]] = None) -> List[str]:
        """Extract topics using zero-shot classification."""
        if not self._topic_classifier:
            return []
        
        # Default candidate labels
        if candidate_labels is None:
            candidate_labels = [
                'technology', 'science', 'programming', 'machine_learning',
                'business', 'health', 'education', 'entertainment',
                'politics', 'sports', 'travel', 'food'
            ]
        
        try:
            # Use a sample for efficiency
            sample = text[:1000]
            
            result = self._topic_classifier(
                sample,
                candidate_labels=candidate_labels,
                multi_label=True
            )
            
            # Return topics with score > 0.3
            topics = [
                label for label, score in zip(result['labels'], result['scores'])
                if score > 0.3
            ]
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.debug(f"Topic extraction failed: {e}")
            return []
    
    def _compute_quality_score(self, document: Document) -> float:
        """Compute overall quality score."""
        scores = []
        
        # Readability (normalized)
        readability = self.metrics.compute_readability(document.content)
        scores.append(min(1.0, readability / 100))
        
        # Grammar score
        grammar = self.metrics.compute_grammar_score(document.content)
        scores.append(grammar)
        
        # Information density
        density = self.metrics.compute_information_density(document.content)
        scores.append(density)
        
        # Low repetition (inverted)
        repetition = self.metrics.compute_repetition_score(document.content)
        scores.append(1 - repetition)
        
        # Word count (favor longer documents up to a point)
        word_count = self.metrics.count_words(document.content)
        word_score = min(1.0, word_count / 500)  # Optimal around 500 words
        scores.append(word_score)
        
        # Average all scores
        return sum(scores) / len(scores) if scores else 0.5
    
    def _get_readability_level(self, score: float) -> str:
        """Convert readability score to level."""
        if score >= 90:
            return 'very_easy'
        elif score >= 80:
            return 'easy'
        elif score >= 70:
            return 'fairly_easy'
        elif score >= 60:
            return 'standard'
        elif score >= 50:
            return 'fairly_difficult'
        elif score >= 30:
            return 'difficult'
        else:
            return 'very_difficult'


class KeywordExtractor:
    """
    Extract keywords from text using TF-IDF or RAKE.
    """
    
    def __init__(self, method: str = 'tfidf', top_k: int = 10):
        self.method = method
        self.top_k = top_k
    
    def extract(self, text: str) -> List[str]:
        """Extract keywords from text."""
        if self.method == 'tfidf':
            return self._extract_tfidf(text)
        elif self.method == 'frequency':
            return self._extract_frequency(text)
        else:
            return self._extract_frequency(text)
    
    def _extract_frequency(self, text: str) -> List[str]:
        """Extract keywords by frequency."""
        # Simple word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                    'can', 'her', 'was', 'one', 'our', 'out', 'has', 'have',
                    'been', 'were', 'said', 'each', 'she', 'which', 'their',
                    'will', 'way', 'about', 'many', 'then', 'them', 'would',
                    'like', 'into', 'this', 'that', 'with', 'from'}
        
        words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_words[:self.top_k]]
    
    def _extract_tfidf(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=self.top_k, stop_words='english')
            vectorizer.fit_transform([text])
            
            return list(vectorizer.get_feature_names_out())
            
        except ImportError:
            return self._extract_frequency(text)
