"""
Toxicity detection for text content.
"""

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False


class ToxicityDetector:
    """
    Detect toxic content using the Detoxify library.
    
    Detects:
    - Toxicity
    - Severe toxicity
    - Obscene content
    - Threats
    - Insults
    - Identity attacks
    """
    
    def __init__(self, model_type: str = "original"):
        """
        Initialize toxicity detector.
        
        Args:
            model_type: Detoxify model type ('original', 'unbiased', 'multilingual')
        """
        self.model_type = model_type
        self.model = None
        self._initialized = False
    
    def _init_model(self):
        """Initialize the model lazily."""
        if self._initialized:
            return
        
        if not DETOXIFY_AVAILABLE:
            logger.warning("Detoxify not installed. Toxicity detection disabled.")
            self._initialized = True
            return
        
        try:
            logger.info(f"Loading Detoxify model: {self.model_type}")
            self.model = Detoxify(self.model_type)
            self._initialized = True
            logger.info("Detoxify model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Detoxify model: {e}")
            self._initialized = True
    
    def detect(self, text: str) -> float:
        """
        Detect toxicity in text.
        
        Args:
            text: Input text
            
        Returns:
            Toxicity score (0-1)
        """
        self._init_model()
        
        if self.model is None:
            return 0.0
        
        try:
            # Truncate very long texts
            if len(text) > 5000:
                text = text[:5000]
            
            results = self.model.predict(text)
            
            # Return the maximum toxicity score
            return max(results.values())
            
        except Exception as e:
            logger.error(f"Error detecting toxicity: {e}")
            return 0.0
    
    def detect_detailed(self, text: str) -> Dict[str, float]:
        """
        Get detailed toxicity scores.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of toxicity scores by category
        """
        self._init_model()
        
        if self.model is None:
            return {}
        
        try:
            if len(text) > 5000:
                text = text[:5000]
            
            return self.model.predict(text)
            
        except Exception as e:
            logger.error(f"Error detecting toxicity: {e}")
            return {}
    
    def detect_batch(self, texts: List[str]) -> List[float]:
        """
        Detect toxicity in multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of toxicity scores
        """
        return [self.detect(text) for text in texts]


class SimpleToxicityDetector:
    """
    Simple rule-based toxicity detector.
    Use when Detoxify is not available.
    """
    
    def __init__(self):
        # Basic word lists (very simplified)
        self.toxic_patterns = [
            # Add appropriate patterns for production use
        ]
    
    def detect(self, text: str) -> float:
        """
        Detect toxicity using simple rules.
        
        Args:
            text: Input text
            
        Returns:
            Toxicity score (0-1)
        """
        text_lower = text.lower()
        
        matches = 0
        for pattern in self.toxic_patterns:
            if pattern in text_lower:
                matches += 1
        
        # Normalize score
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        return min(1.0, matches / (word_count * 0.1))
