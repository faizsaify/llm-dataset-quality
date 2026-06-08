"""
Personally Identifiable Information (PII) detection.
"""

import logging
import re
from typing import Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


class PIIDetector:
    """
    Detect and optionally redact PII from text.
    
    Uses Microsoft Presidio or regex-based fallback.
    
    Detects:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social Security Numbers
    - Names
    - Addresses
    - IP addresses
    """
    
    def __init__(self, use_presidio: bool = True, language: str = "en"):
        """
        Initialize PII detector.
        
        Args:
            use_presidio: Use Presidio if available
            language: Language for detection
        """
        self.language = language
        self.analyzer = None
        self.anonymizer = None
        
        if use_presidio and PRESIDIO_AVAILABLE:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio PII detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio: {e}")
        
        # Regex patterns for fallback
        self.patterns = {
            'EMAIL': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'PHONE': re.compile(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'CREDIT_CARD': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'IP_ADDRESS': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }
    
    def detect(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect if text contains PII.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (has_pii, list_of_pii_types)
        """
        if self.analyzer:
            return self._detect_presidio(text)
        else:
            return self._detect_regex(text)
    
    def _detect_presidio(self, text: str) -> Tuple[bool, List[str]]:
        """Detect PII using Presidio."""
        try:
            results = self.analyzer.analyze(text=text, language=self.language)
            
            if results:
                pii_types = list(set(r.entity_type for r in results))
                return True, pii_types
            
            return False, []
            
        except Exception as e:
            logger.error(f"Presidio detection error: {e}")
            return self._detect_regex(text)
    
    def _detect_regex(self, text: str) -> Tuple[bool, List[str]]:
        """Detect PII using regex patterns."""
        found_types = []
        
        for pii_type, pattern in self.patterns.items():
            if pattern.search(text):
                found_types.append(pii_type)
        
        return len(found_types) > 0, found_types
    
    def detect_detailed(self, text: str) -> List[Dict]:
        """
        Get detailed PII detection results.
        
        Args:
            text: Input text
            
        Returns:
            List of PII detections with location and type
        """
        if self.analyzer:
            try:
                results = self.analyzer.analyze(text=text, language=self.language)
                return [
                    {
                        'type': r.entity_type,
                        'start': r.start,
                        'end': r.end,
                        'score': r.score,
                        'text': text[r.start:r.end]
                    }
                    for r in results
                ]
            except Exception as e:
                logger.error(f"Presidio detection error: {e}")
        
        # Fallback to regex
        results = []
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                results.append({
                    'type': pii_type,
                    'start': match.start(),
                    'end': match.end(),
                    'score': 1.0,
                    'text': match.group()
                })
        
        return results
    
    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Redact PII from text.
        
        Args:
            text: Input text
            replacement: Replacement string for PII
            
        Returns:
            Text with PII redacted
        """
        if self.anonymizer and self.analyzer:
            try:
                results = self.analyzer.analyze(text=text, language=self.language)
                anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized.text
            except Exception as e:
                logger.error(f"Presidio anonymization error: {e}")
        
        # Fallback to regex replacement
        result = text
        for pii_type, pattern in self.patterns.items():
            result = pattern.sub(f"[{pii_type}]", result)
        
        return result
    
    def count_pii(self, text: str) -> int:
        """Count PII instances in text."""
        detections = self.detect_detailed(text)
        return len(detections)
