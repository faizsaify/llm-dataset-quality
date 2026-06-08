"""
Data cleaning pipeline for text documents.
"""

import logging
import re
from typing import List, Optional, Tuple
from datetime import datetime

from ..models import Document
from ..config import CleaningConfig
from .normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and normalize text documents.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.normalizer = TextNormalizer()
        
        # Initialize language detector
        self._langdetect_available = False
        try:
            from langdetect import detect, detect_langs
            self._langdetect_available = True
        except ImportError:
            logger.warning("langdetect not installed. Language detection disabled.")
    
    def clean(self, document: Document) -> Document:
        """
        Clean a single document.
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaned document
        """
        if not document.content:
            return document
        
        content = document.content
        
        # Unicode normalization
        if self.config.normalize_unicode:
            content = self.normalizer.normalize_unicode(content)
        
        # Remove control characters
        content = self.normalizer.remove_control_characters(content)
        
        # HTML cleanup
        if self.config.remove_html_tags:
            content = self.normalizer.remove_html_tags(content)
        
        # Whitespace normalization
        if self.config.normalize_whitespace:
            content = self.normalizer.normalize_whitespace(content)
        
        # Clean up special characters
        content = self.normalizer.clean_special_characters(content)
        
        # Language detection
        if self.config.detect_language and self._langdetect_available:
            lang, confidence = self._detect_language(content)
            document.language = lang
            document.language_confidence = confidence
            
            # Filter by language if needed
            if (self.config.target_languages and 
                lang not in self.config.target_languages and
                confidence >= self.config.min_language_confidence):
                document.passed_quality_filter = False
                logger.debug(f"Document filtered: language {lang} not in target languages")
        
        document.content = content
        document.processed_at = datetime.now()
        
        return document
    
    def clean_batch(self, documents: List[Document]) -> List[Document]:
        """Clean multiple documents."""
        cleaned = []
        for doc in documents:
            try:
                cleaned.append(self.clean(doc))
            except Exception as e:
                logger.error(f"Error cleaning document {doc.doc_id}: {e}")
                cleaned.append(doc)
        return cleaned
    
    def _detect_language(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not self._langdetect_available:
            return None, 0.0
        
        try:
            from langdetect import detect_langs
            
            # Use a sample of text for efficiency
            sample = text[:5000]
            
            results = detect_langs(sample)
            if results:
                top_result = results[0]
                return top_result.lang, top_result.prob
            
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        return None, 0.0
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text.
        
        Args:
            text: Input text
            
        Returns:
            Text with boilerplate removed
        """
        # Common boilerplate patterns
        boilerplate_patterns = [
            r'Copyright\s*©?\s*\d{4}.*?(?:All rights reserved\.?)?',
            r'Privacy Policy\s*\|?\s*Terms of Service',
            r'Subscribe to our newsletter',
            r'Share this article',
            r'Follow us on',
            r'Click here to.*?(?:\.|$)',
            r'Read more\.\.\.',
            r'Advertisement',
            r'Loading\.\.\.',
            r'Please enable JavaScript',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract clean sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        # For production, use spaCy or NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter and clean sentences
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # Basic quality filters
            if len(sent) < 10:
                continue
            if len(sent.split()) < 3:
                continue
            if sent.count(' ') / len(sent) > 0.5:  # Too many spaces
                continue
            clean_sentences.append(sent)
        
        return clean_sentences


class TextCleaner:
    """
    Specialized text cleaning utilities.
    """
    
    @staticmethod
    def clean_code(code: str) -> str:
        """Clean code blocks."""
        # Remove excessive blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)
        # Normalize indentation
        lines = code.split('\n')
        # Find minimum indentation (excluding empty lines)
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        if min_indent != float('inf') and min_indent > 0:
            lines = [line[min_indent:] if line.strip() else line for line in lines]
        
        return '\n'.join(lines)
    
    @staticmethod
    def clean_markdown(text: str) -> str:
        """Clean markdown text while preserving structure."""
        # Normalize headers
        text = re.sub(r'^(#+)\s+', r'\1 ', text, flags=re.MULTILINE)
        # Clean up lists
        text = re.sub(r'^(\s*[-*+])\s+', r'\1 ', text, flags=re.MULTILINE)
        # Normalize links
        text = re.sub(r'\[([^\]]+)\]\s+\(([^)]+)\)', r'[\1](\2)', text)
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers from text."""
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+[0-9]{1,3}[-.\s]?[0-9]{6,12}',
        ]
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        return text
