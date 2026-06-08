"""
Text normalization utilities.
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Text normalization utilities.
    """
    
    def normalize_unicode(self, text: str, form: str = 'NFKC') -> str:
        """
        Normalize unicode characters.
        
        Args:
            text: Input text
            form: Unicode normalization form (NFC, NFKC, NFD, NFKD)
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize(form, text)
    
    def remove_control_characters(self, text: str) -> str:
        """
        Remove control characters except newlines and tabs.
        
        Args:
            text: Input text
            
        Returns:
            Text without control characters
        """
        return ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\n\t\r'
        )
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without HTML tags
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace various whitespace characters with space
        text = re.sub(r'[\t\f\v]', ' ', text)
        
        # Normalize multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at line start/end
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_special_characters(self, text: str) -> str:
        """
        Clean up special characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Replace smart quotes with regular quotes
        replacements = {
            '"': '"', '"': '"',  # Smart double quotes
            ''': "'", ''': "'",  # Smart single quotes
            '–': '-', '—': '-',  # Dashes
            '…': '...',  # Ellipsis
            '•': '-',  # Bullet
            '\xa0': ' ',  # Non-breaking space
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized punctuation
        """
        # Ensure single space after punctuation
        text = re.sub(r'([.!?,;:])\s*', r'\1 ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Fix spacing issues
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        return text
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_accents(self, text: str) -> str:
        """
        Remove accents from characters.
        
        Args:
            text: Input text
            
        Returns:
            Text without accents
        """
        # Decompose unicode characters
        text = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        text = ''.join(
            char for char in text 
            if unicodedata.category(char) != 'Mn'
        )
        return text
    
    def fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed encoding
        """
        # Common mojibake patterns
        fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Â ': ' ',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
        }
        
        for bad, good in fixes.items():
            text = text.replace(bad, good)
        
        return text
    
    def clean_for_tokenization(self, text: str) -> str:
        """
        Clean text for tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text ready for tokenization
        """
        # Apply all normalizations
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.normalize_whitespace(text)
        text = self.clean_special_characters(text)
        
        return text
