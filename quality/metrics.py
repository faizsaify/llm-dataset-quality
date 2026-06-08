"""
Quality metrics computation.
"""

import logging
import re
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class QualityMetrics:
    """
    Compute various quality metrics for text.
    """
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Simple sentence counting
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def count_syllables(self, word: str) -> int:
        """
        Estimate syllable count for a word.
        Uses a simplified algorithm.
        """
        word = word.lower()
        
        # Remove non-alpha characters
        word = re.sub(r'[^a-z]', '', word)
        
        if len(word) == 0:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        # Ensure at least 1 syllable
        return max(1, count)
    
    def compute_readability(self, text: str) -> float:
        """
        Compute Flesch Reading Ease score.
        
        Higher scores indicate easier readability:
        - 90-100: Very Easy
        - 80-89: Easy
        - 70-79: Fairly Easy
        - 60-69: Standard
        - 50-59: Fairly Difficult
        - 30-49: Difficult
        - 0-29: Very Confusing
        
        Args:
            text: Input text
            
        Returns:
            Flesch Reading Ease score
        """
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0
        
        sentence_count = self.count_sentences(text)
        if sentence_count == 0:
            sentence_count = 1
        
        syllable_count = sum(self.count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        score = (
            206.835 
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (syllable_count / word_count)
        )
        
        return max(0, min(100, score))
    
    def compute_flesch_kincaid_grade(self, text: str) -> float:
        """
        Compute Flesch-Kincaid Grade Level.
        
        Returns the US grade level needed to understand the text.
        
        Args:
            text: Input text
            
        Returns:
            Grade level (0-18+)
        """
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0
        
        sentence_count = max(1, self.count_sentences(text))
        syllable_count = sum(self.count_syllables(word) for word in words)
        
        # Flesch-Kincaid Grade Level formula
        grade = (
            0.39 * (word_count / sentence_count)
            + 11.8 * (syllable_count / word_count)
            - 15.59
        )
        
        return max(0, grade)
    
    def compute_grammar_score(self, text: str) -> float:
        """
        Compute a simple grammar quality score.
        
        For production, use LanguageTool or similar.
        
        Args:
            text: Input text
            
        Returns:
            Grammar score (0-1)
        """
        score = 1.0
        issues = 0
        
        # Check for common issues
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for capitalization
            if sentence and sentence[0].islower():
                issues += 1
            
            # Check for double spaces
            if '  ' in sentence:
                issues += 1
            
            # Check for repeated words
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    issues += 1
        
        # Normalize score
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        score = max(0, 1 - (issues / (sentence_count * 2)))
        
        return score
    
    def compute_information_density(self, text: str) -> float:
        """
        Compute information density score.
        
        Higher scores indicate more information-rich content.
        
        Args:
            text: Input text
            
        Returns:
            Information density score (0-1)
        """
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return 0
        
        # Unique words ratio
        unique_ratio = len(set(words)) / word_count
        
        # Average word length (longer words often carry more information)
        avg_word_length = sum(len(w) for w in words) / word_count
        length_score = min(1, avg_word_length / 8)  # Normalize to 8 chars
        
        # Combine scores
        density = (unique_ratio * 0.6 + length_score * 0.4)
        
        return density
    
    def compute_perplexity_proxy(self, text: str) -> float:
        """
        Compute a proxy for text perplexity based on character distributions.
        
        Lower scores indicate more natural text.
        For accurate perplexity, use a language model.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity proxy score
        """
        if not text:
            return float('inf')
        
        # Character frequency distribution
        char_freq = {}
        for char in text.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        if not char_freq:
            return float('inf')
        
        total = sum(char_freq.values())
        
        # Compute entropy
        entropy = 0
        for count in char_freq.values():
            p = count / total
            entropy -= p * math.log2(p)
        
        # English text typically has entropy around 4.0-4.5
        # Return normalized score (lower = more natural)
        return abs(entropy - 4.2)
    
    def compute_repetition_score(self, text: str, ngram_size: int = 3) -> float:
        """
        Compute repetition score (lower = less repetition).
        
        Args:
            text: Input text
            ngram_size: Size of n-grams to check
            
        Returns:
            Repetition score (0-1, where 0 is no repetition)
        """
        words = text.lower().split()
        
        if len(words) < ngram_size:
            return 0
        
        # Generate n-grams
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i:i + ngram_size])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0
        
        # Count unique n-grams
        unique_ratio = len(set(ngrams)) / len(ngrams)
        
        # Repetition score (1 - unique_ratio)
        return 1 - unique_ratio
