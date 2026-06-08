"""
Exact deduplication using content hashing.
"""

import hashlib
import logging
from typing import Set, Optional

logger = logging.getLogger(__name__)


class ExactDeduplicator:
    """
    Exact deduplication using cryptographic hashing.
    Supports SHA256, MD5, and other hash algorithms.
    """
    
    def __init__(self, algorithm: str = "sha256"):
        """
        Initialize exact deduplicator.
        
        Args:
            algorithm: Hash algorithm to use (sha256, md5, sha1, etc.)
        """
        self.algorithm = algorithm
        self.seen_hashes: Set[str] = set()
        
        # Validate algorithm
        try:
            hashlib.new(algorithm)
        except ValueError:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def compute_hash(self, content: str, normalize: bool = True) -> str:
        """
        Compute hash of content.
        
        Args:
            content: Text content to hash
            normalize: Whether to normalize text before hashing
            
        Returns:
            Hex digest of hash
        """
        if normalize:
            content = self._normalize_for_hash(content)
        
        hasher = hashlib.new(self.algorithm)
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()
    
    def _normalize_for_hash(self, content: str) -> str:
        """
        Normalize content before hashing to catch near-exact duplicates.
        
        Args:
            content: Text content
            
        Returns:
            Normalized content
        """
        # Convert to lowercase
        content = content.lower()
        
        # Normalize whitespace
        content = ' '.join(content.split())
        
        # Remove punctuation (optional, makes matching more aggressive)
        # content = ''.join(c for c in content if c.isalnum() or c.isspace())
        
        return content
    
    def is_duplicate(self, content: str) -> bool:
        """
        Check if content is a duplicate.
        
        Args:
            content: Text content to check
            
        Returns:
            True if duplicate, False otherwise
        """
        content_hash = self.compute_hash(content)
        
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def add_hash(self, content_hash: str):
        """Add a hash to the seen set."""
        self.seen_hashes.add(content_hash)
    
    def reset(self):
        """Reset the seen hashes."""
        self.seen_hashes.clear()
    
    def __len__(self) -> int:
        """Return number of unique hashes seen."""
        return len(self.seen_hashes)
