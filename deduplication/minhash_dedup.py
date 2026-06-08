"""
Near-duplicate detection using MinHash and Locality Sensitive Hashing (LSH).
"""

import logging
import re
from typing import List, Optional, Set, Tuple, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    MinHash = None  # type: ignore
    MinHashLSH = None  # type: ignore
    logger.warning("datasketch not installed. MinHash deduplication will be limited.")


class MinHashDeduplicator:
    """
    Near-duplicate detection using MinHash LSH.
    
    MinHash creates a compact signature of document content,
    and LSH enables efficient similarity search.
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8,
                 ngram_size: int = 5):
        """
        Initialize MinHash deduplicator.
        
        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold for duplicates
            ngram_size: Size of character n-grams
        """
        if not DATASKETCH_AVAILABLE:
            raise ImportError("datasketch not installed. Install with: pip install datasketch")
        
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        
        # LSH index
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        # Document ID to MinHash mapping
        self.minhashes: Dict[str, MinHash] = {}
        
        # Document ID to Document mapping
        self.documents: Dict[str, any] = {}
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """
        Get character n-grams from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of n-grams
        """
        # Normalize text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Generate n-grams
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        
        return ngrams
    
    def _get_word_shingles(self, text: str, k: int = 3) -> Set[str]:
        """
        Get word-level shingles from text.
        
        Args:
            text: Input text
            k: Shingle size (number of words)
            
        Returns:
            Set of shingles
        """
        words = text.lower().split()
        shingles = set()
        
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i + k])
            shingles.add(shingle)
        
        return shingles
    
    def compute_minhash(self, text: str, use_words: bool = False) -> Any:
        """
        Compute MinHash signature for text.
        
        Args:
            text: Input text
            use_words: Use word shingles instead of character n-grams
            
        Returns:
            MinHash object
        """
        mh = MinHash(num_perm=self.num_perm)
        
        if use_words:
            shingles = self._get_word_shingles(text)
        else:
            shingles = self._get_ngrams(text)
        
        for shingle in shingles:
            mh.update(shingle.encode('utf-8'))
        
        return mh
    
    def build_index(self, documents: List) -> None:
        """
        Build LSH index from documents.
        
        Args:
            documents: List of Document objects
        """
        for doc in documents:
            if not doc.content:
                continue
            
            try:
                mh = self.compute_minhash(doc.content)
                self.minhashes[doc.doc_id] = mh
                self.documents[doc.doc_id] = doc
                self.lsh.insert(doc.doc_id, mh)
            except Exception as e:
                logger.error(f"Error computing MinHash for {doc.doc_id}: {e}")
    
    def find_similar(self, document, return_similarity: bool = False) -> List:
        """
        Find similar documents to the given document.
        
        Args:
            document: Document to find duplicates for
            return_similarity: Whether to return similarity scores
            
        Returns:
            List of similar documents (or tuples with similarity if return_similarity=True)
        """
        if not document.content:
            return []
        
        mh = self.compute_minhash(document.content)
        
        # Query LSH
        similar_ids = self.lsh.query(mh)
        
        results = []
        for doc_id in similar_ids:
            if doc_id == document.doc_id:
                continue
            
            similar_doc = self.documents.get(doc_id)
            if similar_doc:
                if return_similarity:
                    # Compute exact Jaccard similarity
                    similarity = mh.jaccard(self.minhashes[doc_id])
                    results.append((similar_doc, similarity))
                else:
                    results.append(similar_doc)
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity (0-1)
        """
        mh1 = self.compute_minhash(text1)
        mh2 = self.compute_minhash(text2)
        return mh1.jaccard(mh2)
    
    def reset(self):
        """Reset the LSH index."""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes.clear()
        self.documents.clear()
    
    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.minhashes)


class SimHashDeduplicator:
    """
    Alternative near-duplicate detection using SimHash.
    SimHash is better for very large documents.
    """
    
    def __init__(self, hash_bits: int = 64, threshold: int = 3):
        """
        Initialize SimHash deduplicator.
        
        Args:
            hash_bits: Number of bits in hash
            threshold: Maximum Hamming distance for duplicates
        """
        self.hash_bits = hash_bits
        self.threshold = threshold
        self.hashes: Dict[str, int] = {}
        self.documents: Dict[str, any] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\w+', text.lower())
    
    def compute_simhash(self, text: str) -> int:
        """
        Compute SimHash of text.
        
        Args:
            text: Input text
            
        Returns:
            SimHash value as integer
        """
        tokens = self._tokenize(text)
        
        # Initialize bit vector
        v = [0] * self.hash_bits
        
        for token in tokens:
            # Hash the token
            token_hash = hash(token) & ((1 << self.hash_bits) - 1)
            
            # Update bit vector
            for i in range(self.hash_bits):
                if (token_hash >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Convert to binary hash
        simhash = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1
        return distance
    
    def is_duplicate(self, text: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is a duplicate.
        
        Args:
            text: Text to check
            doc_id: Document ID
            
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
        """
        new_hash = self.compute_simhash(text)
        
        for existing_id, existing_hash in self.hashes.items():
            if self.hamming_distance(new_hash, existing_hash) <= self.threshold:
                return True, existing_id
        
        self.hashes[doc_id] = new_hash
        return False, None
    
    def reset(self):
        """Reset the hash index."""
        self.hashes.clear()
        self.documents.clear()
