"""
Main deduplication pipeline combining multiple strategies.
"""

import logging
from typing import List, Optional, Set
from datetime import datetime

from ..models import Document, Batch
from ..config import DeduplicationConfig
from .exact_dedup import ExactDeduplicator
from .minhash_dedup import MinHashDeduplicator, DATASKETCH_AVAILABLE
from .semantic_dedup import SemanticDeduplicator, FAISS_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Main deduplication pipeline that combines:
    - Exact deduplication (SHA256 hash)
    - Near-duplicate detection (MinHash LSH)
    - Semantic deduplication (embedding similarity)
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        
        # Initialize deduplicators
        self.exact_dedup = ExactDeduplicator(
            algorithm=self.config.hash_algorithm
        ) if self.config.use_exact_dedup else None
        
        # Only initialize MinHash if datasketch is available
        self.minhash_dedup = None
        if self.config.use_minhash and DATASKETCH_AVAILABLE:
            self.minhash_dedup = MinHashDeduplicator(
                num_perm=self.config.minhash_num_perm,
                threshold=self.config.minhash_threshold
            )
        elif self.config.use_minhash:
            logger.warning("MinHash deduplication disabled: datasketch not installed")
        
        # Only initialize semantic dedup if dependencies are available
        self.semantic_dedup = None
        if self.config.use_semantic_dedup and FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.semantic_dedup = SemanticDeduplicator(
                model_name=self.config.embedding_model,
                threshold=self.config.semantic_threshold,
                batch_size=self.config.batch_size
            )
        elif self.config.use_semantic_dedup:
            logger.warning("Semantic deduplication disabled: FAISS or sentence-transformers not installed")
        
        # Track seen documents
        self.seen_hashes: Set[str] = set()
        self.total_duplicates = 0
    
    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents using configured strategies.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            List of unique documents
        """
        if not documents:
            return []
        
        original_count = len(documents)
        
        # Stage 1: Exact deduplication
        if self.exact_dedup:
            documents = self._exact_deduplicate(documents)
            logger.info(f"After exact dedup: {len(documents)}/{original_count} documents")
        
        # Stage 2: Near-duplicate detection
        if self.minhash_dedup and len(documents) > 1:
            documents = self._minhash_deduplicate(documents)
            logger.info(f"After MinHash dedup: {len(documents)}/{original_count} documents")
        
        # Stage 3: Semantic deduplication
        if self.semantic_dedup and len(documents) > 1:
            documents = self._semantic_deduplicate(documents)
            logger.info(f"After semantic dedup: {len(documents)}/{original_count} documents")
        
        self.total_duplicates += (original_count - len(documents))
        logger.info(f"Deduplication complete: {len(documents)}/{original_count} documents retained")
        
        return documents
    
    def deduplicate_batch(self, batch: Batch) -> Batch:
        """Deduplicate a batch of documents."""
        unique_docs = self.deduplicate(batch.documents)
        return Batch(documents=unique_docs)
    
    def _exact_deduplicate(self, documents: List[Document]) -> List[Document]:
        """Remove exact duplicates based on content hash."""
        unique_docs = []
        
        for doc in documents:
            if not doc.content:
                unique_docs.append(doc)
                continue
            
            content_hash = self.exact_dedup.compute_hash(doc.content)
            
            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                doc.content_hash = content_hash
                unique_docs.append(doc)
            else:
                doc.is_duplicate = True
                doc.duplicate_of = f"hash:{content_hash[:16]}"
                logger.debug(f"Exact duplicate found: {doc.doc_id}")
        
        return unique_docs
    
    def _minhash_deduplicate(self, documents: List[Document]) -> List[Document]:
        """Remove near-duplicates using MinHash LSH."""
        # Build index
        self.minhash_dedup.build_index(documents)
        
        # Find duplicates
        duplicate_ids = set()
        
        for i, doc in enumerate(documents):
            if doc.doc_id in duplicate_ids:
                continue
            
            # Find similar documents
            similar = self.minhash_dedup.find_similar(doc)
            
            for similar_doc in similar:
                if similar_doc.doc_id != doc.doc_id:
                    duplicate_ids.add(similar_doc.doc_id)
                    similar_doc.is_duplicate = True
                    similar_doc.duplicate_of = doc.doc_id
                    logger.debug(f"Near-duplicate found: {similar_doc.doc_id} -> {doc.doc_id}")
        
        # Filter duplicates
        unique_docs = [doc for doc in documents if doc.doc_id not in duplicate_ids]
        
        return unique_docs
    
    def _semantic_deduplicate(self, documents: List[Document]) -> List[Document]:
        """Remove semantic duplicates using embeddings."""
        # Build index
        self.semantic_dedup.build_index(documents)
        
        # Find duplicates
        duplicate_ids = set()
        
        for doc in documents:
            if doc.doc_id in duplicate_ids:
                continue
            
            # Find similar documents
            similar = self.semantic_dedup.find_similar(doc, top_k=10)
            
            for similar_doc, similarity in similar:
                if similar_doc.doc_id != doc.doc_id and similarity >= self.config.semantic_threshold:
                    duplicate_ids.add(similar_doc.doc_id)
                    similar_doc.is_duplicate = True
                    similar_doc.duplicate_of = doc.doc_id
                    logger.debug(f"Semantic duplicate found: {similar_doc.doc_id} -> {doc.doc_id} (sim={similarity:.3f})")
        
        # Filter duplicates
        unique_docs = [doc for doc in documents if doc.doc_id not in duplicate_ids]
        
        return unique_docs
    
    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "total_duplicates_removed": self.total_duplicates,
            "unique_hashes": len(self.seen_hashes),
        }
    
    def reset(self):
        """Reset deduplication state."""
        self.seen_hashes.clear()
        self.total_duplicates = 0
        
        if self.minhash_dedup:
            self.minhash_dedup.reset()
        
        if self.semantic_dedup:
            self.semantic_dedup.reset()
