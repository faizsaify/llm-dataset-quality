"""
Semantic deduplication using embeddings and vector similarity.
"""

import logging
from typing import List, Optional, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Semantic deduplication will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Semantic deduplication disabled.")


class SemanticDeduplicator:
    """
    Semantic deduplication using sentence embeddings and FAISS.
    
    Uses embedding models like:
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-small-en-v1.5
    - intfloat/e5-small-v2
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 threshold: float = 0.95, batch_size: int = 32,
                 max_seq_length: int = 512):
        """
        Initialize semantic deduplicator.
        
        Args:
            model_name: Embedding model name
            threshold: Cosine similarity threshold for duplicates
            batch_size: Batch size for embedding computation
            max_seq_length: Maximum sequence length for embedding
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        # Initialize embedding model
        self.model = None
        self.dimension = None
        
        # FAISS index
        self.index = None
        
        # Document mappings
        self.doc_ids: List[str] = []
        self.documents: Dict[str, any] = {}
        self.embeddings: Optional[np.ndarray] = None
        
        self._initialized = False
    
    def _init_model(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_length
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self._initialized = True
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def _init_index(self, dimension: int):
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Using brute-force search.")
            return
        
        # Use IndexFlatIP for cosine similarity (with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        self._init_model()
        
        # Truncate if needed
        if len(text) > self.max_seq_length * 4:  # Rough character estimate
            text = text[:self.max_seq_length * 4]
        
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of embeddings
        """
        self._init_model()
        
        # Truncate texts if needed
        texts = [t[:self.max_seq_length * 4] if len(t) > self.max_seq_length * 4 else t 
                 for t in texts]
        
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings
    
    def build_index(self, documents: List) -> None:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of Document objects
        """
        self._init_model()
        
        # Filter documents with content
        valid_docs = [(doc.doc_id, doc.content, doc) for doc in documents if doc.content]
        
        if not valid_docs:
            logger.warning("No valid documents to index")
            return
        
        doc_ids, texts, docs = zip(*valid_docs)
        
        logger.info(f"Computing embeddings for {len(texts)} documents...")
        self.embeddings = self.compute_embeddings(list(texts))
        
        # Store mappings
        self.doc_ids = list(doc_ids)
        for doc_id, doc in zip(doc_ids, docs):
            self.documents[doc_id] = doc
        
        # Build FAISS index
        if FAISS_AVAILABLE:
            self._init_index(self.dimension)
            self.index.add(self.embeddings.astype(np.float32))
            logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        else:
            logger.info(f"Stored {len(self.embeddings)} embeddings for brute-force search")
    
    def find_similar(self, document, top_k: int = 10) -> List[Tuple[any, float]]:
        """
        Find similar documents.
        
        Args:
            document: Query document
            top_k: Number of similar documents to return
            
        Returns:
            List of (document, similarity) tuples
        """
        if not document.content:
            return []
        
        query_embedding = self.compute_embedding(document.content)
        
        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            similarities, indices = self.index.search(query_embedding, min(top_k + 1, self.index.ntotal))
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                
                doc_id = self.doc_ids[idx]
                if doc_id != document.doc_id:
                    doc = self.documents.get(doc_id)
                    if doc:
                        results.append((doc, float(sim)))
            
            return results[:top_k]
        
        else:
            # Brute-force search
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k + 1]
            
            results = []
            for idx in top_indices:
                doc_id = self.doc_ids[idx]
                if doc_id != document.doc_id:
                    doc = self.documents.get(doc_id)
                    if doc:
                        results.append((doc, float(similarities[idx])))
            
            return results[:top_k]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity (0-1)
        """
        emb1 = self.compute_embedding(text1)
        emb2 = self.compute_embedding(text2)
        
        return float(np.dot(emb1, emb2))
    
    def find_clusters(self, threshold: Optional[float] = None) -> List[List[str]]:
        """
        Find clusters of similar documents.
        
        Args:
            threshold: Similarity threshold (uses instance threshold if None)
            
        Returns:
            List of document ID clusters
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        threshold = threshold or self.threshold
        
        # Compute pairwise similarities
        similarities = np.dot(self.embeddings, self.embeddings.T)
        
        # Find clusters using union-find
        n = len(self.doc_ids)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union similar documents
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] >= threshold:
                    union(i, j)
        
        # Group by cluster
        clusters: Dict[int, List[str]] = {}
        for i, doc_id in enumerate(self.doc_ids):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(doc_id)
        
        # Return clusters with more than one document
        return [cluster for cluster in clusters.values() if len(cluster) > 1]
    
    def reset(self):
        """Reset the index."""
        if FAISS_AVAILABLE:
            self.index = None
        self.doc_ids = []
        self.documents.clear()
        self.embeddings = None
    
    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.doc_ids)
