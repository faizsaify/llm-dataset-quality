"""
Semantic search index using FAISS.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import json
import numpy as np

from ..models import Document
from ..config import SearchConfig

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Semantic search will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed.")


class SemanticIndex:
    """
    Semantic search index using FAISS and sentence transformers.
    
    Supports:
    - Document indexing
    - Similarity search
    - Corpus analysis
    - Index persistence
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        
        # Model and index
        self.model = None
        self.index = None
        self.dimension = self.config.dimension
        
        # Document storage
        self.doc_ids: List[str] = []
        self.documents: Dict[str, Document] = {}
        self.embeddings: Optional[np.ndarray] = None
        
        self._initialized = False
    
    def _init_model(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(self.config.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self._initialized = True
        logger.info(f"Model loaded. Dimension: {self.dimension}")
    
    def _init_index(self):
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available")
            return
        
        if self.config.index_type == "Flat":
            # Exact search (slower but most accurate)
            self.index = faiss.IndexFlatIP(self.dimension)
        
        elif self.config.index_type == "IVF":
            # Inverted file index (faster for large datasets)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.dimension, 
                self.config.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
        
        elif self.config.index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
        
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"FAISS index initialized: {self.config.index_type}")
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the index.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Number of documents added
        """
        self._init_model()
        
        # Filter documents with content
        valid_docs = [doc for doc in documents if doc.content]
        
        if not valid_docs:
            return 0
        
        # Compute embeddings
        texts = [doc.content for doc in valid_docs]
        logger.info(f"Computing embeddings for {len(texts)} documents...")
        
        new_embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        
        # Store documents
        for doc in valid_docs:
            self.doc_ids.append(doc.doc_id)
            self.documents[doc.doc_id] = doc
        
        # Update embeddings array
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update FAISS index
        if FAISS_AVAILABLE:
            if self.index is None:
                self._init_index()
            
            # Train IVF index if needed
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                logger.info("Training IVF index...")
                self.index.train(self.embeddings.astype(np.float32))
            
            self.index.add(new_embeddings.astype(np.float32))
            logger.info(f"Index now contains {self.index.ntotal} vectors")
        
        return len(valid_docs)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        self._init_model()
        
        if len(self.doc_ids) == 0:
            return []
        
        # Compute query embedding
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1).astype(np.float32)
        
        if FAISS_AVAILABLE and self.index is not None:
            # Set search parameters for IVF
            if isinstance(self.index, faiss.IndexIVFFlat):
                self.index.nprobe = self.config.nprobe
            
            # Search
            similarities, indices = self.index.search(
                query_embedding, 
                min(top_k, self.index.ntotal)
            )
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                
                doc_id = self.doc_ids[idx]
                doc = self.documents.get(doc_id)
                if doc:
                    results.append((doc, float(sim)))
            
            return results
        
        else:
            # Brute-force search
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                doc_id = self.doc_ids[idx]
                doc = self.documents.get(doc_id)
                if doc:
                    results.append((doc, float(similarities[idx])))
            
            return results
    
    def search_by_document(self, document: Document, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Find similar documents to a given document.
        
        Args:
            document: Query document
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not document.content:
            return []
        
        results = self.search(document.content, top_k + 1)
        
        # Filter out the query document itself
        return [(doc, sim) for doc, sim in results if doc.doc_id != document.doc_id][:top_k]
    
    def find_clusters(self, n_clusters: int = 10) -> List[List[str]]:
        """
        Cluster documents using K-means.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            List of document ID clusters
        """
        if self.embeddings is None or len(self.embeddings) < n_clusters:
            return []
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available for clustering")
            return []
        
        # K-means clustering
        kmeans = faiss.Kmeans(
            self.dimension, 
            n_clusters,
            niter=20,
            verbose=False
        )
        kmeans.train(self.embeddings.astype(np.float32))
        
        # Get cluster assignments
        _, assignments = kmeans.index.search(self.embeddings.astype(np.float32), 1)
        assignments = assignments.flatten()
        
        # Group by cluster
        clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for doc_idx, cluster_idx in enumerate(assignments):
            clusters[cluster_idx].append(self.doc_ids[doc_idx])
        
        return list(clusters.values())
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed corpus."""
        if self.embeddings is None:
            return {"total_documents": 0}
        
        return {
            "total_documents": len(self.doc_ids),
            "embedding_dimension": self.dimension,
            "index_type": self.config.index_type if FAISS_AVAILABLE else "numpy",
        }
    
    def save(self, path: Path):
        """
        Save index to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        
        # Save document IDs
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        
        # Save documents
        docs_data = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: Path):
        """
        Load index from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        
        # Load FAISS index
        if FAISS_AVAILABLE and (path / "index.faiss").exists():
            self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load embeddings
        if (path / "embeddings.npy").exists():
            self.embeddings = np.load(path / "embeddings.npy")
            self.dimension = self.embeddings.shape[1]
        
        # Load document IDs
        if (path / "doc_ids.json").exists():
            with open(path / "doc_ids.json", "r") as f:
                self.doc_ids = json.load(f)
        
        # Load documents
        if (path / "documents.json").exists():
            with open(path / "documents.json", "r", encoding="utf-8") as f:
                docs_data = json.load(f)
                self.documents = {
                    doc_id: Document.from_dict(data) 
                    for doc_id, data in docs_data.items()
                }
        
        logger.info(f"Index loaded from {path}: {len(self.doc_ids)} documents")
    
    def clear(self):
        """Clear the index."""
        self.doc_ids = []
        self.documents.clear()
        self.embeddings = None
        self.index = None
    
    def __len__(self) -> int:
        return len(self.doc_ids)
