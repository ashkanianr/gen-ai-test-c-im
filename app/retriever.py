"""Vector database abstraction for retrieval operations.

This module provides a clean interface for vector store operations,
currently implemented with FAISS, but abstracted for future swaps.

RETRIEVAL QUALITY:
- Returns top-K chunks with cosine similarity scores (0-1)
- Each result includes:
  * chunk: Full text content
  * score: Similarity score (used for confidence calculation)
  * metadata: section_id, page_number, policy_name, chunk_index
- Metadata is used for citations and traceability
- Similarity scores feed into runtime evaluation confidence calculation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorRetriever(ABC):
    """Abstract base class for vector retrievers."""

    @abstractmethod
    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add document embeddings to the vector store.

        Args:
            embeddings: Numpy array of shape (n_docs, embedding_dim)
            metadata: List of metadata dicts, one per document
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector of shape (embedding_dim,)
            k: Number of results to return

        Returns:
            List of dicts with 'chunk', 'score', and 'metadata' keys
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """Return the number of documents in the store."""
        pass


class FAISSRetriever(VectorRetriever):
    """FAISS-based vector retriever implementation."""

    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS retriever.

        Args:
            embedding_dim: Dimension of embeddings (e.g., 768 for Gemini)
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not available. Install with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata_store: List[Dict[str, Any]] = []
        self._is_initialized = False

    def _initialize_index(self):
        """Initialize FAISS index if not already done."""
        if not self._is_initialized:
            # Use L2 distance (Euclidean) - FAISS uses inner product for cosine similarity
            # We'll normalize embeddings for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
            self._is_initialized = True

    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add document embeddings to FAISS index."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) "
                f"must match number of metadata items ({len(metadata)})"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) "
                f"must match expected dimension ({self.embedding_dim})"
            )

        self._initialize_index()

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms

        # Convert to float32 for FAISS
        normalized_embeddings = normalized_embeddings.astype(np.float32)

        # Add to index
        self.index.add(normalized_embeddings)

        # Store metadata
        self.metadata_store.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of dicts with 'chunk', 'score', and 'metadata' keys
        """
        if not self._is_initialized or self.index.ntotal == 0:
            return []

        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[0]}) "
                f"must match expected dimension ({self.embedding_dim})"
            )

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        normalized_query = (query_embedding / query_norm).astype(np.float32)

        # Reshape for FAISS (needs 2D array)
        query_2d = normalized_query.reshape(1, -1)

        # Search
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_2d, k)

        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            # Convert inner product to cosine similarity (already normalized)
            # Inner product of normalized vectors = cosine similarity
            cosine_similarity = float(distance)

            metadata = self.metadata_store[idx].copy()
            chunk_text = metadata.get("text", "")

            results.append({
                "chunk": chunk_text,
                "score": cosine_similarity,
                "metadata": metadata,
            })

        return results

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.index = None
        self.metadata_store = []
        self._is_initialized = False

    def get_document_count(self) -> int:
        """Return the number of documents in the store."""
        if not self._is_initialized:
            return 0
        return self.index.ntotal


def create_retriever(embedding_dim: int) -> VectorRetriever:
    """
    Factory function to create a vector retriever.

    Args:
        embedding_dim: Dimension of embeddings

    Returns:
        A VectorRetriever instance (currently FAISS)

    Raises:
        RuntimeError: If no retriever implementation is available
    """
    if FAISS_AVAILABLE:
        return FAISSRetriever(embedding_dim)

    raise RuntimeError(
        "No vector retriever available. Install FAISS: pip install faiss-cpu"
    )
