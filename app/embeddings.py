"""Embedding service wrapper for generating vector embeddings.

This module provides a model-agnostic interface for generating embeddings,
currently implemented with Gemini embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this service."""
        pass


class GeminiEmbeddingService(EmbeddingService):
    """Gemini embeddings implementation."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initialize Gemini embedding service.

        Args:
            model_name: Embedding model name (default: 'gemini-embedding-001')
        """
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-genai not installed. Install with: pip install google-genai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Initialize client with API key
        self.client = genai.Client(api_key=api_key)
        # Gemini embeddings model
        self.model = model_name
        self._dimension = 768  # Gemini embedding dimension

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            task_type: Task type for embedding (default: 'RETRIEVAL_QUERY')
        """
        try:
            # Try different possible API structures for embedding
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    content=text,
                    task_type=task_type,
                )
            except (AttributeError, TypeError):
                # Fallback: try alternative API structure
                try:
                    result = self.client.embed_content(
                        model=self.model,
                        content=text,
                        task_type=task_type,
                    )
                except Exception:
                    # Last fallback: try with model object
                    model_obj = self.client.models.get(self.model)
                    result = model_obj.embed_content(content=text, task_type=task_type)
            
            # Extract embedding from response - handle different response structures
            if hasattr(result, 'embedding'):
                return result.embedding
            elif hasattr(result, 'values'):
                return result.values
            elif isinstance(result, (list, tuple)):
                return list(result)
            elif isinstance(result, dict):
                return result.get("embedding", result.get("values", []))
            else:
                # Try to convert to list
                return list(result) if hasattr(result, '__iter__') else []
        except Exception as e:
            raise RuntimeError(f"Gemini embedding error: {str(e)}")

    def embed_batch(self, texts: List[str], task_type: str = "RETRIEVAL_QUERY") -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            task_type: Task type for embedding (default: 'RETRIEVAL_QUERY')
        """
        try:
            # Try batch embedding first
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    content=texts,  # Pass list directly
                    task_type=task_type,
                )
                # Handle batch response
                if isinstance(result, list):
                    return [self._extract_embedding(r) for r in result]
                elif hasattr(result, 'embeddings'):
                    return [self._extract_embedding(e) for e in result.embeddings]
                else:
                    # Fallback to individual calls
                    raise AttributeError("Batch not supported, falling back to individual calls")
            except (AttributeError, TypeError):
                # Fallback: call individually
                embeddings = []
                for text in texts:
                    embedding = self.embed_text(text, task_type=task_type)
                    embeddings.append(embedding)
                return embeddings
        except Exception as e:
            raise RuntimeError(f"Gemini batch embedding error: {str(e)}")
    
    def _extract_embedding(self, result) -> List[float]:
        """Helper to extract embedding from various response formats."""
        if hasattr(result, 'embedding'):
            return result.embedding
        elif hasattr(result, 'values'):
            return result.values
        elif isinstance(result, (list, tuple)):
            return list(result)
        elif isinstance(result, dict):
            return result.get("embedding", result.get("values", []))
        else:
            return list(result) if hasattr(result, '__iter__') else []

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings."""
        return self._dimension


def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """
    Factory function to get an available embedding service.

    Args:
        model_name: Optional embedding model name (default: 'gemini-embedding-001')

    Returns:
        An initialized EmbeddingService instance

    Raises:
        RuntimeError: If no embedding service is available
    """
    try:
        return GeminiEmbeddingService(model_name=model_name or "gemini-embedding-001")
    except (ValueError, ImportError) as e:
        raise RuntimeError(
            f"Embedding service not available: {str(e)}. "
            "Please configure GEMINI_API_KEY."
        )
