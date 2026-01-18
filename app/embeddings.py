"""Embedding service wrapper for generating vector embeddings.

This module provides a model-agnostic interface for generating embeddings,
currently implemented with Gemini embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os
import time
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
        # gemini-embedding-001 returns 3072 dimensions by default
        # Can be truncated to 768 or 1536 if needed, but default is 3072
        self._dimension = 3072  # Gemini embedding dimension

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            task_type: Task type for embedding (default: 'RETRIEVAL_QUERY')
        """
        try:
            # google-genai API structure: client.models.embed_content(model=..., contents=[...])
            # Note: 'contents' is plural and expects a list
            # Note: task_type may not be supported in this API version, using contents only
            result = self.client.models.embed_content(
                model=self.model,
                contents=[text],  # Note: contents is plural and expects a list
            )
            
            # Extract embedding from response
            # The google-genai API returns EmbedContentResponse with embeddings list
            # Each item in embeddings is ContentEmbedding with .values attribute
            embedding = None
            
            # Standard structure: result.embeddings[0].values
            if hasattr(result, 'embeddings') and isinstance(result.embeddings, (list, tuple)) and len(result.embeddings) > 0:
                emb_obj = result.embeddings[0]
                if hasattr(emb_obj, 'values'):
                    embedding = emb_obj.values
                elif hasattr(emb_obj, 'embedding'):
                    embedding = emb_obj.embedding
            
            # Fallback: check for direct embedding attribute
            if embedding is None and hasattr(result, 'embedding'):
                embedding = result.embedding
            elif embedding is None and hasattr(result, 'values'):
                embedding = result.values
            
            # Convert to list of floats, filtering out non-numeric values
            if embedding is None:
                raise RuntimeError(f"Could not extract embedding from API response. Response type: {type(result)}")
            
            # Ensure we return a list of floats
            result_list = []
            if isinstance(embedding, (list, tuple)):
                for x in embedding:
                    if isinstance(x, (int, float)):
                        result_list.append(float(x))
                    elif isinstance(x, (list, tuple)):
                        # If element is itself a tuple/list, flatten it
                        for y in x:
                            if isinstance(y, (int, float)):
                                result_list.append(float(y))
            elif isinstance(embedding, (int, float)):
                result_list.append(float(embedding))
            
            if not result_list:
                raise RuntimeError(f"Could not extract numeric embedding values from response. Response type: {type(result)}, embedding type: {type(embedding)}")
            
            return result_list
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
            # google-genai API: client.models.embed_content(model=..., contents=[...])
            # The API accepts a list of texts in 'contents'
            # Note: task_type may not be supported in this API version
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,  # Pass list of texts
            )
            
            # Handle batch response
            # Standard structure: result.embeddings is a list of ContentEmbedding objects
            # Each has .values attribute with the actual embedding vector
            if hasattr(result, 'embeddings') and isinstance(result.embeddings, (list, tuple)):
                embeddings = []
                for emb_obj in result.embeddings:
                    if hasattr(emb_obj, 'values'):
                        emb_values = emb_obj.values
                        if isinstance(emb_values, (list, tuple)):
                            embeddings.append([float(x) for x in emb_values if isinstance(x, (int, float))])
                        else:
                            embeddings.append([float(emb_values)])
                    elif hasattr(emb_obj, 'embedding'):
                        emb_values = emb_obj.embedding
                        if isinstance(emb_values, (list, tuple)):
                            embeddings.append([float(x) for x in emb_values if isinstance(x, (int, float))])
                        else:
                            embeddings.append([float(emb_values)])
                if embeddings:
                    return embeddings
            
            # Fallback: call individually if batch structure is different
            embeddings = []
            for text in texts:
                embedding = self.embed_text(text, task_type=task_type)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            # Check if it's a rate limit error (429)
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                # Rate limit exceeded - wait and retry with individual calls
                print("[WARNING] Rate limit exceeded. Waiting 20 seconds and retrying with individual calls...")
                time.sleep(20)
                try:
                    embeddings = []
                    for i, text in enumerate(texts):
                        if i > 0 and i % 10 == 0:  # Add small delay every 10 requests
                            time.sleep(1)
                        embedding = self.embed_text(text, task_type=task_type)
                        embeddings.append(embedding)
                    return embeddings
                except Exception as e2:
                    raise RuntimeError(f"Gemini batch embedding error (rate limited): {str(e)} (retry also failed: {str(e2)})")
            else:
                # Fallback: call individually if batch fails for other reasons
                try:
                    embeddings = []
                    for text in texts:
                        embedding = self.embed_text(text, task_type=task_type)
                        embeddings.append(embedding)
                    return embeddings
                except Exception as e2:
                    raise RuntimeError(f"Gemini batch embedding error: {str(e)} (fallback also failed: {str(e2)})")
    
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
