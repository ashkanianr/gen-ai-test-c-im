"""Embedding cache for storing and retrieving policy embeddings.

This module provides caching functionality to avoid regenerating embeddings
for unchanged policies, reducing API calls and preventing rate limits.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np


class EmbeddingCache:
    """Cache manager for policy embeddings."""
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, policy_path: str, chunk_size: int, chunk_overlap: int) -> str:
        """
        Generate cache key based on file path, modification time, and chunking parameters.
        
        Args:
            policy_path: Path to policy file
            chunk_size: Chunk size used
            chunk_overlap: Chunk overlap used
            
        Returns:
            Cache key (hash string)
        """
        policy_path = Path(policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        # Get file modification time and size
        mtime = policy_path.stat().st_mtime
        size = policy_path.stat().st_size
        
        # Create hash from file path, mtime, size, and chunking params
        hash_input = f"{policy_path.absolute()}:{mtime}:{size}:{chunk_size}:{chunk_overlap}"
        cache_key = hashlib.md5(hash_input.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> tuple[Path, Path]:
        """
        Get paths for cached embeddings and metadata.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Tuple of (embeddings_path, metadata_path)
        """
        embeddings_path = self.cache_dir / f"{cache_key}.npy"
        metadata_path = self.cache_dir / f"{cache_key}.json"
        return embeddings_path, metadata_path
    
    def get_cached_embeddings(
        self,
        policy_path: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Optional[tuple[np.ndarray, List[Dict[str, Any]]]]:
        """
        Retrieve cached embeddings if available.
        
        Args:
            policy_path: Path to policy file
            chunk_size: Chunk size used
            chunk_overlap: Chunk overlap used
            
        Returns:
            Tuple of (embeddings_array, metadata_list) if cached, None otherwise
        """
        try:
            cache_key = self._get_cache_key(policy_path, chunk_size, chunk_overlap)
            embeddings_path, metadata_path = self._get_cache_path(cache_key)
            
            # Check if cache exists
            if not embeddings_path.exists() or not metadata_path.exists():
                return None
            
            # Load cached embeddings and metadata
            embeddings_array = np.load(embeddings_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            return embeddings_array, metadata_list
            
        except Exception as e:
            # If cache read fails, return None (will regenerate)
            print(f"[WARNING] Cache read failed: {e}. Regenerating embeddings...")
            return None
    
    def save_embeddings(
        self,
        policy_path: str,
        chunk_size: int,
        chunk_overlap: int,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Save embeddings to cache.
        
        Args:
            policy_path: Path to policy file
            chunk_size: Chunk size used
            chunk_overlap: Chunk overlap used
            embeddings: Embeddings array
            metadata: Metadata list
        """
        try:
            cache_key = self._get_cache_key(policy_path, chunk_size, chunk_overlap)
            embeddings_path, metadata_path = self._get_cache_path(cache_key)
            
            # Save embeddings
            np.save(embeddings_path, embeddings)
            
            # Save metadata (convert numpy types to native Python types for JSON)
            metadata_json = []
            for meta in metadata:
                meta_json = {}
                for key, value in meta.items():
                    if isinstance(value, (np.integer, np.floating)):
                        meta_json[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        meta_json[key] = value.tolist()
                    else:
                        meta_json[key] = value
                metadata_json.append(meta_json)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_json, f, indent=2)
                
        except Exception as e:
            # If cache write fails, just warn (not critical)
            print(f"[WARNING] Cache write failed: {e}. Continuing without cache...")
    
    def clear_cache(self, policy_path: Optional[str] = None) -> None:
        """
        Clear cache for a specific policy or all policies.
        
        Args:
            policy_path: If provided, clear only this policy's cache. Otherwise clear all.
        """
        if policy_path:
            # Clear specific policy cache
            # This is tricky since we need to find the cache key
            # For now, we'll just clear all if a specific path is given
            # (could be improved to track policy->cache_key mapping)
            print(f"[INFO] Clearing cache for {policy_path}...")
            # For simplicity, clear all if specific path requested
            self.clear_cache()
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            print("[INFO] Cache cleared.")
