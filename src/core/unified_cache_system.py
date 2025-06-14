#!/usr/bin/env python3
"""
Unified Cache System - TASK-005 Implementation
==============================================

Consolidates multiple cache systems into a single, unified caching solution:
- optimized_cache.py: High-performance cache with compression
- smart_claude_cache.py: Semantic cache for Claude API responses
- emergency_embeddings.py: Emergency fallback caching

This unified system provides:
- Semantic caching for API responses
- High-performance embedding cache
- Compressed storage with LRU eviction
- Thread-safe operations
- Automatic cleanup and maintenance
- Cache statistics and monitoring

Eliminates 3 parallel cache systems identified in TASK-005 audit.

Author: Pablo Emanuel Romero Almada, Ph.D.
Date: 2025-06-14
Version: 5.0.0
"""

import gzip
import hashlib
import json
import logging
import lz4.frame
import pickle
import threading
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class CacheRequest:
    """Unified cache request supporting multiple cache types"""
    key: str
    content: Any
    cache_type: str  # 'api', 'embedding', 'search', 'general'
    stage: Optional[str] = None
    operation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ttl_hours: int = 24


@dataclass 
class CacheEntry:
    """Unified cache entry with metadata and compression"""
    data: Any
    cache_type: str
    timestamp: float
    ttl_seconds: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    compressed: bool = False
    size_bytes: int = 0


class UnifiedCacheSystem:
    """
    Unified cache system consolidating all caching needs
    
    Features:
    - Semantic caching for API responses (replaces smart_claude_cache)
    - High-performance embedding cache (replaces optimized_cache) 
    - Emergency fallback capabilities (replaces emergency_embeddings)
    - Compressed storage with multiple compression algorithms
    - LRU eviction with intelligent cache policies
    - Thread-safe operations
    - Cache statistics and monitoring
    - Automatic cleanup and maintenance
    """
    
    def __init__(self, 
                 cache_dir: Union[str, Path] = "cache/unified",
                 max_memory_mb: int = 1024,
                 max_disk_gb: int = 5,
                 compression_algorithm: str = "lz4",
                 semantic_similarity_threshold: float = 0.85,
                 default_ttl_hours: int = 24):
        
        # Directory setup
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different cache types
        self.api_cache_dir = self.cache_dir / "api_responses"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.search_cache_dir = self.cache_dir / "search_results"
        self.general_cache_dir = self.cache_dir / "general"
        
        for cache_dir in [self.api_cache_dir, self.embedding_cache_dir, 
                         self.search_cache_dir, self.general_cache_dir]:
            cache_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_gb * 1024 * 1024 * 1024
        self.compression_algorithm = compression_algorithm
        self.semantic_threshold = semantic_similarity_threshold
        self.default_ttl_seconds = default_ttl_hours * 3600
        
        # In-memory caches by type
        self.memory_caches = {
            'api': OrderedDict(),
            'embedding': OrderedDict(),
            'search': OrderedDict(),
            'general': OrderedDict()
        }
        
        # Memory usage tracking
        self.memory_usage = {'api': 0, 'embedding': 0, 'search': 0, 'general': 0}
        self.total_memory_usage = 0
        
        # Semantic search for API cache
        self.api_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.api_vectors = {}
        self.api_keys = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'stores': defaultdict(int),
            'evictions': defaultdict(int),
            'semantic_hits': 0,
            'compression_savings': 0
        }
        
        # Start maintenance thread
        self._start_maintenance_thread()
        
        logger.info(f"Unified cache system initialized: {cache_dir}")
    
    def get(self, key: str, cache_type: str = 'general') -> Optional[Any]:
        """
        Get item from cache with semantic search for API responses
        
        Args:
            key: Cache key
            cache_type: Type of cache ('api', 'embedding', 'search', 'general')
            
        Returns:
            Cached data or None if not found
        """
        with self.lock:
            # Try exact match first
            entry = self._get_exact_match(key, cache_type)
            if entry:
                if self._is_expired(entry):
                    self._remove_entry(key, cache_type)
                    self.stats['misses'][cache_type] += 1
                    return None
                
                # Update access stats
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                # Move to end for LRU
                self.memory_caches[cache_type].move_to_end(key)
                
                self.stats['hits'][cache_type] += 1
                return self._decompress_data(entry.data, entry.compressed)
            
            # For API cache, try semantic search
            if cache_type == 'api':
                semantic_result = self._semantic_search(key)
                if semantic_result:
                    self.stats['semantic_hits'] += 1
                    return semantic_result
            
            self.stats['misses'][cache_type] += 1
            return None
    
    def put(self, key: str, data: Any, cache_type: str = 'general', 
            ttl_hours: Optional[int] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Store item in cache with compression and metadata
        
        Args:
            key: Cache key
            data: Data to cache
            cache_type: Type of cache
            ttl_hours: Time to live in hours
            metadata: Additional metadata
            
        Returns:
            True if stored successfully
        """
        with self.lock:
            try:
                # Prepare entry
                ttl_seconds = (ttl_hours or (self.default_ttl_seconds // 3600)) * 3600
                
                entry = CacheEntry(
                    data=data,
                    cache_type=cache_type,
                    timestamp=time.time(),
                    ttl_seconds=ttl_seconds,
                    metadata=metadata or {}
                )
                
                # Compress data if beneficial
                compressed_data, is_compressed, size_bytes = self._compress_data(data)
                entry.data = compressed_data
                entry.compressed = is_compressed
                entry.size_bytes = size_bytes
                
                # Check if we need to evict
                self._ensure_space(cache_type, size_bytes)
                
                # Store in memory
                self.memory_caches[cache_type][key] = entry
                self.memory_usage[cache_type] += size_bytes
                self.total_memory_usage += size_bytes
                
                # For API cache, update semantic search vectors
                if cache_type == 'api':
                    self._update_semantic_vectors(key)
                
                # Store on disk for persistence
                self._store_on_disk(key, entry, cache_type)
                
                self.stats['stores'][cache_type] += 1
                return True
                
            except Exception as e:
                logger.error(f"Error storing cache entry {key}: {e}")
                return False
    
    def invalidate(self, key: str, cache_type: str = 'general') -> bool:
        """Remove specific entry from cache"""
        with self.lock:
            return self._remove_entry(key, cache_type)
    
    def clear(self, cache_type: Optional[str] = None):
        """Clear cache (specific type or all)"""
        with self.lock:
            if cache_type:
                self.memory_caches[cache_type].clear()
                self.memory_usage[cache_type] = 0
                # Clear disk cache
                cache_dir = self._get_cache_dir(cache_type)
                for file in cache_dir.glob("*.cache"):
                    file.unlink()
            else:
                for ct in self.memory_caches:
                    self.memory_caches[ct].clear()
                    self.memory_usage[ct] = 0
                # Clear all disk caches
                for cache_dir in [self.api_cache_dir, self.embedding_cache_dir,
                                self.search_cache_dir, self.general_cache_dir]:
                    for file in cache_dir.glob("*.cache"):
                        file.unlink()
            
            self._recalculate_memory_usage()
            logger.info(f"Cache cleared: {cache_type or 'all'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_hits = sum(self.stats['hits'].values())
            total_misses = sum(self.stats['misses'].values())
            hit_rate = total_hits / max(1, total_hits + total_misses)
            
            return {
                'hit_rate': hit_rate,
                'semantic_hits': self.stats['semantic_hits'],
                'memory_usage_mb': self.total_memory_usage / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'cache_counts': {ct: len(self.memory_caches[ct]) for ct in self.memory_caches},
                'hit_stats': dict(self.stats['hits']),
                'miss_stats': dict(self.stats['misses']),
                'compression_savings_mb': self.stats['compression_savings'] / (1024 * 1024)
            }
    
    def _get_exact_match(self, key: str, cache_type: str) -> Optional[CacheEntry]:
        """Get exact key match from memory cache"""
        if key in self.memory_caches[cache_type]:
            return self.memory_caches[cache_type][key]
        
        # Try loading from disk
        return self._load_from_disk(key, cache_type)
    
    def _semantic_search(self, query: str) -> Optional[Any]:
        """Search for semantically similar API responses"""
        if not self.api_keys or len(self.api_keys) < 2:
            return None
        
        try:
            # Vectorize query
            query_vector = self.api_vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, 
                                           np.vstack(list(self.api_vectors.values())))
            
            max_similarity = similarities.max()
            if max_similarity >= self.semantic_threshold:
                best_match_idx = similarities.argmax()
                best_key = self.api_keys[best_match_idx]
                
                # Return the cached response
                entry = self.memory_caches['api'].get(best_key)
                if entry and not self._is_expired(entry):
                    return self._decompress_data(entry.data, entry.compressed)
        
        except Exception as e:
            logger.debug(f"Semantic search error: {e}")
        
        return None
    
    def _update_semantic_vectors(self, key: str):
        """Update semantic search vectors for API cache"""
        try:
            # Add key to list
            if key not in self.api_keys:
                self.api_keys.append(key)
            
            # Refit vectorizer if needed
            if len(self.api_keys) >= 2:
                vectors = self.api_vectorizer.fit_transform(self.api_keys)
                self.api_vectors = {k: vectors[i] for i, k in enumerate(self.api_keys)}
        
        except Exception as e:
            logger.debug(f"Error updating semantic vectors: {e}")
    
    def _compress_data(self, data: Any) -> Tuple[Any, bool, int]:
        """Compress data using configured algorithm"""
        try:
            # Serialize first
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Skip compression for small data
            if original_size < 1024:
                return data, False, original_size
            
            # Compress
            if self.compression_algorithm == "lz4":
                compressed = lz4.frame.compress(serialized)
            elif self.compression_algorithm == "gzip":
                compressed = gzip.compress(serialized)
            else:
                return data, False, original_size
            
            compressed_size = len(compressed)
            
            # Only use compressed version if it's significantly smaller
            if compressed_size < original_size * 0.8:
                self.stats['compression_savings'] += (original_size - compressed_size)
                return compressed, True, compressed_size
            else:
                return data, False, original_size
                
        except Exception as e:
            logger.debug(f"Compression failed: {e}")
            return data, False, len(pickle.dumps(data))
    
    def _decompress_data(self, data: Any, is_compressed: bool) -> Any:
        """Decompress data if needed"""
        if not is_compressed:
            return data
        
        try:
            if self.compression_algorithm == "lz4":
                decompressed = lz4.frame.decompress(data)
            elif self.compression_algorithm == "gzip":
                decompressed = gzip.decompress(data)
            else:
                return data
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data
    
    def _ensure_space(self, cache_type: str, needed_bytes: int):
        """Ensure sufficient space by evicting LRU entries"""
        cache = self.memory_caches[cache_type]
        
        # Calculate available space
        available = self.max_memory_bytes - self.total_memory_usage
        
        if available >= needed_bytes:
            return
        
        # Evict LRU entries
        to_evict = needed_bytes - available + (self.max_memory_bytes // 10)  # Extra 10% buffer
        evicted = 0
        
        while evicted < to_evict and cache:
            # Get LRU entry (first in OrderedDict)
            key = next(iter(cache))
            entry = cache[key]
            
            evicted += entry.size_bytes
            self._remove_entry(key, cache_type)
            self.stats['evictions'][cache_type] += 1
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return time.time() - entry.timestamp > entry.ttl_seconds
    
    def _remove_entry(self, key: str, cache_type: str) -> bool:
        """Remove entry from both memory and disk"""
        cache = self.memory_caches[cache_type]
        
        if key in cache:
            entry = cache[key]
            self.memory_usage[cache_type] -= entry.size_bytes
            self.total_memory_usage -= entry.size_bytes
            del cache[key]
            
            # Remove from disk
            cache_file = self._get_cache_dir(cache_type) / f"{self._hash_key(key)}.cache"
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove from semantic vectors if API cache
            if cache_type == 'api' and key in self.api_keys:
                self.api_keys.remove(key)
                if key in self.api_vectors:
                    del self.api_vectors[key]
            
            return True
        
        return False
    
    def _get_cache_dir(self, cache_type: str) -> Path:
        """Get cache directory for specific type"""
        return {
            'api': self.api_cache_dir,
            'embedding': self.embedding_cache_dir,
            'search': self.search_cache_dir,
            'general': self.general_cache_dir
        }.get(cache_type, self.general_cache_dir)
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for key"""
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _store_on_disk(self, key: str, entry: CacheEntry, cache_type: str):
        """Store entry on disk for persistence"""
        try:
            cache_dir = self._get_cache_dir(cache_type)
            cache_file = cache_dir / f"{self._hash_key(key)}.cache"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
                
        except Exception as e:
            logger.debug(f"Failed to store on disk: {e}")
    
    def _load_from_disk(self, key: str, cache_type: str) -> Optional[CacheEntry]:
        """Load entry from disk"""
        try:
            cache_dir = self._get_cache_dir(cache_type)
            cache_file = cache_dir / f"{self._hash_key(key)}.cache"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if expired
                if self._is_expired(entry):
                    cache_file.unlink()
                    return None
                
                # Load into memory
                self.memory_caches[cache_type][key] = entry
                self.memory_usage[cache_type] += entry.size_bytes
                self.total_memory_usage += entry.size_bytes
                
                return entry
                
        except Exception as e:
            logger.debug(f"Failed to load from disk: {e}")
        
        return None
    
    def _recalculate_memory_usage(self):
        """Recalculate total memory usage"""
        self.total_memory_usage = sum(self.memory_usage.values())
    
    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        def maintenance_worker():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Maintenance thread error: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        with self.lock:
            for cache_type in self.memory_caches:
                cache = self.memory_caches[cache_type]
                expired_keys = []
                
                for key, entry in cache.items():
                    if self._is_expired(entry):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_entry(key, cache_type)
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired entries from {cache_type} cache")


# Global instance
_global_cache = None


def get_unified_cache() -> UnifiedCacheSystem:
    """Get global unified cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCacheSystem()
    return _global_cache


def configure_unified_cache(**kwargs) -> UnifiedCacheSystem:
    """Configure and get unified cache with custom settings"""
    global _global_cache
    _global_cache = UnifiedCacheSystem(**kwargs)
    return _global_cache