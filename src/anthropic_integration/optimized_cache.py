"""
Optimized Cache System for Embeddings and Search Results
Provides compressed storage, fast retrieval, and memory-efficient operations
"""

import gzip
import hashlib
import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class OptimizedCache:
    """
    High-performance cache system with compression and smart eviction

    Features:
    - Compressed storage (gzip)
    - LRU eviction policy
    - Memory-efficient operations
    - Thread-safe operations
    - Automatic cleanup
    - Cache statistics
    """

    def __init__(self, cache_dir_or_config: Union[str, Path, Dict[str, Any]], max_memory_mb: int = 512,
                 compression_level: int = 6, ttl_hours: int = 24):
        # Handle both config dict and path string for backward compatibility
        if isinstance(cache_dir_or_config, dict):
            # Config dict passed (from tests)
            self.config = cache_dir_or_config
            # Use a default cache directory for tests
            self.cache_dir = Path.cwd() / 'cache' / 'test_cache'
            # Extract cache settings from config if available
            cache_config = self.config.get('cache', {})
            max_memory_mb = cache_config.get('max_memory_mb', max_memory_mb)
            compression_level = cache_config.get('compression_level', compression_level)
            ttl_hours = cache_config.get('ttl_hours', ttl_hours)
        else:
            # Path passed (legacy)
            self.config = {}
            self.cache_dir = Path(cache_dir_or_config)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_level = compression_level
        self.ttl_seconds = ttl_hours * 3600

        # In-memory cache with LRU
        self.memory_cache = OrderedDict()
        self.memory_usage = 0

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'compressions': 0,
            'decompressions': 0,
            'evictions': 0
        }

        logger.info(f"OptimizedCache initialized: {self.cache_dir}, max_memory={max_memory_mb}MB")

    def _generate_key(self, key: str) -> str:
        """Generate standardized cache key"""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{cache_key}.cache.gz"

    def _compress_data(self, data: Any) -> bytes:
        """Compress data using gzip"""
        try:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = gzip.compress(serialized, compresslevel=self.compression_level)
            self.stats['compressions'] += 1
            return compressed
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            raise

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from gzip"""
        try:
            decompressed = gzip.decompress(compressed_data)
            data = pickle.loads(decompressed)
            self.stats['decompressions'] += 1
            return data
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            raise

    def _evict_lru(self):
        """Evict least recently used items from memory"""
        with self.lock:
            while self.memory_usage > self.max_memory_bytes and self.memory_cache:
                key, (data, size, timestamp) = self.memory_cache.popitem(last=False)
                self.memory_usage -= size
                self.stats['evictions'] += 1
                logger.debug(f"Evicted from memory cache: {key}")

    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 1024  # Default estimate

    def put(self, key: str, data: Any, force_disk: bool = False) -> bool:
        """
        Store data in cache

        Args:
            key: Cache key
            data: Data to store
            force_disk: Force storage to disk only

        Returns:
            Success status
        """
        try:
            cache_key = self._generate_key(key)
            timestamp = time.time()

            # Store to disk
            file_path = self._get_file_path(cache_key)
            compressed_data = self._compress_data({
                'data': data,
                'timestamp': timestamp,
                'key': key
            })

            with open(file_path, 'wb') as f:
                f.write(compressed_data)

            # Store in memory if not forcing disk-only
            if not force_disk:
                data_size = self._estimate_size(data)

                with self.lock:
                    # Remove if already exists
                    if cache_key in self.memory_cache:
                        _, old_size, _ = self.memory_cache[cache_key]
                        self.memory_usage -= old_size

                    # Add to memory cache
                    self.memory_cache[cache_key] = (data, data_size, timestamp)
                    self.memory_usage += data_size

                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(cache_key)

                    # Evict if necessary
                    self._evict_lru()

            return True

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    def set(self, key: str, data: Any) -> bool:
        """
        Store data in cache (test compatibility method).
        
        Args:
            key: Cache key
            data: Data to store
            
        Returns:
            Success status
        """
        return self.put(key, data)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        try:
            cache_key = self._generate_key(key)
            current_time = time.time()

            # Check memory cache first
            with self.lock:
                if cache_key in self.memory_cache:
                    data, size, timestamp = self.memory_cache[cache_key]

                    # Check TTL
                    if current_time - timestamp < self.ttl_seconds:
                        # Move to end (most recently used)
                        self.memory_cache.move_to_end(cache_key)
                        self.stats['hits'] += 1
                        self.stats['memory_hits'] += 1
                        return data
                    else:
                        # Expired, remove from memory
                        del self.memory_cache[cache_key]
                        self.memory_usage -= size

            # Check disk cache
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        compressed_data = f.read()

                    cache_entry = self._decompress_data(compressed_data)

                    # Check TTL
                    if current_time - cache_entry['timestamp'] < self.ttl_seconds:
                        data = cache_entry['data']

                        # Add back to memory cache
                        data_size = self._estimate_size(data)
                        with self.lock:
                            self.memory_cache[cache_key] = (data, data_size, cache_entry['timestamp'])
                            self.memory_usage += data_size
                            self.memory_cache.move_to_end(cache_key)
                            self._evict_lru()

                        self.stats['hits'] += 1
                        self.stats['disk_hits'] += 1
                        return data
                    else:
                        # Expired, remove file
                        file_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.error(f"Error reading cache file {file_path}: {e}")
                    file_path.unlink(missing_ok=True)

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            self.stats['misses'] += 1
            return None

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            cache_key = self._generate_key(key)

            # Remove from memory
            with self.lock:
                if cache_key in self.memory_cache:
                    _, size, _ = self.memory_cache[cache_key]
                    del self.memory_cache[cache_key]
                    self.memory_usage -= size

            # Remove from disk
            file_path = self._get_file_path(cache_key)
            file_path.unlink(missing_ok=True)

            return True

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache"""
        try:
            # Clear memory
            with self.lock:
                self.memory_cache.clear()
                self.memory_usage = 0

            # Clear disk
            for file_path in self.cache_dir.glob("*.cache.gz"):
                file_path.unlink(missing_ok=True)

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        try:
            current_time = time.time()
            cleaned_count = 0

            # Clean memory cache
            with self.lock:
                expired_keys = []
                for cache_key, (data, size, timestamp) in self.memory_cache.items():
                    if current_time - timestamp >= self.ttl_seconds:
                        expired_keys.append(cache_key)

                for cache_key in expired_keys:
                    _, size, _ = self.memory_cache[cache_key]
                    del self.memory_cache[cache_key]
                    self.memory_usage -= size
                    cleaned_count += 1

            # Clean disk cache
            for file_path in self.cache_dir.glob("*.cache.gz"):
                try:
                    with open(file_path, 'rb') as f:
                        compressed_data = f.read()

                    cache_entry = self._decompress_data(compressed_data)

                    if current_time - cache_entry['timestamp'] >= self.ttl_seconds:
                        file_path.unlink(missing_ok=True)
                        cleaned_count += 1

                except Exception as e:
                    logger.debug(f"Error checking cache file {file_path}: {e}")
                    file_path.unlink(missing_ok=True)
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0

            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'memory_usage_percent': (self.memory_usage / self.max_memory_bytes) * 100,
                'memory_items': len(self.memory_cache),
                'disk_files': len(list(self.cache_dir.glob("*.cache.gz")))
            }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        stats = self.get_stats()

        return {
            'cache_dir': str(self.cache_dir),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'ttl_hours': self.ttl_seconds / 3600,
            'compression_level': self.compression_level,
            'statistics': stats
        }

class EmbeddingCache(OptimizedCache):
    """
    Specialized cache for embeddings with additional features
    """

    def __init__(self, cache_dir: Union[str, Path], **kwargs):
        super().__init__(cache_dir, **kwargs)
        self.embedding_stats = {
            'total_embeddings': 0,
            'embedding_dimensions': {},
            'models_used': set()
        }

    def put_embeddings(self, key: str, embeddings: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Store embeddings with metadata"""
        try:
            # Validate embeddings
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)

            # Ensure float32 for memory efficiency
            embeddings = embeddings.astype(np.float32)

            data = {
                'embeddings': embeddings,
                'metadata': metadata or {},
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype)
            }

            success = self.put(key, data)

            if success:
                # Update stats
                self.embedding_stats['total_embeddings'] += embeddings.shape[0]
                if len(embeddings.shape) > 1:
                    dim = embeddings.shape[1]
                    self.embedding_stats['embedding_dimensions'][dim] = self.embedding_stats['embedding_dimensions'].get(dim, 0) + embeddings.shape[0]

                if metadata and 'model' in metadata:
                    self.embedding_stats['models_used'].add(metadata['model'])

            return success

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False

    def get_embeddings(self, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve embeddings with metadata"""
        try:
            data = self.get(key)
            if data:
                return data['embeddings'], data['metadata']
            return None

        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return None

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding-specific statistics"""
        return {
            **self.embedding_stats,
            'models_used': list(self.embedding_stats['models_used']),
            'cache_stats': self.get_stats()
        }

def get_optimized_cache(cache_dir: Union[str, Path], cache_type: str = 'general', **kwargs) -> OptimizedCache:
    """
    Factory function to create optimized cache instances

    Args:
        cache_dir: Cache directory
        cache_type: Type of cache ('general', 'embeddings')
        **kwargs: Additional cache parameters

    Returns:
        Cache instance
    """
    if cache_type == 'embeddings':
        return EmbeddingCache(cache_dir, **kwargs)
    else:
        return OptimizedCache(cache_dir, **kwargs)
