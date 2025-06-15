"""
Unified Cache Framework - Academic Research Optimization
======================================================

Consolidates all cache implementations into a single, comprehensive framework
optimized for academic research with 4GB memory targets and cost efficiency.

CONSOLIDATES:
- EmergencyEmbeddingsCache (src/optimized/emergency_embeddings.py)
- SmartSemanticCache (src/optimized/smart_claude_cache.py) 
- UnifiedCacheSystem (src/core/unified_cache_system.py)
- OptimizedCache (src/anthropic_integration/optimized_cache.py)

ACADEMIC FEATURES:
- Portuguese text normalization for better cache hits
- Cost-aware caching for research budgets
- Memory-efficient storage for 4GB environments
- Research reproducibility through consistent caching

Author: Academic Research Optimization Team
Date: 2025-06-15
Status: CORE CONSOLIDATION IMPLEMENTATION
"""

import hashlib
import json
import logging
import lz4.frame
import pickle
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

# ========================================
# UNIFIED DATA STRUCTURES
# ========================================

@dataclass
class CacheEntry:
    """Unified cache entry with comprehensive metadata"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    cache_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        expires_at = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expires_at

@dataclass 
class CacheStats:
    """Comprehensive cache statistics for academic monitoring"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    entries_count: int = 0
    expired_entries: int = 0
    cost_savings_usd: float = 0.0
    compute_time_saved_seconds: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100.0

# ========================================
# ABSTRACT BASE CLASSES
# ========================================

class CacheKeyGenerator(ABC):
    """Abstract base for cache key generation strategies"""
    
    @abstractmethod
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from inputs"""
        pass

class CacheBackend(ABC):
    """Abstract base for cache storage backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all entries, return count deleted"""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys"""
        pass

# ========================================
# KEY GENERATORS
# ========================================

class HashKeyGenerator(CacheKeyGenerator):
    """Hash-based key generation for content-dependent caching"""
    
    def __init__(self, algorithm: str = "md5", include_brazilian_portuguese_normalization: bool = True):
        self.algorithm = algorithm
        self.include_brazilian_portuguese_normalization = include_brazilian_portuguese_normalization
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate hash-based cache key"""
        # Convert all inputs to string representation
        content_parts = []
        
        # Process positional arguments
        for arg in args:
            if isinstance(arg, (list, tuple)):
                # Handle text lists (common for embeddings)
                if self.include_brazilian_portuguese_normalization:
                    normalized_texts = [self._normalize_brazilian_portuguese_text(str(item)) for item in arg]
                    content_parts.extend(sorted(normalized_texts))
                else:
                    content_parts.extend([str(item) for item in arg])
            else:
                content_parts.append(str(arg))
        
        # Process keyword arguments
        for key, value in sorted(kwargs.items()):
            content_parts.append(f"{key}:{value}")
        
        # Create content string
        content = "|".join(content_parts)
        
        # Generate hash
        if self.algorithm == "md5":
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        elif self.algorithm == "sha256":
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.algorithm}")
    
    def _normalize_brazilian_portuguese_text(self, text: str) -> str:
        """Normalize Brazilian Portuguese text for better cache hits in Brazilian political research"""
        if not isinstance(text, str):
            return str(text)
        
        # Academic normalization for Brazilian political discourse
        text = text.lower().strip()
        
        # Normalize common Brazilian political terms for better caching
        political_normalizations = {
            'bolsonaro': 'political_figure_1',
            'lula': 'political_figure_2', 
            'dilma': 'political_figure_3',
            'pt': 'political_party_1',
            'psdb': 'political_party_2',
            'mdb': 'political_party_3'
        }
        
        for original, normalized in political_normalizations.items():
            text = text.replace(original, normalized)
        
        return text

class DirectKeyGenerator(CacheKeyGenerator):
    """Direct string key generation for simple caching"""
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate direct string key"""
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:
            return args[0]
        
        # Fallback to hash generation for complex inputs
        content_parts = [str(arg) for arg in args]
        for key, value in sorted(kwargs.items()):
            content_parts.append(f"{key}:{value}")
        
        return "_".join(content_parts)

# ========================================
# STORAGE BACKENDS  
# ========================================

class MemoryBackend(CacheBackend):
    """In-memory cache backend with LRU eviction"""
    
    def __init__(self, max_entries: int = 10000, max_memory_mb: float = 500.0):
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry with LRU update"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                return None
            
            # Update LRU order
            self._cache.move_to_end(key)
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            return entry
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry with memory management"""
        with self._lock:
            # Calculate memory usage
            current_memory = self._calculate_memory_usage()
            
            # Evict if necessary
            while (len(self._cache) >= self.max_entries or 
                   current_memory > self.max_memory_mb):
                if not self._cache:
                    break
                self._evict_lru()
                current_memory = self._calculate_memory_usage()
            
            self._cache[key] = entry
            return True
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """Clear all entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self._lock:
            return list(self._cache.keys())
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self._cache:
            self._cache.popitem(last=False)
    
    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB"""
        total_bytes = 0
        for entry in self._cache.values():
            try:
                # Rough size estimation
                if hasattr(entry.value, 'nbytes'):  # numpy arrays
                    total_bytes += entry.value.nbytes
                else:
                    total_bytes += len(pickle.dumps(entry.value))
            except:
                total_bytes += 1024  # Default 1KB estimate
        
        return total_bytes / (1024 * 1024)  # Convert to MB

class DiskBackend(CacheBackend):
    """Disk-based cache backend with compression"""
    
    def __init__(self, cache_dir: str = "cache/unified", 
                 use_compression: bool = True,
                 max_disk_mb: float = 1000.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression
        self.max_disk_mb = max_disk_mb
        self._lock = threading.RLock()
        
        # Index file for metadata
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from disk"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            try:
                # Load entry from disk
                with open(file_path, 'rb') as f:
                    if self.use_compression:
                        data = lz4.frame.decompress(f.read())
                        entry = pickle.loads(data)
                    else:
                        entry = pickle.load(f)
                
                # Check expiration
                if entry.is_expired:
                    self.delete(key)
                    return None
                
                # Update access info
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                return entry
                
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry to disk"""
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                
                # Serialize entry
                data = pickle.dumps(entry)
                if self.use_compression:
                    data = lz4.frame.compress(data)
                
                # Write to disk
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update index
                self._update_index(key, entry)
                
                # Check disk usage
                self._manage_disk_usage()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry from disk"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                try:
                    file_path.unlink()
                    self._remove_from_index(key)
                    return True
                except Exception as e:
                    logger.warning(f"Failed to delete cache entry {key}: {e}")
            
            return False
    
    def clear(self) -> int:
        """Clear all cache entries"""
        with self._lock:
            count = 0
            for file_path in self.cache_dir.glob("cache_*.pkl*"):
                try:
                    file_path.unlink()
                    count += 1
                except:
                    pass
            
            # Clear index
            self.index = {}
            self._save_index()
            
            return count
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self._lock:
            return list(self.index.keys())
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        extension = ".pkl.lz4" if self.use_compression else ".pkl"
        return self.cache_dir / f"cache_{safe_key}{extension}"
    
    def _load_index(self) -> None:
        """Load cache index from disk"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            self.index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _update_index(self, key: str, entry: CacheEntry) -> None:
        """Update cache index with entry info"""
        self.index[key] = {
            'created_at': entry.created_at.isoformat(),
            'size_bytes': entry.size_bytes,
            'cache_type': entry.cache_type
        }
        self._save_index()
    
    def _remove_from_index(self, key: str) -> None:
        """Remove key from index"""
        if key in self.index:
            del self.index[key]
            self._save_index()
    
    def _manage_disk_usage(self) -> None:
        """Manage disk usage by removing old entries"""
        try:
            total_size = sum(f.stat().st_size 
                           for f in self.cache_dir.rglob("cache_*") 
                           if f.is_file())
            
            total_mb = total_size / (1024 * 1024)
            
            if total_mb > self.max_disk_mb:
                # Remove oldest files
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob("cache_*.pkl*")]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                for file_path, _ in files[:len(files)//4]:  # Remove 25% oldest
                    try:
                        file_path.unlink()
                    except:
                        pass
                
                # Rebuild index
                self._rebuild_index()
                
        except Exception as e:
            logger.warning(f"Failed to manage disk usage: {e}")
    
    def _rebuild_index(self) -> None:
        """Rebuild index from existing files"""
        self.index = {}
        for file_path in self.cache_dir.glob("cache_*.pkl*"):
            try:
                # Extract key from filename and update index
                # This is a simplified rebuild - in production would be more sophisticated
                key = file_path.stem.replace("cache_", "")
                self.index[key] = {
                    'created_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'size_bytes': file_path.stat().st_size,
                    'cache_type': 'unknown'
                }
            except:
                continue
        
        self._save_index()

# ========================================
# UNIFIED CACHE SYSTEM
# ========================================

class UnifiedCacheFramework:
    """
    Comprehensive cache framework consolidating all cache implementations
    Optimized for academic research with Portuguese text processing
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/unified",
                 memory_backend_mb: float = 200.0,
                 disk_backend_mb: float = 800.0,
                 default_ttl_hours: int = 24,
                 enable_brazilian_portuguese_optimization: bool = True,
                 academic_mode: bool = True):
        
        self.cache_dir = cache_dir
        self.default_ttl_hours = default_ttl_hours
        self.enable_brazilian_portuguese_optimization = enable_brazilian_portuguese_optimization
        self.academic_mode = academic_mode
        
        # Initialize backends
        self.memory_backend = MemoryBackend(
            max_entries=10000,
            max_memory_mb=memory_backend_mb
        )
        
        self.disk_backend = DiskBackend(
            cache_dir=cache_dir,
            use_compression=True,
            max_disk_mb=disk_backend_mb
        )
        
        # Initialize key generators
        self.hash_generator = HashKeyGenerator(
            algorithm="md5",
            include_brazilian_portuguese_normalization=enable_brazilian_portuguese_optimization
        )
        
        self.direct_generator = DirectKeyGenerator()
        
        # Statistics tracking
        self.stats = CacheStats()
        self._stats_lock = threading.RLock()
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._start_cleanup_thread()
        
        logger.info(f"🚀 UnifiedCacheFramework initialized")
        logger.info(f"📁 Cache directory: {cache_dir}")
        logger.info(f"🧠 Memory limit: {memory_backend_mb}MB")
        logger.info(f"💾 Disk limit: {disk_backend_mb}MB")
        logger.info(f"🇧🇷 Brazilian Portuguese optimization: {enable_brazilian_portuguese_optimization}")
    
    def get(self, *args, key_type: str = "hash", **kwargs) -> Optional[Any]:
        """
        Get value from cache using unified interface
        
        Args:
            *args: Arguments for key generation
            key_type: "hash" or "direct" key generation
            **kwargs: Additional arguments for key generation
            
        Returns:
            Cached value or None if not found
        """
        # Generate cache key
        if key_type == "hash":
            cache_key = self.hash_generator.generate_key(*args, **kwargs)
        else:
            cache_key = self.direct_generator.generate_key(*args, **kwargs)
        
        with self._stats_lock:
            self.stats.total_requests += 1
        
        # Try memory cache first (L1)
        entry = self.memory_backend.get(cache_key)
        if entry is not None:
            with self._stats_lock:
                self.stats.hits += 1
            return entry.value
        
        # Try disk cache (L2)
        entry = self.disk_backend.get(cache_key)
        if entry is not None:
            # Promote to memory cache
            self.memory_backend.set(cache_key, entry)
            with self._stats_lock:
                self.stats.hits += 1
            return entry.value
        
        # Cache miss
        with self._stats_lock:
            self.stats.misses += 1
        
        return None
    
    def set(self, *args, value: Any, 
            ttl_hours: Optional[int] = None,
            key_type: str = "hash",
            cache_type: str = "general",
            cost_saved_usd: float = 0.0,
            compute_time_saved: float = 0.0,
            **kwargs) -> bool:
        """
        Set value in cache using unified interface
        
        Args:
            *args: Arguments for key generation  
            value: Value to cache
            ttl_hours: Time to live in hours (None for default)
            key_type: "hash" or "direct" key generation
            cache_type: Type of cache entry for statistics
            cost_saved_usd: Cost savings for academic budget tracking
            compute_time_saved: Compute time saved in seconds
            **kwargs: Additional arguments for key generation
            
        Returns:
            True if successfully cached
        """
        # Generate cache key
        if key_type == "hash":
            cache_key = self.hash_generator.generate_key(*args, **kwargs)
        else:
            cache_key = self.direct_generator.generate_key(*args, **kwargs)
        
        # Calculate TTL
        if ttl_hours is None:
            ttl_hours = self.default_ttl_hours
        ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None
        
        # Calculate size
        try:
            if hasattr(value, 'nbytes'):  # numpy arrays
                size_bytes = value.nbytes
            else:
                size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes,
            cache_type=cache_type,
            metadata={
                'cost_saved_usd': cost_saved_usd,
                'compute_time_saved': compute_time_saved,
                'brazilian_portuguese_optimized': self.enable_brazilian_portuguese_optimization,
                'academic_mode': self.academic_mode
            }
        )
        
        # Save to both backends
        memory_success = self.memory_backend.set(cache_key, entry)
        disk_success = self.disk_backend.set(cache_key, entry)
        
        # Update statistics
        if memory_success or disk_success:
            with self._stats_lock:
                self.stats.cost_savings_usd += cost_saved_usd
                self.stats.compute_time_saved_seconds += compute_time_saved
        
        return memory_success or disk_success
    
    def get_or_compute(self, compute_func: Callable, *args, 
                       ttl_hours: Optional[int] = None,
                       key_type: str = "hash",
                       cache_type: str = "general",
                       **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Get from cache or compute if not found (cache-or-compute pattern)
        
        Args:
            compute_func: Function to call if cache miss
            *args: Arguments for key generation and compute function
            ttl_hours: Time to live in hours
            key_type: Key generation type
            cache_type: Cache entry type
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (result, metadata) where metadata includes cache info
        """
        start_time = time.time()
        
        # Try cache first
        cached_result = self.get(*args, key_type=key_type, **kwargs)
        
        if cached_result is not None:
            compute_time = time.time() - start_time
            return cached_result, {
                'cache_hit': True,
                'compute_time': compute_time,
                'cost_saved': True,
                'source': 'cache'
            }
        
        # Compute result
        try:
            result = compute_func(*args, **kwargs)
            compute_time = time.time() - start_time
            
            # Cache the result
            # Estimate cost savings (academic research benefit)
            if cache_type == "embeddings":
                cost_saved = 0.002  # Rough estimate for Voyage.ai call
            elif cache_type == "api_response":
                cost_saved = 0.01   # Rough estimate for Claude API call
            else:
                cost_saved = 0.001  # General compute cost
            
            self.set(*args, value=result, 
                    ttl_hours=ttl_hours,
                    key_type=key_type,
                    cache_type=cache_type,
                    cost_saved_usd=cost_saved,
                    compute_time_saved=compute_time,
                    **kwargs)
            
            return result, {
                'cache_hit': False,
                'compute_time': compute_time,
                'cost_saved': False,
                'source': 'computed'
            }
            
        except Exception as e:
            logger.error(f"Failed to compute result: {e}")
            raise
    
    def invalidate(self, *args, key_type: str = "hash", **kwargs) -> bool:
        """Invalidate specific cache entry"""
        if key_type == "hash":
            cache_key = self.hash_generator.generate_key(*args, **kwargs)
        else:
            cache_key = self.direct_generator.generate_key(*args, **kwargs)
        
        memory_deleted = self.memory_backend.delete(cache_key)
        disk_deleted = self.disk_backend.delete(cache_key)
        
        return memory_deleted or disk_deleted
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        memory_keys = self.memory_backend.keys()
        disk_keys = self.disk_backend.keys()
        all_keys = set(memory_keys + disk_keys)
        
        deleted_count = 0
        
        for key in all_keys:
            if pattern in key or (pattern.endswith('*') and key.startswith(pattern[:-1])):
                memory_deleted = self.memory_backend.delete(key)
                disk_deleted = self.disk_backend.delete(key)
                if memory_deleted or disk_deleted:
                    deleted_count += 1
        
        return deleted_count
    
    def clear(self) -> Dict[str, int]:
        """Clear all cache entries"""
        memory_count = self.memory_backend.clear()
        disk_count = self.disk_backend.clear()
        
        # Reset statistics
        with self._stats_lock:
            self.stats = CacheStats()
        
        return {
            'memory_entries_cleared': memory_count,
            'disk_entries_cleared': disk_count,
            'total_cleared': memory_count + disk_count
        }
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired cache entries"""
        memory_keys = self.memory_backend.keys()
        disk_keys = self.disk_backend.keys()
        all_keys = set(memory_keys + disk_keys)
        
        expired_count = 0
        
        for key in all_keys:
            # Check memory backend
            entry = self.memory_backend.get(key)
            if entry and entry.is_expired:
                self.memory_backend.delete(key)
                expired_count += 1
            
            # Check disk backend
            entry = self.disk_backend.get(key)
            if entry and entry.is_expired:
                self.disk_backend.delete(key)
                expired_count += 1
        
        with self._stats_lock:
            self.stats.expired_entries += expired_count
        
        return {
            'expired_entries_removed': expired_count
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for academic monitoring"""
        with self._stats_lock:
            current_stats = CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                total_requests=self.stats.total_requests,
                memory_usage_mb=self.memory_backend._calculate_memory_usage(),
                entries_count=len(self.memory_backend.keys()) + len(self.disk_backend.keys()),
                expired_entries=self.stats.expired_entries,
                cost_savings_usd=self.stats.cost_savings_usd,
                compute_time_saved_seconds=self.stats.compute_time_saved_seconds
            )
        
        return {
            'performance': {
                'hit_rate_percent': current_stats.hit_rate,
                'total_requests': current_stats.total_requests,
                'cache_hits': current_stats.hits,
                'cache_misses': current_stats.misses
            },
            'resource_usage': {
                'memory_usage_mb': current_stats.memory_usage_mb,
                'disk_usage_mb': 0.0,  # Would need disk backend method
                'total_entries': current_stats.entries_count
            },
            'academic_benefits': {
                'cost_savings_usd': current_stats.cost_savings_usd,
                'compute_time_saved_hours': current_stats.compute_time_saved_seconds / 3600,
                'portuguese_optimization_enabled': self.enable_portuguese_optimization,
                'academic_mode': self.academic_mode
            },
            'maintenance': {
                'expired_entries_cleaned': current_stats.expired_entries
            }
        }
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self.cleanup_expired()
                except Exception as e:
                    logger.warning(f"Background cleanup failed: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

# ========================================
# SPECIALIZED CACHE IMPLEMENTATIONS
# ========================================

class EmbeddingsCacheSystem(UnifiedCacheFramework):
    """Specialized cache for embeddings with numpy array optimization"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('cache_dir', 'cache/embeddings')
        kwargs.setdefault('default_ttl_hours', 48)  # Longer TTL for expensive embeddings
        super().__init__(**kwargs)
    
    def get_embeddings(self, texts: List[str], model: str = "voyage-3.5-lite") -> Optional[np.ndarray]:
        """Get embeddings with model-specific caching"""
        return self.get(texts, model=model, cache_type="embeddings", key_type="hash")
    
    def save_embeddings(self, texts: List[str], embeddings: np.ndarray, 
                       model: str = "voyage-3.5-lite", 
                       cost_saved_usd: float = 0.002) -> bool:
        """Save embeddings with academic cost tracking"""
        return self.set(texts, model=model, value=embeddings,
                       cache_type="embeddings",
                       cost_saved_usd=cost_saved_usd,
                       key_type="hash")

class APICacheSystem(UnifiedCacheFramework):
    """Specialized cache for API responses with semantic similarity"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('cache_dir', 'cache/api_responses')
        kwargs.setdefault('default_ttl_hours', 24)
        super().__init__(**kwargs)
    
    def get_api_response(self, prompt: str, stage: str, model: str = "claude-3-5-haiku-20241022") -> Optional[str]:
        """Get API response with stage and model awareness"""
        return self.get(prompt, stage=stage, model=model, cache_type="api_response", key_type="hash")
    
    def save_api_response(self, prompt: str, response: str, stage: str,
                         model: str = "claude-3-5-haiku-20241022",
                         cost_saved_usd: float = 0.01) -> bool:
        """Save API response with academic cost tracking"""
        return self.set(prompt, stage=stage, model=model, value=response,
                       cache_type="api_response",
                       cost_saved_usd=cost_saved_usd,
                       key_type="hash")

# ========================================
# FACTORY FUNCTIONS
# ========================================

def get_unified_cache(cache_type: str = "general", **kwargs) -> UnifiedCacheFramework:
    """Factory function to get appropriate cache instance"""
    if cache_type == "embeddings":
        return EmbeddingsCacheSystem(**kwargs)
    elif cache_type == "api":
        return APICacheSystem(**kwargs)
    else:
        return UnifiedCacheFramework(**kwargs)

# Global cache instances for academic research
_global_caches = {}

def get_academic_cache(cache_type: str = "general") -> UnifiedCacheFramework:
    """Get global cache instance optimized for academic research"""
    if cache_type not in _global_caches:
        _global_caches[cache_type] = get_unified_cache(
            cache_type=cache_type,
            academic_mode=True,
            enable_portuguese_optimization=True,
            memory_backend_mb=150.0,  # Conservative for 4GB target
            disk_backend_mb=500.0,
            default_ttl_hours=24
        )
    
    return _global_caches[cache_type]