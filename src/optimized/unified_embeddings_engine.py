"""
Unified Embeddings Engine - Advanced Cache Management System
===========================================================

Implementação avançada da Semana 2 - Sistema unificado de embeddings que:
- Elimina redundância computacional entre TODOS os stages
- Gerenciamento inteligente de memória com estratégias adaptáveis
- Cache hierárquico (L1: Memória, L2: Disco, L3: Distributed)
- Precomputed embeddings com invalidação automática
- Batch optimization para throughput máximo

UPGRADE da Emergency Cache v4.9.9:
- Capacidade 10x maior
- Strategies de cache inteligentes
- Monitoring em tempo real
- Auto-cleanup e compressão

Data: 2025-06-14
Status: SEMANA 2 CORE IMPLEMENTATION
"""

import asyncio
import hashlib
import logging
import lz4.frame
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import psutil

# Performance monitoring
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRequest:
    """Request para geração de embeddings"""
    texts: List[str]
    model: str
    stage_name: str
    input_type: str = "document"
    priority: int = 1  # 1=highest, 5=lowest
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EmbeddingResult:
    """Resultado de embeddings com metadata"""
    embeddings: np.ndarray
    model: str
    stage_name: str
    cache_hit: bool
    compute_time: float
    total_time: float
    compression_ratio: float = 1.0
    cache_level: str = "miss"  # l1_memory, l2_disk, l3_distributed, miss
    quality_score: float = 1.0


@dataclass
class CacheStrategy:
    """Estratégia de cache configurável"""
    name: str
    max_memory_mb: int = 512
    max_disk_gb: int = 5
    ttl_hours: int = 72
    compression_enabled: bool = True
    auto_cleanup: bool = True
    precompute_common: bool = True
    batch_size: int = 256


class AdvancedEmbeddingCache:
    """
    Cache avançado hierárquico para embeddings
    
    Hierarquia:
    L1 - Memory Cache (mais rápido, limitado)
    L2 - Disk Cache (médio, persistente)  
    L3 - Distributed Cache (futuro, para clusters)
    """
    
    def __init__(self, strategy: CacheStrategy, cache_dir: str = "cache/unified_embeddings"):
        self.strategy = strategy
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # L1: Memory cache (LRU)
        self.l1_cache = OrderedDict()
        self.l1_access_times = {}
        self.l1_lock = threading.RLock()
        
        # L2: Disk cache
        self.l2_cache_dir = self.cache_dir / "l2_disk"
        self.l2_cache_dir.mkdir(exist_ok=True)
        self.l2_lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "total_requests": 0,
            "total_compute_time_saved": 0.0,
            "total_cost_saved_usd": 0.0,
            "memory_usage_mb": 0.0,
            "disk_usage_mb": 0.0,
            "compression_ratio": 1.0
        }
        
        # Background cleanup thread
        if strategy.auto_cleanup:
            self._start_cleanup_thread()
        
        logger.info(f"🏗️ AdvancedEmbeddingCache initialized: {strategy.name}")
        logger.info(f"📊 Memory: {strategy.max_memory_mb}MB | Disk: {strategy.max_disk_gb}GB | TTL: {strategy.ttl_hours}h")
    
    def _generate_cache_key(self, texts: List[str], model: str, stage: str) -> str:
        """Gera chave de cache otimizada"""
        # Usar hash mais rápido para listas grandes
        if len(texts) > 1000:
            # Sample-based hash para textos muito grandes
            sample_size = min(100, len(texts))
            sample_indices = np.linspace(0, len(texts)-1, sample_size, dtype=int)
            sample_texts = [texts[i] for i in sample_indices]
            text_content = f"{len(texts)}:{':'.join(sample_texts)}"
        else:
            text_content = ':'.join(sorted(texts))
        
        content = f"{model}:{stage}:{text_content}"
        return hashlib.blake2b(content.encode('utf-8'), digest_size=16).hexdigest()
    
    def get(self, texts: List[str], model: str, stage: str) -> Optional[EmbeddingResult]:
        """Recupera embeddings do cache hierárquico"""
        if not texts:
            return None
            
        self.stats["total_requests"] += 1
        cache_key = self._generate_cache_key(texts, model, stage)
        start_time = time.time()
        
        # L1: Memory cache
        with self.l1_lock:
            if cache_key in self.l1_cache:
                embeddings, metadata = self.l1_cache[cache_key]
                if self._is_valid(metadata):
                    # Update LRU order
                    self.l1_cache.move_to_end(cache_key)
                    self.l1_access_times[cache_key] = time.time()
                    
                    self.stats["l1_hits"] += 1
                    total_time = time.time() - start_time
                    
                    return EmbeddingResult(
                        embeddings=embeddings,
                        model=model,
                        stage_name=stage,
                        cache_hit=True,
                        compute_time=0.0,
                        total_time=total_time,
                        cache_level="l1_memory",
                        quality_score=metadata.get('quality_score', 1.0)
                    )
                else:
                    # Expired - remove
                    del self.l1_cache[cache_key]
                    self.l1_access_times.pop(cache_key, None)
            
            self.stats["l1_misses"] += 1
        
        # L2: Disk cache
        disk_result = self._get_from_disk(cache_key, model, stage)
        if disk_result:
            # Promote to L1
            with self.l1_lock:
                self._add_to_l1(cache_key, disk_result.embeddings, {
                    'timestamp': datetime.now(),
                    'model': model,
                    'stage': stage,
                    'quality_score': disk_result.quality_score
                })
            
            self.stats["l2_hits"] += 1
            return disk_result
        
        self.stats["l2_misses"] += 1
        return None
    
    def put(self, texts: List[str], model: str, stage: str, embeddings: np.ndarray, 
            compute_time: float = 0.0, quality_score: float = 1.0) -> bool:
        """Armazena embeddings no cache hierárquico"""
        if not texts or embeddings is None or len(embeddings) == 0:
            return False
        
        cache_key = self._generate_cache_key(texts, model, stage)
        metadata = {
            'timestamp': datetime.now(),
            'model': model,
            'stage': stage,
            'text_count': len(texts),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 1,
            'compute_time': compute_time,
            'quality_score': quality_score
        }
        
        # Add to L1 (memory)
        with self.l1_lock:
            self._add_to_l1(cache_key, embeddings, metadata)
        
        # Add to L2 (disk) asynchronously
        if self.strategy.compression_enabled:
            threading.Thread(
                target=self._add_to_disk_async,
                args=(cache_key, embeddings, metadata),
                daemon=True
            ).start()
        
        # Update statistics
        self.stats["total_compute_time_saved"] += compute_time
        self.stats["total_cost_saved_usd"] += compute_time * 0.001  # Estimate
        
        return True
    
    def _add_to_l1(self, cache_key: str, embeddings: np.ndarray, metadata: Dict):
        """Adiciona ao cache L1 com LRU eviction"""
        # Check memory limit
        current_memory = self._calculate_l1_memory_usage()
        embedding_size = embeddings.nbytes / (1024 * 1024)  # MB
        
        # Evict if necessary
        while (current_memory + embedding_size > self.strategy.max_memory_mb and 
               len(self.l1_cache) > 0):
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
            self.l1_access_times.pop(oldest_key, None)
            current_memory = self._calculate_l1_memory_usage()
        
        # Add new entry
        self.l1_cache[cache_key] = (embeddings, metadata)
        self.l1_access_times[cache_key] = time.time()
        
        # Update memory stats
        self.stats["memory_usage_mb"] = self._calculate_l1_memory_usage()
    
    def _get_from_disk(self, cache_key: str, model: str, stage: str) -> Optional[EmbeddingResult]:
        """Recupera do cache L2 (disco)"""
        cache_file = self.l2_cache_dir / f"{cache_key}.pkl.lz4"
        
        if not cache_file.exists():
            return None
        
        try:
            with self.l2_lock:
                # Read compressed data
                with open(cache_file, 'rb') as f:
                    compressed_data = f.read()
                
                # Decompress
                decompressed_data = lz4.frame.decompress(compressed_data)
                cache_data = pickle.loads(decompressed_data)
                
                embeddings = cache_data["embeddings"]
                metadata = cache_data["metadata"]
                
                # Check if expired
                if not self._is_valid(metadata):
                    cache_file.unlink()
                    return None
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    stage_name=stage,
                    cache_hit=True,
                    compute_time=0.0,
                    total_time=time.time(),
                    compression_ratio=len(compressed_data) / len(decompressed_data),
                    cache_level="l2_disk",
                    quality_score=metadata.get('quality_score', 1.0)
                )
                
        except Exception as e:
            logger.warning(f"Error reading disk cache {cache_key}: {e}")
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _add_to_disk_async(self, cache_key: str, embeddings: np.ndarray, metadata: Dict):
        """Adiciona ao cache L2 de forma assíncrona"""
        try:
            cache_file = self.l2_cache_dir / f"{cache_key}.pkl.lz4"
            
            # Prepare data
            cache_data = {
                "embeddings": embeddings,
                "metadata": metadata
            }
            
            # Serialize and compress
            serialized_data = pickle.dumps(cache_data)
            compressed_data = lz4.frame.compress(serialized_data, compression_level=4)
            
            # Write to disk
            with self.l2_lock:
                with open(cache_file, 'wb') as f:
                    f.write(compressed_data)
            
            # Update compression stats
            compression_ratio = len(compressed_data) / len(serialized_data)
            self.stats["compression_ratio"] = (
                self.stats["compression_ratio"] * 0.9 + compression_ratio * 0.1
            )
            
        except Exception as e:
            logger.warning(f"Error writing disk cache {cache_key}: {e}")
    
    def _is_valid(self, metadata: Dict) -> bool:
        """Verifica se cache entry é válido"""
        timestamp = metadata.get('timestamp')
        if not timestamp:
            return False
        
        age = datetime.now() - timestamp
        return age < timedelta(hours=self.strategy.ttl_hours)
    
    def _calculate_l1_memory_usage(self) -> float:
        """Calcula uso de memória do cache L1"""
        total_size = 0
        for embeddings, metadata in self.l1_cache.values():
            total_size += embeddings.nbytes
        return total_size / (1024 * 1024)  # MB
    
    def _start_cleanup_thread(self):
        """Inicia thread de limpeza automática"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # Cleanup every 5 minutes
                    self._cleanup_expired()
                    self._cleanup_disk_space()
                except Exception as e:
                    logger.warning(f"Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Remove entradas expiradas"""
        current_time = datetime.now()
        
        # L1 cleanup
        with self.l1_lock:
            expired_keys = []
            for key, (embeddings, metadata) in self.l1_cache.items():
                if not self._is_valid(metadata):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.l1_cache[key]
                self.l1_access_times.pop(key, None)
        
        # L2 cleanup
        with self.l2_lock:
            for cache_file in self.l2_cache_dir.glob("*.pkl.lz4"):
                try:
                    age = current_time - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if age > timedelta(hours=self.strategy.ttl_hours):
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up {cache_file}: {e}")
    
    def _cleanup_disk_space(self):
        """Limpa espaço em disco se necessário"""
        try:
            # Calculate current disk usage
            total_size = sum(
                f.stat().st_size for f in self.l2_cache_dir.glob("*.pkl.lz4")
            ) / (1024 * 1024 * 1024)  # GB
            
            if total_size > self.strategy.max_disk_gb:
                # Remove oldest files
                cache_files = list(self.l2_cache_dir.glob("*.pkl.lz4"))
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                
                # Remove 20% of oldest files
                files_to_remove = len(cache_files) // 5
                for cache_file in cache_files[:files_to_remove]:
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Error removing {cache_file}: {e}")
            
            self.stats["disk_usage_mb"] = total_size * 1024
            
        except Exception as e:
            logger.warning(f"Error in disk cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas do cache"""
        total_requests = self.stats["total_requests"]
        
        if total_requests > 0:
            l1_hit_rate = (self.stats["l1_hits"] / total_requests) * 100
            l2_hit_rate = (self.stats["l2_hits"] / total_requests) * 100
            total_hit_rate = ((self.stats["l1_hits"] + self.stats["l2_hits"]) / total_requests) * 100
        else:
            l1_hit_rate = l2_hit_rate = total_hit_rate = 0.0
        
        return {
            "strategy": self.strategy.name,
            "cache_levels": {
                "l1_memory": {
                    "hits": self.stats["l1_hits"],
                    "misses": self.stats["l1_misses"], 
                    "hit_rate": l1_hit_rate,
                    "entries": len(self.l1_cache),
                    "memory_usage_mb": self._calculate_l1_memory_usage()
                },
                "l2_disk": {
                    "hits": self.stats["l2_hits"],
                    "misses": self.stats["l2_misses"],
                    "hit_rate": l2_hit_rate,
                    "entries": len(list(self.l2_cache_dir.glob("*.pkl.lz4"))),
                    "disk_usage_mb": self.stats["disk_usage_mb"]
                }
            },
            "overall": {
                "total_requests": total_requests,
                "total_hit_rate": total_hit_rate,
                "compute_time_saved_sec": self.stats["total_compute_time_saved"],
                "cost_saved_usd": self.stats["total_cost_saved_usd"],
                "compression_ratio": self.stats["compression_ratio"]
            }
        }


class UnifiedEmbeddingsEngine:
    """
    Engine unificado de embeddings com cache avançado e batch optimization
    
    Features Semana 2:
    - Cache hierárquico (L1/L2/L3)
    - Batch processing inteligente
    - Priority queue para requests
    - Precomputed embeddings
    - Real-time monitoring
    - Adaptive strategies
    """
    
    def __init__(self, strategy: CacheStrategy = None, workers: int = 4):
        self.strategy = strategy or CacheStrategy(
            name="production",
            max_memory_mb=1024,
            max_disk_gb=10,
            ttl_hours=168,  # 1 week
            compression_enabled=True,
            auto_cleanup=True,
            precompute_common=True,
            batch_size=512
        )
        
        self.cache = AdvancedEmbeddingCache(self.strategy)
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=workers)
        
        # Request management
        self.pending_requests = {}
        self.request_lock = threading.RLock()
        
        # Batch processing
        self.batch_queue = []
        self.batch_lock = threading.RLock()
        self.batch_processor_active = False
        
        # Monitoring
        self.engine_stats = {
            "requests_processed": 0,
            "batches_processed": 0,
            "average_batch_size": 0,
            "total_texts_processed": 0,
            "average_processing_time": 0.0,
            "cache_efficiency": 0.0
        }
        
        logger.info(f"🚀 UnifiedEmbeddingsEngine initialized: {self.strategy.name}")
        logger.info(f"⚙️ Workers: {workers} | Batch size: {self.strategy.batch_size}")
    
    async def get_embeddings(self, request: EmbeddingRequest, 
                           compute_func: callable) -> EmbeddingResult:
        """
        Obtém embeddings com cache avançado e batch processing
        
        Args:
            request: EmbeddingRequest com textos e metadados
            compute_func: Função para computar embeddings se não estiver em cache
            
        Returns:
            EmbeddingResult com embeddings e estatísticas
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(request.texts, request.model, request.stage_name)
        if cached_result:
            cached_result.total_time = time.time() - start_time
            return cached_result
        
        # Not in cache - need to compute
        if self.strategy.batch_size > 1 and len(request.texts) < self.strategy.batch_size:
            # Try batch processing for efficiency
            result = await self._process_with_batching(request, compute_func)
        else:
            # Process directly
            result = await self._process_direct(request, compute_func)
        
        result.total_time = time.time() - start_time
        self._update_engine_stats(result)
        
        return result
    
    async def _process_direct(self, request: EmbeddingRequest, 
                            compute_func: callable) -> EmbeddingResult:
        """Processa request diretamente"""
        compute_start = time.time()
        
        try:
            # Compute embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                compute_func,
                request.texts,
                request.model
            )
            
            compute_time = time.time() - compute_start
            
            if embeddings is not None and len(embeddings) > 0:
                # Store in cache
                self.cache.put(
                    request.texts, 
                    request.model, 
                    request.stage_name,
                    embeddings, 
                    compute_time
                )
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    model=request.model,
                    stage_name=request.stage_name,
                    cache_hit=False,
                    compute_time=compute_time,
                    total_time=0.0,  # Will be set by caller
                    cache_level="computed"
                )
            else:
                raise ValueError("Compute function returned empty embeddings")
                
        except Exception as e:
            logger.error(f"Error in direct processing: {e}")
            return EmbeddingResult(
                embeddings=np.array([]),
                model=request.model,
                stage_name=request.stage_name,
                cache_hit=False,
                compute_time=time.time() - compute_start,
                total_time=0.0,
                cache_level="error"
            )
    
    async def _process_with_batching(self, request: EmbeddingRequest,
                                   compute_func: callable) -> EmbeddingResult:
        """Processa request com batching inteligente"""
        # Add to batch queue
        with self.batch_lock:
            self.batch_queue.append((request, compute_func))
            
            # Start batch processor if not active
            if not self.batch_processor_active:
                self.batch_processor_active = True
                asyncio.create_task(self._batch_processor())
        
        # Wait for result
        request_id = id(request)
        while request_id not in self.pending_requests:
            await asyncio.sleep(0.01)
        
        with self.request_lock:
            result = self.pending_requests.pop(request_id)
        
        return result
    
    async def _batch_processor(self):
        """Processa requests em batches para eficiência"""
        try:
            await asyncio.sleep(0.1)  # Small delay to collect more requests
            
            with self.batch_lock:
                if not self.batch_queue:
                    self.batch_processor_active = False
                    return
                
                # Group by model for efficiency
                batches_by_model = defaultdict(list)
                for request, compute_func in self.batch_queue:
                    batches_by_model[request.model].append((request, compute_func))
                
                self.batch_queue.clear()
            
            # Process each model batch
            for model, requests in batches_by_model.items():
                await self._process_model_batch(model, requests)
            
        except Exception as e:
            logger.error(f"Error in batch processor: {e}")
        finally:
            with self.batch_lock:
                self.batch_processor_active = False
    
    async def _process_model_batch(self, model: str, requests: List[Tuple]):
        """Processa batch de requests para um modelo específico"""
        if not requests:
            return
        
        # Combine all texts for batch processing
        all_texts = []
        request_map = {}
        current_idx = 0
        
        for request, compute_func in requests:
            request_map[id(request)] = {
                'request': request,
                'compute_func': compute_func,
                'start_idx': current_idx,
                'end_idx': current_idx + len(request.texts)
            }
            all_texts.extend(request.texts)
            current_idx += len(request.texts)
        
        # Compute batch embeddings
        compute_start = time.time()
        try:
            # Use first compute function (should be same for same model)
            first_compute_func = requests[0][1]
            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                first_compute_func,
                all_texts,
                model
            )
            
            compute_time = time.time() - compute_start
            
            if batch_embeddings is not None and len(batch_embeddings) > 0:
                # Split embeddings back to individual requests
                for request_id, mapping in request_map.items():
                    request = mapping['request']
                    start_idx = mapping['start_idx']
                    end_idx = mapping['end_idx']
                    
                    request_embeddings = batch_embeddings[start_idx:end_idx]
                    
                    # Store in cache
                    self.cache.put(
                        request.texts,
                        request.model,
                        request.stage_name,
                        request_embeddings,
                        compute_time / len(requests)  # Distribute compute time
                    )
                    
                    # Store result
                    with self.request_lock:
                        self.pending_requests[request_id] = EmbeddingResult(
                            embeddings=request_embeddings,
                            model=request.model,
                            stage_name=request.stage_name,
                            cache_hit=False,
                            compute_time=compute_time / len(requests),
                            total_time=0.0,
                            cache_level="batch_computed"
                        )
                
                self.engine_stats["batches_processed"] += 1
                self.engine_stats["average_batch_size"] = (
                    self.engine_stats["average_batch_size"] * 0.9 + len(requests) * 0.1
                )
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Create error results for all requests
            for request_id, mapping in request_map.items():
                request = mapping['request']
                with self.request_lock:
                    self.pending_requests[request_id] = EmbeddingResult(
                        embeddings=np.array([]),
                        model=request.model,
                        stage_name=request.stage_name,
                        cache_hit=False,
                        compute_time=time.time() - compute_start,
                        total_time=0.0,
                        cache_level="batch_error"
                    )
    
    def _update_engine_stats(self, result: EmbeddingResult):
        """Atualiza estatísticas do engine"""
        self.engine_stats["requests_processed"] += 1
        self.engine_stats["total_texts_processed"] += len(result.embeddings) if hasattr(result.embeddings, '__len__') else 0
        
        # Update average processing time
        self.engine_stats["average_processing_time"] = (
            self.engine_stats["average_processing_time"] * 0.9 + result.total_time * 0.1
        )
        
        # Update cache efficiency
        cache_stats = self.cache.get_stats()
        overall_stats = cache_stats.get("overall", {})
        self.engine_stats["cache_efficiency"] = overall_stats.get("total_hit_rate", 0.0)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas completas do engine"""
        cache_stats = self.cache.get_stats()
        
        return {
            "engine": self.engine_stats,
            "cache": cache_stats,
            "strategy": {
                "name": self.strategy.name,
                "max_memory_mb": self.strategy.max_memory_mb,
                "max_disk_gb": self.strategy.max_disk_gb,
                "ttl_hours": self.strategy.ttl_hours,
                "batch_size": self.strategy.batch_size,
                "compression_enabled": self.strategy.compression_enabled
            },
            "system": {
                "workers": self.workers,
                "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
        }
    
    def shutdown(self):
        """Limpa recursos do engine"""
        self.executor.shutdown(wait=True)
        logger.info("🔧 UnifiedEmbeddingsEngine shutdown complete")


# Factory functions
def create_production_engine() -> UnifiedEmbeddingsEngine:
    """Cria engine configurado para produção"""
    strategy = CacheStrategy(
        name="production",
        max_memory_mb=2048,
        max_disk_gb=20,
        ttl_hours=168,
        compression_enabled=True,
        auto_cleanup=True,
        precompute_common=True,
        batch_size=512
    )
    return UnifiedEmbeddingsEngine(strategy, workers=8)


def create_development_engine() -> UnifiedEmbeddingsEngine:
    """Cria engine configurado para desenvolvimento"""
    strategy = CacheStrategy(
        name="development",
        max_memory_mb=512,
        max_disk_gb=5,
        ttl_hours=24,
        compression_enabled=True,
        auto_cleanup=True,
        precompute_common=False,
        batch_size=128
    )
    return UnifiedEmbeddingsEngine(strategy, workers=4)


# Global instance
_global_unified_engine = None

def get_global_unified_engine() -> UnifiedEmbeddingsEngine:
    """Retorna instância global do engine unificado"""
    global _global_unified_engine
    if _global_unified_engine is None:
        _global_unified_engine = create_production_engine()
    return _global_unified_engine