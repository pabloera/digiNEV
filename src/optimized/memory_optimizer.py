"""
Memory profiler and optimizer for production pipeline execution.

Provides memory monitoring, garbage collection optimization, and adaptive
memory management to maintain target memory usage under 4GB.
"""

import gc
import logging
import psutil
import threading
import time
import tracemalloc
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

# Suppress pandas warnings for memory optimization
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Snapshot de uso de memória em um momento específico"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_gb: float
    stage_name: str = ""
    operation_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryLeak:
    """Detecção de vazamento de memória"""
    detection_time: datetime
    stage_name: str
    memory_growth_mb: float
    growth_rate_mb_per_min: float
    severity: str  # low, medium, high, critical
    recommended_action: str
    stack_trace: Optional[str] = None

@dataclass
class GCStats:
    """Estatísticas de garbage collection"""
    collections_count: Dict[int, int]
    collected_objects: Dict[int, int]
    uncollectable_objects: Dict[int, int]
    total_time_seconds: float
    efficiency_score: float

class MemoryProfiler:
    """Profiler detalhado de memória por stage"""
    
    def __init__(self, sampling_interval: float = 1.0, history_limit: int = 10000):
        self.sampling_interval = sampling_interval
        self.history_limit = history_limit
        
        # Memory tracking
        self.memory_snapshots = deque(maxlen=history_limit)
        self.stage_memory_usage = defaultdict(list)
        self.peak_memory_by_stage = {}
        
        # Profiling state
        self.is_profiling = False
        self.current_stage = "idle"
        self.profiling_thread = None
        
        # System info
        self.process = psutil.Process()
        self.system_memory = psutil.virtual_memory()
        
        logger.info("🧠 MemoryProfiler initialized")
    
    def start_profiling(self, enable_tracemalloc: bool = True):
        """Inicia profiling de memória"""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        
        # Enable tracemalloc for detailed tracking
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("🔍 Tracemalloc enabled for detailed memory tracking")
        
        # Start profiling thread
        self.profiling_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profiling_thread.start()
        
        logger.info("🧠 Memory profiling started")
    
    def stop_profiling(self):
        """Para profiling de memória"""
        if not self.is_profiling:
            return
        
        self.is_profiling = False
        
        if self.profiling_thread:
            self.profiling_thread.join(timeout=2)
        
        # Stop tracemalloc if we started it
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logger.info("🧠 Memory profiling stopped")
    
    def _profiling_loop(self):
        """Loop principal de profiling"""
        while self.is_profiling:
            try:
                snapshot = self._capture_memory_snapshot()
                self.memory_snapshots.append(snapshot)
                
                # Update stage-specific tracking
                if snapshot.stage_name:
                    self.stage_memory_usage[snapshot.stage_name].append(snapshot.rss_mb)
                    
                    # Track peak memory for stage
                    current_peak = self.peak_memory_by_stage.get(snapshot.stage_name, 0)
                    if snapshot.rss_mb > current_peak:
                        self.peak_memory_by_stage[snapshot.stage_name] = snapshot.rss_mb
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in memory profiling loop: {e}")
                time.sleep(self.sampling_interval)
    
    def _capture_memory_snapshot(self) -> MemorySnapshot:
        """Captura snapshot atual de memória"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            system_memory = psutil.virtual_memory()
            
            metadata = {}
            
            # Add tracemalloc info if available
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                metadata['tracemalloc_current_mb'] = current / (1024 * 1024)
                metadata['tracemalloc_peak_mb'] = peak / (1024 * 1024)
            
            return MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=memory_percent,
                available_gb=system_memory.available / (1024**3),
                stage_name=self.current_stage,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error capturing memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=0,
                vms_mb=0,
                percent=0,
                available_gb=0
            )
    
    def set_current_stage(self, stage_name: str, operation_type: str = ""):
        """Define o stage atual para tracking"""
        self.current_stage = stage_name
        logger.debug(f"🏷️ Memory profiling stage: {stage_name}")
    
    def get_stage_memory_stats(self, stage_name: str) -> Dict[str, float]:
        """Retorna estatísticas de memória para um stage específico"""
        if stage_name not in self.stage_memory_usage:
            return {}
        
        memory_values = self.stage_memory_usage[stage_name]
        
        return {
            'peak_memory_mb': max(memory_values),
            'average_memory_mb': np.mean(memory_values),
            'min_memory_mb': min(memory_values),
            'memory_variance': np.var(memory_values),
            'samples_count': len(memory_values)
        }
    
    def get_memory_trend_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """Analisa tendências de uso de memória"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.memory_snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trend
        times = [(s.timestamp - recent_snapshots[0].timestamp).total_seconds() for s in recent_snapshots]
        memory_values = [s.rss_mb for s in recent_snapshots]
        
        # Linear regression for trend
        if len(times) > 1:
            coeffs = np.polyfit(times, memory_values, 1)
            trend_slope = coeffs[0]  # MB per second
        else:
            trend_slope = 0
        
        return {
            'trend_slope_mb_per_hour': trend_slope * 3600,
            'current_memory_mb': memory_values[-1] if memory_values else 0,
            'peak_memory_mb': max(memory_values) if memory_values else 0,
            'memory_growth_total_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0,
            'samples_analyzed': len(recent_snapshots)
        }
    
    def detect_memory_leaks(self, threshold_mb_per_hour: float = 50.0) -> List[MemoryLeak]:
        """Detecta vazamentos de memória"""
        leaks = []
        
        # Analyze trend for each stage
        for stage_name, memory_values in self.stage_memory_usage.items():
            if len(memory_values) < 10:  # Need sufficient data
                continue
            
            # Calculate growth rate
            if len(memory_values) > 1:
                time_span_hours = len(memory_values) * self.sampling_interval / 3600
                memory_growth = memory_values[-1] - memory_values[0]
                growth_rate = memory_growth / time_span_hours if time_span_hours > 0 else 0
                
                if growth_rate > threshold_mb_per_hour:
                    # Determine severity
                    if growth_rate > threshold_mb_per_hour * 4:
                        severity = "critical"
                        action = "Immediate investigation required - possible major memory leak"
                    elif growth_rate > threshold_mb_per_hour * 2:
                        severity = "high"
                        action = "Investigate memory allocation patterns in stage"
                    else:
                        severity = "medium"
                        action = "Monitor continued growth and optimize if persistent"
                    
                    leak = MemoryLeak(
                        detection_time=datetime.now(),
                        stage_name=stage_name,
                        memory_growth_mb=memory_growth,
                        growth_rate_mb_per_min=growth_rate / 60,
                        severity=severity,
                        recommended_action=action
                    )
                    
                    leaks.append(leak)
        
        return leaks
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Gera relatório completo de memória"""
        current_snapshot = self._capture_memory_snapshot()
        trend_analysis = self.get_memory_trend_analysis()
        
        # Stage summaries
        stage_summaries = {}
        for stage_name in self.stage_memory_usage.keys():
            stage_summaries[stage_name] = self.get_stage_memory_stats(stage_name)
        
        # Memory leaks
        detected_leaks = self.detect_memory_leaks()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'current_status': {
                'memory_usage_mb': current_snapshot.rss_mb,
                'memory_percent': current_snapshot.percent,
                'available_memory_gb': current_snapshot.available_gb,
                'current_stage': current_snapshot.stage_name
            },
            'trend_analysis': trend_analysis,
            'stage_summaries': stage_summaries,
            'memory_leaks': [
                {
                    'stage_name': leak.stage_name,
                    'growth_rate_mb_per_min': leak.growth_rate_mb_per_min,
                    'severity': leak.severity,
                    'recommended_action': leak.recommended_action
                }
                for leak in detected_leaks
            ],
            'optimization_opportunities': self._identify_optimization_opportunities(stage_summaries),
            'total_snapshots': len(self.memory_snapshots)
        }
    
    def _identify_optimization_opportunities(self, stage_summaries: Dict[str, Dict]) -> List[str]:
        """Identifica oportunidades de otimização"""
        opportunities = []
        
        for stage_name, stats in stage_summaries.items():
            peak_memory = stats.get('peak_memory_mb', 0)
            avg_memory = stats.get('average_memory_mb', 0)
            variance = stats.get('memory_variance', 0)
            
            # High peak memory
            if peak_memory > 2000:  # 2GB
                opportunities.append(f"{stage_name}: High peak memory ({peak_memory:.0f}MB) - consider data chunking")
            
            # High variance (unstable memory usage)
            if variance > 10000:  # High variance
                opportunities.append(f"{stage_name}: Unstable memory usage - investigate memory allocation patterns")
            
            # High average memory (constant high usage)
            if avg_memory > 1500:  # 1.5GB
                opportunities.append(f"{stage_name}: Consistently high memory usage - optimize data structures")
        
        return opportunities

class GarbageCollectionOptimizer:
    """Otimizador inteligente de garbage collection"""
    
    def __init__(self):
        self.gc_stats_history = deque(maxlen=1000)
        self.auto_gc_enabled = True
        self.gc_thresholds = self._get_optimal_gc_thresholds()
        
        # GC monitoring
        self.collections_count = defaultdict(int)
        self.last_gc_time = time.time()
        
        logger.info("🗑️ GarbageCollectionOptimizer initialized")
    
    def _get_optimal_gc_thresholds(self) -> Tuple[int, int, int]:
        """Calcula thresholds ótimos de GC baseados no sistema"""
        # Get current thresholds
        current = gc.get_threshold()
        
        # Optimize based on available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 16:  # High memory system
            # Less frequent GC for better performance
            return (current[0] * 2, current[1] * 2, current[2] * 2)
        elif memory_gb >= 8:  # Medium memory system
            # Balanced approach
            return (int(current[0] * 1.5), int(current[1] * 1.5), int(current[2] * 1.5))
        else:  # Low memory system
            # More aggressive GC
            return (int(current[0] * 0.7), int(current[1] * 0.7), int(current[2] * 0.7))
    
    def enable_auto_gc(self):
        """Habilita garbage collection automático otimizado"""
        self.auto_gc_enabled = True
        gc.set_threshold(*self.gc_thresholds)
        gc.enable()
        
        logger.info(f"🗑️ Auto GC enabled with thresholds: {self.gc_thresholds}")
    
    def disable_auto_gc(self):
        """Desabilita garbage collection automático"""
        self.auto_gc_enabled = False
        gc.disable()
        
        logger.info("🗑️ Auto GC disabled")
    
    def force_gc_collection(self, generation: Optional[int] = None) -> GCStats:
        """Força garbage collection com estatísticas"""
        start_time = time.time()
        
        # Get stats before collection
        before_stats = gc.get_stats()
        before_objects = gc.get_count()
        
        # Force collection
        if generation is not None:
            collected = gc.collect(generation)
        else:
            collected = gc.collect()
        
        # Get stats after collection
        after_stats = gc.get_stats()
        after_objects = gc.get_count()
        
        collection_time = time.time() - start_time
        
        # Calculate efficiency
        objects_before = sum(before_objects)
        objects_after = sum(after_objects)
        objects_collected = objects_before - objects_after
        
        efficiency = (objects_collected / objects_before * 100) if objects_before > 0 else 0
        
        # Create stats object
        stats = GCStats(
            collections_count={i: after_stats[i]['collections'] for i in range(len(after_stats))},
            collected_objects={i: after_stats[i]['collected'] for i in range(len(after_stats))},
            uncollectable_objects={i: after_stats[i]['uncollectable'] for i in range(len(after_stats))},
            total_time_seconds=collection_time,
            efficiency_score=efficiency
        )
        
        self.gc_stats_history.append(stats)
        self.last_gc_time = time.time()
        
        logger.info(f"🗑️ GC completed: {collected} objects collected in {collection_time:.3f}s "
                   f"(efficiency: {efficiency:.1f}%)")
        
        return stats
    
    def adaptive_gc_trigger(self, memory_usage_mb: float, memory_threshold_mb: float = 4000):
        """Trigger adaptativo de garbage collection baseado em uso de memória"""
        current_time = time.time()
        time_since_last_gc = current_time - self.last_gc_time
        
        # Conditions for triggering GC
        high_memory = memory_usage_mb > memory_threshold_mb
        long_time_since_gc = time_since_last_gc > 300  # 5 minutes
        
        if high_memory or long_time_since_gc:
            logger.info(f"🗑️ Adaptive GC triggered: memory={memory_usage_mb:.0f}MB, "
                       f"time_since_gc={time_since_last_gc:.0f}s")
            return self.force_gc_collection()
        
        return None
    
    def get_gc_performance_analysis(self) -> Dict[str, Any]:
        """Analisa performance do garbage collection"""
        if not self.gc_stats_history:
            return {'error': 'No GC stats available'}
        
        recent_stats = list(self.gc_stats_history)
        
        # Calculate averages
        avg_efficiency = np.mean([s.efficiency_score for s in recent_stats])
        avg_time = np.mean([s.total_time_seconds for s in recent_stats])
        total_collections = sum([sum(s.collections_count.values()) for s in recent_stats])
        
        # Performance assessment
        if avg_efficiency > 50 and avg_time < 0.1:
            performance_rating = "excellent"
        elif avg_efficiency > 30 and avg_time < 0.5:
            performance_rating = "good"
        elif avg_efficiency > 15:
            performance_rating = "acceptable"
        else:
            performance_rating = "poor"
        
        return {
            'average_efficiency_percent': avg_efficiency,
            'average_collection_time_seconds': avg_time,
            'total_collections': total_collections,
            'performance_rating': performance_rating,
            'current_thresholds': self.gc_thresholds,
            'recommendations': self._generate_gc_recommendations(avg_efficiency, avg_time)
        }
    
    def _generate_gc_recommendations(self, avg_efficiency: float, avg_time: float) -> List[str]:
        """Gera recomendações para otimização de GC"""
        recommendations = []
        
        if avg_efficiency < 20:
            recommendations.append("Low GC efficiency - consider reducing object creation or improving data structures")
        
        if avg_time > 0.5:
            recommendations.append("High GC time - consider tuning GC thresholds or reducing heap size")
        
        if avg_efficiency > 60 and avg_time < 0.05:
            recommendations.append("GC performance is excellent - current configuration is optimal")
        
        return recommendations

class AdaptiveMemoryManager:
    """Gerenciador adaptativo de memória que integra profiling e otimização"""
    
    def __init__(self, target_memory_gb: float = 4.0, emergency_threshold_gb: float = 6.0):
        self.target_memory_gb = target_memory_gb
        self.emergency_threshold_gb = emergency_threshold_gb
        
        # Components
        self.profiler = MemoryProfiler()
        self.gc_optimizer = GarbageCollectionOptimizer()
        
        # Management state
        self.is_managing = False
        self.management_thread = None
        
        # Statistics
        self.optimizations_performed = 0
        self.memory_savings_mb = 0.0
        self.emergency_interventions = 0
        
        logger.info(f"🎯 AdaptiveMemoryManager initialized: target={target_memory_gb}GB, "
                   f"emergency={emergency_threshold_gb}GB")
    
    def start_adaptive_management(self):
        """Inicia gerenciamento adaptativo de memória"""
        if self.is_managing:
            return
        
        self.is_managing = True
        
        # Start profiling
        self.profiler.start_profiling()
        
        # Enable optimized GC
        self.gc_optimizer.enable_auto_gc()
        
        # Start management thread
        self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.management_thread.start()
        
        logger.info("🎯 Adaptive memory management started")
    
    def stop_adaptive_management(self):
        """Para gerenciamento adaptativo de memória"""
        if not self.is_managing:
            return
        
        self.is_managing = False
        
        # Stop components
        self.profiler.stop_profiling()
        
        if self.management_thread:
            self.management_thread.join(timeout=2)
        
        logger.info("🎯 Adaptive memory management stopped")
    
    def _management_loop(self):
        """Loop principal de gerenciamento"""
        while self.is_managing:
            try:
                # Get current memory status
                memory_info = psutil.Process().memory_info()
                current_memory_gb = memory_info.rss / (1024**3)
                
                # Check if intervention is needed
                if current_memory_gb > self.emergency_threshold_gb:
                    self._emergency_memory_intervention(current_memory_gb)
                elif current_memory_gb > self.target_memory_gb:
                    self._proactive_memory_optimization(current_memory_gb)
                
                # Adaptive GC based on memory pressure
                memory_mb = current_memory_gb * 1024
                gc_stats = self.gc_optimizer.adaptive_gc_trigger(memory_mb, self.target_memory_gb * 1024)
                
                if gc_stats:
                    self.optimizations_performed += 1
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in memory management loop: {e}")
                time.sleep(10)
    
    def _emergency_memory_intervention(self, current_memory_gb: float):
        """Intervenção de emergência para alto uso de memória"""
        logger.warning(f"🚨 Emergency memory intervention: {current_memory_gb:.2f}GB > {self.emergency_threshold_gb:.2f}GB")
        
        before_memory = current_memory_gb
        
        # Aggressive garbage collection
        self.gc_optimizer.force_gc_collection()
        
        # Clear caches if available
        self._clear_system_caches()
        
        # Force pandas to release memory
        self._optimize_pandas_memory()
        
        # Check memory after intervention
        after_memory = psutil.Process().memory_info().rss / (1024**3)
        memory_freed = before_memory - after_memory
        
        if memory_freed > 0:
            self.memory_savings_mb += memory_freed * 1024
            logger.info(f"Emergency intervention freed {memory_freed:.2f}GB memory")
        
        self.emergency_interventions += 1
    
    def _proactive_memory_optimization(self, current_memory_gb: float):
        """Otimização proativa de memória"""
        logger.info(f"🔧 Proactive memory optimization: {current_memory_gb:.2f}GB > {self.target_memory_gb:.2f}GB")
        
        before_memory = current_memory_gb
        
        # Gentle garbage collection
        self.gc_optimizer.force_gc_collection(generation=0)  # Only gen 0
        
        # Optimize data structures
        self._optimize_data_structures()
        
        # Check memory after optimization
        after_memory = psutil.Process().memory_info().rss / (1024**3)
        memory_freed = before_memory - after_memory
        
        if memory_freed > 0:
            self.memory_savings_mb += memory_freed * 1024
            
        self.optimizations_performed += 1
    
    def _clear_system_caches(self):
        """Limpa caches do sistema"""
        try:
            # Clear any available optimization caches
            # This would integrate with Week 1-3 optimizations
            pass
        except Exception as e:
            logger.debug(f"Cache clearing failed: {e}")
    
    def _optimize_pandas_memory(self):
        """Otimiza uso de memória do pandas"""
        try:
            # Force pandas garbage collection
            import pandas as pd
            
            # This would be more sophisticated in a real implementation
            # For now, just trigger standard GC
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Pandas memory optimization failed: {e}")
    
    def _optimize_data_structures(self):
        """Otimiza estruturas de dados em memória"""
        try:
            # This would implement specific optimizations
            # like converting data types, releasing unnecessary objects, etc.
            
            # Force collection of generation 0 and 1
            self.gc_optimizer.force_gc_collection(generation=1)
            
        except Exception as e:
            logger.debug(f"Data structure optimization failed: {e}")
    
    def set_current_stage(self, stage_name: str):
        """Define stage atual para profiling"""
        self.profiler.set_current_stage(stage_name)
    
    def get_management_summary(self) -> Dict[str, Any]:
        """Retorna resumo do gerenciamento de memória"""
        
        # Get current memory status
        memory_info = psutil.Process().memory_info()
        current_memory_gb = memory_info.rss / (1024**3)
        
        # Get profiler report
        memory_report = self.profiler.generate_memory_report()
        
        # Get GC analysis
        gc_analysis = self.gc_optimizer.get_gc_performance_analysis()
        
        # Calculate efficiency metrics
        target_achievement = min(100, (self.target_memory_gb / current_memory_gb) * 100) if current_memory_gb > 0 else 100
        
        return {
            'management_status': {
                'is_managing': self.is_managing,
                'current_memory_gb': current_memory_gb,
                'target_memory_gb': self.target_memory_gb,
                'target_achievement_percent': target_achievement,
                'memory_within_target': current_memory_gb <= self.target_memory_gb
            },
            'optimization_stats': {
                'optimizations_performed': self.optimizations_performed,
                'memory_savings_mb': self.memory_savings_mb,
                'emergency_interventions': self.emergency_interventions,
                'average_savings_per_optimization': self.memory_savings_mb / max(1, self.optimizations_performed)
            },
            'profiler_report': memory_report,
            'gc_analysis': gc_analysis,
            'recommendations': self._generate_management_recommendations(current_memory_gb, target_achievement)
        }
    
    def _generate_management_recommendations(self, current_memory_gb: float, 
                                          target_achievement: float) -> List[str]:
        """Gera recomendações de gerenciamento"""
        recommendations = []
        
        if current_memory_gb > self.emergency_threshold_gb:
            recommendations.append("CRITICAL: Memory usage above emergency threshold - immediate intervention needed")
        elif current_memory_gb > self.target_memory_gb:
            recommendations.append("Memory usage above target - consider optimizing data processing")
        
        if target_achievement < 80:
            recommendations.append("Target achievement below 80% - review memory allocation strategies")
        
        if self.emergency_interventions > 5:
            recommendations.append("Multiple emergency interventions - consider increasing target memory limit")
        
        if self.memory_savings_mb > 1000:  # 1GB
            recommendations.append(f"Excellent memory optimization: {self.memory_savings_mb:.0f}MB total savings achieved")
        
        return recommendations
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory for test compatibility."""
        memory_info = psutil.Process().memory_info()
        current_memory_gb = memory_info.rss / (1024**3)
        
        # Perform optimization
        self._optimize_pandas_memory()
        self._optimize_data_structures()
        self.gc_optimizer.run_aggressive_gc()
        
        # Calculate savings
        new_memory_info = psutil.Process().memory_info()
        new_memory_gb = new_memory_info.rss / (1024**3)
        savings_mb = (current_memory_gb - new_memory_gb) * 1024
        
        self.optimizations_performed += 1
        self.memory_savings_mb += max(0, savings_mb)
        
        return {
            'memory_before_gb': current_memory_gb,
            'memory_after_gb': new_memory_gb,
            'savings_mb': max(0, savings_mb),
            'optimizations_performed': self.optimizations_performed
        }
    
    def manage(self) -> Dict[str, Any]:
        """Manage memory for test compatibility."""
        return self.get_management_summary()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for test compatibility."""
        memory_info = psutil.Process().memory_info()
        return {
            'current_memory_gb': memory_info.rss / (1024**3),
            'target_memory_gb': self.target_memory_gb,
            'memory_utilization': (memory_info.rss / (1024**3)) / self.target_memory_gb
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory stats for test compatibility."""
        return self.get_management_summary()

# Factory functions
def create_production_memory_manager() -> AdaptiveMemoryManager:
    """Cria memory manager configurado para produção"""
    return AdaptiveMemoryManager(
        target_memory_gb=4.0,      # 4GB target as per optimization goals
        emergency_threshold_gb=6.0  # 6GB emergency threshold
    )

def create_development_memory_manager() -> AdaptiveMemoryManager:
    """Cria memory manager configurado para desenvolvimento"""
    return AdaptiveMemoryManager(
        target_memory_gb=2.0,      # 2GB target for development
        emergency_threshold_gb=3.0  # 3GB emergency threshold
    )

# Global instance
_global_memory_manager = None

def get_global_memory_manager() -> AdaptiveMemoryManager:
    """Retorna instância global do memory manager"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = create_production_memory_manager()
    return _global_memory_manager