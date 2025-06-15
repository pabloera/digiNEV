"""
Real-time Performance Monitor - Week 4 Advanced Monitoring System
================================================================

Sistema de monitoramento em tempo real para acompanhar execução do pipeline:
- Métricas de performance ao vivo
- Alertas automáticos para problemas
- Dashboard de monitoramento
- Trending de recursos

BENEFÍCIOS SEMANA 4:
- Visibilidade completa do sistema em execução
- Detecção precoce de problemas de performance
- Otimização baseada em dados em tempo real
- Alertas automáticos para intervenção

Sistema enterprise-grade para monitoramento production-ready.

Data: 2025-06-14
Status: SEMANA 4 REALTIME MONITORING
"""

import asyncio
import json
import logging
import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil

# Dashboard components availability flag
PLOTLY_AVAILABLE = False
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Níveis de alerta do sistema"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Tipos de métricas monitoradas"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    SYSTEM = "system"

@dataclass
class Alert:
    """Alerta do sistema de monitoramento"""
    alert_id: str
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class MetricValue:
    """Valor de uma métrica"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSnapshot:
    """Snapshot de performance em um momento específico"""
    timestamp: datetime
    pipeline_stage: str
    metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    quality_indicators: Dict[str, float]
    active_optimizations: List[str]

class MetricsCollector:
    """Coletador de métricas do sistema"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.metrics_queue = queue.Queue()
        self.collection_thread = None
        
        # System info
        self.process = psutil.Process()
        self.start_time = time.time()
        
        logger.info("📊 MetricsCollector initialized")
    
    def start_collection(self):
        """Inicia coleta de métricas"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("📈 Metrics collection started")
    
    def stop_collection(self):
        """Para coleta de métricas"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2)
        
        logger.info("📈 Metrics collection stopped")
    
    def _collection_loop(self):
        """Loop principal de coleta"""
        while self.is_collecting:
            try:
                metrics = self._collect_current_metrics()
                
                for metric in metrics:
                    self.metrics_queue.put(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_current_metrics(self) -> List[MetricValue]:
        """Coleta métricas atuais do sistema"""
        current_time = datetime.now()
        metrics = []
        
        try:
            # Memory metrics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            metrics.extend([
                MetricValue("memory_rss_mb", memory_info.rss / (1024 * 1024), "MB", current_time),
                MetricValue("memory_vms_mb", memory_info.vms / (1024 * 1024), "MB", current_time),
                MetricValue("memory_percent", memory_percent, "%", current_time)
            ])
            
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            cpu_count = psutil.cpu_count()
            
            metrics.extend([
                MetricValue("cpu_percent", cpu_percent, "%", current_time),
                MetricValue("cpu_count", cpu_count, "cores", current_time)
            ])
            
            # System metrics
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=None)
            
            metrics.extend([
                MetricValue("system_memory_percent", system_memory.percent, "%", current_time),
                MetricValue("system_memory_available_gb", system_memory.available / (1024**3), "GB", current_time),
                MetricValue("system_cpu_percent", system_cpu, "%", current_time)
            ])
            
            # Uptime metrics
            uptime_seconds = time.time() - self.start_time
            metrics.append(MetricValue("uptime_seconds", uptime_seconds, "seconds", current_time))
            
            # Disk I/O (if available)
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.extend([
                        MetricValue("disk_read_mb", disk_io.read_bytes / (1024 * 1024), "MB", current_time),
                        MetricValue("disk_write_mb", disk_io.write_bytes / (1024 * 1024), "MB", current_time)
                    ])
            except Exception:
                pass  # Disk I/O not available on all systems
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def get_metrics(self, count: int = None) -> List[MetricValue]:
        """Retorna métricas coletadas"""
        metrics = []
        
        try:
            while not self.metrics_queue.empty() and (count is None or len(metrics) < count):
                metrics.append(self.metrics_queue.get_nowait())
        except queue.Empty:
            pass
        
        return metrics
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Returns metrics as a dictionary for test compatibility."""
        metrics_list = self.get_metrics()
        metrics_dict = {}
        
        for metric in metrics_list:
            metrics_dict[metric.name] = {
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp,
                'metadata': metric.metadata
            }
        
        # Also include current system metrics for immediate testing
        current_metrics = self.collect_current_metrics()
        for metric in current_metrics:
            metrics_dict[metric.name] = {
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp,
                'metadata': metric.metadata
            }
        
        return metrics_dict

class AlertSystem:
    """Sistema de alertas baseado em thresholds"""
    
    def __init__(self):
        self.thresholds = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Default thresholds
        self._setup_default_thresholds()
        
        logger.info("🚨 AlertSystem initialized")
    
    def _setup_default_thresholds(self):
        """Configura thresholds padrão"""
        self.thresholds = {
            'memory_percent': {'medium': 70.0, 'high': 85.0, 'critical': 95.0},
            'cpu_percent': {'medium': 70.0, 'high': 85.0, 'critical': 95.0},
            'system_memory_percent': {'medium': 80.0, 'high': 90.0, 'critical': 95.0},
            'pipeline_error_rate': {'medium': 0.10, 'high': 0.25, 'critical': 0.50},
            'cache_hit_rate': {'low': 0.30, 'medium': 0.50},  # Inverted (low is bad)
            'execution_time_minutes': {'medium': 30.0, 'high': 60.0, 'critical': 120.0}
        }
    
    def set_threshold(self, metric_name: str, level: AlertLevel, value: float):
        """Define threshold para uma métrica"""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        self.thresholds[metric_name][level.value] = value
        logger.info(f"🎯 Threshold set: {metric_name} {level.value} = {value}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Adiciona callback para alertas"""
        self.alert_callbacks.append(callback)
    
    def check_metric(self, metric: MetricValue):
        """Verifica métrica contra thresholds"""
        if metric.name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric.name]
        alert_level = None
        threshold_value = None
        
        # Check in order of severity
        for level in ['critical', 'high', 'medium', 'low']:
            if level in thresholds:
                threshold = thresholds[level]
                
                # Normal thresholds (higher is worse)
                if level != 'low' and metric.value >= threshold:
                    alert_level = AlertLevel(level)
                    threshold_value = threshold
                    break
                # Inverted thresholds (lower is worse) 
                elif level == 'low' and metric.value <= threshold:
                    alert_level = AlertLevel.MEDIUM  # Convert low to medium
                    threshold_value = threshold
                    break
        
        if alert_level:
            self._create_alert(metric, alert_level, threshold_value)
        else:
            # Check if we should resolve an existing alert
            self._check_alert_resolution(metric)
    
    def _create_alert(self, metric: MetricValue, level: AlertLevel, threshold: float):
        """Cria novo alerta"""
        alert_id = f"{metric.name}_{level.value}"
        
        # Check if alert already exists and is active
        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
            return  # Don't create duplicate alerts
        
        message = self._generate_alert_message(metric, level, threshold)
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=threshold,
            message=message
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"🚨 ALERT [{level.value.upper()}]: {message}")
    
    def _check_alert_resolution(self, metric: MetricValue):
        """Verifica se alertas podem ser resolvidos"""
        alert_id_prefix = f"{metric.name}_"
        
        for alert_id, alert in self.active_alerts.items():
            if alert_id.startswith(alert_id_prefix) and not alert.resolved:
                # Check if metric is now below threshold
                if metric.value < alert.threshold_value * 0.9:  # 10% hysteresis
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    logger.info(f"ALERT RESOLVED: {alert.message}")
    
    def _generate_alert_message(self, metric: MetricValue, level: AlertLevel, threshold: float) -> str:
        """Gera mensagem de alerta"""
        
        messages = {
            'memory_percent': f"High memory usage (Current: {metric.value:.1f}%, Threshold: {threshold:.1f}%)",
            'cpu_percent': f"High CPU usage (Current: {metric.value:.1f}%, Threshold: {threshold:.1f}%)",
            'system_memory_percent': f"High system memory (Current: {metric.value:.1f}%, Threshold: {threshold:.1f}%)",
            'pipeline_error_rate': f"High pipeline error rate (Current: {metric.value:.1%}, Threshold: {threshold:.1%})",
            'cache_hit_rate': f"Low cache hit rate (Current: {metric.value:.1%}, Threshold: {threshold:.1%})",
            'execution_time_minutes': f"Long execution time (Current: {metric.value:.1f}min, Threshold: {threshold:.1f}min)"
        }
        
        return messages.get(metric.name, f"{metric.name}: {metric.value:.2f} {metric.unit} (threshold: {threshold})")
    
    def get_active_alerts(self) -> List[Alert]:
        """Retorna alertas ativos"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Retorna histórico de alertas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

class PerformanceMonitor:
    """Monitor principal de performance em tempo real"""
    
    def __init__(self, metrics_interval: float = 1.0, snapshot_interval: float = 10.0):
        self.metrics_interval = metrics_interval
        self.snapshot_interval = snapshot_interval
        
        # Components
        self.metrics_collector = MetricsCollector(metrics_interval)
        self.alert_system = AlertSystem()
        
        # Data storage
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        self.snapshots = deque(maxlen=1000)  # Keep last 1k snapshots
        
        # State tracking
        self.is_monitoring = False
        self.current_pipeline_stage = "idle"
        self.pipeline_start_time = None
        
        # Performance tracking
        self.stage_metrics = {}
        self.optimization_stats = {}
        
        # Setup alert callbacks
        self.alert_system.add_callback(self._handle_alert)
        
        logger.info("🎯 PerformanceMonitor initialized")
    
    def start_monitoring(self):
        """Inicia monitoramento completo"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.pipeline_start_time = datetime.now()
        
        # Start components
        self.metrics_collector.start_collection()
        
        # Start monitoring loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._monitoring_loop())
            else:
                # Start new event loop if none running
                asyncio.ensure_future(self._monitoring_loop())
        except RuntimeError:
            # No event loop running, start monitoring in thread
            import threading
            self.monitoring_thread = threading.Thread(target=self._run_monitoring_thread, daemon=True)
            self.monitoring_thread.start()
        
        logger.info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.metrics_collector.stop_collection()
        
        logger.info("🛑 Performance monitoring stopped")
    
    def _run_monitoring_thread(self):
        """Run monitoring loop in separate thread when no async event loop available"""
        try:
            asyncio.run(self._monitoring_loop())
        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")
    
    async def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        last_snapshot_time = time.time()
        
        while self.is_monitoring:
            try:
                # Process new metrics
                new_metrics = self.metrics_collector.get_metrics()
                
                for metric in new_metrics:
                    self.metrics_history.append(metric)
                    self.alert_system.check_metric(metric)
                
                # Create performance snapshot if needed
                current_time = time.time()
                if current_time - last_snapshot_time >= self.snapshot_interval:
                    snapshot = self._create_performance_snapshot()
                    self.snapshots.append(snapshot)
                    last_snapshot_time = current_time
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    def _create_performance_snapshot(self) -> PerformanceSnapshot:
        """Cria snapshot de performance atual"""
        current_time = datetime.now()
        
        # Get recent metrics for analysis
        recent_metrics = [m for m in self.metrics_history if (current_time - m.timestamp).total_seconds() <= self.snapshot_interval]
        
        # Aggregate metrics
        metrics_dict = {}
        resource_usage = {}
        
        for metric in recent_metrics:
            if metric.name not in metrics_dict:
                metrics_dict[metric.name] = []
            metrics_dict[metric.name].append(metric.value)
        
        # Calculate averages
        for name, values in metrics_dict.items():
            avg_value = np.mean(values)
            
            if 'memory' in name or 'cpu' in name:
                resource_usage[name] = avg_value
            else:
                metrics_dict[name] = avg_value
        
        # Quality indicators (would be populated by pipeline)
        quality_indicators = {
            'cache_hit_rate': self.optimization_stats.get('cache_hit_rate', 0.0),
            'success_rate': self.optimization_stats.get('success_rate', 0.0),
            'error_rate': self.optimization_stats.get('error_rate', 0.0)
        }
        
        # Active optimizations
        active_optimizations = self.optimization_stats.get('active_optimizations', [])
        
        return PerformanceSnapshot(
            timestamp=current_time,
            pipeline_stage=self.current_pipeline_stage,
            metrics=metrics_dict,
            resource_usage=resource_usage,
            quality_indicators=quality_indicators,
            active_optimizations=active_optimizations
        )
    
    def _handle_alert(self, alert: Alert):
        """Manipula alertas recebidos"""
        # Could implement automatic remediation here
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"🔥 CRITICAL ALERT: {alert.message}")
            # Could trigger emergency procedures
        
        # Store alert for dashboard
        self.optimization_stats['last_alert'] = {
            'level': alert.level.value,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat()
        }
    
    def record_stage_completion(self, stage_name: str, records_processed: int, 
                              processing_time: float, success_rate: float,
                              api_calls: int = 0, cost_usd: float = 0.0):
        """Registra conclusão de um stage"""
        self.current_pipeline_stage = stage_name
        
        # Record stage metrics
        self.stage_metrics[stage_name] = {
            'records_processed': records_processed,
            'processing_time': processing_time,
            'success_rate': success_rate,
            'api_calls': api_calls,
            'cost_usd': cost_usd,
            'timestamp': datetime.now().isoformat(),
            'records_per_second': records_processed / processing_time if processing_time > 0 else 0
        }
        
        # Update optimization stats
        self.optimization_stats.update({
            'last_stage': stage_name,
            'total_records_processed': sum(m.get('records_processed', 0) for m in self.stage_metrics.values()),
            'total_api_calls': sum(m.get('api_calls', 0) for m in self.stage_metrics.values()),
            'total_cost_usd': sum(m.get('cost_usd', 0) for m in self.stage_metrics.values()),
            'average_success_rate': np.mean([m.get('success_rate', 0) for m in self.stage_metrics.values()]) if self.stage_metrics else 0
        })
        
        logger.info(f"📈 Stage completed: {stage_name} ({records_processed} records, {processing_time:.2f}s, {success_rate:.1%} success)")
    
    def record_optimization_event(self, optimization_type: str, details: Dict[str, Any]):
        """Registra evento de otimização"""
        event_key = f"{optimization_type}_events"
        
        if event_key not in self.optimization_stats:
            self.optimization_stats[event_key] = []
        
        event = {
            'type': optimization_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_stats[event_key].append(event)
        
        # Keep only last 100 events per type
        self.optimization_stats[event_key] = self.optimization_stats[event_key][-100:]
        
        logger.info(f"🔧 Optimization event: {optimization_type}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Retorna status atual do sistema"""
        current_time = datetime.now()
        
        # Get recent metrics
        recent_metrics = [m for m in self.metrics_history if (current_time - m.timestamp).total_seconds() <= 60]
        
        # Aggregate current metrics
        current_metrics = {}
        for metric in recent_metrics:
            if metric.name not in current_metrics:
                current_metrics[metric.name] = []
            current_metrics[metric.name].append(metric.value)
        
        # Calculate current values
        current_values = {name: np.mean(values) for name, values in current_metrics.items()}
        
        # System health score
        health_score = self._calculate_health_score(current_values)
        
        return {
            'timestamp': current_time.isoformat(),
            'monitoring_active': self.is_monitoring,
            'current_stage': self.current_pipeline_stage,
            'health_score': health_score,
            'active_alerts': len(self.alert_system.get_active_alerts()),
            'current_metrics': current_values,
            'pipeline_uptime_minutes': (current_time - self.pipeline_start_time).total_seconds() / 60 if self.pipeline_start_time else 0,
            'optimization_stats': self.optimization_stats,
            'stages_completed': len(self.stage_metrics)
        }
    
    def _calculate_health_score(self, current_metrics: Dict[str, float]) -> float:
        """Calcula score de saúde do sistema (0-100)"""
        score_components = []
        
        # Memory health (weight: 30%)
        memory_percent = current_metrics.get('memory_percent', 0)
        if memory_percent <= 50:
            memory_score = 100
        elif memory_percent <= 75:
            memory_score = 100 - (memory_percent - 50) * 2
        else:
            memory_score = max(0, 50 - (memory_percent - 75) * 2)
        score_components.append(memory_score * 0.3)
        
        # CPU health (weight: 20%)
        cpu_percent = current_metrics.get('cpu_percent', 0)
        if 30 <= cpu_percent <= 70:  # Optimal range
            cpu_score = 100
        elif cpu_percent < 30:
            cpu_score = 50 + cpu_percent * 50/30  # Underutilization
        else:
            cpu_score = max(0, 100 - (cpu_percent - 70) * 2)
        score_components.append(cpu_score * 0.2)
        
        # System stability (weight: 25%)
        alerts = self.alert_system.get_active_alerts()
        critical_alerts = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
        high_alerts = sum(1 for a in alerts if a.level == AlertLevel.HIGH)
        
        stability_score = max(0, 100 - critical_alerts * 50 - high_alerts * 20)
        score_components.append(stability_score * 0.25)
        
        # Performance (weight: 25%)
        success_rate = self.optimization_stats.get('average_success_rate', 0)
        performance_score = success_rate * 100
        score_components.append(performance_score * 0.25)
        
        return sum(score_components)
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Gera resumo executivo da performance"""
        current_time = datetime.now()
        
        summary = {
            'report_timestamp': current_time.isoformat(),
            'monitoring_period': {
                'start': self.pipeline_start_time.isoformat() if self.pipeline_start_time else None,
                'duration_minutes': (current_time - self.pipeline_start_time).total_seconds() / 60 if self.pipeline_start_time else 0
            },
            'overall_health': self._calculate_health_score(self.get_current_status()['current_metrics']),
            'stages_summary': self._summarize_stages(),
            'resource_utilization': self._summarize_resource_usage(),
            'optimization_effectiveness': self._summarize_optimizations(),
            'alerts_summary': self._summarize_alerts(),
            'recommendations': self._generate_performance_recommendations()
        }
        
        return summary
    
    def get_metrics(self, count: int = None) -> List[MetricValue]:
        """Get recent metrics for test compatibility."""
        return self.metrics_collector.get_metrics(count)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for test compatibility."""
        return self.get_current_status()
    
    def _summarize_stages(self) -> Dict[str, Any]:
        """Sumariza performance dos stages"""
        if not self.stage_metrics:
            return {'total_stages': 0}
        
        total_time = sum(m.get('processing_time', 0) for m in self.stage_metrics.values())
        total_records = sum(m.get('records_processed', 0) for m in self.stage_metrics.values())
        avg_success_rate = np.mean([m.get('success_rate', 0) for m in self.stage_metrics.values()])
        
        # Find slowest and fastest stages
        stage_times = [(name, metrics.get('processing_time', 0)) for name, metrics in self.stage_metrics.items()]
        slowest_stage = max(stage_times, key=lambda x: x[1]) if stage_times else ("", 0)
        fastest_stage = min(stage_times, key=lambda x: x[1]) if stage_times else ("", 0)
        
        return {
            'total_stages': len(self.stage_metrics),
            'total_processing_time': total_time,
            'total_records_processed': total_records,
            'average_success_rate': avg_success_rate,
            'records_per_second_overall': total_records / total_time if total_time > 0 else 0,
            'slowest_stage': {'name': slowest_stage[0], 'time': slowest_stage[1]},
            'fastest_stage': {'name': fastest_stage[0], 'time': fastest_stage[1]}
        }
    
    def _summarize_resource_usage(self) -> Dict[str, Any]:
        """Sumariza uso de recursos"""
        if not self.metrics_history:
            return {}
        
        # Get resource metrics from last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= one_hour_ago]
        
        resource_summary = {}
        for metric_name in ['memory_percent', 'cpu_percent', 'memory_rss_mb']:
            values = [m.value for m in recent_metrics if m.name == metric_name]
            if values:
                resource_summary[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'peak': max(values),
                    'min': min(values)
                }
        
        return resource_summary
    
    def _summarize_optimizations(self) -> Dict[str, Any]:
        """Sumariza efetividade das otimizações"""
        return {
            'cache_hit_rate': self.optimization_stats.get('cache_hit_rate', 0),
            'total_api_calls': self.optimization_stats.get('total_api_calls', 0),
            'total_cost_usd': self.optimization_stats.get('total_cost_usd', 0),
            'parallel_stages_active': len([stage for stage in self.stage_metrics.keys() if 'parallel' in stage.lower()]),
            'optimization_events': sum(len(events) for key, events in self.optimization_stats.items() if key.endswith('_events'))
        }
    
    def _summarize_alerts(self) -> Dict[str, Any]:
        """Sumariza alertas"""
        active_alerts = self.alert_system.get_active_alerts()
        recent_alerts = self.alert_system.get_alert_history(24)
        
        alert_counts = {level.value: 0 for level in AlertLevel}
        for alert in active_alerts:
            alert_counts[alert.level.value] += 1
        
        return {
            'active_alerts': len(active_alerts),
            'alerts_last_24h': len(recent_alerts),
            'alert_breakdown': alert_counts,
            'most_common_alert': self._find_most_common_alert(recent_alerts)
        }
    
    def _find_most_common_alert(self, alerts: List[Alert]) -> str:
        """Encontra tipo de alerta mais comum"""
        if not alerts:
            return "none"
        
        alert_types = {}
        for alert in alerts:
            metric = alert.metric_name
            alert_types[metric] = alert_types.get(metric, 0) + 1
        
        return max(alert_types, key=alert_types.get)
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Gera recomendações de performance"""
        recommendations = []
        
        # Analyze resource usage
        current_status = self.get_current_status()
        current_metrics = current_status.get('current_metrics', {})
        
        memory_percent = current_metrics.get('memory_percent', 0)
        if memory_percent > 80:
            recommendations.append("Consider increasing memory allocation or optimizing memory-intensive stages")
        
        cpu_percent = current_metrics.get('cpu_percent', 0)
        if cpu_percent < 30:
            recommendations.append("CPU underutilized - consider increasing parallel processing")
        elif cpu_percent > 85:
            recommendations.append("High CPU usage detected - consider reducing concurrent operations")
        
        # Analyze stage performance
        if self.stage_metrics:
            avg_success_rate = np.mean([m.get('success_rate', 0) for m in self.stage_metrics.values()])
            if avg_success_rate < 0.95:
                recommendations.append("Success rate below target (95%) - review error handling and fallback mechanisms")
        
        # Analyze alerts
        active_alerts = self.alert_system.get_active_alerts()
        if len(active_alerts) > 5:
            recommendations.append("Multiple active alerts - investigate system stability issues")
        
        # Optimization effectiveness
        cache_hit_rate = self.optimization_stats.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate - review caching strategies and data patterns")
        
        return recommendations

# Factory functions
def create_production_monitor() -> PerformanceMonitor:
    """Cria monitor configurado para produção"""
    return PerformanceMonitor(
        metrics_interval=1.0,  # Collect every second
        snapshot_interval=10.0  # Snapshot every 10 seconds
    )

def create_development_monitor() -> PerformanceMonitor:
    """Cria monitor configurado para desenvolvimento"""
    return PerformanceMonitor(
        metrics_interval=2.0,  # Collect every 2 seconds  
        snapshot_interval=30.0  # Snapshot every 30 seconds
    )

# Global instance
_global_performance_monitor = None

def get_global_performance_monitor() -> PerformanceMonitor:
    """Retorna instância global do performance monitor"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = create_production_monitor()
    return _global_performance_monitor