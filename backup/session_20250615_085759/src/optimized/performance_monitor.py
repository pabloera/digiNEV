"""
Advanced Performance Monitor - Real-time System Monitoring
=======================================================

Sistema de monitoramento avançado da Semana 2 que fornece:
- Monitoramento em tempo real de performance do pipeline
- Métricas detalhadas de cache, API e sistema
- Alertas automáticos para problemas de performance
- Relatórios executivos com insights acionáveis
- Integração com todos os componentes do pipeline

BENEFÍCIOS SEMANA 2:
- Visibilidade completa do sistema em tempo real
- Detecção proativa de bottlenecks
- Otimização automática baseada em métricas
- Relatórios executivos para tomada de decisão

Data: 2025-06-14
Status: SEMANA 2 CORE IMPLEMENTATION
"""

import json
import logging
import psutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Métricas de sistema coletadas em tempo real"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io_bytes: Tuple[int, int]  # (bytes_sent, bytes_recv)
    active_threads: int
    open_files: int

@dataclass
class PipelineMetrics:
    """Métricas específicas do pipeline"""
    timestamp: datetime
    stage_name: str
    operation_type: str
    records_processed: int
    processing_time: float
    success_rate: float
    error_count: int
    cache_hit_rate: float
    api_calls_made: int
    cost_estimate_usd: float

@dataclass
class CacheMetrics:
    """Métricas de performance de cache"""
    timestamp: datetime
    cache_type: str  # embeddings, claude, emergency
    cache_level: str  # l1_memory, l2_disk, l3_distributed
    hit_count: int
    miss_count: int
    hit_rate: float
    size_mb: float
    entries_count: int
    evictions_count: int
    compression_ratio: float

@dataclass
class APIMetrics:
    """Métricas de uso de APIs"""
    timestamp: datetime
    api_provider: str  # anthropic, voyage
    model_name: str
    requests_count: int
    tokens_used: int
    cost_usd: float
    avg_response_time: float
    error_rate: float
    quota_remaining: Optional[float]

@dataclass
class PerformanceAlert:
    """Alerta de performance"""
    timestamp: datetime
    severity: str  # low, medium, high, critical
    category: str  # system, cache, api, pipeline
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    recommended_action: str

class MetricsCollector:
    """
    Coletador de métricas em tempo real do sistema
    """
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.running = False
        self.collector_thread = None
        
        # Storage for metrics
        self.system_metrics = deque(maxlen=1000)
        self.pipeline_metrics = deque(maxlen=1000)
        self.cache_metrics = deque(maxlen=1000)
        self.api_metrics = deque(maxlen=1000)
        
        # Locks for thread safety
        self.metrics_lock = threading.RLock()
        
        # Performance baselines
        self.baselines = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'cache_hit_rate': 0.70,
            'api_response_time': 10.0,
            'processing_rate': 100.0  # records per second
        }
        
        logger.info("📊 MetricsCollector initialized")
    
    def start_collection(self):
        """Inicia coleta de métricas em background"""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self.collector_thread.start()
        
        logger.info("🚀 Performance metrics collection started")
    
    def stop_collection(self):
        """Para coleta de métricas"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        
        logger.info("⏹️ Performance metrics collection stopped")
    
    def _collection_worker(self):
        """Worker thread para coleta contínua"""
        while self.running:
            try:
                # Collect system metrics
                system_metric = self._collect_system_metrics()
                
                with self.metrics_lock:
                    self.system_metrics.append(system_metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.warning(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval * 2)  # Back off on error
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Coleta métricas do sistema"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network IO
        net_io = psutil.net_io_counters()
        
        # Process info
        process = psutil.Process()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=(disk.used / disk.total) * 100,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            network_io_bytes=(net_io.bytes_sent, net_io.bytes_recv),
            active_threads=process.num_threads(),
            open_files=len(process.open_files())
        )
    
    def record_pipeline_metrics(self, stage_name: str, operation_type: str,
                              records_processed: int, processing_time: float,
                              success_rate: float, error_count: int = 0,
                              cache_hit_rate: float = 0.0, api_calls: int = 0,
                              cost_estimate: float = 0.0):
        """Registra métricas do pipeline"""
        metric = PipelineMetrics(
            timestamp=datetime.now(),
            stage_name=stage_name,
            operation_type=operation_type,
            records_processed=records_processed,
            processing_time=processing_time,
            success_rate=success_rate,
            error_count=error_count,
            cache_hit_rate=cache_hit_rate,
            api_calls_made=api_calls,
            cost_estimate_usd=cost_estimate
        )
        
        with self.metrics_lock:
            self.pipeline_metrics.append(metric)
    
    def record_cache_metrics(self, cache_type: str, cache_level: str,
                           hit_count: int, miss_count: int,
                           size_mb: float, entries_count: int,
                           evictions: int = 0, compression_ratio: float = 1.0):
        """Registra métricas de cache"""
        hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0.0
        
        metric = CacheMetrics(
            timestamp=datetime.now(),
            cache_type=cache_type,
            cache_level=cache_level,
            hit_count=hit_count,
            miss_count=miss_count,
            hit_rate=hit_rate,
            size_mb=size_mb,
            entries_count=entries_count,
            evictions_count=evictions,
            compression_ratio=compression_ratio
        )
        
        with self.metrics_lock:
            self.cache_metrics.append(metric)
    
    def record_api_metrics(self, api_provider: str, model_name: str,
                         requests_count: int, tokens_used: int,
                         cost_usd: float, avg_response_time: float,
                         error_rate: float = 0.0, quota_remaining: float = None):
        """Registra métricas de API"""
        metric = APIMetrics(
            timestamp=datetime.now(),
            api_provider=api_provider,
            model_name=model_name,
            requests_count=requests_count,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            quota_remaining=quota_remaining
        )
        
        with self.metrics_lock:
            self.api_metrics.append(metric)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais consolidadas"""
        with self.metrics_lock:
            # Get latest metrics
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            
            # Aggregate pipeline metrics (last 10 minutes)
            recent_pipeline = [
                m for m in self.pipeline_metrics 
                if (datetime.now() - m.timestamp).total_seconds() < 600
            ]
            
            # Aggregate cache metrics
            recent_cache = [
                m for m in self.cache_metrics
                if (datetime.now() - m.timestamp).total_seconds() < 600
            ]
            
            # Aggregate API metrics
            recent_api = [
                m for m in self.api_metrics
                if (datetime.now() - m.timestamp).total_seconds() < 600
            ]
        
        return {
            "system": {
                "cpu_percent": latest_system.cpu_percent if latest_system else 0,
                "memory_percent": latest_system.memory_percent if latest_system else 0,
                "memory_used_mb": latest_system.memory_used_mb if latest_system else 0,
                "disk_usage_percent": latest_system.disk_usage_percent if latest_system else 0,
                "active_threads": latest_system.active_threads if latest_system else 0
            },
            "pipeline": {
                "total_records_processed": sum(m.records_processed for m in recent_pipeline),
                "avg_processing_time": np.mean([m.processing_time for m in recent_pipeline]) if recent_pipeline else 0,
                "avg_success_rate": np.mean([m.success_rate for m in recent_pipeline]) if recent_pipeline else 0,
                "total_errors": sum(m.error_count for m in recent_pipeline),
                "stages_active": len(set(m.stage_name for m in recent_pipeline))
            },
            "cache": {
                "overall_hit_rate": np.mean([m.hit_rate for m in recent_cache]) if recent_cache else 0,
                "total_size_mb": sum(m.size_mb for m in recent_cache),
                "total_entries": sum(m.entries_count for m in recent_cache),
                "avg_compression": np.mean([m.compression_ratio for m in recent_cache]) if recent_cache else 1.0
            },
            "api": {
                "total_requests": sum(m.requests_count for m in recent_api),
                "total_tokens": sum(m.tokens_used for m in recent_api),
                "total_cost_usd": sum(m.cost_usd for m in recent_api),
                "avg_response_time": np.mean([m.avg_response_time for m in recent_api]) if recent_api else 0,
                "avg_error_rate": np.mean([m.error_rate for m in recent_api]) if recent_api else 0
            }
        }

class AlertSystem:
    """
    Sistema de alertas baseado em thresholds de performance
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts = deque(maxlen=500)
        self.alert_rules = self._create_default_alert_rules()
        self.alert_lock = threading.RLock()
        
        # Start monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info("🚨 AlertSystem initialized")
    
    def _create_default_alert_rules(self) -> Dict[str, Dict]:
        """Cria regras de alerta padrão"""
        return {
            "high_cpu": {
                "metric_path": "system.cpu_percent",
                "threshold": 85.0,
                "severity": "high",
                "message": "High CPU usage detected",
                "action": "Consider reducing batch sizes or adding processing delays"
            },
            "high_memory": {
                "metric_path": "system.memory_percent", 
                "threshold": 90.0,
                "severity": "critical",
                "message": "Critical memory usage",
                "action": "Clear caches or restart pipeline"
            },
            "low_cache_hit_rate": {
                "metric_path": "cache.overall_hit_rate",
                "threshold": 0.50,
                "severity": "medium",
                "message": "Low cache hit rate",
                "action": "Review cache configuration and TTL settings"
            },
            "slow_api_response": {
                "metric_path": "api.avg_response_time",
                "threshold": 15.0,
                "severity": "medium", 
                "message": "Slow API response times",
                "action": "Check network connection or reduce request frequency"
            },
            "high_error_rate": {
                "metric_path": "pipeline.avg_success_rate",
                "threshold": 0.90,
                "severity": "high",
                "message": "High pipeline error rate",
                "action": "Check logs for error patterns and fix underlying issues"
            },
            "high_cost": {
                "metric_path": "api.total_cost_usd",
                "threshold": 10.0,
                "severity": "medium",
                "message": "High API costs in last 10 minutes",
                "action": "Review cost optimization settings and sampling strategies"
            }
        }
    
    def start_monitoring(self):
        """Inicia monitoramento de alertas"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🔍 Alert monitoring started")
    
    def stop_monitoring(self):
        """Para monitoramento de alertas"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("⏹️ Alert monitoring stopped")
    
    def _monitoring_worker(self):
        """Worker thread para monitoramento contínuo"""
        while self.monitoring:
            try:
                self._check_alerts()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.warning(f"Error in alert monitoring: {e}")
                time.sleep(60)  # Back off on error
    
    def _check_alerts(self):
        """Verifica todas as regras de alerta"""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                metric_value = self._get_nested_value(current_metrics, rule["metric_path"])
                
                if metric_value is None:
                    continue
                
                # Check threshold (handle both > and < conditions)
                threshold_exceeded = False
                if "success_rate" in rule["metric_path"] or "hit_rate" in rule["metric_path"]:
                    # These should be high (alert if below threshold)
                    threshold_exceeded = metric_value < rule["threshold"]
                else:
                    # These should be low (alert if above threshold)
                    threshold_exceeded = metric_value > rule["threshold"]
                
                if threshold_exceeded:
                    alert = PerformanceAlert(
                        timestamp=datetime.now(),
                        severity=rule["severity"],
                        category=rule["metric_path"].split(".")[0],
                        message=rule["message"],
                        metric_name=rule["metric_path"],
                        current_value=metric_value,
                        threshold_value=rule["threshold"],
                        recommended_action=rule["action"]
                    )
                    
                    with self.alert_lock:
                        self.alerts.append(alert)
                    
                    logger.warning(f"🚨 ALERT [{rule['severity'].upper()}]: {rule['message']} "
                                 f"(Current: {metric_value:.2f}, Threshold: {rule['threshold']:.2f})")
                    
            except Exception as e:
                logger.warning(f"Error checking alert rule {rule_name}: {e}")
    
    def _get_nested_value(self, data: Dict, path: str) -> Optional[float]:
        """Extrai valor aninhado usando notação de ponto"""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                value = value[key]
            return float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None
    
    def get_recent_alerts(self, severity_filter: str = None, 
                         hours_back: int = 24) -> List[PerformanceAlert]:
        """Retorna alertas recentes"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self.alert_lock:
            filtered_alerts = [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]
            
            if severity_filter:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert.severity == severity_filter
                ]
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

class PerformanceMonitor:
    """
    Monitor principal de performance com relatórios executivos
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "performance_reports"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.alert_system = AlertSystem(self.metrics_collector)
        
        # Performance tracking
        self.session_start = datetime.now()
        self.session_stats = {
            "total_records_processed": 0,
            "total_api_calls": 0,
            "total_cost_usd": 0.0,
            "peak_memory_mb": 0.0,
            "peak_cpu_percent": 0.0
        }
        
        logger.info("🎯 PerformanceMonitor initialized")
    
    def start_monitoring(self):
        """Inicia monitoramento completo"""
        self.metrics_collector.start_collection()
        self.alert_system.start_monitoring()
        
        logger.info("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Para monitoramento completo"""
        self.metrics_collector.stop_collection()
        self.alert_system.stop_monitoring()
        
        logger.info("📊 Performance monitoring stopped")
    
    def record_stage_completion(self, stage_name: str, records_processed: int,
                              processing_time: float, success_rate: float,
                              api_calls: int = 0, cost_usd: float = 0.0):
        """Registra conclusão de stage do pipeline"""
        self.metrics_collector.record_pipeline_metrics(
            stage_name=stage_name,
            operation_type="stage_completion",
            records_processed=records_processed,
            processing_time=processing_time,
            success_rate=success_rate,
            api_calls=api_calls,
            cost_estimate=cost_usd
        )
        
        # Update session stats
        self.session_stats["total_records_processed"] += records_processed
        self.session_stats["total_api_calls"] += api_calls
        self.session_stats["total_cost_usd"] += cost_usd
        
        # Update peaks
        current_metrics = self.metrics_collector.get_current_metrics()
        system_metrics = current_metrics.get("system", {})
        
        self.session_stats["peak_memory_mb"] = max(
            self.session_stats["peak_memory_mb"],
            system_metrics.get("memory_used_mb", 0)
        )
        self.session_stats["peak_cpu_percent"] = max(
            self.session_stats["peak_cpu_percent"],
            system_metrics.get("cpu_percent", 0)
        )
        
        logger.info(f"📈 Stage completed: {stage_name} ({records_processed} records, "
                   f"{processing_time:.2f}s, {success_rate*100:.1f}% success)")
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Gera resumo executivo de performance"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        current_metrics = self.metrics_collector.get_current_metrics()
        recent_alerts = self.alert_system.get_recent_alerts(hours_back=1)
        
        # Calculate key performance indicators
        throughput = (self.session_stats["total_records_processed"] / 
                     session_duration * 60) if session_duration > 0 else 0  # records per minute
        
        cost_per_record = (self.session_stats["total_cost_usd"] / 
                          self.session_stats["total_records_processed"]) if self.session_stats["total_records_processed"] > 0 else 0
        
        # Performance grade
        cache_hit_rate = current_metrics.get("cache", {}).get("overall_hit_rate", 0)
        success_rate = current_metrics.get("pipeline", {}).get("avg_success_rate", 0)
        
        performance_score = (
            (cache_hit_rate * 0.3) +
            (success_rate * 0.3) +
            (min(1.0, throughput / 1000) * 0.2) +  # normalize throughput
            (max(0, 1.0 - self.session_stats["peak_cpu_percent"] / 100) * 0.2)
        ) * 100
        
        performance_grade = "A" if performance_score >= 85 else \
                           "B" if performance_score >= 70 else \
                           "C" if performance_score >= 55 else "D"
        
        return {
            "session_overview": {
                "start_time": self.session_start.isoformat(),
                "duration_minutes": session_duration / 60,
                "records_processed": self.session_stats["total_records_processed"],
                "throughput_records_per_minute": throughput,
                "performance_grade": performance_grade,
                "performance_score": performance_score
            },
            "resource_utilization": {
                "peak_memory_mb": self.session_stats["peak_memory_mb"],
                "peak_cpu_percent": self.session_stats["peak_cpu_percent"],
                "current_memory_percent": current_metrics.get("system", {}).get("memory_percent", 0),
                "current_cpu_percent": current_metrics.get("system", {}).get("cpu_percent", 0)
            },
            "api_usage": {
                "total_api_calls": self.session_stats["total_api_calls"],
                "total_cost_usd": self.session_stats["total_cost_usd"],
                "cost_per_record": cost_per_record,
                "avg_response_time": current_metrics.get("api", {}).get("avg_response_time", 0)
            },
            "cache_performance": {
                "overall_hit_rate": cache_hit_rate,
                "total_size_mb": current_metrics.get("cache", {}).get("total_size_mb", 0),
                "total_entries": current_metrics.get("cache", {}).get("total_entries", 0),
                "compression_ratio": current_metrics.get("cache", {}).get("avg_compression", 1.0)
            },
            "quality_indicators": {
                "avg_success_rate": success_rate,
                "total_errors": current_metrics.get("pipeline", {}).get("total_errors", 0),
                "active_stages": current_metrics.get("pipeline", {}).get("stages_active", 0)
            },
            "alerts_summary": {
                "critical_alerts": len([a for a in recent_alerts if a.severity == "critical"]),
                "high_alerts": len([a for a in recent_alerts if a.severity == "high"]),
                "medium_alerts": len([a for a in recent_alerts if a.severity == "medium"]),
                "total_alerts": len(recent_alerts)
            }
        }
    
    def save_performance_report(self, report_name: str = None) -> str:
        """Salva relatório de performance completo"""
        if not report_name:
            report_name = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        executive_summary = self.generate_executive_summary()
        current_metrics = self.metrics_collector.get_current_metrics()
        recent_alerts = self.alert_system.get_recent_alerts()
        
        report = {
            "report_metadata": {
                "report_name": report_name,
                "generated_at": datetime.now().isoformat(),
                "report_type": "comprehensive_performance_analysis"
            },
            "executive_summary": executive_summary,
            "detailed_metrics": current_metrics,
            "alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "category": alert.category,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "recommended_action": alert.recommended_action
                }
                for alert in recent_alerts
            ]
        }
        
        # Save report
        report_file = self.output_dir / f"{report_name}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Performance report saved: {report_file}")
        return str(report_file)
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Retorna dados para dashboard em tempo real"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics_collector.get_current_metrics(),
            "alerts": [
                {
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alert_system.get_recent_alerts(hours_back=1)
            ],
            "session_stats": self.session_stats,
            "performance_grade": self.generate_executive_summary()["session_overview"]["performance_grade"]
        }

# Factory functions
def create_production_monitor(config: Dict[str, Any]) -> PerformanceMonitor:
    """Cria monitor configurado para produção"""
    return PerformanceMonitor(config, "performance_reports/production")

def create_development_monitor(config: Dict[str, Any]) -> PerformanceMonitor:
    """Cria monitor configurado para desenvolvimento"""
    return PerformanceMonitor(config, "performance_reports/development")

# Global instance
_global_performance_monitor = None

def get_global_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Retorna instância global do monitor de performance"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        if config is None:
            config = {}
        _global_performance_monitor = create_production_monitor(config)
    return _global_performance_monitor