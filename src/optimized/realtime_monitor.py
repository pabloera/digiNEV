"""
Simple realtime monitor fallback for pipeline compatibility
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SimpleRealtimeMonitor:
    """Minimal realtime monitor for pipeline compatibility"""
    
    def __init__(self):
        self.alerts = []
        self.metrics = {}
        
    def log_alert(self, level: AlertLevel, message: str):
        """Log an alert"""
        self.alerts.append({'level': level.value, 'message': message})
        logger.info(f"Alert [{level.value}]: {message}")
    
    def record_metric(self, name: str, value: Any):
        """Record a metric"""
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        return {
            'alerts_count': len(self.alerts),
            'metrics_count': len(self.metrics),
            'latest_alerts': self.alerts[-5:] if self.alerts else []
        }

# Global instance
_global_monitor = None

def get_global_realtime_monitor() -> SimpleRealtimeMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SimpleRealtimeMonitor()
    return _global_monitor