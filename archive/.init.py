"""
Batch Analyzer - Sistema Independente de Análise em Lote
Análise de discurso político brasileiro com IA
"""

from .batch_analysis import (
    IntegratedBatchAnalyzer,
    BatchConfig,
    AnalysisStage
)

__version__ = "1.0.0"
__author__ = "Academic Research Team"
__all__ = [
    "IntegratedBatchAnalyzer",
    "BatchConfig",
    "AnalysisStage"
]

# Configuração padrão
DEFAULT_CONFIG = {
    "use_apis": False,  # Modo sem APIs por padrão
    "language": "pt-BR",  # Português brasileiro
    "academic_mode": True,  # Otimizações acadêmicas
    "monthly_budget": 50.0,  # Orçamento acadêmico
}