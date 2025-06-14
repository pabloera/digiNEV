"""
Pipeline Monitor v4.9.1 - Sistema de Monitoramento em Tempo Real
================================================================

Monitoramento completo das 22 etapas do pipeline com gráficos de controle,
métricas de qualidade e alertas em tempo real.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Configuração de logging
logger = logging.getLogger(__name__)

class StageStatus(Enum):
    """Status possíveis de uma etapa"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PROTECTED = "protected"

@dataclass
class StageMetrics:
    """Métricas de uma etapa do pipeline"""
    stage_id: str
    stage_name: str
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    records_processed: int = 0
    records_input: int = 0
    records_output: int = 0
    success_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_messages: List[str] = None
    quality_score: float = 0.0
    api_calls_made: int = 0
    api_cost_usd: float = 0.0
    processing_rate: float = 0.0  # registros por segundo

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []

@dataclass
class PipelineSession:
    """Sessão completa de execução do pipeline"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    stages_metrics: Dict[str, StageMetrics] = None
    overall_status: StageStatus = StageStatus.PENDING
    total_records: int = 0
    overall_success_rate: float = 0.0
    total_api_cost: float = 0.0
    dataset_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.stages_metrics is None:
            self.stages_metrics = {}
        if self.dataset_info is None:
            self.dataset_info = {}

class PipelineMonitor:
    """Monitor principal do pipeline com todas as 22 etapas"""

    # Definição das 22 etapas do pipeline v4.9.1
    PIPELINE_STAGES = {
        "01_chunk_processing": {
            "name": "Chunk Processing",
            "category": "preprocessing",
            "expected_duration": 180,
            "critical": True,
            "description": "Dataset division into processable chunks"
        },
        "02_encoding_validation": {
            "name": "Encoding Validation",
            "category": "data_quality",
            "expected_duration": 300,
            "critical": True,
            "description": "Encoding detection and correction with chardet"
        },
        "03_deduplication": {
            "name": "Global Deduplication",
            "category": "data_quality",
            "expected_duration": 240,
            "critical": True,
            "description": "Duplicate removal with multiple strategies"
        },
        "04_feature_validation": {
            "name": "Feature Validation",
            "category": "data_quality",
            "expected_duration": 180,
            "critical": True,
            "description": "Column integrity verification"
        },
        "04b_statistical_analysis_pre": {
            "name": "Statistical Analysis (Pre)",
            "category": "analysis",
            "expected_duration": 240,
            "critical": False,
            "description": "Statistics before text cleaning"
        },
        "05_political_analysis": {
            "name": "Political Analysis",
            "category": "ai_processing",
            "expected_duration": 900,
            "critical": True,
            "description": "Political classification with claude-3-5-haiku-20241022"
        },
        "06_text_cleaning": {
            "name": "Text Cleaning",
            "category": "preprocessing",
            "expected_duration": 300,
            "critical": True,
            "description": "Graduated cleaning with robust validation"
        },
        "06b_statistical_analysis_post": {
            "name": "Statistical Analysis (Post)",
            "category": "analysis",
            "expected_duration": 300,
            "critical": False,
            "description": "Statistics after cleaning and comparison"
        },
        "07_linguistic_processing": {
            "name": "Linguistic Processing",
            "category": "nlp_processing",
            "expected_duration": 600,
            "critical": True,
            "description": "Analysis with spaCy pt_core_news_lg"
        },
        "08_sentiment_analysis": {
            "name": "Sentiment Analysis",
            "category": "ai_processing",
            "expected_duration": 1200,
            "critical": True,
            "description": "Optimized sentiment analysis"
        },
        "09_topic_modeling": {
            "name": "Topic Modeling",
            "category": "ai_processing",
            "expected_duration": 800,
            "critical": True,
            "description": "Topic modeling com Voyage.ai"
        },
        "10_tfidf_extraction": {
            "name": "TF-IDF Extraction",
            "category": "feature_engineering",
            "expected_duration": 400,
            "critical": False,
            "description": "Semantic TF-IDF with Voyage.ai"
        },
        "11_clustering": {
            "name": "Semantic Clustering",
            "category": "ai_processing",
            "expected_duration": 600,
            "critical": False,
            "description": "Clustering with Voyage.ai embeddings"
        },
        "12_hashtag_normalization": {
            "name": "Hashtag Normalization",
            "category": "preprocessing",
            "expected_duration": 300,
            "critical": False,
            "description": "Political hashtag standardization"
        },
        "13_domain_analysis": {
            "name": "Domain Analysis",
            "category": "analysis",
            "expected_duration": 360,
            "critical": False,
            "description": "Domain and source analysis"
        },
        "14_temporal_analysis": {
            "name": "Temporal Analysis",
            "category": "analysis",
            "expected_duration": 420,
            "critical": False,
            "description": "Temporal patterns and seasonality"
        },
        "15_network_analysis": {
            "name": "Network Analysis",
            "category": "analysis",
            "expected_duration": 480,
            "critical": False,
            "description": "Social network and interaction analysis"
        },
        "16_qualitative_analysis": {
            "name": "Qualitative Analysis",
            "category": "ai_processing",
            "expected_duration": 540,
            "critical": False,
            "description": "AI-powered qualitative analysis"
        },
        "17_smart_pipeline_review": {
            "name": "Smart Review",
            "category": "validation",
            "expected_duration": 300,
            "critical": False,
            "description": "Automated quality review"
        },
        "18_topic_interpretation": {
            "name": "Topic Interpretation",
            "category": "ai_processing",
            "expected_duration": 360,
            "critical": False,
            "description": "Semantic topic interpretation"
        },
        "19_semantic_search": {
            "name": "Semantic Search",
            "category": "ai_processing",
            "expected_duration": 480,
            "critical": False,
            "description": "Semantic search engine with Voyage.ai"
        },
        "20_pipeline_validation": {
            "name": "Final Validation",
            "category": "validation",
            "expected_duration": 240,
            "critical": True,
            "description": "Final quality and integrity validation"
        }
    }

    def __init__(self, project_root: Path):
        """Initialize pipeline monitor"""
        self.project_root = project_root
        self.checkpoints_dir = project_root / "checkpoints"
        self.logs_dir = project_root / "logs"
        self.dashboard_data_dir = project_root / "src" / "dashboard" / "data"

        # Criar diretórios se não existirem
        for directory in [self.checkpoints_dir, self.logs_dir, self.dashboard_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[PipelineSession] = None
        self.sessions_history: List[PipelineSession] = []

    def load_current_session(self) -> Optional[PipelineSession]:
        """Carrega a sessão atual do pipeline"""
        try:
            # Carregar checkpoints principais
            checkpoints_file = self.checkpoints_dir / "checkpoints.json"
            if checkpoints_file.exists():
                with open(checkpoints_file, 'r', encoding='utf-8') as f:
                    checkpoints_data = json.load(f)

                # Converter para sessão
                session = self._convert_checkpoints_to_session(checkpoints_data)
                self.current_session = session
                return session

        except Exception as e:
            logger.error(f"Erro carregando sessão atual: {e}")

        return None

    def _convert_checkpoints_to_session(self, checkpoints_data: Dict) -> PipelineSession:
        """Converte dados de checkpoints para objeto de sessão"""
        execution_summary = checkpoints_data.get('execution_summary', {})

        session = PipelineSession(
            session_id=execution_summary.get('session_id', f"session_{int(time.time())}"),
            start_time=datetime.fromisoformat(execution_summary.get('start_time', datetime.now().isoformat())),
            total_records=execution_summary.get('total_records', 0),
            overall_success_rate=execution_summary.get('overall_progress', 0.0)
        )

        # Converter etapas
        stages_data = checkpoints_data.get('stages', {})
        for stage_id, stage_data in stages_data.items():
            if stage_id in self.PIPELINE_STAGES:
                stage_info = self.PIPELINE_STAGES[stage_id]

                # Determinar status
                status = StageStatus.PENDING
                if stage_data.get('status') == 'completed':
                    status = StageStatus.COMPLETED
                elif stage_data.get('status') == 'running':
                    status = StageStatus.RUNNING
                elif stage_data.get('status') == 'failed':
                    status = StageStatus.FAILED

                # Criar métricas da etapa
                metrics = StageMetrics(
                    stage_id=stage_id,
                    stage_name=stage_info['name'],
                    status=status,
                    start_time=datetime.fromisoformat(stage_data.get('start_time', datetime.now().isoformat())),
                    end_time=datetime.fromisoformat(stage_data.get('end_time', datetime.now().isoformat())) if stage_data.get('end_time') else None,
                    duration=stage_data.get('duration', 0),
                    records_processed=stage_data.get('records_processed', 0),
                    success_rate=stage_data.get('success_rate', 0.0),
                    quality_score=stage_data.get('quality_score', 0.0)
                )

                session.stages_metrics[stage_id] = metrics

        return session

    def get_pipeline_overview(self) -> Dict[str, Any]:
        """Retorna visão geral do pipeline"""
        if not self.current_session:
            self.load_current_session()

        if not self.current_session:
            return self._get_empty_overview()

        session = self.current_session

        # Calcular estatísticas gerais
        total_stages = len(self.PIPELINE_STAGES)
        completed_stages = sum(1 for metrics in session.stages_metrics.values()
                             if metrics.status == StageStatus.COMPLETED)
        running_stages = sum(1 for metrics in session.stages_metrics.values()
                           if metrics.status == StageStatus.RUNNING)
        failed_stages = sum(1 for metrics in session.stages_metrics.values()
                          if metrics.status == StageStatus.FAILED)

        # Calcular tempo total estimado
        total_estimated_time = sum(stage['expected_duration'] for stage in self.PIPELINE_STAGES.values())
        elapsed_time = sum(metrics.duration or 0 for metrics in session.stages_metrics.values())

        # Calcular progresso por categoria
        categories = {}
        for stage_id, stage_info in self.PIPELINE_STAGES.items():
            category = stage_info['category']
            if category not in categories:
                categories[category] = {'total': 0, 'completed': 0}

            categories[category]['total'] += 1
            if stage_id in session.stages_metrics and session.stages_metrics[stage_id].status == StageStatus.COMPLETED:
                categories[category]['completed'] += 1

        return {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'running_stages': running_stages,
            'failed_stages': failed_stages,
            'overall_progress': completed_stages / total_stages,
            'total_records': session.total_records,
            'elapsed_time': elapsed_time,
            'estimated_total_time': total_estimated_time,
            'estimated_remaining_time': max(0, total_estimated_time - elapsed_time),
            'categories_progress': categories,
            'current_stage': self._get_current_stage(),
            'next_stage': self._get_next_stage(),
            'critical_stages_status': self._get_critical_stages_status()
        }

    def _get_empty_overview(self) -> Dict[str, Any]:
        """Retorna overview vazio quando não há sessão ativa"""
        return {
            'session_id': 'Nenhuma sessão ativa',
            'start_time': None,
            'total_stages': len(self.PIPELINE_STAGES),
            'completed_stages': 0,
            'running_stages': 0,
            'failed_stages': 0,
            'overall_progress': 0.0,
            'total_records': 0,
            'elapsed_time': 0,
            'estimated_total_time': sum(stage['expected_duration'] for stage in self.PIPELINE_STAGES.values()),
            'estimated_remaining_time': sum(stage['expected_duration'] for stage in self.PIPELINE_STAGES.values()),
            'categories_progress': {},
            'current_stage': None,
            'next_stage': list(self.PIPELINE_STAGES.keys())[0] if self.PIPELINE_STAGES else None,
            'critical_stages_status': {}
        }

    def _get_current_stage(self) -> Optional[str]:
        """Identifica a etapa atualmente em execução"""
        if not self.current_session:
            return None

        for stage_id, metrics in self.current_session.stages_metrics.items():
            if metrics.status == StageStatus.RUNNING:
                return stage_id
        return None

    def _get_next_stage(self) -> Optional[str]:
        """Identifica a próxima etapa a ser executada"""
        if not self.current_session:
            return list(self.PIPELINE_STAGES.keys())[0]

        for stage_id in self.PIPELINE_STAGES.keys():
            if stage_id not in self.current_session.stages_metrics:
                return stage_id
            if self.current_session.stages_metrics[stage_id].status == StageStatus.PENDING:
                return stage_id
        return None

    def _get_critical_stages_status(self) -> Dict[str, str]:
        """Status das etapas críticas"""
        critical_status = {}

        for stage_id, stage_info in self.PIPELINE_STAGES.items():
            if stage_info.get('critical', False):
                if self.current_session and stage_id in self.current_session.stages_metrics:
                    status = self.current_session.stages_metrics[stage_id].status.value
                else:
                    status = 'pending'
                critical_status[stage_id] = status

        return critical_status

    def get_stage_details(self, stage_id: str) -> Dict[str, Any]:
        """Retorna detalhes específicos de uma etapa"""
        if stage_id not in self.PIPELINE_STAGES:
            return {}

        stage_info = self.PIPELINE_STAGES[stage_id]

        if self.current_session and stage_id in self.current_session.stages_metrics:
            metrics = self.current_session.stages_metrics[stage_id]

            return {
                'stage_id': stage_id,
                'name': stage_info['name'],
                'category': stage_info['category'],
                'description': stage_info['description'],
                'critical': stage_info.get('critical', False),
                'expected_duration': stage_info['expected_duration'],
                'status': metrics.status.value,
                'start_time': metrics.start_time,
                'end_time': metrics.end_time,
                'duration': metrics.duration,
                'records_processed': metrics.records_processed,
                'records_input': metrics.records_input,
                'records_output': metrics.records_output,
                'success_rate': metrics.success_rate,
                'quality_score': metrics.quality_score,
                'processing_rate': metrics.processing_rate,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'api_calls_made': metrics.api_calls_made,
                'api_cost_usd': metrics.api_cost_usd,
                'error_messages': metrics.error_messages,
                'efficiency': self._calculate_stage_efficiency(metrics, stage_info)
            }
        else:
            return {
                'stage_id': stage_id,
                'name': stage_info['name'],
                'category': stage_info['category'],
                'description': stage_info['description'],
                'critical': stage_info.get('critical', False),
                'expected_duration': stage_info['expected_duration'],
                'status': 'pending',
                'start_time': None,
                'end_time': None,
                'duration': 0,
                'records_processed': 0,
                'efficiency': 0.0
            }

    def _calculate_stage_efficiency(self, metrics: StageMetrics, stage_info: Dict) -> float:
        """Calcula eficiência da etapa (tempo real vs esperado)"""
        if not metrics.duration or metrics.duration == 0:
            return 0.0

        expected_duration = stage_info['expected_duration']
        efficiency = expected_duration / metrics.duration
        return min(efficiency, 2.0)  # Cap at 200% efficiency

    def get_timeline_data(self) -> List[Dict[str, Any]]:
        """Retorna dados para timeline do pipeline"""
        if not self.current_session:
            self.load_current_session()

        timeline_data = []

        for stage_id, stage_info in self.PIPELINE_STAGES.items():
            stage_data = {
                'stage_id': stage_id,
                'name': stage_info['name'],
                'category': stage_info['category'],
                'expected_duration': stage_info['expected_duration'],
                'critical': stage_info.get('critical', False)
            }

            if self.current_session and stage_id in self.current_session.stages_metrics:
                metrics = self.current_session.stages_metrics[stage_id]
                stage_data.update({
                    'status': metrics.status.value,
                    'start_time': metrics.start_time,
                    'end_time': metrics.end_time,
                    'duration': metrics.duration,
                    'success_rate': metrics.success_rate,
                    'quality_score': metrics.quality_score
                })
            else:
                stage_data.update({
                    'status': 'pending',
                    'start_time': None,
                    'end_time': None,
                    'duration': 0,
                    'success_rate': 0.0,
                    'quality_score': 0.0
                })

            timeline_data.append(stage_data)

        return timeline_data
