#!/usr/bin/env python3
"""
DashboardPipelineRunner: Sistema para executar pipeline diretamente pelo dashboard
Function: Executa run_pipeline.py via subprocess com monitoramento em tempo real
Usage: Interface integrada no dashboard para execução completa do pipeline de análise
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st


class PipelineStatus(Enum):
    """Status da execução do pipeline"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StageProgress:
    """Progresso de uma etapa do pipeline"""
    stage_id: str
    stage_name: str
    status: str  # pending, running, completed, error
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percent: float = 0.0
    error_message: Optional[str] = None


@dataclass
class PipelineProgress:
    """Progresso geral do pipeline"""
    status: PipelineStatus
    current_stage: int = 0
    total_stages: int = 22
    stages: List[StageProgress] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class DashboardPipelineRunner:
    """Executor de pipeline integrado ao dashboard"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Inicializa o executor do pipeline"""
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.logger = logging.getLogger(__name__)
        
        # Estado da execução
        self.progress = PipelineProgress(status=PipelineStatus.IDLE)
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_requested = False
        
        # Callbacks para atualizações
        self.progress_callbacks: List[Callable] = []
        self.log_callbacks: List[Callable] = []
        
        # Definir etapas do pipeline
        self._init_pipeline_stages()
        
        # Lock para thread safety
        self._lock = threading.Lock()
    
    def _init_pipeline_stages(self):
        """Inicializa as etapas do pipeline"""
        stages = [
            ("01_chunk_processing", "Processamento de Chunks"),
            ("02_encoding_validation", "Validação de Encoding"),
            ("03_deduplication", "Deduplicação Global"),
            ("04_feature_validation", "Validação de Features"),
            ("04b_statistical_analysis_pre", "Análise Estatística (Pré)"),
            ("05_political_analysis", "Análise Política"),
            ("06_text_cleaning", "Limpeza de Texto"),
            ("06b_statistical_analysis_post", "Análise Estatística (Pós)"),
            ("07_linguistic_processing", "Processamento Linguístico"),
            ("08_sentiment_analysis", "Análise de Sentimento"),
            ("08_5_hashtag_normalization", "Normalização de Hashtags"),
            ("09_topic_modeling", "Modelagem de Tópicos"),
            ("10_tfidf_extraction", "Extração TF-IDF"),
            ("11_clustering", "Clustering"),
            ("12_domain_analysis", "Análise de Domínio"),
            ("13_temporal_analysis", "Análise Temporal"),
            ("14_network_analysis", "Análise de Rede"),
            ("15_qualitative_analysis", "Análise Qualitativa"),
            ("16_smart_pipeline_review", "Revisão Inteligente"),
            ("17_topic_interpretation", "Interpretação de Tópicos"),
            ("18_semantic_search", "Busca Semântica"),
            ("19_pipeline_validation", "Validação do Pipeline")
        ]
        
        self.progress.stages = [
            StageProgress(stage_id=stage_id, stage_name=stage_name, status="pending")
            for stage_id, stage_name in stages
        ]
        self.progress.total_stages = len(stages)
    
    def add_progress_callback(self, callback: Callable):
        """Adiciona callback para atualizações de progresso"""
        self.progress_callbacks.append(callback)
    
    def add_log_callback(self, callback: Callable):
        """Adiciona callback para logs"""
        self.log_callbacks.append(callback)
    
    def _notify_progress_update(self):
        """Notifica callbacks sobre atualização de progresso"""
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                self.logger.error(f"Erro no callback de progresso: {e}")
    
    def _notify_log_update(self, log_line: str):
        """Notifica callbacks sobre novo log"""
        with self._lock:
            self.progress.logs.append(log_line)
        
        for callback in self.log_callbacks:
            try:
                callback(log_line)
            except Exception as e:
                self.logger.error(f"Erro no callback de log: {e}")
    
    def _parse_pipeline_output(self, line: str):
        """Analisa saída do pipeline para extrair progresso"""
        line = line.strip()
        if not line:
            return
        
        # Padrões para detectar início/fim de etapas
        stage_patterns = [
            r"Starting stage (\d+[a-z]*_\w+)",
            r"Executing stage (\d+[a-z]*_\w+)",
            r"✅.*?(\d+[a-z]*_\w+).*?completed",
            r"❌.*?(\d+[a-z]*_\w+).*?failed",
            r"Stage (\d+[a-z]*_\w+) finished"
        ]
        
        for pattern in stage_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                stage_id = match.group(1)
                self._update_stage_progress(stage_id, line)
                break
    
    def _update_stage_progress(self, stage_id: str, log_line: str):
        """Atualiza progresso de uma etapa específica"""
        with self._lock:
            # Encontrar a etapa correspondente
            stage_found = None
            stage_index = 0
            for i, stage in enumerate(self.progress.stages):
                if stage.stage_id == stage_id:
                    stage_found = stage
                    stage_index = i
                    break
            
            if not stage_found:
                return
            
            # Determinar status baseado no log
            if any(keyword in log_line.lower() for keyword in ["starting", "executing", "processing"]):
                stage_found.status = "running"
                stage_found.start_time = datetime.now()
                self.progress.current_stage = stage_index + 1
                
            elif any(keyword in log_line.lower() for keyword in ["completed", "finished", "success", "✅"]):
                stage_found.status = "completed"
                stage_found.end_time = datetime.now()
                stage_found.progress_percent = 100.0
                
            elif any(keyword in log_line.lower() for keyword in ["failed", "error", "❌"]):
                stage_found.status = "error"
                stage_found.end_time = datetime.now()
                stage_found.error_message = log_line
        
        self._notify_progress_update()
    
    def _execute_pipeline(self):
        """Executa o pipeline em thread separada"""
        try:
            with self._lock:
                self.progress.status = PipelineStatus.RUNNING
                self.progress.start_time = datetime.now()
                self.progress.logs.clear()
            
            self._notify_progress_update()
            self._notify_log_update("🚀 Iniciando execução do pipeline digiNEV...")
            
            # Preparar comando
            script_path = self.project_root / "run_pipeline.py"
            if not script_path.exists():
                raise FileNotFoundError(f"Script não encontrado: {script_path}")
            
            # Comando para executar o pipeline (usar sys.executable para robustez)
            import sys
            cmd = [sys.executable, str(script_path)]
            
            self._notify_log_update(f"📋 Executando comando: {' '.join(cmd)}")
            self._notify_log_update(f"📁 Diretório de trabalho: {self.project_root}")
            
            # Executar processo
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Ler saída em tempo real
            if self.process.stdout:
                while True:
                    if self.stop_requested:
                        self._notify_log_update("⏹️ Cancelamento solicitado...")
                        self.process.terminate()
                        break
                    
                    output = self.process.stdout.readline()
                    if output == '' and self.process.poll() is not None:
                        break
                    
                    if output:
                        self._notify_log_update(output.strip())
                        self._parse_pipeline_output(output)
            
            # Verificar resultado final
            return_code = self.process.wait()
            
            with self._lock:
                if self.stop_requested:
                    self.progress.status = PipelineStatus.CANCELLED
                    self._notify_log_update("❌ Pipeline cancelado pelo usuário")
                elif return_code == 0:
                    self.progress.status = PipelineStatus.COMPLETED
                    self.progress.end_time = datetime.now()
                    self._notify_log_update("✅ Pipeline executado com sucesso!")
                else:
                    self.progress.status = PipelineStatus.ERROR
                    self.progress.end_time = datetime.now()
                    self.progress.error_message = f"Pipeline falhou com código {return_code}"
                    self._notify_log_update(f"❌ Pipeline falhou com código de saída: {return_code}")
            
            self._notify_progress_update()
            
        except Exception as e:
            error_msg = f"Erro na execução do pipeline: {str(e)}"
            self.logger.error(error_msg)
            
            with self._lock:
                self.progress.status = PipelineStatus.ERROR
                self.progress.error_message = error_msg
                self.progress.end_time = datetime.now()
            
            self._notify_log_update(f"❌ {error_msg}")
            self._notify_progress_update()
        
        finally:
            self.process = None
    
    def start_pipeline(self) -> bool:
        """Inicia a execução do pipeline"""
        with self._lock:
            if self.progress.status == PipelineStatus.RUNNING:
                return False  # Já está executando
            
            # Reset do estado
            self.stop_requested = False
            self.progress = PipelineProgress(status=PipelineStatus.IDLE)
            self._init_pipeline_stages()
        
        # Iniciar thread de execução
        self.thread = threading.Thread(target=self._execute_pipeline, daemon=True)
        self.thread.start()
        return True
    
    def stop_pipeline(self) -> bool:
        """Para a execução do pipeline"""
        with self._lock:
            if self.progress.status != PipelineStatus.RUNNING:
                return False
            
            self.stop_requested = True
        
        if self.process:
            try:
                self.process.terminate()
                # Dar tempo para terminar gracefully
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                self.logger.error(f"Erro ao parar pipeline: {e}")
        
        return True
    
    def get_progress(self) -> PipelineProgress:
        """Retorna o progresso atual"""
        with self._lock:
            return self.progress
    
    def is_running(self) -> bool:
        """Verifica se o pipeline está executando"""
        return self.progress.status == PipelineStatus.RUNNING
    
    def get_completion_percentage(self) -> float:
        """Calcula porcentagem de conclusão"""
        if not self.progress.stages:
            return 0.0
        
        completed_stages = sum(1 for stage in self.progress.stages if stage.status == "completed")
        return (completed_stages / len(self.progress.stages)) * 100.0
    
    def get_current_stage_name(self) -> str:
        """Retorna nome da etapa atual"""
        if self.progress.current_stage > 0 and self.progress.current_stage <= len(self.progress.stages):
            return self.progress.stages[self.progress.current_stage - 1].stage_name
        return "Preparando..."
    
    def get_logs_summary(self, max_lines: int = 50) -> List[str]:
        """Retorna resumo dos logs mais recentes"""
        with self._lock:
            return self.progress.logs[-max_lines:] if self.progress.logs else []


# Instância global para uso no dashboard
_pipeline_runner = None

def get_pipeline_runner() -> DashboardPipelineRunner:
    """Retorna instância singleton do pipeline runner"""
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = DashboardPipelineRunner()
    return _pipeline_runner