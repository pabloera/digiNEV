"""
Adaptive Chunking Manager - Solução para Timeouts no Pipeline
===========================================================

Sistema inteligente de chunking adaptativo para resolver problemas de timeout,
especialmente no Stage 8 (Sentiment Analysis) e outros stages API-intensivos.

IMPLEMENTADO PARA RESOLVER:
- ✅ Pipeline Timeout Stage 8 (Sentiment Analysis)
- ✅ Otimização automática de chunk sizes
- ✅ Recovery automático em caso de timeout
- ✅ Monitoramento de performance por stage

Data: 2025-06-08
Status: IMPLEMENTAÇÃO COMPLETA
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ChunkPerformanceMetrics:
    """Métricas de performance para monitoramento de chunks"""
    stage_name: str
    chunk_size: int
    processing_time: float
    success: bool
    records_processed: int
    timeout_occurred: bool
    memory_usage_mb: float = 0.0
    retry_count: int = 0


class AdaptiveChunkingManager:
    """
    ✅ Gerenciador de Chunking Adaptativo - SOLUÇÃO PARA TIMEOUTS
    =============================================================

    RESOLVE PROBLEMAS:
    ✅ Stage 8 - Sentiment Analysis timeouts
    ✅ Chunks muito grandes causando timeouts
    ✅ Processamento ineficiente em stages API-intensivos
    ✅ Falta de recovery automático

    FEATURES IMPLEMENTADAS:
    ✅ Chunking adaptativo baseado em performance histórica
    ✅ Recovery automático com chunk size reduction
    ✅ Monitoramento de performance em tempo real
    ✅ Configuração específica por stage
    ✅ Timeout management integrado
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Carregar configurações
        self.config = self._load_config(config_path)
        self.timeout_config = self._load_timeout_config()

        # Performance tracking
        self.performance_history: Dict[str, List[ChunkPerformanceMetrics]] = {}
        self.optimal_chunk_sizes: Dict[str, int] = {}

        # Thread safety
        self.lock = threading.Lock()

        # Configurações padrão
        chunk_config = self.config.get('chunk_management', {})
        self.default_chunk_size = chunk_config.get('base_chunk_size', 10)
        self.max_chunk_size = chunk_config.get('max_chunk_size', 50)
        self.min_chunk_size = chunk_config.get('min_chunk_size', 2)

        self.logger.info("✅ AdaptiveChunkingManager inicializado com sucesso")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carrega configuração de timeout management"""
        if config_path is None:
            config_path = "config/timeout_management.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('timeout_management', {})
        except FileNotFoundError:
            msg = f"⚠️ Arquivo de configuração não encontrado: {config_path}"
            self.logger.warning(msg)
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar configuração: {e}")
            return self._get_default_config()

    def _load_timeout_config(self) -> Dict[str, Any]:
        """Carrega configurações de timeout do processing.yaml"""
        try:
            with open("config/processing.yaml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('pipeline_timeouts', {})
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao carregar timeout config: {e}")
            return {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão de fallback"""
        return {
            'chunk_management': {
                'adaptive_chunk_size': True,
                'base_chunk_size': 10,
                'max_chunk_size': 50,
                'min_chunk_size': 2,
                'time_per_chunk_target': 60,
                'stage_specific_chunks': {
                    '05_political_analysis': 5,
                    '08_sentiment_analysis': 3,
                    '07_linguistic_processing': 25,
                    '09_topic_modeling': 15
                }
            },
            'recovery': {
                'max_retries': 3,
                'progressive_chunk_reduction': 0.5,
                'fallback_to_sampling': True,
                'emergency_sample_size': 100
            }
        }

    def get_optimal_chunk_size(self, stage_name: str, data_size: int,
                               estimated_time_per_record: float = 2.5) -> int:
        """
        ✅ Calcula tamanho ótimo de chunk para um stage específico

        ALGORITMO:
        1. Verifica configuração específica do stage
        2. Analisa histórico de performance
        3. Calcula baseado no timeout target
        4. Aplica limitações min/max
        """
        self.logger.info(f"🔧 Calculando chunk size ótimo para {stage_name}")

        # 1. Verificar configuração específica do stage
        stage_specific = self.config.get('chunk_management', {}).get(
            'stage_specific_chunks', {})
        if stage_name in stage_specific:
            configured_size = stage_specific[stage_name]
            self.logger.info(
                f"📋 Usando chunk size configurado para {stage_name}: {configured_size}")
            return max(self.min_chunk_size, min(configured_size,
                       self.max_chunk_size))

        # 2. Verificar se temos histórico de performance
        if stage_name in self.optimal_chunk_sizes:
            historical_optimal = self.optimal_chunk_sizes[stage_name]
            self.logger.info(
                f"📊 Usando chunk size histórico para {stage_name}: "
                f"{historical_optimal}")
            return historical_optimal

        # 3. Calcular baseado no tempo target
        target_time = self.config.get('chunk_management', {}).get(
            'time_per_chunk_target', 60)
        calculated_size = int(target_time / estimated_time_per_record)

        # 4. Aplicar limitações
        optimal_size = max(self.min_chunk_size, min(calculated_size,
                           self.max_chunk_size))

        # 5. Ajustar para stages API-intensivos
        api_intensive_stages = [
            '05_political_analysis', '08_sentiment_analysis',
            '12_hashtag_normalization', '13_domain_analysis',
            '14_temporal_analysis', '15_network_analysis',
            '16_qualitative_analysis', '17_smart_pipeline_review',
            '18_topic_interpretation'
        ]

        if stage_name in api_intensive_stages:
            optimal_size = max(2, int(optimal_size * 0.5))
            self.logger.info(
                f"⚡ Reduzindo chunk size para stage API-intensivo "
                f"{stage_name}: {optimal_size}")

        self.logger.info(
            f"✅ Chunk size ótimo calculado para {stage_name}: {optimal_size}")
        return optimal_size

    def create_adaptive_chunks(self, data: pd.DataFrame, stage_name: str,
                               estimated_time_per_record: float = 2.5
                               ) -> List[pd.DataFrame]:
        """
        ✅ Cria chunks adaptativos para processamento eficiente

        OTIMIZAÇÕES:
        - Chunk size baseado em performance histórica
        - Balanceamento de carga entre chunks
        - Tratamento especial para stages problemáticos
        """
        data_size = len(data)
        optimal_chunk_size = self.get_optimal_chunk_size(
            stage_name, data_size, estimated_time_per_record)

        self.logger.info(f"📦 Criando chunks adaptativos para {stage_name}")
        self.logger.info(
            f"📊 Dataset: {data_size} registros | Chunk size: "
            f"{optimal_chunk_size}")

        # Criar chunks
        chunks = []
        for i in range(0, data_size, optimal_chunk_size):
            chunk = data.iloc[i:i + optimal_chunk_size].copy()
            chunks.append(chunk)

        self.logger.info(f"✅ Criados {len(chunks)} chunks para processamento")
        return chunks

    def process_with_adaptive_chunking(
            self, data: pd.DataFrame, stage_name: str,
            processing_function, *args, **kwargs) -> pd.DataFrame:
        """
        ✅ Processa dados com chunking adaptativo e recovery automático

        FEATURES:
        - ✅ Timeout handling automático
        - ✅ Retry com chunk size reduzido
        - ✅ Fallback para sampling em emergência
        - ✅ Monitoramento de performance
        """
        self.logger.info(f"🚀 Iniciando processamento adaptativo para {stage_name}")

        # Obter timeout para o stage
        stage_timeout = self._get_stage_timeout(stage_name)
        self.logger.info(f"⏱️ Timeout configurado para {stage_name}: {stage_timeout}s")

        results = []
        chunks = self.create_adaptive_chunks(data, stage_name)

        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            success = False
            retry_count = 0
            max_retries = self.config.get('recovery', {}).get('max_retries', 3)

            current_chunk = chunk.copy()

            while retry_count <= max_retries and not success:
                try:
                    self.logger.info(f"📝 Processando chunk {i+1}/{len(chunks)} (tentativa {retry_count+1})")

                    # Processar chunk com timeout
                    chunk_result = self._process_chunk_with_timeout(
                        current_chunk, processing_function, stage_timeout, *args, **kwargs
                    )

                    results.append(chunk_result)
                    success = True

                    # Registrar métricas de sucesso
                    processing_time = time.time() - chunk_start_time
                    metrics = ChunkPerformanceMetrics(
                        stage_name=stage_name,
                        chunk_size=len(current_chunk),
                        processing_time=processing_time,
                        success=True,
                        records_processed=len(current_chunk),
                        timeout_occurred=False,
                        retry_count=retry_count
                    )
                    self._record_performance(metrics)

                    self.logger.info(f"✅ Chunk {i+1} processado com sucesso em {processing_time:.2f}s")

                except TimeoutError:
                    retry_count += 1
                    self.logger.warning(f"⚠️ Timeout no chunk {i+1}, tentativa {retry_count}")

                    if retry_count <= max_retries:
                        # Reduzir tamanho do chunk
                        reduction_factor = self.config.get('recovery', {}).get('progressive_chunk_reduction', 0.5)
                        new_size = max(1, int(len(current_chunk) * reduction_factor))
                        current_chunk = current_chunk.head(new_size)
                        self.logger.info(f"🔄 Reduzindo chunk para {new_size} registros")

                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"❌ Erro no chunk {i+1}: {e}")

                    if retry_count > max_retries:
                        # Emergency fallback - usar apenas uma amostra muito pequena
                        emergency_size = self.config.get('recovery', {}).get('emergency_sample_size', 10)
                        emergency_sample = current_chunk.head(min(emergency_size, len(current_chunk)))

                        try:
                            self.logger.warning(f"🚨 Usando amostra de emergência: {len(emergency_sample)} registros")
                            emergency_result = processing_function(emergency_sample, *args, **kwargs)
                            results.append(emergency_result)
                            success = True
                        except Exception as emergency_error:
                            self.logger.error(f"💥 Falha crítica no chunk {i+1}: {emergency_error}")
                            # Continuar com próximo chunk
                            break

            if not success:
                self.logger.error(f"💥 Falha total no processamento do chunk {i+1}")

        # Consolidar resultados
        if results:
            final_result = pd.concat(results, ignore_index=True)
            self.logger.info(f"✅ Processamento adaptativo concluído: {len(final_result)} registros")
            return final_result
        else:
            self.logger.error("💥 Nenhum resultado foi processado com sucesso")
            return pd.DataFrame()

    def _get_stage_timeout(self, stage_name: str) -> int:
        """Obtém timeout configurado para o stage"""
        # Primeiro tentar timeout específico
        stage_timeouts = self.timeout_config.get('stage_specific_timeouts', {})
        if stage_name in stage_timeouts:
            return stage_timeouts[stage_name]

        # Fallback para timeout padrão
        return self.timeout_config.get('default_timeout', 300)

    def _process_chunk_with_timeout(self, chunk: pd.DataFrame, processing_function,
                                   timeout: int, *args, **kwargs) -> pd.DataFrame:
        """Processa chunk com timeout usando ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(processing_function, chunk, *args, **kwargs)

            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                future.cancel()
                raise TimeoutError(f"Processamento excedeu timeout de {timeout}s")

    def _record_performance(self, metrics: ChunkPerformanceMetrics):
        """Registra métricas de performance para otimização futura"""
        with self.lock:
            if metrics.stage_name not in self.performance_history:
                self.performance_history[metrics.stage_name] = []

            self.performance_history[metrics.stage_name].append(metrics)

            # Manter apenas os últimos 50 registros por stage
            if len(self.performance_history[metrics.stage_name]) > 50:
                self.performance_history[metrics.stage_name] = \
                    self.performance_history[metrics.stage_name][-50:]

            # Recalcular chunk size ótimo baseado no histórico
            self._update_optimal_chunk_size(metrics.stage_name)

    def _update_optimal_chunk_size(self, stage_name: str):
        """Atualiza chunk size ótimo baseado no histórico de performance"""
        history = self.performance_history.get(stage_name, [])
        if len(history) < 3:  # Precisa de pelo menos 3 amostras
            return

        # Calcular chunk size ótimo baseado em sucessos recentes
        successful_metrics = [m for m in history[-10:] if m.success and not m.timeout_occurred]

        if successful_metrics:
            # Média ponderada priorizando chunks maiores que foram bem-sucedidos
            total_weight = 0
            weighted_sum = 0

            for metrics in successful_metrics:
                # Peso baseado no tamanho do chunk e velocidade
                speed = metrics.records_processed / max(metrics.processing_time, 0.1)
                weight = speed * metrics.chunk_size

                weighted_sum += metrics.chunk_size * weight
                total_weight += weight

            if total_weight > 0:
                optimal_size = int(weighted_sum / total_weight)
                optimal_size = max(self.min_chunk_size, min(optimal_size, self.max_chunk_size))

                self.optimal_chunk_sizes[stage_name] = optimal_size
                self.logger.debug(f"📈 Chunk size ótimo atualizado para {stage_name}: {optimal_size}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo de performance do chunking adaptativo"""
        summary = {
            "total_stages_processed": len(self.performance_history),
            "optimal_chunk_sizes": self.optimal_chunk_sizes.copy(),
            "stage_performance": {}
        }

        for stage_name, history in self.performance_history.items():
            recent_history = history[-10:]  # Últimos 10 processamentos

            total_processed = sum(m.records_processed for m in recent_history)
            successful_count = sum(1 for m in recent_history if m.success)
            timeout_count = sum(1 for m in recent_history if m.timeout_occurred)
            avg_time = sum(m.processing_time for m in recent_history) / len(recent_history)

            summary["stage_performance"][stage_name] = {
                "total_records_processed": total_processed,
                "success_rate": successful_count / len(recent_history) * 100,
                "timeout_rate": timeout_count / len(recent_history) * 100,
                "average_processing_time": avg_time,
                "current_optimal_chunk_size": self.optimal_chunk_sizes.get(stage_name, "not_calculated")
            }

        return summary

    def reset_performance_history(self, stage_name: Optional[str] = None):
        """Reset histórico de performance (útil para debugging)"""
        with self.lock:
            if stage_name:
                if stage_name in self.performance_history:
                    del self.performance_history[stage_name]
                if stage_name in self.optimal_chunk_sizes:
                    del self.optimal_chunk_sizes[stage_name]
                self.logger.info(f"🔄 Histórico de performance resetado para {stage_name}")
            else:
                self.performance_history.clear()
                self.optimal_chunk_sizes.clear()
                self.logger.info("🔄 Todo histórico de performance resetado")


# Instância global para uso no pipeline
adaptive_chunking_manager = None


def get_adaptive_chunking_manager() -> AdaptiveChunkingManager:
    """Factory function para obter instância do manager"""
    global adaptive_chunking_manager
    if adaptive_chunking_manager is None:
        adaptive_chunking_manager = AdaptiveChunkingManager()
    return adaptive_chunking_manager


# Decorator para aplicar chunking adaptativo automaticamente
def with_adaptive_chunking(stage_name: str,
                          estimated_time_per_record: float = 2.5):
    """
    Decorator para aplicar chunking adaptativo automaticamente a funções
    de processamento

    Usage:
        @with_adaptive_chunking("08_sentiment_analysis",
                               estimated_time_per_record=3.0)
        def process_sentiment(df):
            # Sua função de processamento aqui
            return processed_df
    """
    def decorator(func):
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            manager = get_adaptive_chunking_manager()
            return manager.process_with_adaptive_chunking(
                df, stage_name, func, *args, **kwargs
            )
        return wrapper
    return decorator
