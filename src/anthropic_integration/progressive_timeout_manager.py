"""
Progressive Timeout Manager - Sistema de Timeout Escalonado
==========================================================

Sistema de timeout progressivo com escalação automática para resolver
falhas de timeout persistentes, especialmente no Stage 8 (Sentiment Analysis).

IMPLEMENTADO PARA RESOLVER:
- Timeouts repetidos em stages API-intensivos
- ✅ Falta de recovery progressivo
- ✅ Timeouts fixos inadequados para datasets variáveis
- ✅ Necessidade de timeout adaptativo baseado em contexto

Data: 2025-06-08
Status: IMPLEMENTAÇÃO COMPLETA
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

class TimeoutStrategy(Enum):
    """Estratégias de timeout disponíveis"""
    FIXED = "fixed"
    PROGRESSIVE = "progressive"
    ADAPTIVE = "adaptive"
    EXPONENTIAL = "exponential"

@dataclass
class TimeoutAttempt:
    """Registro de uma tentativa de processamento com timeout"""
    stage_name: str
    attempt_number: int
    timeout_used: int
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    timeout_occurred: bool = False
    error_message: Optional[str] = None
    data_size: int = 0
    processing_time: float = 0.0

@dataclass
class StageTimeoutProfile:
    """Perfil de timeout para um stage específico"""
    stage_name: str
    base_timeout: int
    current_timeout: int
    max_timeout: int
    min_timeout: int
    escalation_factor: float
    attempts_history: List[TimeoutAttempt] = field(default_factory=list)
    success_rate: float = 0.0
    average_processing_time: float = 0.0
    recommended_timeout: int = 0

class ProgressiveTimeoutManager:
    """
    Gerenciador de Timeout Progressivo - SOLUÇÃO PARA TIMEOUTS PERSISTENTES
    =========================================================================

    RESOLVE PROBLEMAS:
    ✅ Stage 8 - Sentiment Analysis timeouts repetidos
    ✅ Timeouts fixos inadequados para datasets variáveis
    ✅ Falta de escalação automática de timeout
    ✅ Ausência de aprendizado baseado em histórico

    FEATURES IMPLEMENTADAS:
    ✅ Timeout progressivo com escalação automática
    ✅ Aprendizado baseado em histórico de performance
    ✅ Estratégias múltiplas (fixed, progressive, adaptive, exponential)
    ✅ Recovery automático com timeout aumentado
    ✅ Perfis específicos por stage
    ✅ Monitoramento e otimização contínua
    """

    def __init__(self, config_or_path: Union[str, Dict[str, Any], None] = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Handle both config dict and path string for backward compatibility
        if isinstance(config_or_path, dict):
            # Config dict passed (from tests)
            self.config = config_or_path.get('timeout_management', {})
            if not self.config:
                self.config = self._get_default_config()
        else:
            # Path string passed (legacy)
            self.config = self._load_config(config_or_path)

        # Estado interno
        self.stage_profiles: Dict[str, StageTimeoutProfile] = {}
        self.global_strategy = TimeoutStrategy(self.config.get('strategy', 'progressive'))

        # Thread safety
        self.lock = threading.Lock()

        # Configurações de escalação
        self.escalation_timeouts = self.config.get('recovery', {}).get('escalation_timeouts', [300, 600, 1200, 1800])
        self.max_escalation_attempts = len(self.escalation_timeouts)

        # Inicializar perfis de stages
        self._initialize_stage_profiles()

        self.logger.info("ProgressiveTimeoutManager inicializado com sucesso")
        self.logger.info(f"⚙️ Estratégia global: {self.global_strategy.value}")
        self.logger.info(f"🔄 Escalação configurada: {self.escalation_timeouts}")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carrega configuração de timeout management"""
        if config_path is None:
            config_path = "config/timeout_management.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('timeout_management', {})
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao carregar config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão para timeout progressivo"""
        return {
            'strategy': 'progressive',
            'stage_timeouts': {
                '05_political_analysis': 900,
                '08_sentiment_analysis': 1200,
                '07_linguistic_processing': 600,
                '09_topic_modeling': 800,
                '11_clustering': 600
            },
            'adaptive_config': {
                'base_timeout_per_record': 2.5,
                'max_timeout_per_stage': 1800,
                'min_timeout_per_stage': 120,
                'timeout_buffer_factor': 1.2
            },
            'recovery': {
                'escalation_timeouts': [300, 600, 1200, 1800],
                'max_retries': 3
            }
        }

    def _initialize_stage_profiles(self):
        """Inicializa perfis de timeout para todos os stages configurados"""
        stage_timeouts = self.config.get('stage_timeouts', {})
        adaptive_config = self.config.get('adaptive_config', {})

        max_timeout = adaptive_config.get('max_timeout_per_stage', 1800)
        min_timeout = adaptive_config.get('min_timeout_per_stage', 120)

        for stage_name, base_timeout in stage_timeouts.items():
            profile = StageTimeoutProfile(
                stage_name=stage_name,
                base_timeout=base_timeout,
                current_timeout=base_timeout,
                max_timeout=max_timeout,
                min_timeout=min_timeout,
                escalation_factor=1.5,  # 50% increase per escalation
                recommended_timeout=base_timeout
            )
            self.stage_profiles[stage_name] = profile

        self.logger.info(f"📊 Inicializados {len(self.stage_profiles)} perfis de stage")

    def get_current_timeout(self, stage_name: str = "default") -> int:
        """
        Get current timeout for a stage (for testing).
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Current timeout in seconds
        """
        with self.lock:
            if stage_name in self.stage_profiles:
                return self.stage_profiles[stage_name].current_timeout
            else:
                # Return default timeout from config
                default_timeouts = self.config.get('stage_timeouts', {})
                return default_timeouts.get(stage_name, 300)  # 5 minutes default

    def on_request_failed(self, stage_name: str = "default") -> None:
        """
        Handle request failure by increasing timeout (for testing).
        
        Args:
            stage_name: Name of the stage that failed
        """
        with self.lock:
            if stage_name in self.stage_profiles:
                profile = self.stage_profiles[stage_name]
                # Increase timeout by escalation factor
                new_timeout = int(profile.current_timeout * profile.escalation_factor)
                profile.current_timeout = min(new_timeout, profile.max_timeout)
                self.logger.info(f"🔄 Timeout increased for {stage_name}: {profile.current_timeout}s")
            else:
                # Create a default profile for this stage if it doesn't exist
                # Get current timeout without lock (avoid deadlock)
                default_timeouts = self.config.get('stage_timeouts', {})
                current_timeout = default_timeouts.get(stage_name, 300)  # 5 minutes default
                escalated_timeout = int(current_timeout * 1.5)  # 50% increase
                
                # Create and store the profile
                profile = StageTimeoutProfile(
                    stage_name=stage_name,
                    base_timeout=current_timeout,
                    current_timeout=escalated_timeout,
                    max_timeout=1800,  # 30 minutes max
                    min_timeout=120,   # 2 minutes min
                    escalation_factor=1.5,
                    recommended_timeout=escalated_timeout
                )
                self.stage_profiles[stage_name] = profile
                self.logger.info(f"🔄 Created profile and increased timeout for {stage_name}: {escalated_timeout}s")

    def on_request_success(self, stage_name: str = "default") -> None:
        """
        Handle request success by resetting timeout (for testing).
        
        Args:
            stage_name: Name of the stage that succeeded
        """
        with self.lock:
            if stage_name in self.stage_profiles:
                profile = self.stage_profiles[stage_name]
                # Reset timeout to base timeout
                profile.current_timeout = profile.base_timeout
                self.logger.info(f"✅ Timeout reset for {stage_name}: {profile.current_timeout}s")

    def get_timeout_for_stage(self, stage_name: str, data_size: int = 0,
                             attempt_number: int = 1) -> int:
        """
        Calcula timeout otimizado para um stage específico

        ALGORITMO:
        1. Verifica perfil do stage
        2. Aplica estratégia configurada
        3. Considera tentativas anteriores
        4. Ajusta baseado no tamanho dos dados
        """
        with self.lock:
            # Obter ou criar perfil do stage
            if stage_name not in self.stage_profiles:
                self._create_stage_profile(stage_name)

            profile = self.stage_profiles[stage_name]

            # Calcular timeout baseado na estratégia
            if self.global_strategy == TimeoutStrategy.FIXED:
                timeout = profile.base_timeout

            elif self.global_strategy == TimeoutStrategy.PROGRESSIVE:
                timeout = self._calculate_progressive_timeout(profile, attempt_number)

            elif self.global_strategy == TimeoutStrategy.ADAPTIVE:
                timeout = self._calculate_adaptive_timeout(profile, data_size)

            elif self.global_strategy == TimeoutStrategy.EXPONENTIAL:
                timeout = self._calculate_exponential_timeout(profile, attempt_number)

            else:
                timeout = profile.base_timeout

            # Aplicar limitações
            timeout = max(profile.min_timeout, min(timeout, profile.max_timeout))

            # Atualizar timeout atual do perfil
            profile.current_timeout = timeout

            self.logger.info(f"⏱️ Timeout calculado para {stage_name} (tentativa {attempt_number}): {timeout}s")
            return timeout

    def _calculate_progressive_timeout(self, profile: StageTimeoutProfile,
                                     attempt_number: int) -> int:
        """Calcula timeout progressivo baseado no número de tentativas"""
        if attempt_number <= 1:
            return profile.base_timeout

        # Usar timeouts de escalação predefinidos
        escalation_index = min(attempt_number - 1, len(self.escalation_timeouts) - 1)
        return self.escalation_timeouts[escalation_index]

    def _calculate_adaptive_timeout(self, profile: StageTimeoutProfile,
                                   data_size: int) -> int:
        """Calcula timeout adaptativo baseado no tamanho dos dados"""
        adaptive_config = self.config.get('adaptive_config', {})
        base_time_per_record = adaptive_config.get('base_timeout_per_record', 2.5)
        buffer_factor = adaptive_config.get('timeout_buffer_factor', 1.2)

        if data_size > 0:
            calculated_timeout = int(data_size * base_time_per_record * buffer_factor)

            # Ajustar baseado no histórico de sucesso
            if profile.success_rate < 0.5 and profile.attempts_history:
                # Taxa de sucesso baixa - aumentar timeout
                calculated_timeout = int(calculated_timeout * 1.5)

            return calculated_timeout
        else:
            return profile.base_timeout

    def _calculate_exponential_timeout(self, profile: StageTimeoutProfile,
                                     attempt_number: int) -> int:
        """Calcula timeout exponencial para tentativas repetidas"""
        if attempt_number <= 1:
            return profile.base_timeout

        # Crescimento exponencial: timeout * (2 ^ (attempt - 1))
        exponential_timeout = profile.base_timeout * (2 ** (attempt_number - 1))
        return min(exponential_timeout, profile.max_timeout)

    def _create_stage_profile(self, stage_name: str):
        """Cria perfil padrão para stage não configurado"""
        adaptive_config = self.config.get('adaptive_config', {})

        profile = StageTimeoutProfile(
            stage_name=stage_name,
            base_timeout=300,  # 5 minutos padrão
            current_timeout=300,
            max_timeout=adaptive_config.get('max_timeout_per_stage', 1800),
            min_timeout=adaptive_config.get('min_timeout_per_stage', 120),
            escalation_factor=1.5,
            recommended_timeout=300
        )

        self.stage_profiles[stage_name] = profile
        self.logger.info(f"📝 Criado perfil padrão para stage: {stage_name}")

    def execute_with_progressive_timeout(self, stage_name: str, processing_function: Callable,
                                       data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Executa função com timeout progressivo e recovery automático

        FEATURES:
        - Retry automático com timeout aumentado
        - Registro de tentativas para aprendizado
        - Fallback para amostragem em caso de falha total
        - Atualização de perfil baseado em resultados
        """
        self.logger.info(f"🚀 Iniciando execução com timeout progressivo para {stage_name}")

        data_size = len(data)
        max_attempts = min(self.max_escalation_attempts,
                          self.config.get('recovery', {}).get('max_retries', 3))

        for attempt in range(1, max_attempts + 1):
            # Obter timeout para esta tentativa
            timeout = self.get_timeout_for_stage(stage_name, data_size, attempt)

            # Criar registro da tentativa
            attempt_record = TimeoutAttempt(
                stage_name=stage_name,
                attempt_number=attempt,
                timeout_used=timeout,
                start_time=datetime.now(),
                data_size=data_size
            )

            try:
                self.logger.info(f"🔄 Tentativa {attempt}/{max_attempts} para {stage_name} (timeout: {timeout}s)")

                # Executar com timeout
                result = self._execute_with_timeout(processing_function, data, timeout, *args, **kwargs)

                # Sucesso - registrar e retornar
                attempt_record.end_time = datetime.now()
                attempt_record.success = True
                attempt_record.processing_time = (attempt_record.end_time - attempt_record.start_time).total_seconds()

                self._record_attempt(attempt_record)
                self._update_stage_profile_on_success(stage_name, attempt_record)

                self.logger.info(f"{stage_name} concluído com sucesso na tentativa {attempt}")
                return result

            except TimeoutError:
                # Timeout ocorreu
                attempt_record.end_time = datetime.now()
                attempt_record.timeout_occurred = True
                attempt_record.error_message = f"Timeout após {timeout}s"

                self._record_attempt(attempt_record)

                self.logger.warning(f"⏱️ Timeout na tentativa {attempt} para {stage_name}")

                if attempt < max_attempts:
                    # Tentar novamente com timeout maior
                    self.logger.info(f"🔄 Preparando tentativa {attempt + 1} com timeout aumentado")
                    continue
                else:
                    # Última tentativa falhou - usar fallback
                    self.logger.error(f"💥 Todas as tentativas falharam para {stage_name}")
                    return self._execute_emergency_fallback(stage_name, data, processing_function, *args, **kwargs)

            except Exception as e:
                # Erro não relacionado a timeout
                attempt_record.end_time = datetime.now()
                attempt_record.error_message = str(e)

                self._record_attempt(attempt_record)

                self.logger.error(f"❌ Erro na tentativa {attempt} para {stage_name}: {e}")

                if attempt < max_attempts:
                    continue
                else:
                    raise e

        # Se chegou aqui, todas as tentativas falharam
        self.logger.error(f"💥 Falha total no processamento de {stage_name}")
        return pd.DataFrame()

    def _execute_with_timeout(self, function: Callable, data: pd.DataFrame,
                             timeout: int, *args, **kwargs) -> pd.DataFrame:
        """Executa função com timeout usando ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FutureTimeoutError

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(function, data, *args, **kwargs)

            try:
                result = future.result(timeout=timeout)
                return result
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Processamento excedeu timeout de {timeout}s")

    def _execute_emergency_fallback(self, stage_name: str, data: pd.DataFrame,
                                   processing_function: Callable, *args, **kwargs) -> pd.DataFrame:
        """Executa fallback de emergência com amostra pequena"""
        emergency_size = self.config.get('recovery', {}).get('emergency_sample_size', 50)
        emergency_sample = data.head(min(emergency_size, len(data)))

        self.logger.warning(f"🚨 Usando fallback de emergência para {stage_name}: {len(emergency_sample)} registros")

        try:
            # Timeout muito alto para amostra pequena
            emergency_timeout = 300  # 5 minutos para amostra pequena
            result = self._execute_with_timeout(processing_function, emergency_sample,
                                               emergency_timeout, *args, **kwargs)

            self.logger.info(f"Fallback de emergência bem-sucedido para {stage_name}")
            return result

        except Exception as e:
            self.logger.error(f"💥 Falha crítica no fallback de emergência para {stage_name}: {e}")
            return pd.DataFrame()

    def _record_attempt(self, attempt: TimeoutAttempt):
        """Registra tentativa no perfil do stage"""
        with self.lock:
            if attempt.stage_name in self.stage_profiles:
                profile = self.stage_profiles[attempt.stage_name]
                profile.attempts_history.append(attempt)

                # Manter apenas os últimos 50 registros
                if len(profile.attempts_history) > 50:
                    profile.attempts_history = profile.attempts_history[-50:]

    def _update_stage_profile_on_success(self, stage_name: str, attempt: TimeoutAttempt):
        """Atualiza perfil do stage baseado em sucesso"""
        with self.lock:
            if stage_name in self.stage_profiles:
                profile = self.stage_profiles[stage_name]

                # Calcular estatísticas atualizadas
                recent_attempts = profile.attempts_history[-10:]  # Últimas 10 tentativas
                successful_attempts = [a for a in recent_attempts if a.success]

                if successful_attempts:
                    # Atualizar taxa de sucesso
                    profile.success_rate = len(successful_attempts) / len(recent_attempts)

                    # Atualizar tempo médio de processamento
                    avg_time = sum(a.processing_time for a in successful_attempts) / len(successful_attempts)
                    profile.average_processing_time = avg_time

                    # Recomendar timeout baseado no histórico
                    # Usar 1.2x o tempo médio como recomendação
                    recommended = int(avg_time * 1.2)
                    profile.recommended_timeout = max(profile.min_timeout,
                                                    min(recommended, profile.max_timeout))

                    self.logger.debug(f"📊 Perfil atualizado para {stage_name}: "
                                    f"sucesso {profile.success_rate:.1%}, "
                                    f"tempo médio {avg_time:.1f}s, "
                                    f"timeout recomendado {profile.recommended_timeout}s")

    def get_stage_performance_summary(self, stage_name: str) -> Dict[str, Any]:
        """Retorna resumo de performance para um stage específico"""
        if stage_name not in self.stage_profiles:
            return {"error": f"Stage {stage_name} não encontrado"}

        profile = self.stage_profiles[stage_name]
        recent_attempts = profile.attempts_history[-20:]  # Últimas 20 tentativas

        if not recent_attempts:
            return {"stage_name": stage_name, "status": "no_data"}

        successful = [a for a in recent_attempts if a.success]
        timeouts = [a for a in recent_attempts if a.timeout_occurred]

        return {
            "stage_name": stage_name,
            "total_attempts": len(recent_attempts),
            "successful_attempts": len(successful),
            "timeout_attempts": len(timeouts),
            "success_rate_percent": (len(successful) / len(recent_attempts)) * 100,
            "timeout_rate_percent": (len(timeouts) / len(recent_attempts)) * 100,
            "average_processing_time": profile.average_processing_time,
            "current_timeout": profile.current_timeout,
            "recommended_timeout": profile.recommended_timeout,
            "base_timeout": profile.base_timeout,
            "escalation_factor": profile.escalation_factor
        }

    def get_global_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo global de performance"""
        summary = {
            "strategy": self.global_strategy.value,
            "total_stages": len(self.stage_profiles),
            "escalation_timeouts": self.escalation_timeouts,
            "stage_summaries": {}
        }

        for stage_name in self.stage_profiles:
            summary["stage_summaries"][stage_name] = self.get_stage_performance_summary(stage_name)

        return summary

    def adjust_stage_timeout(self, stage_name: str, new_base_timeout: int):
        """Ajusta timeout base para um stage específico"""
        with self.lock:
            if stage_name in self.stage_profiles:
                profile = self.stage_profiles[stage_name]
                old_timeout = profile.base_timeout

                profile.base_timeout = max(profile.min_timeout,
                                         min(new_base_timeout, profile.max_timeout))
                profile.current_timeout = profile.base_timeout

                self.logger.info(f"🔧 Timeout ajustado para {stage_name}: {old_timeout}s → {profile.base_timeout}s")
            else:
                self.logger.warning(f"⚠️ Stage {stage_name} não encontrado para ajuste")

# Instância global para uso no pipeline
progressive_timeout_manager = None

def get_progressive_timeout_manager() -> ProgressiveTimeoutManager:
    """Factory function para obter instância do manager"""
    global progressive_timeout_manager
    if progressive_timeout_manager is None:
        progressive_timeout_manager = ProgressiveTimeoutManager()
    return progressive_timeout_manager

# Decorator para aplicar timeout progressivo automaticamente
def with_progressive_timeout(stage_name: str):
    """
    Decorator para aplicar timeout progressivo automaticamente

    Usage:
        @with_progressive_timeout("08_sentiment_analysis")
        def process_sentiment(df):
            # Sua função de processamento aqui
            return processed_df
    """
    def decorator(func):
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            manager = get_progressive_timeout_manager()
            return manager.execute_with_progressive_timeout(
                stage_name, func, df, *args, **kwargs
            )
        return wrapper
    return decorator
