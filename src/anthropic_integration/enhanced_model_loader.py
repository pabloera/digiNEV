"""
Enhanced Model Configuration Loader v4.9.8
==========================================

Carrega configurações específicas por stage do arquivo enhanced_model_settings.yaml
para substituir claude-3-5-haiku-latest por modelos fixos otimizados.

🔧 UPGRADE: Sistema de configuração avançada por stage
✅ REPRODUTIBILIDADE: Versões fixas de modelo
🎯 OTIMIZAÇÃO: Parâmetros específicos por tarefa
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


class EnhancedModelLoader:
    """Carregador de configurações avançadas de modelos por stage"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o loader

        Args:
            config_path: Caminho para enhanced_model_settings.yaml
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Buscar na estrutura padrão do projeto
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            self.config_path = project_root / "config" / "enhanced_model_settings.yaml"

        self.config = self._load_config()
        self.stage_configs = self.config.get('anthropic_enhanced', {}).get('stage_specific_configs', {})
        self.fallback_strategies = self.config.get('anthropic_enhanced', {}).get('fallback_strategies', {})

    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Enhanced model settings carregado: {self.config_path}")
                return config
            else:
                logger.warning(f"⚠️ Arquivo enhanced_model_settings.yaml não encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"❌ Erro ao carregar enhanced_model_settings.yaml: {e}")
            return {}

    def get_stage_config(self, stage_id: str) -> Dict[str, Any]:
        """
        Obtém configuração específica para um stage

        Args:
            stage_id: ID do stage (ex: 'stage_05_political', 'stage_08_sentiment')

        Returns:
            Dicionário com configuração do stage ou configuração padrão
        """
        if stage_id in self.stage_configs:
            config = self.stage_configs[stage_id].copy()
            logger.info(f"🎯 Configuração específica para {stage_id}: {config.get('model', 'N/A')}")
            return config
        else:
            # Configuração padrão se stage específico não encontrado
            default_config = {
                'model': 'claude-3-5-sonnet-20241022',
                'temperature': 0.3,
                'max_tokens': 3000,
                'batch_size': 20
            }
            logger.warning(f"⚠️ Stage {stage_id} não encontrado, usando configuração padrão")
            return default_config

    def get_fallback_models(self, primary_model: str) -> list:
        """
        Obtém lista de modelos fallback para um modelo primário

        Args:
            primary_model: Modelo primário

        Returns:
            Lista de modelos fallback
        """
        fallbacks = self.fallback_strategies.get(primary_model, [])
        if fallbacks:
            logger.info(f"🔄 Fallbacks para {primary_model}: {fallbacks}")
        return fallbacks

    def get_performance_mode_config(self, mode: str = 'balanced') -> Dict[str, Any]:
        """
        Obtém configuração baseada no modo de performance

        Args:
            mode: 'speed', 'balanced', ou 'quality'

        Returns:
            Dicionário com configuração do modo
        """
        performance_modes = self.config.get('anthropic_enhanced', {}).get('performance_modes', {})
        if mode in performance_modes:
            config = performance_modes[mode].copy()
            logger.info(f"⚡ Modo {mode}: {config.get('preferred_model', 'N/A')}")
            return config
        else:
            logger.warning(f"⚠️ Modo {mode} não encontrado, usando balanced")
            return performance_modes.get('balanced', {})

    def get_cost_config(self) -> Dict[str, Any]:
        """Obtém configuração de custos"""
        cost_config = self.config.get('anthropic_enhanced', {}).get('cost_optimization', {})
        logger.info(f"💰 Configuração de custos carregada: budget_limit=${cost_config.get('monthly_budget_limit', 0)}")
        return cost_config

    def should_auto_downgrade(self, current_budget_usage: float) -> bool:
        """
        Verifica se deve fazer downgrade automático baseado no orçamento

        Args:
            current_budget_usage: Uso atual do orçamento (0.0 a 1.0)

        Returns:
            True se deve fazer downgrade
        """
        cost_config = self.get_cost_config()
        auto_downgrade = cost_config.get('auto_downgrade', {})
        
        if auto_downgrade.get('enable', False):
            threshold = auto_downgrade.get('budget_threshold', 0.8)
            if current_budget_usage >= threshold:
                logger.warning(f"⚠️ Auto-downgrade ativado: uso {current_budget_usage:.1%} >= {threshold:.1%}")
                return True
        
        return False

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Obtém configuração de monitoramento"""
        monitoring_config = self.config.get('anthropic_enhanced', {}).get('monitoring', {})
        logger.info(f"📊 Monitoramento configurado: {list(monitoring_config.keys())}")
        return monitoring_config

    def get_stage_from_operation(self, operation: str) -> str:
        """
        Mapeia operação para stage_id

        Args:
            operation: Nome da operação

        Returns:
            Stage ID correspondente
        """
        operation_mapping = {
            'political_analysis': 'stage_05_political',
            'sentiment_analysis': 'stage_08_sentiment',
            'network_analysis': 'stage_15_network',
            'qualitative_analysis': 'stage_16_qualitative',
            'pipeline_review': 'stage_17_review',
            'topic_interpretation': 'stage_18_topics',
            'validation': 'stage_20_validation'
        }
        
        stage_id = operation_mapping.get(operation, f'stage_{operation}')
        logger.debug(f"🔗 Operação '{operation}' mapeada para '{stage_id}'")
        return stage_id


# Singleton instance para uso global
_enhanced_loader_instance = None


def get_enhanced_model_loader(config_path: Optional[str] = None) -> EnhancedModelLoader:
    """
    Obtém instância singleton do EnhancedModelLoader

    Args:
        config_path: Caminho opcional para configuração

    Returns:
        Instância do EnhancedModelLoader
    """
    global _enhanced_loader_instance
    
    if _enhanced_loader_instance is None:
        _enhanced_loader_instance = EnhancedModelLoader(config_path)
        logger.info("🚀 EnhancedModelLoader inicializado")
    
    return _enhanced_loader_instance


def load_stage_config(stage_id: str) -> Dict[str, Any]:
    """
    Função de conveniência para carregar configuração de stage

    Args:
        stage_id: ID do stage

    Returns:
        Configuração do stage
    """
    loader = get_enhanced_model_loader()
    return loader.get_stage_config(stage_id)


def load_operation_config(operation: str) -> Dict[str, Any]:
    """
    Função de conveniência para carregar configuração por operação

    Args:
        operation: Nome da operação

    Returns:
        Configuração da operação
    """
    loader = get_enhanced_model_loader()
    stage_id = loader.get_stage_from_operation(operation)
    return loader.get_stage_config(stage_id)