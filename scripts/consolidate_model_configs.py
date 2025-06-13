#!/usr/bin/env python3
"""
Script de Consolidação de Configurações de Modelos v4.9.8
=========================================================

Consolida as configurações entre settings.yaml e enhanced_model_settings.yaml,
garantindo compatibilidade e atualizando referências nos componentes.

🔧 UPGRADE: Sistema unificado de configuração de modelos
✅ COMPATIBILIDADE: Mantém fallbacks para configurações antigas  
🎯 OTIMIZAÇÃO: Carrega configurações específicas por stage
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Encontrar raiz do projeto
script_dir = Path(__file__).parent
project_root = script_dir.parent

def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Carrega configuração YAML"""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuração carregada: {file_path}")
            return config
        else:
            logger.warning(f"⚠️ Arquivo não encontrado: {file_path}")
            return {}
    except Exception as e:
        logger.error(f"❌ Erro ao carregar {file_path}: {e}")
        return {}

def save_yaml_config(config: Dict[str, Any], file_path: Path) -> bool:
    """Salva configuração YAML"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        logger.info(f"✅ Configuração salva: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao salvar {file_path}: {e}")
        return False

def consolidate_model_configurations():
    """
    Consolida configurações de modelos entre arquivos
    """
    logger.info("🔧 Iniciando consolidação de configurações de modelos...")
    
    # Carregar arquivos de configuração
    settings_path = project_root / "config" / "settings.yaml"
    enhanced_path = project_root / "config" / "enhanced_model_settings.yaml"
    
    settings_config = load_yaml_config(settings_path)
    enhanced_config = load_yaml_config(enhanced_path)
    
    if not settings_config:
        logger.error("❌ Não foi possível carregar settings.yaml")
        return False
    
    if not enhanced_config:
        logger.error("❌ Não foi possível carregar enhanced_model_settings.yaml")
        return False
    
    # Backup das configurações atuais
    backup_path = project_root / "config" / f"settings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    save_yaml_config(settings_config, backup_path)
    logger.info(f"💾 Backup criado: {backup_path}")
    
    # Consolidar configurações do Anthropic
    enhanced_anthropic = enhanced_config.get('anthropic_enhanced', {})
    
    # Atualizar configuração principal do Anthropic
    if 'anthropic' in settings_config:
        # Usar configuração padrão do enhanced como base
        default_config = enhanced_anthropic.get('default_config', {})
        
        # Manter configurações existentes e adicionar novas
        settings_config['anthropic'].update({
            'model': default_config.get('model', 'claude-3-5-sonnet-20241022'),
            'max_tokens': default_config.get('max_tokens', 4000),
            'temperature': default_config.get('temperature', 0.3),
            'enhanced_config_enabled': True,
            'enhanced_config_path': 'config/enhanced_model_settings.yaml'
        })
        
        logger.info("✅ Configuração Anthropic principal atualizada")
    
    # Adicionar referência para enhanced configurations
    settings_config['enhanced_models'] = {
        'enabled': True,
        'config_file': 'config/enhanced_model_settings.yaml',
        'fallback_to_default': True,
        'stage_specific_configs': True
    }
    
    # Atualizar versão
    if 'project' in settings_config:
        settings_config['project']['version'] = '4.9.8'
        settings_config['project']['model_config_version'] = '4.9.8'
    
    # Adicionar comentários sobre enhanced config
    settings_config['# Enhanced Model Configuration'] = {
        'info': 'Stage-specific model configurations are loaded from enhanced_model_settings.yaml',
        'priority': 'enhanced_config > settings.yaml > environment variables',
        'supported_stages': [
            'political_analysis', 'sentiment_analysis', 'network_analysis',
            'qualitative_analysis', 'pipeline_review', 'topic_interpretation', 'validation'
        ]
    }
    
    # Salvar configuração consolidada
    if save_yaml_config(settings_config, settings_path):
        logger.info("✅ Configuração consolidada salva em settings.yaml")
        return True
    else:
        logger.error("❌ Falha ao salvar configuração consolidada")
        return False

def validate_enhanced_config():
    """
    Valida se enhanced_model_settings.yaml está corretamente configurado
    """
    logger.info("🔍 Validando enhanced_model_settings.yaml...")
    
    enhanced_path = project_root / "config" / "enhanced_model_settings.yaml"
    enhanced_config = load_yaml_config(enhanced_path)
    
    if not enhanced_config:
        return False
    
    # Verificar estrutura obrigatória
    required_sections = ['anthropic_enhanced']
    for section in required_sections:
        if section not in enhanced_config:
            logger.error(f"❌ Seção obrigatória ausente: {section}")
            return False
    
    anthropic_enhanced = enhanced_config['anthropic_enhanced']
    
    # Verificar subseções obrigatórias
    required_subsections = ['default_config', 'stage_specific_configs', 'cost_optimization']
    for subsection in required_subsections:
        if subsection not in anthropic_enhanced:
            logger.error(f"❌ Subseção obrigatória ausente: {subsection}")
            return False
    
    # Validar configurações específicas de stage
    stage_configs = anthropic_enhanced['stage_specific_configs']
    expected_stages = [
        'stage_05_political', 'stage_08_sentiment', 'stage_15_network',
        'stage_16_qualitative', 'stage_17_review', 'stage_18_topics', 'stage_20_validation'
    ]
    
    for stage in expected_stages:
        if stage not in stage_configs:
            logger.warning(f"⚠️ Configuração de stage ausente: {stage}")
        else:
            stage_config = stage_configs[stage]
            if 'model' not in stage_config:
                logger.error(f"❌ Modelo não especificado para {stage}")
                return False
    
    logger.info("✅ enhanced_model_settings.yaml validado com sucesso")
    return True

def update_component_imports():
    """
    Verifica se componentes precisam de atualizações adicionais
    """
    logger.info("🔍 Verificando componentes anthropic_integration...")
    
    components_dir = project_root / "src" / "anthropic_integration"
    
    # Lista de componentes que foram atualizados
    updated_components = [
        'political_analyzer.py',
        'sentiment_analyzer.py', 
        'smart_pipeline_reviewer.py',
        'topic_interpreter.py',
        'pipeline_validator.py',
        'qualitative_classifier.py',
        'intelligent_network_analyzer.py'
    ]
    
    missing_components = []
    for component in updated_components:
        component_path = components_dir / component
        if not component_path.exists():
            missing_components.append(component)
    
    if missing_components:
        logger.warning(f"⚠️ Componentes não encontrados: {missing_components}")
        return False
    
    logger.info(f"✅ Todos os {len(updated_components)} componentes principais encontrados")
    return True

def main():
    """Função principal"""
    logger.info("🚀 Iniciando consolidação completa de configurações...")
    
    try:
        # Etapa 1: Validar enhanced config
        if not validate_enhanced_config():
            logger.error("❌ Validação do enhanced config falhou")
            return False
        
        # Etapa 2: Consolidar configurações
        if not consolidate_model_configurations():
            logger.error("❌ Consolidação de configurações falhou")
            return False
        
        # Etapa 3: Verificar componentes
        if not update_component_imports():
            logger.error("❌ Verificação de componentes falhou")
            return False
        
        logger.info("🎉 Consolidação completa realizada com sucesso!")
        logger.info("")
        logger.info("📋 Próximos passos:")
        logger.info("1. Testar pipeline com: poetry run python run_pipeline.py")
        logger.info("2. Verificar logs de enhanced config loading")
        logger.info("3. Monitorar custos com os novos modelos")
        logger.info("4. Validar qualidade das análises")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado durante consolidação: {e}")
        return False

if __name__ == "__main__":
    from datetime import datetime
    success = main()
    sys.exit(0 if success else 1)