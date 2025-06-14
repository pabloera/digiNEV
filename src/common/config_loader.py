#!/usr/bin/env python3
"""
Configuration Loader - Carregador de Configurações Centralizado
==============================================================

TASK-023 v5.0.0: Consolida carregamento de todas as configurações
Substitui valores hardcoded por configurações centralizadas

Funcionalidades implementadas:
- Carregamento unificado de todos os arquivos YAML de configuração
- Cache inteligente para evitar recarregamento desnecessário
- Fallbacks automáticos para valores padrão
- Validação de configurações obrigatórias
- Interpolação de variáveis de ambiente
- Suporte a múltiplos ambientes (dev/test/prod)
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from functools import lru_cache

logger = logging.getLogger(__name__)

class ConfigurationLoader:
    """
    Carregador centralizado de configurações para eliminar valores hardcoded
    """
    
    _instance = None
    _configs_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.project_root = self._find_project_root()
        self.config_dir = self.project_root / "config"
        self.environment = os.getenv("BOLSONARISMO_ENV", "development")
        
        # Verificar se diretório de configuração existe
        if not self.config_dir.exists():
            logger.warning(f"Diretório de configuração não encontrado: {self.config_dir}")
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        
        logger.info(f"ConfigurationLoader initialized - Environment: {self.environment}")
    
    def _find_project_root(self) -> Path:
        """Encontra a raiz do projeto procurando por arquivos característicos"""
        current = Path(__file__).parent
        
        # Procurar por arquivos característicos do projeto
        markers = ['pyproject.toml', 'run_pipeline.py', 'CLAUDE.md']
        
        while current != current.parent:
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent
        
        # Fallback para estrutura padrão
        return Path(__file__).parent.parent.parent
    
    @lru_cache(maxsize=32)
    def load_config(self, config_name: str, required: bool = True) -> Dict[str, Any]:
        """
        Carrega um arquivo de configuração específico
        
        Args:
            config_name: Nome do arquivo (sem extensão .yaml)
            required: Se True, lança erro se arquivo não existir
            
        Returns:
            Dict com configurações carregadas
        """
        cache_key = f"{config_name}_{self.environment}"
        
        if cache_key in self._configs_cache:
            return self._configs_cache[cache_key]
        
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            if required:
                logger.error(f"Arquivo de configuração obrigatório não encontrado: {config_file}")
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            else:
                logger.warning(f"Arquivo de configuração opcional não encontrado: {config_file}")
                return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Interpolação de variáveis de ambiente
            config = self._interpolate_env_vars(config)
            
            # Aplicar configurações específicas do ambiente
            config = self._apply_environment_overrides(config)
            
            self._configs_cache[cache_key] = config
            logger.debug(f"Configuração carregada: {config_name}")
            
            return config
            
        except Exception as e:
            logger.error(f"Erro carregando configuração {config_name}: {e}")
            if required:
                raise
            return {}
    
    def _interpolate_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Interpola variáveis de ambiente nas configurações"""
        if isinstance(config, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            default_value = None
            
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            
            return os.getenv(env_var, default_value or config)
        
        return config
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica configurações específicas do ambiente"""
        if not isinstance(config, dict):
            return config
        
        # Verificar se há seção de ambientes
        environments = config.get('environments', {})
        env_config = environments.get(self.environment, {})
        
        if env_config:
            # Merge recursivo das configurações do ambiente
            config = self._deep_merge(config, env_config)
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge recursivo de dois dicionários"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_api_limits(self) -> Dict[str, Any]:
        """Carrega configurações de limites de API"""
        return self.load_config('api_limits', required=False)
    
    def get_network_config(self) -> Dict[str, Any]:
        """Carrega configurações de rede"""
        return self.load_config('network', required=False)
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Carrega configurações de caminhos"""
        return self.load_config('paths', required=False)
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Carrega configurações de processamento"""
        return self.load_config('processing', required=True)
    
    def get_timeout_config(self) -> Dict[str, Any]:
        """Carrega configurações de timeout"""
        return self.load_config('timeout_management', required=False)
    
    def get_settings(self) -> Dict[str, Any]:
        """Carrega configurações principais"""
        return self.load_config('settings', required=True)
    
    def get_anthropic_config(self) -> Dict[str, Any]:
        """Carrega configurações do Anthropic"""
        return self.load_config('anthropic', required=False)
    
    def get_voyage_config(self) -> Dict[str, Any]:
        """Carrega configurações do Voyage AI"""
        return self.load_config('voyage_embeddings', required=False)
    
    def get_value(self, config_path: str, default: Any = None) -> Any:
        """
        Obtém um valor específico usando notação de ponto
        
        Args:
            config_path: Caminho no formato "config_file.section.key"
            default: Valor padrão se não encontrado
            
        Returns:
            Valor da configuração ou default
        """
        parts = config_path.split('.')
        if len(parts) < 2:
            logger.error(f"Caminho de configuração inválido: {config_path}")
            return default
        
        config_name = parts[0]
        key_path = parts[1:]
        
        try:
            config = self.load_config(config_name, required=False)
            
            current = config
            for key in key_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception as e:
            logger.warning(f"Erro obtendo valor {config_path}: {e}")
            return default
    
    def get_model_config(self, service: str, key: Optional[str] = None) -> Any:
        """
        Obtém configuração de modelo específico
        
        Args:
            service: "anthropic" ou "voyage"  
            key: Chave específica (opcional)
            
        Returns:
            Configuração do modelo
        """
        if service == "anthropic":
            config = self.get_api_limits().get('api_limits', {}).get('anthropic', {})
        elif service == "voyage":
            config = self.get_api_limits().get('api_limits', {}).get('voyage', {})
        else:
            logger.error(f"Serviço desconhecido: {service}")
            return {} if key is None else None
        
        if key:
            return config.get(key)
        return config
    
    def get_path(self, path_key: str, create_if_missing: bool = False) -> Optional[Path]:
        """
        Obtém um caminho configurado
        
        Args:
            path_key: Chave do caminho (ex: "data.uploads", "logs.pipeline")
            create_if_missing: Criar diretório se não existir
            
        Returns:
            Path object ou None se não encontrado
        """
        path_str = self.get_value(f"paths.{path_key}")
        
        if not path_str:
            return None
        
        # Resolver caminho relativo ao projeto
        if not os.path.isabs(path_str):
            path = self.project_root / path_str
        else:
            path = Path(path_str)
        
        if create_if_missing and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Diretório criado: {path}")
            except Exception as e:
                logger.error(f"Erro criando diretório {path}: {e}")
        
        return path
    
    def clear_cache(self):
        """Limpa o cache de configurações"""
        self._configs_cache.clear()
        self.load_config.cache_clear()
        logger.info("Cache de configurações limpo")
    
    def validate_required_configs(self) -> bool:
        """
        Valida se todas as configurações obrigatórias estão presentes
        
        Returns:
            True se todas configurações estão OK
        """
        required_configs = [
            'settings',
            'processing'
        ]
        
        optional_configs = [
            'api_limits',
            'network', 
            'paths',
            'timeout_management'
        ]
        
        all_valid = True
        
        # Verificar configurações obrigatórias
        for config_name in required_configs:
            try:
                config = self.load_config(config_name, required=True)
                if not config:
                    logger.error(f"Configuração obrigatória vazia: {config_name}")
                    all_valid = False
                else:
                    logger.info(f"✅ Configuração obrigatória OK: {config_name}")
            except Exception as e:
                logger.error(f"❌ Erro na configuração obrigatória {config_name}: {e}")
                all_valid = False
        
        # Verificar configurações opcionais
        for config_name in optional_configs:
            try:
                config = self.load_config(config_name, required=False)
                if config:
                    logger.info(f"✅ Optional configuration loaded: {config_name}")
                else:
                    logger.warning(f"⚠️  Configuração opcional não encontrada: {config_name}")
            except Exception as e:
                logger.warning(f"⚠️  Erro na configuração opcional {config_name}: {e}")
        
        return all_valid


# Instância global do carregador
_config_loader = None

def get_config_loader() -> ConfigurationLoader:
    """Obtém instância global do carregador de configurações"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader()
    return _config_loader

# Funções de conveniência
def get_config(config_name: str, required: bool = True) -> Dict[str, Any]:
    """Função de conveniência para carregar configuração"""
    return get_config_loader().load_config(config_name, required)

def get_config_value(config_path: str, default: Any = None) -> Any:
    """Função de conveniência para obter valor específico"""
    return get_config_loader().get_value(config_path, default)

def get_model_setting(service: str, key: str, default: Any = None) -> Any:
    """Função de conveniência para configurações de modelo"""
    value = get_config_loader().get_model_config(service, key)
    return value if value is not None else default

def get_path_config(path_key: str, create_if_missing: bool = False) -> Optional[Path]:
    """Função de conveniência para obter caminhos"""
    return get_config_loader().get_path(path_key, create_if_missing)


if __name__ == "__main__":
    # Teste básico das funcionalidades
    print("🧪 Testando ConfigurationLoader...")
    
    loader = get_config_loader()
    
    # Validar configurações
    if loader.validate_required_configs():
        print("✅ Todas as configurações obrigatórias estão OK")
    else:
        print("❌ Problemas encontrados nas configurações")
    
    # Testar carregamento de configurações
    try:
        api_limits = loader.get_api_limits()
        print(f"📊 API Limits carregado: {len(api_limits)} seções")
        
        network = loader.get_network_config()
        print(f"🌐 Network config carregado: {len(network)} seções")
        
        paths = loader.get_paths_config()
        print(f"📁 Paths config carregado: {len(paths)} seções")
        
        # Testar valores específicos
        anthropic_model = get_model_setting("anthropic", "default_model", "claude-3-5-sonnet-20241022")
        print(f"🤖 Modelo Anthropic: {anthropic_model}")
        
        batch_size = get_config_value("processing.batch_processing.chunk_size", 10000)
        print(f"📦 Batch size: {batch_size}")
        
        print("✅ ConfigurationLoader funcionando corretamente!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")