#!/usr/bin/env python3
"""
LoggingMixin - Padronização de Logging
=====================================

Mixin para padronizar logging em todos os módulos do projeto.
Elimina inconsistências de formatação identificadas na auditoria v5.0.0 - TASK-019

61 arquivos com logging inconsistente → formatação padronizada
"""

import logging
import functools
import time
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime
from pathlib import Path


class LoggingMixin:
    """
    Mixin para padronização de logging em todos os módulos
    
    Fornece métodos consistentes de logging com formatação padronizada
    e funcionalidades avançadas como timing automático e contexto.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_logger()
    
    def _setup_logger(self):
        """Configura logger específico para a classe"""
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        
        # Criar logger hierárquico baseado no módulo
        logger_name = f"monitor_discurso_digital.{module_name.split('.')[-1]}.{class_name}"
        self._logger = logging.getLogger(logger_name)
        
        # Evitar duplicação de handlers
        if not self._logger.handlers:
            self._setup_class_specific_logging()
    
    def _setup_class_specific_logging(self):
        """Configura logging específico para a classe se necessário"""
        # Por padrão, usa a configuração global
        # Classes específicas podem sobrescrever este método
        pass
    
    # =============================================================================
    # MÉTODOS DE LOGGING PADRONIZADOS
    # =============================================================================
    
    def log_info(self, message: str, **context):
        """Log de informação com formatação padronizada"""
        formatted_msg = self._format_message(message, "INFO", **context)
        self._logger.info(formatted_msg)
    
    def log_debug(self, message: str, **context):
        """Log de debug com formatação padronizada"""
        formatted_msg = self._format_message(message, "DEBUG", **context)
        self._logger.debug(formatted_msg)
    
    def log_warning(self, message: str, **context):
        """Log de warning com formatação padronizada"""
        formatted_msg = self._format_message(message, "WARNING", **context)
        self._logger.warning(formatted_msg)
    
    def log_error(self, message: str, error: Optional[Exception] = None, **context):
        """Log de erro com formatação padronizada"""
        if error:
            context['error_type'] = type(error).__name__
            context['error_details'] = str(error)
        
        formatted_msg = self._format_message(message, "ERROR", **context)
        self._logger.error(formatted_msg, exc_info=error is not None)
    
    def log_success(self, message: str, **context):
        """Log de sucesso com emoji padronizado"""
        formatted_msg = f"✅ {message}"
        if context:
            formatted_msg += f" | {self._format_context(**context)}"
        self._logger.info(formatted_msg)
    
    def log_start_operation(self, operation: str, **context):
        """Log padronizado de início de operação"""
        formatted_msg = f"🚀 INICIANDO: {operation}"
        if context:
            formatted_msg += f" | {self._format_context(**context)}"
        self._logger.info(formatted_msg)
    
    def log_end_operation(self, operation: str, duration: Optional[float] = None, **context):
        """Log padronizado de fim de operação"""
        formatted_msg = f"✅ CONCLUÍDO: {operation}"
        if duration:
            formatted_msg += f" | Tempo: {duration:.2f}s"
        if context:
            formatted_msg += f" | {self._format_context(**context)}"
        self._logger.info(formatted_msg)
    
    def log_progress(self, current: int, total: int, operation: str = "", **context):
        """Log padronizado de progresso"""
        percentage = (current / total * 100) if total > 0 else 0
        formatted_msg = f"📊 PROGRESSO: {current}/{total} ({percentage:.1f}%)"
        if operation:
            formatted_msg += f" | {operation}"
        if context:
            formatted_msg += f" | {self._format_context(**context)}"
        self._logger.info(formatted_msg)
    
    def log_data_stats(self, df_or_count: Union[int, Any], operation: str = "", **context):
        """Log padronizado de estatísticas de dados"""
        if hasattr(df_or_count, '__len__'):
            count = len(df_or_count)
            if hasattr(df_or_count, 'columns'):
                context['columns'] = len(df_or_count.columns)
        else:
            count = df_or_count
            
        formatted_msg = f"📊 DADOS: {count:,} registros"
        if operation:
            formatted_msg += f" | {operation}"
        if context:
            formatted_msg += f" | {self._format_context(**context)}"
        self._logger.info(formatted_msg)
    
    # =============================================================================
    # DECORATORS PARA LOGGING AUTOMÁTICO
    # =============================================================================
    
    def log_timing(self, operation_name: Optional[str] = None):
        """Decorator para logging automático de tempo de execução"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
                
                self.log_start_operation(op_name)
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.log_end_operation(op_name, duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_error(f"FALHA: {op_name}", error=e, duration=duration)
                    raise
                    
            return wrapper
        return decorator
    
    def log_method_calls(self, include_args: bool = False):
        """Decorator para logging automático de chamadas de método"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                method_name = f"{self.__class__.__name__}.{func.__name__}"
                
                if include_args:
                    # Log apenas tipos para evitar logs muito longos
                    arg_types = [type(arg).__name__ for arg in args[1:]]  # Skip self
                    kwarg_info = {k: type(v).__name__ for k, v in kwargs.items()}
                    self.log_debug(f"CHAMADA: {method_name}", 
                                 arg_types=arg_types, kwargs=kwarg_info)
                else:
                    self.log_debug(f"CHAMADA: {method_name}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # =============================================================================
    # MÉTODOS DE FORMATAÇÃO
    # =============================================================================
    
    def _format_message(self, message: str, level: str, **context) -> str:
        """Formata mensagem com contexto adicional"""
        if not context:
            return message
            
        context_str = self._format_context(**context)
        return f"{message} | {context_str}"
    
    def _format_context(self, **context) -> str:
        """Formata contexto adicional para logs"""
        if not context:
            return ""
            
        formatted_items = []
        for key, value in context.items():
            # Formatação específica por tipo
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_items.append(f"{key}={value:.2f}")
                else:
                    formatted_items.append(f"{key}={value:,}")
            elif isinstance(value, (list, tuple)):
                formatted_items.append(f"{key}=[{len(value)} items]")
            elif isinstance(value, dict):
                formatted_items.append(f"{key}={{{len(value)} keys}}")
            elif isinstance(value, Path):
                formatted_items.append(f"{key}={value.name}")
            else:
                # Truncar strings muito longas
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:47] + "..."
                formatted_items.append(f"{key}={str_value}")
        
        return " | ".join(formatted_items)
    
    # =============================================================================
    # LOGGING ESPECÍFICO PARA PIPELINE
    # =============================================================================
    
    def log_stage_start(self, stage_name: str, stage_number: Optional[int] = None):
        """Log padronizado de início de stage do pipeline"""
        if stage_number:
            self.log_info(f"🎯 STAGE {stage_number:02d}: {stage_name}")
        else:
            self.log_info(f"🎯 STAGE: {stage_name}")
    
    def log_stage_end(self, stage_name: str, duration: float, records_processed: int = 0):
        """Log padronizado de fim de stage do pipeline"""
        self.log_success(f"STAGE CONCLUÍDO: {stage_name}", 
                        duration=duration, records=records_processed)
    
    def log_api_call(self, api_name: str, model: str, tokens_used: Dict[str, int], cost: float):
        """Log padronizado de chamadas de API"""
        self.log_info(f"🤖 API: {api_name}", 
                     model=model, 
                     input_tokens=tokens_used.get('input', 0),
                     output_tokens=tokens_used.get('output', 0),
                     cost_usd=cost)
    
    def log_cache_hit(self, cache_type: str, key: str):
        """Log padronizado de cache hit"""
        self.log_debug(f"💾 CACHE HIT: {cache_type}", key=key[:20] + "..." if len(key) > 20 else key)
    
    def log_cache_miss(self, cache_type: str, key: str):
        """Log padronizado de cache miss"""
        self.log_debug(f"💾 CACHE MISS: {cache_type}", key=key[:20] + "..." if len(key) > 20 else key)
    
    # =============================================================================
    # CONTEXT MANAGERS
    # =============================================================================
    
    class operation_context:
        """Context manager para logging automático de operações"""
        
        def __init__(self, parent: 'LoggingMixin', operation_name: str, **context):
            self.parent = parent
            self.operation_name = operation_name
            self.context = context
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            self.parent.log_start_operation(self.operation_name, **self.context)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            
            if exc_type is None:
                self.parent.log_end_operation(self.operation_name, duration, **self.context)
            else:
                self.parent.log_error(f"FALHA: {self.operation_name}", 
                                    error=exc_val, duration=duration, **self.context)
    
    def operation(self, name: str, **context):
        """Cria context manager para operação"""
        return self.operation_context(self, name, **context)


# =============================================================================
# FUNÇÕES UTILITÁRIAS GLOBAIS
# =============================================================================

def get_standard_logger(name: str) -> logging.Logger:
    """
    Retorna logger padronizado para uso fora de classes
    
    Args:
        name: Nome do logger (geralmente __name__)
        
    Returns:
        logging.Logger: Logger configurado
    """
    # Normalizar nome para hierarquia padrão
    if not name.startswith('monitor_discurso_digital'):
        if '.' in name:
            module_part = name.split('.')[-1]
        else:
            module_part = name
        name = f"monitor_discurso_digital.{module_part}"
    
    return logging.getLogger(name)


def log_system_info():
    """Log de informações do sistema para debugging"""
    import platform
    import sys
    import psutil
    
    logger = get_standard_logger('system_info')
    
    logger.info("=" * 50)
    logger.info("🖥️  INFORMAÇÕES DO SISTEMA")
    logger.info("=" * 50)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Plataforma: {platform.platform()}")
    logger.info(f"Arquitetura: {platform.architecture()}")
    logger.info(f"Processador: {platform.processor()}")
    
    try:
        memory = psutil.virtual_memory()
        logger.info(f"Memória Total: {memory.total / 1024**3:.1f} GB")
        logger.info(f"Memória Disponível: {memory.available / 1024**3:.1f} GB")
        logger.info(f"CPU Cores: {psutil.cpu_count()}")
    except:
        logger.info("Informações de sistema detalhadas não disponíveis")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    # Teste básico do LoggingMixin
    class TestClass(LoggingMixin):
        def test_method(self):
            self.log_info("Teste de método")
            
        @LoggingMixin.log_timing()
        def timed_method(self):
            import time
            time.sleep(0.1)
            return "resultado"
    
    print("🧪 Testando LoggingMixin...")
    
    # Configurar logging básico para teste
    logging.basicConfig(level=logging.INFO)
    
    test_obj = TestClass()
    test_obj.log_info("Mensagem de teste", records=1000, duration=1.5)
    test_obj.log_success("Operação concluída", items_processed=500)
    test_obj.log_data_stats(1000, "processamento")
    
    # Teste com context manager
    with test_obj.operation("operação_teste", batch_size=100):
        time.sleep(0.05)
    
    # Teste com decorator
    result = test_obj.timed_method()
    
    print("✅ LoggingMixin funcionando corretamente!")