"""
Circuit Breaker para APIs Externas - Fase 3 do Plano de Aprimoramento

Este módulo implementa o padrão Circuit Breaker para aumentar a
resiliência do sistema a falhas temporárias em serviços externos
(APIs da Anthropic e Voyage).
"""

import logging
import time
from typing import Dict, Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Estados do Circuit Breaker"""
    CLOSED = "closed"        # Funcionamento normal
    OPEN = "open"           # Circuito aberto, chamadas bloqueadas
    HALF_OPEN = "half_open" # Teste de recuperação

@dataclass
class CircuitBreakerConfig:
    """Configuração do Circuit Breaker"""
    failure_threshold: int = 5          # Número de falhas para abrir circuito
    recovery_timeout: float = 60.0      # Tempo em segundos antes de tentar recovery
    expected_exception: type = Exception # Tipo de exceção que conta como falha
    success_threshold: int = 3          # Sucessos consecutivos para fechar circuito
    call_timeout: float = 30.0          # Timeout para chamadas individuais

@dataclass
class CircuitBreakerStats:
    """Estatísticas do Circuit Breaker"""
    total_calls: int = 0
    failed_calls: int = 0
    successful_calls: int = 0
    timeouts: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

class CircuitBreakerOpenException(Exception):
    """Exceção lançada quando o circuito está aberto"""
    def __init__(self, message: str, retry_after: float):
        self.retry_after = retry_after
        super().__init__(message)

class CircuitBreaker:
    """
    Implementação do padrão Circuit Breaker para APIs externas.
    
    O Circuit Breaker monitora falhas nas chamadas para serviços externos
    e automaticamente "abre o circuito" quando muitas falhas são detectadas,
    evitando sobrecarga dos serviços e permitindo recuperação elegante.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Inicializa o Circuit Breaker.
        
        Args:
            name: Nome identificador do circuit breaker
            config: Configuração do circuit breaker
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_changed_at = datetime.now()
        
        logger.info(f"Circuit Breaker '{name}' inicializado em estado CLOSED")
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator para aplicar o Circuit Breaker a uma função.
        
        Args:
            func: Função a ser protegida
            
        Returns:
            Função decorada com Circuit Breaker
        """
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executa uma função protegida pelo Circuit Breaker.
        
        Args:
            func: Função a ser executada
            *args: Argumentos posicionais para a função
            **kwargs: Argumentos nomeados para a função
            
        Returns:
            Resultado da função
            
        Raises:
            CircuitBreakerOpenException: Se o circuito estiver aberto
            Exception: Outras exceções da função original
        """
        self.stats.total_calls += 1
        
        # Verificar estado do circuito
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._move_to_half_open()
            else:
                retry_after = self._get_retry_after()
                raise CircuitBreakerOpenException(
                    f"Circuit Breaker '{self.name}' está ABERTO. "
                    f"Tente novamente em {retry_after:.1f} segundos.",
                    retry_after
                )
        
        try:
            # Executar função com timeout
            start_time = time.time()
            
            # Aqui poderíamos implementar timeout real, mas por simplicidade
            # vamos apenas executar a função
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Verificar se excedeu timeout
            if execution_time > self.config.call_timeout:
                self.stats.timeouts += 1
                raise TimeoutError(f"Chamada excedeu timeout de {self.config.call_timeout}s")
            
            # Sucesso
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            # Falha esperada
            self._on_failure()
            raise e
        except Exception as e:
            # Falha inesperada - também conta como falha
            logger.warning(f"Falha inesperada no Circuit Breaker '{self.name}': {e}")
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar resetar o circuito"""
        if self.stats.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.stats.last_failure_time
        return time_since_failure >= timedelta(seconds=self.config.recovery_timeout)
    
    def _get_retry_after(self) -> float:
        """Calcula tempo até próxima tentativa"""
        if self.stats.last_failure_time is None:
            return 0.0
        
        time_since_failure = datetime.now() - self.stats.last_failure_time
        remaining = self.config.recovery_timeout - time_since_failure.total_seconds()
        return max(0.0, remaining)
    
    def _move_to_half_open(self):
        """Move circuito para estado HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self._state_changed_at = datetime.now()
        self.stats.consecutive_successes = 0
        logger.info(f"Circuit Breaker '{self.name}' movido para HALF_OPEN")
    
    def _on_success(self):
        """Lida com sucesso na execução"""
        self.stats.successful_calls += 1
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes += 1
        self.stats.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._move_to_closed()
    
    def _on_failure(self):
        """Lida com falha na execução"""
        self.stats.failed_calls += 1
        self.stats.consecutive_successes = 0
        self.stats.consecutive_failures += 1
        self.stats.last_failure_time = datetime.now()
        
        if self.stats.consecutive_failures >= self.config.failure_threshold:
            self._move_to_open()
    
    def _move_to_closed(self):
        """Move circuito para estado CLOSED"""
        self.state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        logger.info(f"Circuit Breaker '{self.name}' movido para CLOSED - funcionamento normal restaurado")
    
    def _move_to_open(self):
        """Move circuito para estado OPEN"""
        self.state = CircuitState.OPEN
        self._state_changed_at = datetime.now()
        logger.warning(
            f"Circuit Breaker '{self.name}' movido para OPEN - "
            f"{self.stats.consecutive_failures} falhas consecutivas detectadas"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status completo do Circuit Breaker.
        
        Returns:
            Dicionário com status e estatísticas
        """
        return {
            'name': self.name,
            'state': self.state.value,
            'state_changed_at': self._state_changed_at.isoformat(),
            'stats': {
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'timeouts': self.stats.timeouts,
                'consecutive_failures': self.stats.consecutive_failures,
                'consecutive_successes': self.stats.consecutive_successes,
                'last_failure_time': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
                'last_success_time': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
                'success_rate': self._get_success_rate()
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'call_timeout': self.config.call_timeout
            },
            'retry_after': self._get_retry_after() if self.state == CircuitState.OPEN else 0.0
        }
    
    def _get_success_rate(self) -> float:
        """Calcula taxa de sucesso"""
        if self.stats.total_calls == 0:
            return 0.0
        return self.stats.successful_calls / self.stats.total_calls
    
    def reset(self):
        """Reseta o Circuit Breaker para estado inicial"""
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_changed_at = datetime.now()
        logger.info(f"Circuit Breaker '{self.name}' resetado")

class CircuitBreakerManager:
    """
    Gerenciador de múltiplos Circuit Breakers.
    
    Permite gerenciar Circuit Breakers para diferentes serviços
    (Anthropic, Voyage, etc.) de forma centralizada.
    """
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """
        Cria um novo Circuit Breaker.
        
        Args:
            name: Nome único do breaker
            config: Configuração do breaker
            
        Returns:
            Circuit Breaker criado
        """
        if name in self._breakers:
            logger.warning(f"Circuit Breaker '{name}' já existe, substituindo")
        
        breaker = CircuitBreaker(name, config)
        self._breakers[name] = breaker
        
        logger.info(f"Circuit Breaker '{name}' criado e registrado")
        return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Obtém um Circuit Breaker pelo nome.
        
        Args:
            name: Nome do breaker
            
        Returns:
            Circuit Breaker ou None se não encontrado
        """
        return self._breakers.get(name)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna status de todos os Circuit Breakers.
        
        Returns:
            Dicionário com status de todos os breakers
        """
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self):
        """Reseta todos os Circuit Breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("Todos os Circuit Breakers foram resetados")

# Instância global do gerenciador
_circuit_manager = CircuitBreakerManager()

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Obtém instância global do gerenciador de Circuit Breakers"""
    return _circuit_manager

def create_api_circuit_breaker(api_name: str, config_dict: Dict[str, Any]) -> CircuitBreaker:
    """
    Cria um Circuit Breaker para uma API específica.
    
    Args:
        api_name: Nome da API (ex: "anthropic", "voyage")
        config_dict: Configuração como dicionário
        
    Returns:
        Circuit Breaker configurado
    """
    config = CircuitBreakerConfig(
        failure_threshold=config_dict.get('failure_threshold', 5),
        recovery_timeout=config_dict.get('recovery_timeout', 60.0),
        success_threshold=config_dict.get('success_threshold', 3),
        call_timeout=config_dict.get('call_timeout', 30.0)
    )
    
    return _circuit_manager.create_breaker(api_name, config)

def setup_circuit_breakers_from_config(network_config: Dict[str, Any]) -> Dict[str, CircuitBreaker]:
    """
    Configura Circuit Breakers a partir da configuração de rede.
    
    Args:
        network_config: Configuração de rede contendo circuit_breaker
        
    Returns:
        Dicionário de Circuit Breakers configurados
    """
    breakers = {}
    
    circuit_config = network_config.get('circuit_breaker', {})
    
    for service_name, service_config in circuit_config.items():
        if service_name != 'default':
            breaker = create_api_circuit_breaker(service_name, service_config)
            breakers[service_name] = breaker
            logger.info(f"Circuit Breaker configurado para {service_name}")
    
    return breakers