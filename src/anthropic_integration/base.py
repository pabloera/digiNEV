"""
Módulo base para integração com API Anthropic

Este módulo fornece classes base e utilitários comuns para todos os módulos
de integração com a API Anthropic.
"""

import json
import logging
import os

# Carregar variáveis de ambiente do arquivo .env
# Buscar .env a partir do diretório do projeto (subindo 2 níveis do src/anthropic_integration)
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

# Encontrar diretório raiz do projeto
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # sobe 2 níveis: src/anthropic_integration -> src -> projeto
env_file = project_root / '.env'

if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback para carregar do diretório atual
    load_dotenv()

# Import cost monitor (depois para evitar circular import)
try:
    from .cost_monitor import get_cost_monitor
    COST_MONITOR_AVAILABLE = True
except ImportError:
    COST_MONITOR_AVAILABLE = False


@dataclass
class AnthropicConfig:
    """Configuração para API Anthropic"""
    api_key: str
    model: str = "claude-3-5-haiku-latest"
    max_tokens: int = 2000
    temperature: float = 0.3
    

class AnthropicBase:
    """Classe base para integração com API Anthropic"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa cliente Anthropic
        
        Args:
            config: Dicionário de configuração (se None, usa variáveis de ambiente)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configurar API
        if config and 'anthropic' in config:
            anthro_config = config['anthropic']
            config_api_key = anthro_config.get('api_key', '')
            
            # Verificar se é uma referência de variável de ambiente (${VAR_NAME})
            if config_api_key.startswith('${') and config_api_key.endswith('}'):
                var_name = config_api_key[2:-1]  # Remove ${ e }
                api_key = os.getenv(var_name)
            elif config_api_key and not config_api_key.startswith('${'):
                api_key = config_api_key
            else:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            
            self.model = anthro_config.get('model', 'claude-3-5-haiku-latest')
            self.max_tokens = anthro_config.get('max_tokens_per_request', 2000)
            self.temperature = anthro_config.get('temperature', 0.3)
        else:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            self.model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-latest')
            self.max_tokens = 2000
            self.temperature = 0.3
        
        if not api_key:
            self.logger.warning("API key Anthropic não encontrada. Modo tradicional será usado.")
            self.client = None
            self.api_available = False
        else:
            try:
                self.client = Anthropic(api_key=api_key)
                self.api_available = True
                self.logger.info(f"Cliente Anthropic inicializado com modelo: {self.model}")
            except Exception as e:
                self.logger.error(f"Falha ao inicializar cliente Anthropic: {e}")
                self.client = None
                self.api_available = False
        
        # Configurar monitor de custos
        self.cost_monitor = None
        if COST_MONITOR_AVAILABLE:
            try:
                self.cost_monitor = get_cost_monitor(project_root)
            except Exception as e:
                self.logger.warning(f"Não foi possível inicializar monitor de custos: {e}")
    
    def create_message(self, prompt: str, stage: str = 'unknown', operation: str = 'general', **kwargs) -> str:
        """
        Cria mensagem usando API Anthropic com fallback
        
        Args:
            prompt: Texto do prompt
            stage: Etapa do pipeline (para rastreamento)
            operation: Operação específica (para rastreamento)
            **kwargs: Parâmetros adicionais para API
            
        Returns:
            Resposta da API como string ou mensagem de fallback
        """
        # Verificar se API está disponível
        if not self.api_available or not self.client:
            fallback_message = kwargs.get('fallback_response', 
                f"API indisponível para {stage}:{operation}. Usando processamento tradicional.")
            self.logger.warning(f"API indisponível, usando fallback para {stage}:{operation}")
            return fallback_message
        
        try:
            model = kwargs.get('model', self.model)
            
            response = self.client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Rastrear custos se monitor disponível
            if self.cost_monitor and response.usage:
                try:
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    self.cost_monitor.record_usage(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stage=stage,
                        operation=operation
                    )
                except Exception as e:
                    self.logger.warning(f"Erro ao registrar custos: {e}")
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Erro na API Anthropic: {e}")
            # Marcar API como indisponível temporariamente
            self.api_available = False
            
            # Retornar fallback em caso de erro
            fallback_message = kwargs.get('fallback_response', 
                f"Erro na API para {stage}:{operation}. Usando processamento tradicional.")
            self.logger.warning(f"API falhou, usando fallback para {stage}:{operation}")
            return fallback_message
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Extrai JSON de resposta da API com tratamento robusto
        
        Args:
            response: Resposta da API
            
        Returns:
            Dicionário com dados parseados
        """
        if not response or not response.strip():
            if hasattr(self, 'logger') and self.logger:
                self.logger.error("Resposta vazia da API")
            return {"error": "Resposta vazia da API", "encoding_issues": [], "overall_assessment": {}, "results": []}
        
        # Log para debug de respostas truncadas
        if hasattr(self, 'logger') and self.logger:
            self.logger.debug(f"Parsing JSON de {len(response)} chars: {response[:100]}...")
        
        try:
            # Tentar parse direto
            return json.loads(response)
        except json.JSONDecodeError as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Parse JSON direto falhou: {e}. Aplicando correções robustas...")
            
            # NOVA CORREÇÃO: Verificar se resposta é Claude introdutório
            import re
            
            # CORREÇÃO PRINCIPAL: Detectar e remover texto introdutório do Claude
            claude_intro_patterns = [
                r"^Aqui está a análise detalhada[^{]*",
                r"^Vou analisar[^{]*",
                r"^Análise detalhada[^{]*", 
                r"^Segue a análise[^{]*",
                r"^Com base[^{]*",
                r"^Baseado[^{]*"
            ]
            
            cleaned_response = response
            for pattern in claude_intro_patterns:
                match = re.match(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if match:
                    # Encontrar onde o JSON realmente começa
                    json_start = cleaned_response.find('{')
                    if json_start > 0:
                        if hasattr(self, 'logger') and self.logger:
                            intro_text = cleaned_response[:json_start].strip()
                            self.logger.info(f"🔧 Removendo texto introdutório: '{intro_text[:50]}...'")
                        cleaned_response = cleaned_response[json_start:]
                        try:
                            result = json.loads(cleaned_response)
                            if hasattr(self, 'logger') and self.logger:
                                self.logger.info("✅ JSON parseado após remoção de introdução")
                            return result
                        except json.JSONDecodeError:
                            pass
                    break

            # Remover marcadores de código se existirem
            if cleaned_response.strip().startswith('```') or cleaned_response.strip().startswith('"""'):
                # Remove marcadores ```json, ```, """, etc
                cleaned_response = re.sub(r'^(```(?:json)?|""")?\s*', '', cleaned_response.strip())
                cleaned_response = re.sub(r'\s*(```|""")$', '', cleaned_response.strip())
                try:
                    result = json.loads(cleaned_response)
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info("✅ JSON parseado após remoção de marcadores")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Tentar encontrar um bloco de código JSON mais robusto
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', cleaned_response, re.DOTALL)
            if code_block_match:
                try:
                    result = json.loads(code_block_match.group(1))
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info("✅ JSON parseado de bloco de código")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # MÉTODO MELHORADO: Extrair JSON balanceado mais robusto
            start_idx = cleaned_response.find('{')
            if start_idx != -1:
                brace_count = 0
                in_string = False
                escape_next = False
                json_end = -1
                
                for i in range(start_idx, len(cleaned_response)):
                    char = cleaned_response[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                        
                    if char == '\\' and in_string:
                        escape_next = True
                        continue
                        
                    if char == '"':
                        in_string = not in_string
                        continue
                        
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                
                if json_end > start_idx:
                    try:
                        json_str = cleaned_response[start_idx:json_end]
                        result = json.loads(json_str)
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.info("✅ JSON extraído com método balanceado")
                        return result
                    except json.JSONDecodeError as e:
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.warning(f"JSON balanceado falhou: {e}")
                        pass
            
            # Tentar regex específica para estruturas conhecidas
            for structure_pattern in [
                r'\{[^{}]*"results"[^{}]*:\s*\[[^\]]*\][^{}]*\}',
                r'\{[^{}]*"analysis"[^{}]*:\s*\{[^{}]*\}[^{}]*\}',
                r'\{[^{}]*"assessment"[^{}]*:\s*\{[^{}]*\}[^{}]*\}'
            ]:
                json_match = re.search(structure_pattern, cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.info("✅ JSON extraído com regex específica")
                        return result
                    except json.JSONDecodeError:
                        continue
            
            # CORREÇÃO ESPECÍFICA: Tentar corrigir resposta truncada
            if response.strip().endswith('"'):
                # Se termina com aspas, pode estar truncado no meio de um campo
                last_brace = response.rfind('}')
                if last_brace == -1:
                    # Tentar fechar o JSON adicionando fechamentos
                    potential_fixes = [
                        response + '"}]}',  # Fechar reasoning + array + objeto
                        response + '"}]',   # Fechar reasoning + array
                        response + '"}',    # Fechar reasoning
                        response + '}',     # Fechar objeto
                        response + ']}'     # Fechar array + objeto
                    ]
                    
                    for fixed_json in potential_fixes:
                        try:
                            result = json.loads(fixed_json)
                            if hasattr(self, 'logger') and self.logger:
                                self.logger.info("JSON truncado corrigido com sucesso")
                            return result
                        except json.JSONDecodeError:
                            continue
            
            # ÚLTIMO RECURSO: Log detalhado para debug
            if hasattr(self, 'logger') and self.logger:
                # Log mais detalhado para ajudar no debug
                self.logger.error(f"Não foi possível parsear JSON após todas as tentativas")
                self.logger.error(f"Tamanho da resposta: {len(response)} caracteres")
                self.logger.error(f"Primeiros 200 chars: {response[:200]}")
                self.logger.error(f"Últimos 100 chars: {response[-100:]}")
                
                # Verificar se há caracteres problemáticos
                non_printable = [char for char in response[:200] if ord(char) < 32 and char not in ['\n', '\r', '\t']]
                if non_printable:
                    self.logger.error(f"Caracteres não imprimíveis encontrados: {non_printable}")
            
            # Retornar estrutura padrão que não quebra o processamento
            return {
                "error": "JSON parse failed after all attempts",
                "response_length": len(response),
                "response_preview": response[:100],
                "encoding_issues": [],
                "overall_assessment": {},
                "results": []
            }
    
    def parse_json_response_robust(self, response: str, expected_structure: str = "results") -> Dict[str, Any]:
        """
        Parser JSON ultra-robusto para uso em todos os componentes
        
        Args:
            response: Resposta da API
            expected_structure: Estrutura esperada ("results", "analysis", etc.)
            
        Returns:
            Dicionário parseado com estrutura garantida
        """
        if not response or not response.strip():
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning("Resposta vazia da API")
            return {expected_structure: []}
        
        try:
            # Usar o parser principal já melhorado
            result = self.parse_json_response(response)
            
            # Verificar se tem a estrutura esperada
            if isinstance(result, dict):
                if expected_structure in result:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"✅ JSON parseado com sucesso - estrutura '{expected_structure}' encontrada")
                    return result
                else:
                    # Tentar adaptar estrutura
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.warning(f"Estrutura '{expected_structure}' não encontrada, adaptando...")
                    
                    # Se result parece ser um item individual, transformar em lista
                    if isinstance(result, dict) and len(result) > 0:
                        adapted_result = {expected_structure: [result]}
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.info(f"📝 Estrutura adaptada: item individual -> lista")
                        return adapted_result
                    else:
                        # Estrutura vazia ou inesperada
                        return {expected_structure: []}
            else:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Resposta não é dicionário: {type(result)}")
                return {expected_structure: []}
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Erro no parser robusto: {e}")
                self.logger.error(f"Resposta (primeiros 200 chars): {response[:200]}")
            return {expected_structure: []}
    
    def parse_claude_response_safe(self, response: str, expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parser ultra-seguro para qualquer resposta da Claude API
        
        Args:
            response: Resposta da Claude API
            expected_keys: Lista de chaves esperadas na resposta
            
        Returns:
            Dicionário parseado com estrutura garantida
        """
        if expected_keys is None:
            expected_keys = ["results"]
            
        try:
            # Usar o parser robusto existente
            parsed = self.parse_json_response(response)
            
            # Verificar se é um dicionário válido
            if not isinstance(parsed, dict):
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Resposta não é dicionário: {type(parsed)}")
                return self._create_safe_response(expected_keys)
            
            # Verificar se tem as chaves esperadas
            missing_keys = [key for key in expected_keys if key not in parsed]
            if missing_keys:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Chaves ausentes: {missing_keys}")
                # Adicionar chaves faltantes
                for key in missing_keys:
                    parsed[key] = [] if key in ['results', 'items', 'data'] else {}
            
            if hasattr(self, 'logger'):
                self.logger.info("✅ Claude response parseada com sucesso")
            return parsed
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Erro crítico no parsing Claude: {e}")
            return self._create_safe_response(expected_keys)
    
    def _create_safe_response(self, expected_keys: List[str]) -> Dict[str, Any]:
        """Cria resposta segura com estrutura padrão"""
        safe_response = {
            "error": "Failed to parse Claude response",
            "parsed_successfully": False
        }
        
        # Adicionar chaves esperadas com valores padrão
        for key in expected_keys:
            if key in ['results', 'items', 'data', 'analysis_results']:
                safe_response[key] = []
            elif key in ['analysis', 'assessment', 'statistics', 'summary']:
                safe_response[key] = {}
            else:
                safe_response[key] = None
                
        return safe_response
    
    def process_batch(self, items: List[Any], batch_size: int, process_func, **kwargs) -> List[Any]:
        """
        Processa itens em lotes
        
        Args:
            items: Lista de itens para processar
            batch_size: Tamanho do lote
            process_func: Função para processar cada lote
            **kwargs: Argumentos adicionais para process_func
            
        Returns:
            Lista com resultados processados
        """
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.logger.info(f"Processando lote {batch_num}/{total_batches}")
            
            try:
                batch_results = process_func(batch, **kwargs)
                if batch_results:
                    results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Erro no lote {batch_num}: {e}")
                # Continuar com próximo lote
                continue
        
        return results


class APIUsageTracker:
    """Rastreador de uso da API para controle de custos"""
    
    def __init__(self):
        self.usage = {
            'total_requests': 0,
            'total_tokens': 0,
            'cost_estimate': 0.0,
            'requests_by_module': {}
        }
    
    def log_request(self, module: str, tokens: int, model: str = "claude-3-5-haiku-20241022"):
        """Registra uso da API"""
        self.usage['total_requests'] += 1
        self.usage['total_tokens'] += tokens
        
        # Estimativa de custo (ajustar conforme pricing atual)
        if model == "claude-sonnet-4-20250514":
            # Claude Sonnet 4: $3.00 entrada, $15.00 saída
            # Assumindo proporção 2:1 (entrada:saída)
            avg_cost_per_token = 7.00 / 1_000_000  # Média ponderada
            cost = tokens * avg_cost_per_token
        elif model == "claude-3-5-haiku-20241022":
            # $0.25 por milhão de tokens de entrada
            cost = (tokens / 1_000_000) * 0.25
        else:
            cost = 0
        
        self.usage['cost_estimate'] += cost
        
        if module not in self.usage['requests_by_module']:
            self.usage['requests_by_module'][module] = {
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }
        
        self.usage['requests_by_module'][module]['requests'] += 1
        self.usage['requests_by_module'][module]['tokens'] += tokens
        self.usage['requests_by_module'][module]['cost'] += cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do uso"""
        return self.usage
    
    def save_report(self, filepath: str):
        """Salva relatório de uso"""
        with open(filepath, 'w') as f:
            json.dump(self.usage, f, indent=2)            