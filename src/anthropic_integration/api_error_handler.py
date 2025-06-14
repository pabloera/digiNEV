"""
Framework de Tratamento de Erros para API Integration
Implementa sistema de retry automático e escalação para usuário conforme especificado.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .base import AnthropicBase

logger = logging.getLogger(__name__)

@dataclass
class APIError:
    """Classe para representar erros da API"""
    stage: str
    operation: str
    error_type: str
    error_message: str
    timestamp: datetime
    retry_count: int = 0
    resolved: bool = False
    resolution_details: Optional[str] = None

@dataclass
class APIOperationResult:
    """Resultado de uma operação da API"""
    success: bool
    data: Any = None
    error: Optional[APIError] = None
    retry_count: int = 0
    total_time: float = 0.0

class APIErrorHandler:
    """
    Framework de tratamento de erros para operações da API

    Implementa:
    - 2 tentativas automáticas para resolução de erros
    - Escalação para usuário após falhas
    - Log detalhado de erros e resoluções
    - Detecção de alucinações e inconsistências
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.error_log_file = self.project_root / "logs" / "api_errors.json"
        self.errors: List[APIError] = []
        self.load_error_history()

        # Configurações de retry
        self.max_retries = 2
        self.retry_delay = 1.0  # segundos entre tentativas

        # Padrões para detectar alucinações e inconsistências
        self.hallucination_patterns = [
            "não tenho informações",
            "não posso acessar",
            "como um modelo de linguagem",
            "não tenho acesso aos dados",
            "desculpe, mas não consigo",
            "preciso de mais contexto"
        ]

        self.inconsistency_patterns = [
            "contradição",
            "conflito",
            "inconsistente",
            "não bate com",
            "diferente do esperado"
        ]

    def load_error_history(self):
        """Carrega histórico de erros do arquivo de log"""
        if self.error_log_file.exists():
            try:
                with open(self.error_log_file, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                    for error_dict in error_data:
                        error_dict['timestamp'] = datetime.fromisoformat(error_dict['timestamp'])
                        self.errors.append(APIError(**error_dict))
            except Exception as e:
                logger.warning(f"Erro ao carregar histórico de erros: {e}")

    def save_error_history(self):
        """Salva histórico de erros no arquivo de log"""
        try:
            self.error_log_file.parent.mkdir(parents=True, exist_ok=True)
            error_data = []
            for error in self.errors:
                error_dict = asdict(error)
                error_dict['timestamp'] = error.timestamp.isoformat()
                error_data.append(error_dict)

            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar histórico de erros: {e}")

    def detect_hallucination(self, response: str) -> bool:
        """Detecta possíveis alucinações na resposta da API"""
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in self.hallucination_patterns)

    def detect_inconsistency(self, response: str, expected_context: str = None) -> bool:
        """Detecta inconsistências na resposta da API"""
        response_lower = response.lower()
        has_inconsistency_markers = any(pattern in response_lower for pattern in self.inconsistency_patterns)

        # Verificações adicionais de consistência se contexto fornecido
        if expected_context:
            expected_lower = expected_context.lower()
            # Verifica se a resposta faz sentido no contexto
            # Implementar lógica específica conforme necessário
            pass

        return has_inconsistency_markers

    def execute_with_retry(
        self,
        operation_func: Callable,
        stage: str,
        operation: str,
        *args,
        **kwargs
    ) -> APIOperationResult:
        """
        Executa operação com sistema de retry automático

        Args:
            operation_func: Função a ser executada
            stage: Etapa do pipeline
            operation: Nome da operação
            *args, **kwargs: Argumentos para a função

        Returns:
            APIOperationResult com resultado da operação
        """
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Tentativa {attempt + 1}/{self.max_retries + 1} - {stage}.{operation}")

                result = operation_func(*args, **kwargs)

                # Verificar se resultado indica alucinação ou inconsistência
                if isinstance(result, str):
                    if self.detect_hallucination(result):
                        raise ValueError(f"Alucinação detectada na resposta: {result[:100]}...")

                    if self.detect_inconsistency(result):
                        raise ValueError(f"Inconsistência detectada na resposta: {result[:100]}...")

                # Sucesso
                total_time = time.time() - start_time
                logger.info(f"Operação {stage}.{operation} concluída com sucesso (tentativa {attempt + 1})")

                return APIOperationResult(
                    success=True,
                    data=result,
                    retry_count=attempt,
                    total_time=total_time
                )

            except Exception as e:
                last_error = APIError(
                    stage=stage,
                    operation=operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    timestamp=datetime.now(),
                    retry_count=attempt
                )

                logger.warning(f"Erro na tentativa {attempt + 1}: {e}")

                if attempt < self.max_retries:
                    logger.info(f"Aguardando {self.retry_delay}s antes da próxima tentativa...")
                    time.sleep(self.retry_delay)
                    # Aumentar delay exponencialmente
                    self.retry_delay *= 1.5
                else:
                    # Todas as tentativas falharam
                    self.errors.append(last_error)
                    self.save_error_history()

                    total_time = time.time() - start_time
                    logger.error(f"Operação {stage}.{operation} falhou após {self.max_retries + 1} tentativas")

                    return APIOperationResult(
                        success=False,
                        error=last_error,
                        retry_count=self.max_retries,
                        total_time=total_time
                    )

        # Não deveria chegar aqui, mas safety net
        total_time = time.time() - start_time
        return APIOperationResult(
            success=False,
            error=last_error,
            retry_count=self.max_retries,
            total_time=total_time
        )

    def escalate_to_user(self, error: APIError, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Escala erro para usuário com informações detalhadas

        Args:
            error: Erro a ser escalado
            context: Contexto adicional sobre o erro

        Returns:
            Dicionário com informações para o usuário
        """
        escalation_info = {
            "timestamp": datetime.now().isoformat(),
            "stage": error.stage,
            "operation": error.operation,
            "error_type": error.error_type,
            "error_message": error.error_message,
            "retry_count": error.retry_count,
            "suggestions": self._generate_error_suggestions(error),
            "context": context or {}
        }

        # Salvar escalação em arquivo específico
        escalation_file = self.project_root / "logs" / "user_escalations.json"
        try:
            escalations = []
            if escalation_file.exists():
                with open(escalation_file, 'r', encoding='utf-8') as f:
                    escalations = json.load(f)

            escalations.append(escalation_info)

            with open(escalation_file, 'w', encoding='utf-8') as f:
                json.dump(escalations, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erro ao salvar escalação: {e}")

        return escalation_info

    def _generate_error_suggestions(self, error: APIError) -> List[str]:
        """Gera sugestões para resolução do erro"""
        suggestions = []

        error_msg_lower = error.error_message.lower()

        if "api key" in error_msg_lower or "authentication" in error_msg_lower:
            suggestions.extend([
                "Verifique se a chave API está configurada corretamente no arquivo .env",
                "Confirme se a chave API não expirou",
                "Teste a chave API com uma requisição simples"
            ])

        elif "rate limit" in error_msg_lower or "quota" in error_msg_lower:
            suggestions.extend([
                "Aguarde alguns minutos antes de tentar novamente",
                "Considere reduzir o tamanho dos lotes de processamento",
                "Verifique se há outros processos usando a API simultaneamente"
            ])

        elif "timeout" in error_msg_lower or "connection" in error_msg_lower:
            suggestions.extend([
                "Verifique sua conexão com a internet",
                "Tente novamente em alguns momentos",
                "Considere dividir dados em chunks menores"
            ])

        elif "alucinação" in error_msg_lower or "inconsistência" in error_msg_lower:
            suggestions.extend([
                "Revise o prompt para ser mais específico",
                "Adicione exemplos claros no prompt",
                "Considere usar temperatura mais baixa (ex: 0.1)",
                "Verifique se os dados de entrada estão limpos"
            ])

        else:
            suggestions.extend([
                "Verifique os logs detalhados para mais informações",
                "Confirme se os dados de entrada estão no formato correto",
                "Considere executar um teste simples primeiro"
            ])

        return suggestions

    def get_error_report(self, stage: str = None) -> Dict[str, Any]:
        """
        Gera relatório de erros

        Args:
            stage: Filtrar por etapa específica (opcional)

        Returns:
            Dicionário com relatório de erros
        """
        filtered_errors = self.errors
        if stage:
            filtered_errors = [e for e in self.errors if e.stage == stage]

        total_errors = len(filtered_errors)
        resolved_errors = len([e for e in filtered_errors if e.resolved])

        error_types = {}
        for error in filtered_errors:
            if error.error_type not in error_types:
                error_types[error.error_type] = 0
            error_types[error.error_type] += 1

        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "unresolved_errors": total_errors - resolved_errors,
            "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0,
            "error_types": error_types,
            "recent_errors": [asdict(e) for e in filtered_errors[-5:]]  # Últimos 5 erros
        }

class APIQualityChecker(AnthropicBase):
    """
    Verificador de qualidade para saídas da API
    Detecta inconsistências, alucinações e problemas de qualidade
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()

    def validate_output_quality(
        self,
        output: Any,
        expected_format: str,
        context: Dict[str, Any] = None,
        stage: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Valida qualidade da saída da API

        Args:
            output: Saída da API para validar
            expected_format: Formato esperado ('json', 'text', 'list', etc.)
            context: Contexto para validação
            stage: Etapa do pipeline

        Returns:
            Dicionário com resultado da validação
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "quality_score": 1.0,
            "suggestions": []
        }

        try:
            # Validar formato
            format_issues = self._validate_format(output, expected_format)
            validation_result["issues"].extend(format_issues)

            # Validar conteúdo se for texto
            if isinstance(output, str):
                content_issues = self._validate_content_quality(output, context)
                validation_result["issues"].extend(content_issues)

            # Calcular score de qualidade
            validation_result["quality_score"] = max(0.0, 1.0 - (len(validation_result["issues"]) * 0.1))
            validation_result["valid"] = validation_result["quality_score"] > 0.5

            # Gerar sugestões se há problemas
            if validation_result["issues"]:
                validation_result["suggestions"] = self._generate_quality_suggestions(validation_result["issues"])

        except Exception as e:
            logger.error(f"Erro na validação de qualidade: {e}")
            validation_result["valid"] = False
            validation_result["issues"].append(f"Erro na validação: {e}")

        return validation_result

    def _validate_format(self, output: Any, expected_format: str) -> List[str]:
        """Valida formato da saída"""
        issues = []

        if expected_format == "json":
            if isinstance(output, str):
                try:
                    json.loads(output)
                except json.JSONDecodeError:
                    issues.append("Saída não é um JSON válido")
            elif not isinstance(output, dict):
                issues.append("Saída deveria ser JSON/dict")

        elif expected_format == "list":
            if not isinstance(output, (list, tuple)):
                issues.append("Saída deveria ser uma lista")

        elif expected_format == "text":
            if not isinstance(output, str):
                issues.append("Saída deveria ser texto")

        return issues

    def _validate_content_quality(self, text: str, context: Dict[str, Any] = None) -> List[str]:
        """Valida qualidade do conteúdo textual"""
        issues = []

        # Detectar alucinações
        if self.error_handler.detect_hallucination(text):
            issues.append("Possível alucinação detectada")

        # Detectar inconsistências
        expected_context = context.get("expected_context", "") if context else ""
        if self.error_handler.detect_inconsistency(text, expected_context):
            issues.append("Inconsistência detectada")

        # Verificar completude
        if len(text.strip()) < 10:
            issues.append("Resposta muito curta")

        # Verificar repetições excessivas
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:  # Mais de 30% de repetição
                issues.append("Repetição excessiva detectada")

        return issues

    def _generate_quality_suggestions(self, issues: List[str]) -> List[str]:
        """Gera sugestões para melhorar qualidade"""
        suggestions = []

        for issue in issues:
            if "alucinação" in issue.lower():
                suggestions.append("Use prompts mais específicos e contextuais")
                suggestions.append("Reduza a temperatura do modelo")

            elif "inconsistência" in issue.lower():
                suggestions.append("Forneça contexto mais claro no prompt")
                suggestions.append("Valide dados de entrada antes do processamento")

            elif "json" in issue.lower():
                suggestions.append("Especifique formato JSON no prompt")
                suggestions.append("Adicione exemplos de JSON válido")

            elif "repetição" in issue.lower():
                suggestions.append("Use penalties para evitar repetição")
                suggestions.append("Revise o prompt para ser mais direto")

        return list(set(suggestions))  # Remove duplicatas
