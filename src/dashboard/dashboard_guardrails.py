#!/usr/bin/env python3
"""
Dashboard Content Guardrails - digiNEV v5.1.0
==============================================

Sistema de guardrails para prevenir inserção de métricas, scores ou indicadores
fictícios nos dashboards de pesquisa acadêmica.

Baseado na estrutura do StageValidator existente, este sistema garante que:
1. Apenas dados reais do pipeline sejam exibidos
2. Nenhuma métrica seja inventada sem autorização
3. Todo conteúdo seja rastreável aos dados de origem
4. Validações sejam automáticas e obrigatórias

Author: Sistema de Validação Acadêmica
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ContentValidationResult:
    """Resultado da validação de conteúdo do dashboard"""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    metrics: Dict[str, Any] = None
    unauthorized_content: List[str] = None
    data_sources: Set[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}
        if self.unauthorized_content is None:
            self.unauthorized_content = []
        if self.data_sources is None:
            self.data_sources = set()


class DashboardContentGuard:
    """
    Guardrail para validação de conteúdo de dashboard acadêmico

    Previne inserção de:
    - Métricas inventadas
    - Scores fictícios
    - Indicadores não solicitados
    - Dados sintéticos não autorizados
    """

    # Colunas autorizadas que vêm diretamente do pipeline
    AUTHORIZED_PIPELINE_COLUMNS = {
        # Dados básicos
        'text', 'chunk_id', 'encoding', 'is_duplicate',

        # Análise política
        'political_category', 'political_confidence', 'political_alignment',
        'political_spectrum', 'political_keywords_found',

        # Análise de sentimento
        'sentiment', 'sentiment_score', 'sentiment_confidence',
        'emotion_scores', 'emotion_category',

        # Processamento linguístico
        'text_cleaned', 'tokens', 'pos_tags', 'entities',
        'language_detected', 'text_quality_score',

        # Análise semântica
        'topic_id', 'topic_words', 'topic_probability',
        'tfidf_scores', 'cluster_id', 'cluster_label',

        # Análise de conteúdo
        'hashtags', 'hashtags_normalized', 'urls', 'domains',
        'mentions', 'reply_count', 'forward_count',

        # Análise temporal
        'date', 'timestamp', 'temporal_pattern', 'time_period',
        'day_of_week', 'hour_of_day',

        # Análise de rede
        'network_metrics', 'centrality_score', 'influence_score',
        'community_id', 'connection_strength',

        # Análise qualitativa
        'qualitative_codes', 'coding_confidence', 'themes',
        'narrative_patterns', 'rhetorical_strategies',

        # Detecção de padrões
        'patterns', 'pattern_confidence', 'anomaly_score',
        'is_anomaly', 'anomaly_type',

        # Busca semântica
        'search_embeddings', 'similarity_scores', 'search_rank',

        # Validação
        'validation_passed', 'validation_score', 'quality_metrics'
    }

    # Métricas proibidas que NÃO devem ser criadas sem autorização
    PROHIBITED_SYNTHETIC_METRICS = {
        'escala_autoritaria', 'autoritario_extremo', 'democratico_moderado',
        'radicalizacao_linguistica', 'intensidade_negacionista',
        'completude_discursiva', 'marcadores_autoritarios',
        'indice_polarizacao', 'score_extremismo',
        'nivel_conspiracao', 'grau_fakenews',
        'intensidade_odio', 'radicalismo_score'
    }

    # Padrões de nomes suspeitos que indicam métricas inventadas
    SUSPICIOUS_PATTERNS = {
        '_score', '_index', '_nivel', '_grau', '_intensidade',
        '_escala', '_rating', '_rank', 'fake_', 'synthetic_',
        'random_', 'dummy_', 'mock_', 'test_score'
    }

    def __init__(self,
                 pipeline_data_path: Optional[str] = None,
                 strict_mode: bool = True,
                 log_violations: bool = True):
        """
        Inicializar guardrails do dashboard

        Args:
            pipeline_data_path: Caminho para dados reais do pipeline
            strict_mode: Se True, falha em qualquer violação
            log_violations: Se True, registra todas as violações
        """
        self.pipeline_data_path = Path(pipeline_data_path) if pipeline_data_path else None
        self.strict_mode = strict_mode
        self.log_violations = log_violations
        self.validation_history: List[ContentValidationResult] = []
        self.authorized_data_sources: Set[str] = set()
        self.violation_log: List[Dict[str, Any]] = []

        # Carregar dados reais se fornecidos
        if self.pipeline_data_path and self.pipeline_data_path.exists():
            self._load_authorized_data_sources()

    def validate_dashboard_content(self,
                                   content_data: Dict[str, Any],
                                   section_name: str = "unknown") -> ContentValidationResult:
        """
        Validar conteúdo antes de exibir no dashboard

        Args:
            content_data: Dados a serem validados
            section_name: Nome da seção do dashboard

        Returns:
            ContentValidationResult com detalhes da validação
        """
        result = ContentValidationResult()

        # Validar DataFrames
        for key, value in content_data.items():
            if isinstance(value, pd.DataFrame):
                self._validate_dataframe_content(value, key, result)
            elif isinstance(value, dict):
                self._validate_dict_content(value, key, result)
            elif isinstance(value, (list, tuple)):
                self._validate_list_content(value, key, result)

        # Verificar se há conteúdo não autorizado
        if result.unauthorized_content:
            violation = {
                'timestamp': datetime.now().isoformat(),
                'section': section_name,
                'violations': result.unauthorized_content,
                'severity': 'ERROR' if self.strict_mode else 'WARNING'
            }
            self.violation_log.append(violation)

            if self.log_violations:
                logger.error(f"VIOLAÇÃO DE GUARDRAIL - Seção: {section_name}")
                logger.error(f"Conteúdo não autorizado: {result.unauthorized_content}")

            if self.strict_mode:
                result.is_valid = False
                result.errors.append(f"Conteúdo não autorizado detectado: {result.unauthorized_content}")

        self.validation_history.append(result)
        return result

    def _validate_dataframe_content(self,
                                    df: pd.DataFrame,
                                    df_name: str,
                                    result: ContentValidationResult):
        """Validar conteúdo de DataFrame"""

        # Verificar colunas não autorizadas
        unauthorized_cols = set(df.columns) - self.AUTHORIZED_PIPELINE_COLUMNS
        for col in unauthorized_cols:
            # Verificar se é métrica proibida
            if col.lower() in self.PROHIBITED_SYNTHETIC_METRICS:
                result.unauthorized_content.append(f"Métrica proibida em {df_name}: {col}")

            # Verificar padrões suspeitos
            elif any(pattern in col.lower() for pattern in self.SUSPICIOUS_PATTERNS):
                result.unauthorized_content.append(f"Métrica suspeita em {df_name}: {col}")

            # Verificar se contém dados sintéticos
            elif self._contains_synthetic_data(df[col]):
                result.unauthorized_content.append(f"Dados sintéticos detectados em {df_name}.{col}")

        # Registrar fonte de dados
        result.data_sources.add(f"dataframe:{df_name}")

    def _validate_dict_content(self,
                               data_dict: Dict[str, Any],
                               dict_name: str,
                               result: ContentValidationResult):
        """Validar conteúdo de dicionário"""

        for key, value in data_dict.items():
            # Verificar chaves proibidas
            if key.lower() in self.PROHIBITED_SYNTHETIC_METRICS:
                result.unauthorized_content.append(f"Métrica proibida em {dict_name}: {key}")

            # Verificar valores sintéticos
            elif isinstance(value, (int, float)) and self._is_synthetic_number(value):
                result.unauthorized_content.append(f"Valor sintético em {dict_name}.{key}: {value}")

        result.data_sources.add(f"dict:{dict_name}")

    def _validate_list_content(self,
                               data_list: Union[List, tuple],
                               list_name: str,
                               result: ContentValidationResult):
        """Validar conteúdo de lista"""

        # Verificar se contém dados fictícios
        if any(isinstance(item, str) and 'fake' in item.lower() for item in data_list):
            result.unauthorized_content.append(f"Dados fictícios detectados em {list_name}")

        result.data_sources.add(f"list:{list_name}")

    def _contains_synthetic_data(self, series: pd.Series) -> bool:
        """Detectar se uma série contém dados sintéticos"""

        # Verificar se todos os valores são idênticos (suspeito)
        if series.nunique() <= 1 and len(series) > 10:
            return True

        # Verificar padrões matemáticos simples (números sequenciais, etc.)
        if series.dtype in ['int64', 'float64']:
            # Diferenças constantes podem indicar dados sintéticos
            if len(series) > 5:
                diffs = series.diff().dropna()
                if diffs.nunique() <= 2 and diffs.abs().max() < 1:
                    return True

        # Verificar se valores estão em ranges típicos de np.random
        if series.dtype == 'float64':
            if ((series >= 0) & (series <= 1)).all() and series.std() > 0.1:
                # Provável np.random.uniform(0, 1)
                return True

        return False

    def _is_synthetic_number(self, value: Union[int, float]) -> bool:
        """Verificar se um número parece sintético"""

        # Valores com muitas casas decimais são suspeitos
        if isinstance(value, float):
            decimal_places = len(str(value).split('.')[-1])
            if decimal_places > 6:
                return True

        # Valores muito específicos em ranges 0-1 são suspeitos
        if isinstance(value, float) and 0 <= value <= 1:
            # Típico de np.random.uniform()
            return True

        return False

    def _load_authorized_data_sources(self):
        """Carregar fontes de dados autorizadas do pipeline"""

        try:
            # Tentar carregar diferentes formatos de dados
            if self.pipeline_data_path.suffix == '.csv':
                df = pd.read_csv(self.pipeline_data_path, sep=';', nrows=10)  # Apenas amostra
                self.authorized_data_sources.update(df.columns)

            elif self.pipeline_data_path.suffix == '.json':
                with open(self.pipeline_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.authorized_data_sources.update(data.keys())

            logger.info(f"Carregadas {len(self.authorized_data_sources)} fontes autorizadas")

        except Exception as e:
            logger.warning(f"Não foi possível carregar dados autorizados: {e}")

    def register_authorized_metric(self, metric_name: str, source: str, justification: str):
        """
        Registrar uma nova métrica como autorizada

        Args:
            metric_name: Nome da métrica
            source: Fonte dos dados (arquivo, API, etc.)
            justification: Justificativa acadêmica para a métrica
        """

        registration = {
            'metric_name': metric_name,
            'source': source,
            'justification': justification,
            'timestamp': datetime.now().isoformat(),
            'authorized_by': 'dashboard_guardrails'
        }

        # Adicionar à lista de colunas autorizadas
        self.AUTHORIZED_PIPELINE_COLUMNS.add(metric_name)

        logger.info(f"Métrica autorizada: {metric_name} - Fonte: {source}")

        # Salvar registro de autorização
        auth_file = Path('dashboard_authorized_metrics.json')
        if auth_file.exists():
            with open(auth_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        else:
            existing = []

        existing.append(registration)

        with open(auth_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def get_violation_report(self) -> Dict[str, Any]:
        """Gerar relatório de violações detectadas"""

        total_validations = len(self.validation_history)
        violations = len(self.violation_log)

        if not self.violation_log:
            return {
                'status': 'LIMPO',
                'total_validations': total_validations,
                'violations_detected': 0,
                'compliance_rate': 100.0
            }

        return {
            'status': 'VIOLAÇÕES_DETECTADAS',
            'total_validations': total_validations,
            'violations_detected': violations,
            'compliance_rate': ((total_validations - violations) / total_validations * 100) if total_validations > 0 else 0,
            'recent_violations': self.violation_log[-5:],  # Últimas 5 violações
            'violation_types': self._categorize_violations()
        }

    def _categorize_violations(self) -> Dict[str, int]:
        """Categorizar tipos de violações"""

        categories = {
            'metricas_proibidas': 0,
            'dados_sinteticos': 0,
            'padroes_suspeitos': 0,
            'outros': 0
        }

        for violation in self.violation_log:
            for content in violation.get('violations', []):
                if 'métrica proibida' in content.lower():
                    categories['metricas_proibidas'] += 1
                elif 'sintético' in content.lower():
                    categories['dados_sinteticos'] += 1
                elif 'suspeita' in content.lower():
                    categories['padroes_suspeitos'] += 1
                else:
                    categories['outros'] += 1

        return categories

    def enforce_guardrails(self, dashboard_function):
        """
        Decorator para aplicar guardrails automaticamente

        Usage:
            @guardrail.enforce_guardrails
            def render_dashboard_section():
                # código do dashboard
                return content
        """

        def wrapper(*args, **kwargs):
            # Executar função original
            result = dashboard_function(*args, **kwargs)

            # Validar resultado
            if isinstance(result, dict):
                validation = self.validate_dashboard_content(
                    result,
                    dashboard_function.__name__
                )

                if not validation.is_valid and self.strict_mode:
                    raise ValueError(f"Guardrail violado em {dashboard_function.__name__}: {validation.errors}")

                elif validation.warnings:
                    logger.warning(f"Avisos de guardrail em {dashboard_function.__name__}: {validation.warnings}")

            return result

        return wrapper


# Instância global do guardrail para uso em dashboards
dashboard_guardrail = DashboardContentGuard(
    pipeline_data_path="pipeline_outputs/dashboard_ready",
    strict_mode=True,
    log_violations=True
)


def validate_dashboard_data(data: Dict[str, Any], section: str = "dashboard") -> bool:
    """
    Função de conveniência para validação rápida

    Args:
        data: Dados a serem validados
        section: Nome da seção

    Returns:
        True se válido, False caso contrário
    """

    result = dashboard_guardrail.validate_dashboard_content(data, section)
    return result.is_valid


def require_real_data_only(func):
    """
    Decorator para garantir que apenas dados reais sejam usados

    Usage:
        @require_real_data_only
        def create_political_chart(df):
            # Esta função só aceitará DataFrames com dados reais
            return chart
    """

    def wrapper(*args, **kwargs):
        # Verificar todos os argumentos DataFrame
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                validation = dashboard_guardrail._validate_dataframe_content(
                    arg, f"{func.__name__}_input", ContentValidationResult()
                )
                if validation.unauthorized_content:
                    raise ValueError(f"Dados não autorizados detectados em {func.__name__}")

        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    # Teste básico do sistema de guardrails
    print("🔒 Testando sistema de guardrails...")

    # Criar dados de teste (válidos)
    valid_data = {
        'political_analysis': pd.DataFrame({
            'text': ['Teste 1', 'Teste 2'],
            'political_category': ['direita', 'esquerda'],
            'sentiment_score': [0.7, 0.3]
        })
    }

    # Criar dados de teste (inválidos)
    invalid_data = {
        'fake_analysis': pd.DataFrame({
            'text': ['Teste 1', 'Teste 2'],
            'escala_autoritaria': ['autoritario_extremo', 'democratico'],
            'fake_score': [0.12345678, 0.87654321]  # Valores sintéticos
        })
    }

    # Testar validação
    print("✅ Testando dados válidos...")
    result_valid = dashboard_guardrail.validate_dashboard_content(valid_data, "teste_valido")
    print(f"Resultado: {'VÁLIDO' if result_valid.is_valid else 'INVÁLIDO'}")

    print("\n❌ Testando dados inválidos...")
    result_invalid = dashboard_guardrail.validate_dashboard_content(invalid_data, "teste_invalido")
    print(f"Resultado: {'VÁLIDO' if result_invalid.is_valid else 'INVÁLIDO'}")
    print(f"Violações: {result_invalid.unauthorized_content}")

    print("\n📊 Relatório de violações:")
    report = dashboard_guardrail.get_violation_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))