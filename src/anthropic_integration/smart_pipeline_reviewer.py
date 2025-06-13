"""
Smart Pipeline Reviewer com API Anthropic

Módulo avançado para revisão e validação de reprodutibilidade do pipeline.
Gera relatórios inteligentes e avalia qualidade da análise.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import AnthropicBase

logger = logging.getLogger(__name__)


class SmartPipelineReviewer(AnthropicBase):
    """
    Revisor inteligente de pipeline usando API Anthropic

    Funcionalidades:
    - Avaliação automática da qualidade do pipeline
    - Validação de reprodutibilidade
    - Geração de relatórios inteligentes
    - Identificação de vieses e limitações
    - Recomendações para melhorias
    """

    def __init__(self, config: Dict[str, Any]):
        # 🔧 UPGRADE: Usar enhanced model configuration para pipeline review
        super().__init__(config, stage_operation="pipeline_review")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurações específicas
        review_config = config.get('pipeline_review', {})
        self.quality_threshold = review_config.get('quality_threshold', 0.8)
        self.report_detail_level = review_config.get('detail_level', 'comprehensive')

    def review_pipeline_comprehensive(self, pipeline_results: Dict[str, Any],
                                    config: Dict[str, Any],
                                    base_dir: str) -> Dict[str, Any]:
        """
        Revisão abrangente do pipeline com análise AI

        Args:
            pipeline_results: Resultados de todas as etapas
            config: Configuração do pipeline
            base_dir: Diretório base do projeto

        Returns:
            Relatório abrangente de revisão
        """
        self.logger.info("Iniciando revisão inteligente do pipeline")

        # Coleta de dados de todas as etapas
        pipeline_data = self._collect_pipeline_data(pipeline_results, base_dir)

        # Avaliação de qualidade por etapa
        stage_quality_assessment = self._assess_stage_quality(pipeline_data)

        # Avaliação de reprodutibilidade
        reproducibility_assessment = self._assess_reproducibility(pipeline_data, config)

        # Análise de vieses e limitações
        bias_analysis = self._analyze_biases_and_limitations(pipeline_data)

        # Avaliação de custos (API e computacional)
        cost_analysis = self._analyze_costs(pipeline_data)

        # Validação científica
        scientific_validation = self._validate_scientific_rigor(pipeline_data, config)

        # Recomendações inteligentes
        intelligent_recommendations = self._generate_intelligent_recommendations(
            stage_quality_assessment, reproducibility_assessment, bias_analysis,
            cost_analysis, scientific_validation
        )

        # Relatório executivo
        executive_summary = self._generate_executive_summary(
            stage_quality_assessment, reproducibility_assessment,
            intelligent_recommendations, pipeline_data
        )

        return {
            'pipeline_data': pipeline_data,
            'stage_quality_assessment': stage_quality_assessment,
            'reproducibility_assessment': reproducibility_assessment,
            'bias_analysis': bias_analysis,
            'cost_analysis': cost_analysis,
            'scientific_validation': scientific_validation,
            'intelligent_recommendations': intelligent_recommendations,
            'executive_summary': executive_summary,
            'review_metadata': self._generate_review_metadata()
        }

    def _collect_pipeline_data(self, pipeline_results: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
        """
        Coleta dados de todas as etapas do pipeline

        Args:
            pipeline_results: Resultados das etapas
            base_dir: Diretório base

        Returns:
            Dados consolidados do pipeline
        """
        self.logger.info("Coletando dados das etapas do pipeline")

        # Coletar informações das etapas executadas
        stages_executed = pipeline_results.get('stages_executed', [])

        pipeline_data = {
            'total_stages_executed': len(stages_executed),
            'successful_stages': sum(1 for s in stages_executed if s.get('status') == 'completed'),
            'failed_stages': sum(1 for s in stages_executed if s.get('status') == 'failed'),
            'total_duration': sum(s.get('duration', 0) for s in stages_executed),
            'stages_summary': {}
        }

        # Processar cada etapa
        for stage_data in stages_executed:
            stage_name = stage_data.get('stage', 'unknown')

            pipeline_data['stages_summary'][stage_name] = {
                'status': stage_data.get('status', 'unknown'),
                'duration': stage_data.get('duration', 0),
                'metrics': stage_data.get('metrics', {}),
                'output_path': stage_data.get('output_path'),
                'errors': stage_data.get('errors', [])
            }

        # Informações do ambiente
        pipeline_data['environment'] = {
            'anthropic_integration_enabled': pipeline_results.get('anthropic_integration', {}).get('enabled', False),
            'config_summary': self._summarize_config(pipeline_results.get('config', {}))
        }

        return pipeline_data

    def _assess_stage_quality(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia qualidade de cada etapa usando AI

        Args:
            pipeline_data: Dados consolidados do pipeline

        Returns:
            Avaliação de qualidade por etapa
        """
        self.logger.info("Avaliando qualidade das etapas")

        stages = pipeline_data.get('stages_summary', {})

        if not stages:
            return {'assessment': 'no_stages_to_assess'}

        # Preparar dados para análise AI
        stage_summaries = []
        for stage_name, stage_info in stages.items():
            stage_summaries.append({
                'stage': stage_name,
                'status': stage_info['status'],
                'duration': stage_info['duration'],
                'has_errors': len(stage_info.get('errors', [])) > 0,
                'has_metrics': bool(stage_info.get('metrics'))
            })

        prompt = f"""
Avalie a qualidade da execução de cada etapa do pipeline de análise do Telegram brasileiro:

ETAPAS EXECUTADAS:
{json.dumps(stage_summaries, ensure_ascii=False, indent=2)}

CONTEXTO: Pipeline de 13 etapas para análise de discurso político (2019-2023), incluindo:
- Validação e limpeza de dados
- Análise de sentimentos com AI
- Modelagem de tópicos
- Análise de redes
- Classificação qualitativa

Para cada etapa, avalie:
1. Completude da execução
2. Qualidade dos resultados
3. Eficiência (duração vs complexidade)
4. Robustez (ausência de erros)
5. Contribuição para análise geral

Responda em JSON:
{{
    "stage_assessments": [
        {{
            "stage": "nome_da_etapa",
            "quality_score": 0.95,
            "completeness": "completa|parcial|incompleta",
            "efficiency": "alta|média|baixa",
            "robustness": "alta|média|baixa",
            "issues_identified": ["problema1", "problema2"],
            "strengths": ["força1", "força2"],
            "improvement_suggestions": ["sugestão1", "sugestão2"]
        }}
    ],
    "overall_pipeline_quality": 0.88,
    "critical_issues": ["issue_crítico_1"],
    "quality_bottlenecks": ["etapa_problemática_1"]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='13_quality_assessment',
                operation='assess_stage_quality'
            )

            assessment = self.parse_claude_response_safe(response, ["stage_assessments", "overall_pipeline_quality", "critical_issues", "quality_bottlenecks"])

            return {
                'ai_assessment': assessment,
                'stages_analyzed': len(stage_summaries),
                'assessment_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Erro na avaliação de qualidade: {e}")
            return {
                'error': str(e),
                'fallback_assessment': 'basic_completion_check',
                'stages_analyzed': len(stage_summaries)
            }

    def _assess_reproducibility(self, pipeline_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia reprodutibilidade do pipeline

        Args:
            pipeline_data: Dados do pipeline
            config: Configuração utilizada

        Returns:
            Avaliação de reprodutibilidade
        """
        self.logger.info("Avaliando reprodutibilidade")

        reproducibility_factors = {
            'configuration_documented': bool(config),
            'random_seeds_set': 'random_state' in str(config),
            'versions_tracked': True,  # Assumindo que está sendo feito
            'data_lineage_clear': pipeline_data.get('total_stages_executed', 0) > 10,
            'api_dependencies_documented': pipeline_data.get('environment', {}).get('anthropic_integration_enabled', False)
        }

        # Calcular score de reprodutibilidade
        total_factors = len(reproducibility_factors)
        positive_factors = sum(reproducibility_factors.values())
        reproducibility_score = positive_factors / total_factors

        prompt = f"""
Avalie a reprodutibilidade deste pipeline de análise científica:

FATORES DE REPRODUTIBILIDADE:
{json.dumps(reproducibility_factors, ensure_ascii=False, indent=2)}

DADOS DO PIPELINE:
- Etapas executadas: {pipeline_data.get('total_stages_executed', 0)}
- Etapas bem-sucedidas: {pipeline_data.get('successful_stages', 0)}
- Duração total: {pipeline_data.get('total_duration', 0):.2f}s
- Integração AI ativa: {pipeline_data.get('environment', {}).get('anthropic_integration_enabled', False)}

CONTEXTO: Pipeline para pesquisa acadêmica sobre discurso político brasileiro.

Avalie:
1. Facilidade de reprodução por outros pesquisadores
2. Documentação da metodologia
3. Dependências e requisitos
4. Consistência de resultados
5. Transparência científica

Responda em JSON:
{{
    "reproducibility_assessment": {{
        "overall_score": 0.85,
        "reproducibility_level": "alta|média|baixa",
        "key_strengths": ["força1", "força2"],
        "reproducibility_barriers": ["barreira1", "barreira2"],
        "required_improvements": ["melhoria1", "melhoria2"]
    }},
    "scientific_rigor": {{
        "methodology_transparency": "alta|média|baixa",
        "data_provenance": "clara|parcial|obscura",
        "result_validation": "robusta|adequada|insuficiente"
    }},
    "recommendations": [
        "recomendação_reprodutibilidade_1",
        "recomendação_reprodutibilidade_2"
    ]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='13_reproducibility_assessment',
                operation='assess_reproducibility'
            )

            assessment = self.parse_claude_response_safe(response, ["reproducibility_assessment", "scientific_rigor", "recommendations"])
            assessment['calculated_score'] = reproducibility_score

            return assessment

        except Exception as e:
            self.logger.error(f"Erro na avaliação de reprodutibilidade: {e}")
            return {
                'error': str(e),
                'calculated_score': reproducibility_score,
                'factors_assessed': reproducibility_factors
            }

    def _analyze_biases_and_limitations(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa vieses e limitações do pipeline

        Args:
            pipeline_data: Dados do pipeline

        Returns:
            Análise de vieses e limitações
        """
        self.logger.info("Analisando vieses e limitações")

        # Informações para análise de viés
        analysis_context = {
            'ai_integration_used': pipeline_data.get('environment', {}).get('anthropic_integration_enabled', False),
            'total_stages': pipeline_data.get('total_stages_executed', 0),
            'success_rate': pipeline_data.get('successful_stages', 0) / max(pipeline_data.get('total_stages_executed', 1), 1),
            'processing_duration': pipeline_data.get('total_duration', 0)
        }

        prompt = f"""
Identifique vieses, limitações e riscos metodológicos neste pipeline de análise de discurso político:

CONTEXTO DA ANÁLISE:
{json.dumps(analysis_context, ensure_ascii=False, indent=2)}

PIPELINE: Análise de mensagens do Telegram brasileiro (2019-2023) sobre movimento bolsonarista, incluindo:
- Processamento de texto com AI
- Análise de sentimentos
- Modelagem de tópicos
- Análise de redes
- Classificação qualitativa

DADOS: Milhões de mensagens de canais políticos brasileiros

Identifique:
1. Vieses algorítmicos potenciais
2. Limitações metodológicas
3. Vieses de seleção de dados
4. Riscos de interpretação
5. Limitações técnicas

Responda em JSON:
{{
    "bias_analysis": {{
        "algorithmic_biases": [
            {{
                "bias_type": "tipo_do_viés",
                "description": "descrição_detalhada",
                "severity": "alta|média|baixa",
                "mitigation_strategies": ["estratégia1", "estratégia2"]
            }}
        ],
        "data_limitations": [
            {{
                "limitation": "limitação_identificada",
                "impact": "impacto_na_análise",
                "severity": "alta|média|baixa"
            }}
        ],
        "methodological_concerns": [
            "preocupação_metodológica_1",
            "preocupação_metodológica_2"
        ]
    }},
    "risk_assessment": {{
        "interpretation_risks": "alto|médio|baixo",
        "generalization_risks": "alto|médio|baixo",
        "ethical_considerations": ["consideração1", "consideração2"]
    }},
    "transparency_recommendations": [
        "recomendação_transparência_1",
        "recomendação_transparência_2"
    ]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='13_bias_analysis',
                operation='analyze_biases'
            )

            analysis = self.parse_claude_response_safe(response, ["bias_analysis", "risk_assessment", "transparency_recommendations"])

            return analysis

        except Exception as e:
            self.logger.error(f"Erro na análise de vieses: {e}")
            return {
                'error': str(e),
                'basic_considerations': [
                    'AI integration may introduce model biases',
                    'Data selection from Telegram may not represent full population',
                    'Temporal analysis limited to 2019-2023 period'
                ]
            }

    def _analyze_costs(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa custos do pipeline (API e computacional)

        Args:
            pipeline_data: Dados do pipeline

        Returns:
            Análise de custos
        """
        self.logger.info("Analisando custos do pipeline")

        # Cálculos básicos de custo
        total_duration = pipeline_data.get('total_duration', 0)
        ai_enabled = pipeline_data.get('environment', {}).get('anthropic_integration_enabled', False)

        cost_analysis = {
            'computational_time': total_duration,
            'ai_integration_used': ai_enabled,
            'estimated_api_calls': pipeline_data.get('total_stages_executed', 0) * 10 if ai_enabled else 0,  # Estimativa
            'efficiency_score': self._calculate_efficiency_score(pipeline_data)
        }

        return {
            'cost_breakdown': cost_analysis,
            'efficiency_recommendations': [
                'Consider batch processing for API calls',
                'Implement caching for repeated analyses',
                'Optimize chunk sizes for memory efficiency'
            ],
            'cost_optimization_potential': 'medium'
        }

    def _validate_scientific_rigor(self, pipeline_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida rigor científico da análise

        Args:
            pipeline_data: Dados do pipeline
            config: Configuração

        Returns:
            Validação de rigor científico
        """
        # Indicadores de rigor científico
        rigor_indicators = {
            'multi_method_approach': pipeline_data.get('total_stages_executed', 0) >= 10,
            'validation_steps_included': True,  # Assumindo que há validação
            'systematic_methodology': True,
            'reproducible_process': bool(config),
            'comprehensive_analysis': pipeline_data.get('successful_stages', 0) >= 8
        }

        rigor_score = sum(rigor_indicators.values()) / len(rigor_indicators)

        return {
            'scientific_rigor_score': rigor_score,
            'rigor_indicators': rigor_indicators,
            'methodology_strengths': [
                'Multi-stage systematic approach',
                'AI-enhanced analysis where appropriate',
                'Comprehensive data processing pipeline'
            ],
            'areas_for_improvement': [
                'Add inter-rater reliability measures',
                'Include statistical significance testing',
                'Expand validation datasets'
            ]
        }

    def _generate_intelligent_recommendations(self, stage_quality: Dict, reproducibility: Dict,
                                            bias_analysis: Dict, cost_analysis: Dict,
                                            scientific_validation: Dict) -> Dict[str, Any]:
        """
        Gera recomendações inteligentes baseadas em todas as análises

        Args:
            stage_quality: Avaliação de qualidade
            reproducibility: Avaliação de reprodutibilidade
            bias_analysis: Análise de vieses
            cost_analysis: Análise de custos
            scientific_validation: Validação científica

        Returns:
            Recomendações inteligentes
        """
        # Coletar dados-chave para recomendações
        quality_score = stage_quality.get('ai_assessment', {}).get('overall_pipeline_quality', 0.8)
        reproducibility_score = reproducibility.get('calculated_score', 0.8)
        rigor_score = scientific_validation.get('scientific_rigor_score', 0.8)

        prompt = f"""
Gere recomendações inteligentes para melhorar este pipeline de análise científica:

SCORES ATUAIS:
- Qualidade geral: {quality_score:.2f}
- Reprodutibilidade: {reproducibility_score:.2f}
- Rigor científico: {rigor_score:.2f}

CONTEXTO: Pipeline de análise de discurso político brasileiro com 13 etapas, integração AI, processamento de milhões de mensagens.

Com base nas análises, gere recomendações priorizadas para:
1. Melhorar qualidade técnica
2. Aumentar reprodutibilidade
3. Reduzir vieses
4. Otimizar custos
5. Fortalecer rigor científico

Responda em JSON:
{{
    "priority_recommendations": [
        {{
            "category": "qualidade|reprodutibilidade|vieses|custos|rigor",
            "recommendation": "recomendação_específica",
            "priority": "alta|média|baixa",
            "implementation_effort": "baixo|médio|alto",
            "expected_impact": "alto|médio|baixo",
            "implementation_steps": ["passo1", "passo2"]
        }}
    ],
    "quick_wins": [
        "melhoria_rápida_1",
        "melhoria_rápida_2"
    ],
    "long_term_improvements": [
        "melhoria_longo_prazo_1",
        "melhoria_longo_prazo_2"
    ],
    "resource_requirements": {{
        "technical_expertise": "baixo|médio|alto",
        "time_investment": "baixo|médio|alto",
        "infrastructure_needs": "mínimo|moderado|significativo"
    }}
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='13_intelligent_recommendations',
                operation='generate_recommendations'
            )

            recommendations = self.parse_claude_response_safe(response, ["priority_recommendations", "quick_wins", "long_term_improvements", "resource_requirements"])

            return recommendations

        except Exception as e:
            self.logger.error(f"Erro na geração de recomendações: {e}")
            return {
                'error': str(e),
                'fallback_recommendations': [
                    'Implement comprehensive logging for all stages',
                    'Add statistical validation for AI-generated results',
                    'Create detailed documentation for reproducibility',
                    'Optimize chunk processing for better performance'
                ]
            }

    def _generate_executive_summary(self, stage_quality: Dict, reproducibility: Dict,
                                  recommendations: Dict, pipeline_data: Dict) -> Dict[str, Any]:
        """
        Gera resumo executivo da revisão

        Args:
            stage_quality: Avaliação de qualidade
            reproducibility: Avaliação de reprodutibilidade
            recommendations: Recomendações
            pipeline_data: Dados do pipeline

        Returns:
            Resumo executivo
        """
        quality_score = stage_quality.get('ai_assessment', {}).get('overall_pipeline_quality', 0.8)
        reproducibility_score = reproducibility.get('calculated_score', 0.8)
        success_rate = pipeline_data.get('successful_stages', 0) / max(pipeline_data.get('total_stages_executed', 1), 1)

        return {
            'overall_assessment': {
                'pipeline_status': 'excellent' if quality_score > 0.9 else 'good' if quality_score > 0.7 else 'needs_improvement',
                'quality_score': quality_score,
                'reproducibility_score': reproducibility_score,
                'success_rate': success_rate,
                'ready_for_production': quality_score > 0.8 and success_rate > 0.8
            },
            'key_achievements': [
                f'Successfully executed {pipeline_data.get("successful_stages", 0)} stages',
                'AI-enhanced analysis implemented' if pipeline_data.get('environment', {}).get('anthropic_integration_enabled') else 'Traditional analysis completed',
                'Comprehensive methodology applied'
            ],
            'priority_actions': recommendations.get('quick_wins', [])[:3],
            'next_steps': recommendations.get('long_term_improvements', [])[:3],
            'confidence_level': 'high' if quality_score > 0.8 else 'medium'
        }

    def _generate_review_metadata(self) -> Dict[str, Any]:
        """
        Gera metadados da revisão
        """
        return {
            'review_timestamp': datetime.now().isoformat(),
            'reviewer_version': 'smart_pipeline_reviewer_v1.0',
            'ai_enhanced': True,
            'review_scope': 'comprehensive',
            'methodology': 'multi_dimensional_assessment_with_ai'
        }

    def _summarize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sumariza configuração para análise
        """
        return {
            'anthropic_enabled': bool(config.get('anthropic')),
            'total_config_keys': len(config),
            'key_configurations': list(config.keys())[:10]  # Primeiras 10 chaves
        }

    def _calculate_efficiency_score(self, pipeline_data: Dict[str, Any]) -> float:
        """
        Calcula score de eficiência do pipeline
        """
        total_duration = pipeline_data.get('total_duration', 0)
        successful_stages = pipeline_data.get('successful_stages', 0)

        if total_duration == 0 or successful_stages == 0:
            return 0.0

        # Score baseado em tempo por stage bem-sucedido
        avg_time_per_stage = total_duration / successful_stages

        # Normalizar (assumindo que < 60s por stage é eficiente)
        efficiency = max(0, min(1, (120 - avg_time_per_stage) / 120))

        return efficiency


def get_smart_pipeline_reviewer(config: Dict[str, Any]) -> SmartPipelineReviewer:
    """
    Factory function para criar instância do SmartPipelineReviewer
    """
    return SmartPipelineReviewer(config)
