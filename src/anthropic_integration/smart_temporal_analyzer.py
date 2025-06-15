"""
Smart Temporal Analyzer com API Anthropic

Módulo avançado para análise temporal com interpretação contextual.
Identifica eventos, tendências e padrões temporais significativos.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import AnthropicBase

logger = logging.getLogger(__name__)

class SmartTemporalAnalyzer(AnthropicBase):
    """
    Analisador temporal inteligente usando API Anthropic

    Funcionalidades:
    - Detecção automática de eventos significativos
    - Análise de tendências e padrões temporais
    - Interpretação contextual de picos e mudanças
    - Correlação com eventos históricos
    - Identificação de campanhas coordenadas
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurações específicas
        temporal_config = config.get('temporal_analysis', {})
        self.window_size = temporal_config.get('analysis_window_days', 7)
        self.significance_threshold = temporal_config.get('significance_threshold', 2.0)  # desvios padrão
        self.event_detection_sensitivity = temporal_config.get('event_sensitivity', 0.8)

        # Períodos de interesse definidos
        self.key_periods = {
            'pre_pandemic': ('2019-01-01', '2020-02-29'),
            'pandemic_start': ('2020-03-01', '2020-12-31'),
            'pandemic_peak': ('2021-01-01', '2021-12-31'),
            'pre_election': ('2022-01-01', '2022-09-30'),
            'election_period': ('2022-10-01', '2022-10-30'),
            'post_election': ('2022-11-01', '2023-12-31')
        }

    def analyze_temporal_patterns(self, df: pd.DataFrame, timestamp_column: str = 'timestamp',
                                text_column: str = 'body') -> Dict[str, Any]:
        """
        Análise temporal completa com interpretação AI

        Args:
            df: DataFrame com dados
            timestamp_column: Coluna de timestamp
            text_column: Coluna de texto para análise de conteúdo

        Returns:
            Análise temporal completa
        """
        self.logger.info("Iniciando análise temporal inteligente")

        # Try to find a suitable timestamp column
        if timestamp_column not in df.columns:
            # Look for alternative column names
            possible_columns = ['date', 'time', 'created_at', 'timestamp', 'datetime']
            found_column = None
            
            for col in possible_columns:
                if col in df.columns:
                    found_column = col
                    break
            
            if found_column:
                timestamp_column = found_column
                self.logger.info(f"Using '{timestamp_column}' as timestamp column")
            else:
                # Return structured fallback result instead of error
                return {
                    'patterns': ['no_temporal_data'],
                    'temporal_trends': {
                        'error': f'No suitable timestamp column found. Available columns: {list(df.columns)}',
                        'available_columns': list(df.columns)
                    },
                    'events': [],
                    'analysis_completed': False,
                    'method': 'fallback'
                }

        # Preparar dados temporais
        temporal_data = self._prepare_temporal_data(df, timestamp_column)

        # Análise estatística de séries temporais
        statistical_analysis = self._analyze_time_series_statistics(temporal_data)

        # Detecção de eventos significativos
        event_detection = self._detect_significant_events(temporal_data, statistical_analysis)

        # Análise contextual de eventos
        event_interpretation = self._interpret_events_with_ai(
            event_detection, df, timestamp_column, text_column
        )

        # Análise de períodos específicos
        period_analysis = self._analyze_key_periods(temporal_data, df, text_column, timestamp_column)

        # Detecção de padrões coordenados
        coordination_analysis = self._detect_coordination_patterns(temporal_data, df, timestamp_column)

        # Insights contextuais
        contextual_insights = self._generate_temporal_insights(
            statistical_analysis, event_interpretation, period_analysis, coordination_analysis
        )

        # Create comprehensive result with TDD compatibility
        result = {
            'temporal_data': temporal_data,
            'statistical_analysis': statistical_analysis,
            'event_detection': event_detection,
            'event_interpretation': event_interpretation,
            'period_analysis': period_analysis,
            'coordination_analysis': coordination_analysis,
            'contextual_insights': contextual_insights,
            'analysis_summary': self._generate_temporal_summary(
                statistical_analysis, event_detection, event_interpretation
            )
        }
        
        # Add TDD-compatible keys for test infrastructure
        result['patterns'] = []
        if statistical_analysis:
            patterns = []
            if 'trends' in statistical_analysis:
                patterns.extend(['daily_trends', 'hourly_patterns', 'weekly_cycles'])
            if event_detection and event_detection.get('events'):
                patterns.extend(['significant_events'])
            if coordination_analysis and coordination_analysis.get('coordination_detected'):
                patterns.extend(['coordination_patterns'])
            result['patterns'] = patterns
            
        result['temporal_trends'] = {
            'daily_patterns': statistical_analysis.get('trends', {}),
            'event_count': len(event_detection.get('events', [])) if event_detection else 0,
            'coordination_detected': coordination_analysis.get('coordination_detected', False) if coordination_analysis else False,
            'analysis_quality': result['analysis_summary'].get('analysis_quality', 'unknown')
        }
        
        return result

    def _prepare_temporal_data(self, df: pd.DataFrame, timestamp_column: str) -> Dict[str, Any]:
        """
        Prepara dados temporais para análise

        Args:
            df: DataFrame original
            timestamp_column: Coluna de timestamp

        Returns:
            Dados temporais estruturados
        """
        self.logger.info("Preparando dados temporais")

        # Converter timestamps
        df_temp = df.copy()
        df_temp[timestamp_column] = pd.to_datetime(df_temp[timestamp_column], errors='coerce')

        # Remover dados sem timestamp válido
        df_temp = df_temp.dropna(subset=[timestamp_column])

        if len(df_temp) == 0:
            return {
                'patterns': ['no_valid_timestamps'],
                'temporal_trends': {
                    'error': 'Nenhum timestamp válido encontrado após conversão',
                    'original_count': len(df),
                    'valid_count': 0
                },
                'events': [],
                'analysis_completed': False,
                'method': 'fallback'
            }

        # Verify datetime conversion worked
        if not pd.api.types.is_datetime64_any_dtype(df_temp[timestamp_column]):
            return {
                'patterns': ['datetime_conversion_failed'],
                'temporal_trends': {
                    'error': f'Failed to convert {timestamp_column} to datetime. Current dtype: {df_temp[timestamp_column].dtype}',
                    'sample_values': df_temp[timestamp_column].head().tolist()
                },
                'events': [],
                'analysis_completed': False,
                'method': 'fallback'
            }

        try:
            # Criar série temporal diária
            df_temp['date'] = df_temp[timestamp_column].dt.date
            daily_counts = df_temp.groupby('date').size().reset_index(name='message_count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            daily_counts = daily_counts.sort_values('date')

            # Criar série temporal horária
            df_temp['hour'] = df_temp[timestamp_column].dt.hour
            hourly_pattern = df_temp.groupby('hour').size().reset_index(name='message_count')

            # Análise semanal
            df_temp['weekday'] = df_temp[timestamp_column].dt.day_name()
            weekly_pattern = df_temp.groupby('weekday').size().reset_index(name='message_count')
        except Exception as e:
            return {
                'patterns': ['temporal_processing_error'],
                'temporal_trends': {
                    'error': f'Error processing temporal data: {e}',
                    'datetime_dtype': str(df_temp[timestamp_column].dtype),
                    'sample_values': df_temp[timestamp_column].head().tolist()
                },
                'events': [],
                'analysis_completed': False,
                'method': 'fallback'
            }

        return {
            'total_messages': len(df_temp),
            'date_range': (df_temp[timestamp_column].min(), df_temp[timestamp_column].max()),
            'daily_series': daily_counts.to_dict('records'),
            'hourly_pattern': hourly_pattern.to_dict('records'),
            'weekly_pattern': weekly_pattern.to_dict('records'),
            'raw_data_prepared': len(df_temp)
        }

    def _analyze_time_series_statistics(self, temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análise estatística da série temporal

        Args:
            temporal_data: Dados temporais preparados

        Returns:
            Estatísticas da série temporal
        """
        if 'daily_series' not in temporal_data:
            return {'error': 'Dados temporais inválidos'}

        daily_series = temporal_data['daily_series']
        if not daily_series:
            return {'error': 'Série temporal vazia'}

        # Converter para array numpy
        counts = np.array([d['message_count'] for d in daily_series])

        # Estatísticas descritivas
        stats = {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'median': float(np.median(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
            'total_days': len(counts)
        }

        # Detectar outliers (usando z-score)
        z_scores = np.abs((counts - stats['mean']) / stats['std']) if stats['std'] > 0 else np.zeros_like(counts)
        outlier_threshold = self.significance_threshold
        outlier_indices = np.where(z_scores > outlier_threshold)[0]

        outliers = []
        for idx in outlier_indices:
            outliers.append({
                'date': daily_series[idx]['date'],
                'count': daily_series[idx]['message_count'],
                'z_score': float(z_scores[idx]),
                'deviation_type': 'high' if counts[idx] > stats['mean'] else 'low'
            })

        # Tendência geral (regressão linear simples)
        if len(counts) > 1:
            x = np.arange(len(counts))
            slope, intercept = np.polyfit(x, counts, 1)
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        else:
            slope, intercept, trend_direction = 0, counts[0], 'stable'

        return {
            'basic_stats': stats,
            'outliers': outliers,
            'trend_analysis': {
                'slope': float(slope),
                'intercept': float(intercept),
                'direction': trend_direction
            },
            'volatility': {
                'coefficient_variation': float(stats['std'] / stats['mean']) if stats['mean'] > 0 else 0,
                'outlier_rate': len(outliers) / len(counts) if len(counts) > 0 else 0
            }
        }

    def _detect_significant_events(self, temporal_data: Dict[str, Any],
                                 statistical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta eventos significativos na série temporal

        Args:
            temporal_data: Dados temporais
            statistical_analysis: Análise estatística

        Returns:
            Eventos detectados
        """
        if 'outliers' not in statistical_analysis:
            return {'events': []}

        outliers = statistical_analysis['outliers']

        # Agrupar outliers próximos temporalmente
        events = []
        current_event = None

        for outlier in sorted(outliers, key=lambda x: x['date']):
            outlier_date = pd.to_datetime(outlier['date'])

            if current_event is None:
                current_event = {
                    'start_date': outlier_date,
                    'end_date': outlier_date,
                    'peak_date': outlier_date,
                    'peak_count': outlier['count'],
                    'max_z_score': outlier['z_score'],
                    'outlier_days': [outlier]
                }
            else:
                # Se outlier está próximo (dentro da janela), adicionar ao evento atual
                days_diff = (outlier_date - current_event['end_date']).days

                if days_diff <= self.window_size:
                    current_event['end_date'] = outlier_date
                    current_event['outlier_days'].append(outlier)

                    # Atualizar pico se necessário
                    if outlier['count'] > current_event['peak_count']:
                        current_event['peak_date'] = outlier_date
                        current_event['peak_count'] = outlier['count']
                        current_event['max_z_score'] = outlier['z_score']
                else:
                    # Finalizar evento atual e começar novo
                    events.append(current_event)
                    current_event = {
                        'start_date': outlier_date,
                        'end_date': outlier_date,
                        'peak_date': outlier_date,
                        'peak_count': outlier['count'],
                        'max_z_score': outlier['z_score'],
                        'outlier_days': [outlier]
                    }

        # Adicionar último evento
        if current_event:
            events.append(current_event)

        # Calcular métricas dos eventos
        for event in events:
            event['duration_days'] = (event['end_date'] - event['start_date']).days + 1
            event['total_outlier_days'] = len(event['outlier_days'])
            event['intensity_score'] = event['max_z_score']

            # Converter datas para string para serialização
            event['start_date'] = event['start_date'].strftime('%Y-%m-%d')
            event['end_date'] = event['end_date'].strftime('%Y-%m-%d')
            event['peak_date'] = event['peak_date'].strftime('%Y-%m-%d')

        return {
            'events_detected': len(events),
            'events': sorted(events, key=lambda x: x['intensity_score'], reverse=True),
            'detection_parameters': {
                'significance_threshold': self.significance_threshold,
                'window_size_days': self.window_size
            }
        }

    def _interpret_events_with_ai(self, event_detection: Dict[str, Any], df: pd.DataFrame,
                                timestamp_column: str, text_column: str) -> Dict[str, Any]:
        """
        Interpreta eventos usando AI

        Args:
            event_detection: Eventos detectados
            df: DataFrame original
            timestamp_column: Coluna de timestamp
            text_column: Coluna de texto

        Returns:
            Interpretação dos eventos
        """
        events = event_detection.get('events', [])
        if not events:
            return {'interpretations': []}

        self.logger.info("Interpretando eventos com AI")

        interpretations = []

        # Interpretar apenas os top 5 eventos mais significativos
        for event in events[:5]:
            interpretation = self._interpret_single_event(event, df, timestamp_column, text_column)
            if interpretation:
                interpretations.append(interpretation)

        return {
            'interpretations': interpretations,
            'total_events_interpreted': len(interpretations)
        }

    def _interpret_single_event(self, event: Dict[str, Any], df: pd.DataFrame,
                               timestamp_column: str, text_column: str) -> Optional[Dict[str, Any]]:
        """
        Interpreta um único evento

        Args:
            event: Dados do evento
            df: DataFrame original
            timestamp_column: Coluna de timestamp
            text_column: Coluna de texto

        Returns:
            Interpretação do evento
        """
        # Extrair mensagens do período do evento
        start_date = pd.to_datetime(event['start_date'])
        end_date = pd.to_datetime(event['end_date'])

        df_temp = df.copy()
        df_temp[timestamp_column] = pd.to_datetime(df_temp[timestamp_column], errors='coerce')

        event_mask = (df_temp[timestamp_column] >= start_date) & (df_temp[timestamp_column] <= end_date)
        event_messages = df_temp[event_mask][text_column].dropna().astype(str)

        if len(event_messages) == 0:
            return None

        # Amostra de mensagens para análise
        sample_messages = event_messages.sample(min(50, len(event_messages)), random_state=42).tolist()

        prompt = f"""
Analise este evento significativo detectado automaticamente nas mensagens do Telegram:

DADOS DO EVENTO:
- Período: {event['start_date']} a {event['end_date']}
- Pico: {event['peak_date']} ({event['peak_count']} mensagens)
- Intensidade: {event['intensity_score']:.2f} desvios padrão
- Duração: {event['duration_days']} dias

AMOSTRA DE MENSAGENS DO PERÍODO:
{chr(10).join([f"- {msg[:200]}" for msg in sample_messages[:20]])}

CONTEXTO HISTÓRICO: Brasil 2019-2023 (governo Bolsonaro, pandemia COVID-19, eleições 2022)

Determine:
1. Que evento real/histórico causou este pico?
2. Qual o tema/assunto predominante?
3. Qual o tom emocional das mensagens?
4. Há indicações de coordenação/campanha?
5. Relevância política/social do evento

Responda em JSON:
{{
    "event_interpretation": {{
        "likely_cause": "evento_histórico_identificado",
        "primary_theme": "tema_principal",
        "emotional_tone": "positivo|negativo|neutro|polarizado",
        "coordination_indicators": true|false,
        "political_relevance": "alta|média|baixa",
        "social_impact": "descrição_do_impacto",
        "key_topics": ["tópico1", "tópico2", "tópico3"]
    }},
    "confidence_level": 0.85,
    "analysis_notes": "observações_adicionais"
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='10_event_interpretation',
                operation='interpret_event'
            )

            result = self.parse_claude_response_safe(response, ["event_interpretation", "confidence_level", "analysis_notes"])

            # Combinar dados do evento com interpretação
            return {
                'event_data': event,
                'interpretation': result.get('event_interpretation', {}),
                'confidence': result.get('confidence_level', 0.5),
                'analysis_notes': result.get('analysis_notes', ''),
                'sample_messages_count': len(sample_messages)
            }

        except Exception as e:
            self.logger.error(f"Erro na interpretação do evento: {e}")
            return {
                'event_data': event,
                'interpretation': {'error': str(e)},
                'confidence': 0.0
            }

    def _analyze_key_periods(self, temporal_data: Dict[str, Any], df: pd.DataFrame,
                           text_column: str, timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Analisa períodos-chave predefinidos

        Args:
            temporal_data: Dados temporais
            df: DataFrame original
            text_column: Coluna de texto

        Returns:
            Análise dos períodos-chave
        """
        period_analyses = {}

        for period_name, (start_date, end_date) in self.key_periods.items():
            # Filtrar dados do período
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            period_mask = (pd.to_datetime(df[timestamp_column], errors='coerce') >= start_dt) & \
                          (pd.to_datetime(df[timestamp_column], errors='coerce') <= end_dt)

            period_df = df[period_mask]

            if len(period_df) > 0:
                period_analyses[period_name] = {
                    'message_count': len(period_df),
                    'start_date': start_date,
                    'end_date': end_date,
                    'avg_daily_messages': len(period_df) / ((end_dt - start_dt).days + 1),
                    'sample_analysis': f'Period {period_name} has {len(period_df)} messages'
                }

        return period_analyses

    def _detect_coordination_patterns(self, temporal_data: Dict[str, Any],
                                    df: pd.DataFrame, timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Detecta padrões que indicam coordenação

        Args:
            temporal_data: Dados temporais
            df: DataFrame original

        Returns:
            Análise de coordenação
        """
        # Análise simplificada de coordenação
        hourly_pattern = temporal_data.get('hourly_pattern', [])

        if not hourly_pattern:
            return {'coordination_detected': False}

        # Verificar padrões não-naturais (muito concentrados em horários específicos)
        hourly_counts = [h['message_count'] for h in hourly_pattern]

        if len(hourly_counts) > 0:
            max_hourly = max(hourly_counts)
            total_messages = sum(hourly_counts)
            concentration_ratio = max_hourly / total_messages if total_messages > 0 else 0

            # Se mais de 30% das mensagens estão em uma única hora, pode indicar coordenação
            coordination_suspected = concentration_ratio > 0.3
        else:
            coordination_suspected = False

        return {
            'coordination_detected': coordination_suspected,
            'concentration_ratio': concentration_ratio if 'concentration_ratio' in locals() else 0,
            'analysis_method': 'hourly_concentration_analysis'
        }

    def _generate_temporal_insights(self, statistical_analysis: Dict, event_interpretation: Dict,
                                  period_analysis: Dict, coordination_analysis: Dict) -> Dict[str, Any]:
        """
        Gera insights temporais contextuais

        Args:
            statistical_analysis: Análise estatística
            event_interpretation: Interpretação de eventos
            period_analysis: Análise de períodos
            coordination_analysis: Análise de coordenação

        Returns:
            Insights contextuais
        """
        # Preparar dados para insights
        events_count = len(event_interpretation.get('interpretations', []))
        total_outliers = len(statistical_analysis.get('outliers', []))
        trend_direction = statistical_analysis.get('trend_analysis', {}).get('direction', 'stable')

        prompt = f"""
Gere insights sobre padrões temporais das mensagens do Telegram brasileiro (2019-2023):

DADOS ANALISADOS:
- Eventos significativos detectados: {events_count}
- Total de dias com atividade anômala: {total_outliers}
- Tendência geral: {trend_direction}
- Coordenação detectada: {coordination_analysis.get('coordination_detected', False)}

CONTEXTO: Período inclui governo Bolsonaro, pandemia COVID-19, eleições 2022.

Analise:
1. Padrões de mobilização e engajamento
2. Relação entre eventos históricos e atividade
3. Indicadores de campanhas organizadas
4. Evolução do discurso ao longo do tempo
5. Implicações para democracia digital

Responda em JSON:
{{
    "temporal_insights": [
        {{
            "insight": "insight_principal",
            "evidence": "evidência_dos_dados",
            "implications": "implicações_para_análise"
        }}
    ],
    "mobilization_patterns": {{
        "spontaneous_events": 3,
        "coordinated_campaigns": 2,
        "reactive_responses": 4
    }},
    "democratic_implications": [
        "implicação_democrática_1",
        "implicação_democrática_2"
    ],
    "research_recommendations": [
        "recomendação_pesquisa_1",
        "recomendação_pesquisa_2"
    ]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='10_temporal_insights',
                operation='generate_insights'
            )

            insights = self.parse_claude_response_safe(response, ["temporal_insights", "mobilization_patterns", "democratic_implications", "research_recommendations"])
            return insights

        except Exception as e:
            self.logger.error(f"Erro na geração de insights temporais: {e}")
            return {
                'error': str(e),
                'fallback_insights': ['Análise temporal concluída com métodos estatísticos']
            }

    def _generate_temporal_summary(self, statistical_analysis: Dict, event_detection: Dict,
                                 event_interpretation: Dict) -> Dict[str, Any]:
        """
        Gera resumo da análise temporal
        """
        return {
            'total_days_analyzed': statistical_analysis.get('basic_stats', {}).get('total_days', 0),
            'significant_events_detected': event_detection.get('events_detected', 0),
            'events_interpreted': len(event_interpretation.get('interpretations', [])),
            'trend_direction': statistical_analysis.get('trend_analysis', {}).get('direction', 'unknown'),
            'volatility_level': 'high' if statistical_analysis.get('volatility', {}).get('coefficient_variation', 0) > 1 else 'low',
            'analysis_quality': 'ai_enhanced' if event_interpretation.get('interpretations') else 'statistical_only',
            'methodology': 'smart_temporal_analysis_with_ai'
        }

def get_smart_temporal_analyzer(config: Dict[str, Any]) -> SmartTemporalAnalyzer:
    """
    Factory function para criar instância do SmartTemporalAnalyzer
    """
    return SmartTemporalAnalyzer(config)
