"""
Stage 12 Semantic Analysis Dashboard
Análise Semântica com Análise de Sentimento e Intensidade Emocional

Foco: Visualização de padrões semânticos no discurso político brasileiro
- Gauge charts para distribuição de sentimentos
- Timeline de evolução temporal do sentimento
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Analisador semântico com foco em sentimento e emoções."""

    def __init__(self):
        # Configurações de cores para consistência visual
        self.sentiment_colors = {
            'positive': '#2E8B57',    # Verde escuro
            'negative': '#DC143C',    # Vermelho escuro
            'neutral': '#708090'      # Cinza escuro
        }

        # Configurações para gauge charts
        self.gauge_config = {
            'height': 350,
            'showlegend': False,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }

    def validate_semantic_columns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Valida se as colunas semânticas necessárias estão presentes."""
        required_columns = [
            'sentiment_polarity',
            'sentiment_label',
            'emotion_intensity',
            'has_aggressive_language',
            'semantic_diversity'
        ]

        validation = {}
        for col in required_columns:
            validation[col] = col in df.columns

        return validation

    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calcula distribuição de sentimentos."""
        if 'sentiment_label' not in df.columns:
            return {'positive': 0, 'negative': 0, 'neutral': 0}

        sentiment_counts = df['sentiment_label'].value_counts().to_dict()

        # Garantir que todas as categorias existam
        distribution = {
            'positive': sentiment_counts.get('positive', 0),
            'negative': sentiment_counts.get('negative', 0),
            'neutral': sentiment_counts.get('neutral', 0)
        }

        return distribution

    def calculate_sentiment_percentages(self, distribution: Dict[str, int]) -> Dict[str, float]:
        """Calcula percentuais de sentimento."""
        total = sum(distribution.values())
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        percentages = {
            sentiment: (count / total) * 100
            for sentiment, count in distribution.items()
        }

        return percentages

    def create_sentiment_gauge_chart(self, percentages: Dict[str, float], sentiment_type: str) -> go.Figure:
        """Cria gauge chart para um tipo específico de sentimento."""
        percentage = percentages.get(sentiment_type, 0.0)
        color = self.sentiment_colors.get(sentiment_type, '#708090')

        # Configurar título
        titles = {
            'positive': 'Sentimento Positivo',
            'negative': 'Sentimento Negativo',
            'neutral': 'Sentimento Neutro'
        }
        title = titles.get(sentiment_type, sentiment_type.title())

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16}},
            delta = {'reference': 33.33, 'position': "top"},  # Referência para distribuição equilibrada
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': 'lightgray'},
                    {'range': [20, 50], 'color': 'gray'},
                    {'range': [50, 100], 'color': 'lightgreen' if sentiment_type == 'positive' else 'lightcoral' if sentiment_type == 'negative' else 'lightblue'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig.update_layout(
            height=self.gauge_config['height'],
            showlegend=self.gauge_config['showlegend'],
            margin=self.gauge_config['margin'],
            font={'color': "darkblue", 'family': "Arial"}
        )

        return fig

    def prepare_temporal_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepara dados para análise temporal."""
        # Verificar se há coluna de data/tempo
        date_columns = ['datetime', 'date', 'timestamp', 'created_at']
        date_col = None

        for col in date_columns:
            if col in df.columns:
                date_col = col
                break

        if not date_col:
            return None

        if 'sentiment_label' not in df.columns:
            return None

        try:
            # Criar cópia para manipulação
            temporal_df = df[[date_col, 'sentiment_label']].copy()

            # Converter para datetime
            temporal_df[date_col] = pd.to_datetime(temporal_df[date_col], errors='coerce')

            # Remover valores nulos
            temporal_df = temporal_df.dropna(subset=[date_col])

            if temporal_df.empty:
                return None

            # Agrupar por data
            temporal_df['date_only'] = temporal_df[date_col].dt.date

            # Calcular distribuição de sentimentos por data
            sentiment_by_date = temporal_df.groupby(['date_only', 'sentiment_label']).size().unstack(fill_value=0)

            # Calcular percentuais
            sentiment_percentages = sentiment_by_date.div(sentiment_by_date.sum(axis=1), axis=0) * 100

            # Resetar índice para facilitar plotagem
            sentiment_percentages = sentiment_percentages.reset_index()

            return sentiment_percentages

        except Exception as e:
            logger.warning(f"Erro ao preparar dados temporais: {e}")
            return None

    def create_sentiment_timeline(self, temporal_data: pd.DataFrame) -> go.Figure:
        """Cria timeline de evolução do sentimento."""
        fig = go.Figure()

        # Adicionar linha para cada sentimento
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in temporal_data.columns:
                fig.add_trace(go.Scatter(
                    x=temporal_data['date_only'],
                    y=temporal_data[sentiment],
                    mode='lines+markers',
                    name=sentiment.title(),
                    line=dict(
                        color=self.sentiment_colors[sentiment],
                        width=2
                    ),
                    marker=dict(
                        size=4,
                        color=self.sentiment_colors[sentiment]
                    ),
                    hovertemplate=f'<b>{sentiment.title()}</b><br>' +
                                  'Data: %{x}<br>' +
                                  'Percentual: %{y:.1f}%<br>' +
                                  '<extra></extra>'
                ))

        fig.update_layout(
            title={
                'text': 'Evolução Temporal do Sentimento',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Data',
            yaxis_title='Percentual (%)',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=450,
            margin={'l': 50, 'r': 50, 't': 70, 'b': 50}
        )

        # Configurar eixos
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[0, 100]
        )

        return fig

    def calculate_semantic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas semânticas avançadas."""
        metrics = {}

        # Polarity média
        if 'sentiment_polarity' in df.columns:
            metrics['avg_polarity'] = df['sentiment_polarity'].mean()
            metrics['polarity_std'] = df['sentiment_polarity'].std()

        # Intensidade emocional média
        if 'emotion_intensity' in df.columns:
            metrics['avg_emotion_intensity'] = df['emotion_intensity'].mean()
            metrics['high_emotion_pct'] = (df['emotion_intensity'] > 0.5).mean() * 100

        # Linguagem agressiva
        if 'has_aggressive_language' in df.columns:
            metrics['aggressive_language_pct'] = df['has_aggressive_language'].mean() * 100

        # Diversidade semântica
        if 'semantic_diversity' in df.columns:
            metrics['avg_semantic_diversity'] = df['semantic_diversity'].mean()
            metrics['high_diversity_pct'] = (df['semantic_diversity'] > 0.7).mean() * 100

        return metrics

def create_emotion_intensity_histogram(df: pd.DataFrame) -> go.Figure:
    """Cria histograma de intensidade emocional."""
    if 'emotion_intensity' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Dados de intensidade emocional não disponíveis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['emotion_intensity'],
        nbinsx=20,
        name='Intensidade Emocional',
        marker_color='#FF6B6B',
        opacity=0.7,
        hovertemplate='Intensidade: %{x:.2f}<br>Frequência: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Distribuição da Intensidade Emocional',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Intensidade Emocional (0-1)',
        yaxis_title='Frequência',
        showlegend=False,
        height=350
    )

    return fig

def create_semantic_diversity_scatter(df: pd.DataFrame) -> go.Figure:
    """Cria scatter plot de diversidade semântica vs polaridade."""
    if 'semantic_diversity' not in df.columns or 'sentiment_polarity' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Dados de diversidade semântica ou polaridade não disponíveis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    # Mapear cores por sentimento se disponível
    colors = df['sentiment_label'].map({
        'positive': '#2E8B57',
        'negative': '#DC143C',
        'neutral': '#708090'
    }) if 'sentiment_label' in df.columns else '#4169E1'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['semantic_diversity'],
        y=df['sentiment_polarity'],
        mode='markers',
        marker=dict(
            color=colors,
            size=6,
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        hovertemplate='Diversidade: %{x:.2f}<br>Polaridade: %{y:.2f}<extra></extra>',
        name='Mensagens'
    ))

    fig.update_layout(
        title={
            'text': 'Diversidade Semântica vs Polaridade de Sentimento',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Diversidade Semântica',
        yaxis_title='Polaridade de Sentimento',
        showlegend=False,
        height=400
    )

    return fig

def main_dashboard(df: pd.DataFrame):
    """Dashboard principal de análise semântica."""
    st.title("🧠 Stage 12 - Análise Semântica")
    st.markdown("**Análise de sentimento e padrões semânticos no discurso político brasileiro**")

    # Inicializar analisador
    analyzer = SemanticAnalyzer()

    # Validar colunas necessárias
    validation = analyzer.validate_semantic_columns(df)
    missing_columns = [col for col, exists in validation.items() if not exists]

    if missing_columns:
        st.error(f"⚠️ Colunas semânticas ausentes: {missing_columns}")
        st.info("Execute o pipeline completo para gerar análise semântica (Stage 12)")
        return

    # Sidebar com configurações
    st.sidebar.header("Configurações")

    # Filtros de análise
    min_emotion_intensity = st.sidebar.slider(
        "Intensidade emocional mínima:",
        min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )

    include_aggressive = st.sidebar.checkbox(
        "Incluir apenas linguagem agressiva",
        value=False
    )

    # Aplicar filtros
    filtered_df = df.copy()

    if 'emotion_intensity' in df.columns:
        filtered_df = filtered_df[filtered_df['emotion_intensity'] >= min_emotion_intensity]

    if include_aggressive and 'has_aggressive_language' in df.columns:
        filtered_df = filtered_df[filtered_df['has_aggressive_language'] == True]

    if filtered_df.empty:
        st.warning("⚠️ Nenhum dado corresponde aos filtros selecionados")
        return

    # Métricas gerais
    st.subheader("📊 Visão Geral Semântica")

    col1, col2, col3, col4 = st.columns(4)

    # Calcular métricas
    metrics = analyzer.calculate_semantic_metrics(filtered_df)

    with col1:
        st.metric(
            "Polaridade Média",
            f"{metrics.get('avg_polarity', 0):.3f}",
            f"σ: {metrics.get('polarity_std', 0):.3f}"
        )

    with col2:
        st.metric(
            "Intensidade Emocional",
            f"{metrics.get('avg_emotion_intensity', 0):.3f}",
            f"{metrics.get('high_emotion_pct', 0):.1f}% alta"
        )

    with col3:
        st.metric(
            "Linguagem Agressiva",
            f"{metrics.get('aggressive_language_pct', 0):.1f}%"
        )

    with col4:
        st.metric(
            "Diversidade Semântica",
            f"{metrics.get('avg_semantic_diversity', 0):.3f}",
            f"{metrics.get('high_diversity_pct', 0):.1f}% alta"
        )

    # Seção principal: Gauge Charts de Sentimento
    st.subheader("🎯 Distribuição de Sentimentos")

    # Calcular distribuição
    sentiment_distribution = analyzer.get_sentiment_distribution(filtered_df)
    sentiment_percentages = analyzer.calculate_sentiment_percentages(sentiment_distribution)

    # Mostrar estatísticas
    total_messages = sum(sentiment_distribution.values())
    st.info(f"📊 Total de mensagens analisadas: {total_messages:,}")

    # Criar gauge charts
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_positive = analyzer.create_sentiment_gauge_chart(sentiment_percentages, 'positive')
        st.plotly_chart(fig_positive, use_container_width=True)

    with col2:
        fig_negative = analyzer.create_sentiment_gauge_chart(sentiment_percentages, 'negative')
        st.plotly_chart(fig_negative, use_container_width=True)

    with col3:
        fig_neutral = analyzer.create_sentiment_gauge_chart(sentiment_percentages, 'neutral')
        st.plotly_chart(fig_neutral, use_container_width=True)

    # Seção temporal: Timeline de Sentimento
    st.subheader("📈 Evolução Temporal do Sentimento")

    temporal_data = analyzer.prepare_temporal_data(filtered_df)

    if temporal_data is not None and not temporal_data.empty:
        fig_timeline = analyzer.create_sentiment_timeline(temporal_data)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Estatísticas temporais
        date_range = temporal_data['date_only'].max() - temporal_data['date_only'].min()
        st.info(f"📅 Período analisado: {date_range.days} dias ({temporal_data['date_only'].min()} até {temporal_data['date_only'].max()})")

    else:
        st.warning("⚠️ Dados temporais não disponíveis ou insuficientes para análise temporal")

    # Seção de análises complementares
    st.subheader("🔍 Análises Complementares")

    tab1, tab2 = st.tabs(["Intensidade Emocional", "Diversidade Semântica"])

    with tab1:
        fig_emotion = create_emotion_intensity_histogram(filtered_df)
        st.plotly_chart(fig_emotion, use_container_width=True)

        if 'emotion_intensity' in filtered_df.columns:
            # Estatísticas da intensidade emocional
            st.markdown("**Estatísticas de Intensidade Emocional:**")
            emotion_stats = filtered_df['emotion_intensity'].describe()

            cols = st.columns(4)
            with cols[0]:
                st.metric("Mínima", f"{emotion_stats['min']:.3f}")
            with cols[1]:
                st.metric("Média", f"{emotion_stats['mean']:.3f}")
            with cols[2]:
                st.metric("Máxima", f"{emotion_stats['max']:.3f}")
            with cols[3]:
                st.metric("Desvio Padrão", f"{emotion_stats['std']:.3f}")

    with tab2:
        fig_diversity = create_semantic_diversity_scatter(filtered_df)
        st.plotly_chart(fig_diversity, use_container_width=True)

        if 'semantic_diversity' in filtered_df.columns:
            # Análise de correlação
            if 'sentiment_polarity' in filtered_df.columns:
                correlation = filtered_df['semantic_diversity'].corr(filtered_df['sentiment_polarity'])
                st.metric("Correlação Diversidade-Polaridade", f"{correlation:.3f}")

    # Tabela de detalhes semânticos
    with st.expander("📋 Detalhes dos Dados Semânticos"):
        semantic_columns = [col for col in filtered_df.columns if col in validation.keys()]
        if semantic_columns:
            display_df = filtered_df[semantic_columns].head(100)
            st.dataframe(display_df, use_container_width=True)

        st.markdown("**Legenda das Colunas:**")
        st.markdown("""
        - **sentiment_polarity**: Polaridade de sentimento (-1 a +1)
        - **sentiment_label**: Classificação categórica (positive/negative/neutral)
        - **emotion_intensity**: Intensidade emocional (0 a 1)
        - **has_aggressive_language**: Presença de linguagem agressiva (True/False)
        - **semantic_diversity**: Diversidade vocabular (0 a 1)
        """)

if __name__ == "__main__":
    # Teste básico
    st.set_page_config(
        page_title="Stage 12 - Análise Semântica",
        page_icon="🧠",
        layout="wide"
    )

    # Carregar dados de teste
    try:
        df = pd.read_csv("data/controlled_test_100.csv")
        main_dashboard(df)
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Execute o dashboard através do sistema principal.")