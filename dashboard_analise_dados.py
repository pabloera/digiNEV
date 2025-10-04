#!/usr/bin/env python3
"""
Dashboard de Análise de Dados - digiNEV v.final
===============================================

Dashboard focado EXCLUSIVAMENTE nos resultados da análise de dados,
sem métricas de sucesso do pipeline. Apresenta insights e visualizações
dos dados políticos processados.

🎯 FOCO: Resultados da análise de dados
📊 OBJETIVO: Visualizar descobertas sobre discurso político brasileiro
🚫 NÃO INCLUI: Métricas de execução do pipeline
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime
from plotly.subplots import make_subplots
import re

# Configuração da página
st.set_page_config(
    page_title="Análise de Dados - Discurso Político Brasil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .analysis-card {
        background: linear-gradient(135deg, #1f4e79 0%, #2c5f7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #ff6b35;
        text-align: center;
        margin: 0.5rem 0;
    }

    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #1f4e79;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .warning-text {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .success-text {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_analysis_data():
    """Carregar dados da análise mais recente."""

    # Procurar por arquivos de saída da análise
    data_files = []

    # Verificar arquivos chunked
    chunked_files = list(Path(".").glob("chunked_*_output_chunk_*.csv"))
    if chunked_files:
        data_files.extend(chunked_files)

    # Verificar outros arquivos de análise
    analysis_files = list(Path(".").glob("*_analysis_*.csv"))
    data_files.extend(analysis_files)

    # Arquivos de validação
    validation_files = list(Path(".").glob("*validation*.csv"))
    data_files.extend(validation_files)

    if not data_files:
        return None, "Nenhum arquivo de análise encontrado"

    # Carregar o arquivo mais recente
    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)

    try:
        # Tentar diferentes separadores
        for sep in [';', ',']:
            try:
                df = pd.read_csv(latest_file, sep=sep, encoding='utf-8')
                if len(df.columns) > 10:  # Validar que tem colunas suficientes
                    return df, f"Carregado: {latest_file.name} ({len(df):,} registros)"
            except:
                continue

        return None, f"Erro ao carregar {latest_file.name}"

    except Exception as e:
        return None, f"Erro: {str(e)}"

def analyze_political_distribution(df):
    """Analisar distribuição política dos dados."""

    if 'political_spectrum' not in df.columns:
        return None

    # Distribuição política
    political_dist = df['political_spectrum'].value_counts()

    # Criar gráfico de pizza
    fig = px.pie(
        values=political_dist.values,
        names=political_dist.index,
        title="Distribuição do Espectro Político",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        font=dict(size=12),
        showlegend=True,
        height=500
    )

    return fig, political_dist

def analyze_temporal_patterns(df):
    """Analisar padrões temporais dos dados."""

    temporal_cols = [col for col in df.columns if any(term in col.lower() for term in ['hour', 'day', 'month', 'time'])]

    if not temporal_cols:
        return None, None

    figures = []

    # Análise por hora do dia
    if 'hour' in df.columns:
        hourly_dist = df['hour'].value_counts().sort_index()

        fig_hour = px.bar(
            x=hourly_dist.index,
            y=hourly_dist.values,
            title="Distribuição de Mensagens por Hora do Dia",
            labels={'x': 'Hora', 'y': 'Número de Mensagens'},
            color=hourly_dist.values,
            color_continuous_scale='Blues'
        )
        fig_hour.update_layout(height=400)
        figures.append(("Distribuição Horária", fig_hour))

    # Análise por dia da semana
    if 'day_of_week' in df.columns:
        days_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sáb', 6: 'Dom'}
        df_temp = df.copy()
        df_temp['day_name'] = df_temp['day_of_week'].map(days_map)

        daily_dist = df_temp['day_name'].value_counts()
        # Reordenar para semana
        day_order = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        daily_dist = daily_dist.reindex([d for d in day_order if d in daily_dist.index])

        fig_day = px.bar(
            x=daily_dist.index,
            y=daily_dist.values,
            title="Distribuição por Dia da Semana",
            labels={'x': 'Dia da Semana', 'y': 'Número de Mensagens'},
            color=daily_dist.values,
            color_continuous_scale='Greens'
        )
        fig_day.update_layout(height=400)
        figures.append(("Distribuição Semanal", fig_day))

    return figures

def analyze_coordination_networks(df):
    """Analisar coordenação e redes."""

    coordination_cols = [col for col in df.columns if 'coordination' in col.lower() or 'network' in col.lower()]

    if not coordination_cols:
        return None

    insights = []

    # Analisar coordenação potencial
    if 'potential_coordination' in df.columns:
        coordinated = df['potential_coordination'].sum()
        total = len(df)
        coord_pct = (coordinated / total) * 100

        insights.append({
            'title': 'Coordenação Detectada',
            'value': f"{coordinated:,} mensagens ({coord_pct:.1f}%)",
            'description': f"De {total:,} mensagens analisadas, {coordinated:,} apresentam indícios de coordenação"
        })

    # Analisar clusters
    if 'cluster' in df.columns:
        cluster_dist = df['cluster'].value_counts()
        num_clusters = len(cluster_dist)

        insights.append({
            'title': 'Clusters Identificados',
            'value': f"{num_clusters} grupos",
            'description': f"Mensagens organizadas em {num_clusters} clusters, maior grupo: {cluster_dist.iloc[0]:,} mensagens"
        })

    return insights

def analyze_text_characteristics(df):
    """Analisar características do texto."""

    text_cols = [col for col in df.columns if any(term in col.lower() for term in ['length', 'words', 'chars', 'tokens'])]

    if not text_cols:
        return None

    # Estatísticas descritivas
    stats = {}

    if 'word_count' in df.columns:
        stats['Palavras por mensagem'] = {
            'média': df['word_count'].mean(),
            'mediana': df['word_count'].median(),
            'máximo': df['word_count'].max()
        }

    if 'text_length' in df.columns:
        stats['Caracteres por mensagem'] = {
            'média': df['text_length'].mean(),
            'mediana': df['text_length'].median(),
            'máximo': df['text_length'].max()
        }

    # Criar gráfico de distribuição
    if 'word_count' in df.columns:
        fig = px.histogram(
            df,
            x='word_count',
            title='Distribuição do Número de Palavras por Mensagem',
            labels={'word_count': 'Número de Palavras', 'count': 'Frequência'},
            nbins=50
        )
        fig.update_layout(height=400)

        return stats, fig

    return stats, None

def analyze_topic_modeling(df):
    """Analisar resultados de topic modeling."""

    topic_cols = [col for col in df.columns if 'topic' in col.lower()]

    if not topic_cols:
        return None

    # Buscar colunas de tópicos
    if 'dominant_topic' in df.columns:
        topic_dist = df['dominant_topic'].value_counts()

        fig = px.bar(
            x=[f"Tópico {i}" for i in topic_dist.index],
            y=topic_dist.values,
            title="Distribuição de Tópicos Dominantes",
            labels={'x': 'Tópico', 'y': 'Número de Mensagens'},
            color=topic_dist.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)

        return fig, topic_dist

    return None

def create_data_overview(df):
    """Criar visão geral dos dados."""

    # Estatísticas básicas
    total_records = len(df)
    total_columns = len(df.columns)

    # Identificar colunas por tipo de análise
    political_cols = [col for col in df.columns if 'political' in col.lower()]
    temporal_cols = [col for col in df.columns if any(term in col.lower() for term in ['hour', 'day', 'month', 'time'])]
    text_cols = [col for col in df.columns if any(term in col.lower() for term in ['length', 'words', 'chars'])]
    network_cols = [col for col in df.columns if 'coordination' in col.lower() or 'cluster' in col.lower()]

    overview = {
        'total_records': total_records,
        'total_columns': total_columns,
        'analysis_types': {
            'Classificação Política': len(political_cols),
            'Análise Temporal': len(temporal_cols),
            'Características Textuais': len(text_cols),
            'Análise de Redes': len(network_cols)
        }
    }

    return overview

def main():
    """Função principal do dashboard."""

    # Título principal
    st.markdown('<h1 class="main-header">📊 Análise de Dados - Discurso Político Brasileiro</h1>', unsafe_allow_html=True)

    # Subtítulo
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <h3>Resultados da Análise de Canais Telegram (2019-2023)</h3>
        <p><strong>Foco:</strong> Visualização de dados processados | <strong>Escopo:</strong> Descobertas sobre discurso político</p>
    </div>
    """, unsafe_allow_html=True)

    # Carregar dados
    with st.spinner("🔄 Carregando dados da análise..."):
        df, status_msg = load_analysis_data()

    # Mostrar status do carregamento
    if df is not None:
        st.success(f"✅ {status_msg}")
    else:
        st.error(f"❌ {status_msg}")
        st.warning("Execute primeiro uma análise para gerar dados para visualização.")
        return

    # Visão geral dos dados
    overview = create_data_overview(df)

    # Layout em colunas para métricas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="analysis-card">
            <h4>📊 Total de Registros</h4>
            <div class="metric-highlight">{overview['total_records']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="analysis-card">
            <h4>🔬 Colunas Geradas</h4>
            <div class="metric-highlight">{overview['total_columns']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        political_cols = overview['analysis_types']['Classificação Política']
        st.markdown(f"""
        <div class="analysis-card">
            <h4>🏛️ Análise Política</h4>
            <div class="metric-highlight">{political_cols} features</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        temporal_cols = overview['analysis_types']['Análise Temporal']
        st.markdown(f"""
        <div class="analysis-card">
            <h4>⏰ Análise Temporal</h4>
            <div class="metric-highlight">{temporal_cols} features</div>
        </div>
        """, unsafe_allow_html=True)

    # Separador
    st.markdown("---")

    # Análises específicas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏛️ Política", "⏰ Temporal", "🕸️ Redes", "📝 Texto", "🎯 Tópicos"
    ])

    # Tab 1: Análise Política
    with tab1:
        st.subheader("🏛️ Análise do Espectro Político")

        political_result = analyze_political_distribution(df)
        if political_result:
            fig_political, political_dist = political_result

            col1, col2 = st.columns([3, 2])

            with col1:
                st.plotly_chart(fig_political, use_container_width=True)

            with col2:
                st.markdown("### 📈 Estatísticas Políticas")
                for category, count in political_dist.head().items():
                    percentage = (count / len(df)) * 100
                    st.markdown(f"**{category}:** {count:,} ({percentage:.1f}%)")

                # Insights
                st.markdown("""
                <div class="insight-box">
                    <h4>💡 Insights Políticos</h4>
                    <ul>
                        <li>Distribuição do espectro político nos dados analisados</li>
                        <li>Identificação de tendências partidárias</li>
                        <li>Análise de polarização política</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Dados de classificação política não encontrados nos resultados.")

    # Tab 2: Análise Temporal
    with tab2:
        st.subheader("⏰ Padrões Temporais")

        temporal_figures = analyze_temporal_patterns(df)
        if temporal_figures:
            for title, fig in temporal_figures:
                st.markdown(f"### {title}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Dados temporais não encontrados nos resultados.")

    # Tab 3: Análise de Redes
    with tab3:
        st.subheader("🕸️ Coordenação e Redes")

        network_insights = analyze_coordination_networks(df)
        if network_insights:
            for insight in network_insights:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>📊 {insight['title']}</h4>
                    <div class="metric-highlight">{insight['value']}</div>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Dados de coordenação não encontrados nos resultados.")

    # Tab 4: Análise de Texto
    with tab4:
        st.subheader("📝 Características do Texto")

        text_result = analyze_text_characteristics(df)
        if text_result:
            stats, fig = text_result

            # Mostrar estatísticas
            if stats:
                for metric_name, metric_stats in stats.items():
                    st.markdown(f"### {metric_name}")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Média", f"{metric_stats['média']:.1f}")
                    with col2:
                        st.metric("Mediana", f"{metric_stats['mediana']:.1f}")
                    with col3:
                        st.metric("Máximo", f"{metric_stats['máximo']:,}")

            # Mostrar gráfico
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Dados de características textuais não encontrados.")

    # Tab 5: Análise de Tópicos
    with tab5:
        st.subheader("🎯 Modelagem de Tópicos")

        topic_result = analyze_topic_modeling(df)
        if topic_result:
            fig_topic, topic_dist = topic_result

            col1, col2 = st.columns([3, 2])

            with col1:
                st.plotly_chart(fig_topic, use_container_width=True)

            with col2:
                st.markdown("### 📊 Distribuição de Tópicos")
                for topic_id, count in topic_dist.head().items():
                    percentage = (count / len(df)) * 100
                    st.markdown(f"**Tópico {topic_id}:** {count:,} ({percentage:.1f}%)")
        else:
            st.warning("⚠️ Dados de tópicos não encontrados nos resultados.")

    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p><strong>digiNEV v.final</strong> | Análise de Discurso Político Brasileiro |
        Dados processados: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()