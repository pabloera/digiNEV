#!/usr/bin/env python3
"""
Dashboard digiNEV v.final - Análise de Discurso Político Brasileiro
Sistema científico completo para análise de dados eleitorais 2022-2023
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import time

# Configuração da página
st.set_page_config(
    page_title="digiNEV v.final - Dashboard Científico",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar src ao path
sys.path.append('src')

try:
    from src.analyzer import Analyzer
except ImportError:
    st.error("❌ Módulo Analyzer não encontrado. Verifique a instalação.")
    st.stop()

# Header principal
st.title("🔬 digiNEV v.final - Dashboard Científico")
st.markdown("**Análise de Discurso Político Brasileiro | Canais Públicos Telegram 2022-2023**")
st.divider()

# Sidebar para controles
with st.sidebar:
    st.header("⚙️ Configurações")

    # Seleção do dataset
    dataset_option = st.selectbox(
        "📊 Dataset:",
        ["4_2022-2023-elec.csv (Telegram Político)", "sample_1000_cases.csv (Amostra Teste)"],
        index=0
    )

    # Tamanho da amostra
    sample_size = st.slider(
        "📏 Tamanho da amostra:",
        min_value=10,
        max_value=5000,
        value=1000,
        step=50,
        help="Número de registros para análise (recomendado: 500-2000)"
    )

    # Botão de análise
    run_analysis = st.button("🚀 Executar Análise", type="primary")

    st.divider()
    st.markdown("### 📋 Sistema")
    st.info("""
    **digiNEV v.final**
    - 14 stages científicos
    - spaCy 3.8.7
    - 86+ colunas geradas
    - Canais Telegram políticos
    """)

# Container principal
if run_analysis:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Carregar dados
    status_text.text("📊 Carregando dataset...")
    progress_bar.progress(10)

    try:
        if "4_2022-2023-elec.csv" in dataset_option:
            data_path = "data/4_2022-2023-elec.csv"
            df = pd.read_csv(data_path, low_memory=False)
            df = df.rename(columns={'body': 'text', 'datetime': 'date'})
        else:
            data_path = "data/sample_1000_cases_20250928_025745.csv"
            df = pd.read_csv(data_path, sep=';')

        df_sample = df.head(sample_size).copy()

        status_text.text(f"✅ Dataset carregado: {len(df_sample):,} registros")
        progress_bar.progress(20)

    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset: {e}")
        st.stop()

    # Executar análise
    status_text.text("🔬 Executando pipeline digiNEV...")
    progress_bar.progress(30)

    try:
        analyzer = Analyzer()
        start_time = time.time()

        results = analyzer.analyze_dataset(df_sample)
        analysis_time = time.time() - start_time

        if isinstance(results, dict) and 'data' in results:
            df_result = results['data']
            stats = results['stats']
        else:
            df_result = results
            stats = {'stages_completed': 14}

        status_text.text("✅ Análise concluída!")
        progress_bar.progress(100)

    except Exception as e:
        st.error(f"❌ Erro na análise: {e}")
        st.stop()

    # Remover progress bar
    progress_bar.empty()
    status_text.empty()

    # === DASHBOARD PRINCIPAL ===

    # Métricas principais
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "📊 Registros",
            f"{len(df_result):,}",
            f"+{len(df_result):,}"
        )

    with col2:
        st.metric(
            "🔧 Colunas",
            len(df_result.columns),
            f"+{len(df_result.columns) - len(df_sample.columns)}"
        )

    with col3:
        st.metric(
            "🎯 Stages",
            f"{stats.get('stages_completed', 14)}/14",
            "✅ Completo"
        )

    with col4:
        st.metric(
            "⚡ Performance",
            f"{len(df_result)/analysis_time:.1f}/s",
            f"{analysis_time:.1f}s total"
        )

    with col5:
        political_classified = (df_result['political_spectrum'] != 'unknown').sum()
        st.metric(
            "🏛️ Classificados",
            f"{political_classified:,}",
            f"{(political_classified/len(df_result)*100):.1f}%"
        )

    st.divider()

    # === ANÁLISE POLÍTICA ===
    st.header("🏛️ Análise Política")

    col1, col2 = st.columns(2)

    with col1:
        # Distribuição política
        pol_counts = df_result['political_spectrum'].value_counts()

        fig_pol = px.pie(
            values=pol_counts.values,
            names=pol_counts.index,
            title="Distribuição do Espectro Político",
            color_discrete_map={
                'extrema-direita': '#8B0000',
                'direita': '#DC143C',
                'centro-direita': '#FFA500',
                'centro': '#FFD700',
                'centro-esquerda': '#90EE90',
                'esquerda': '#228B22',
                'neutral': '#808080',
                'unknown': '#D3D3D3'
            }
        )
        fig_pol.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pol, use_container_width=True)

    with col2:
        # Análise temporal ou distribuição alternativa
        try:
            if 'date' in df_result.columns and 'political_spectrum' in df_result.columns:
                df_result['date'] = pd.to_datetime(df_result['date'], errors='coerce')
                df_result['month'] = df_result['date'].dt.to_period('M').astype(str)

                pol_time = df_result.groupby(['month', 'political_spectrum']).size().unstack(fill_value=0)

                # Verificar se há dados suficientes e colunas disponíveis
                if len(pol_time) > 1:
                    available_cols = [col for col in ['extrema-direita', 'direita', 'esquerda']
                                    if col in pol_time.columns and pol_time[col].sum() > 0]

                    if available_cols:
                        pol_time_reset = pol_time.reset_index()
                        fig_time = px.line(
                            pol_time_reset,
                            x='month',
                            y=available_cols,
                            title="Evolução da Polarização Política",
                            labels={'value': 'Número de Mensagens', 'month': 'Período'}
                        )
                        fig_time.update_xaxis(tickangle=45)
                        st.plotly_chart(fig_time, use_container_width=True)
                    else:
                        # Fallback: gráfico de barras da distribuição política
                        pol_counts = df_result['political_spectrum'].value_counts().head(5)
                        fig_bar = px.bar(
                            x=pol_counts.index,
                            y=pol_counts.values,
                            title="Top 5 Categorias Políticas",
                            labels={'x': 'Categoria', 'y': 'Quantidade'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    # Fallback: distribuição por canal se disponível
                    if 'channel' in df_result.columns:
                        channel_counts = df_result['channel'].value_counts().head(5)
                        fig_channel = px.bar(
                            x=channel_counts.index,
                            y=channel_counts.values,
                            title="Top 5 Canais",
                            labels={'x': 'Canal', 'y': 'Mensagens'}
                        )
                        fig_channel.update_xaxis(tickangle=45)
                        st.plotly_chart(fig_channel, use_container_width=True)
                    else:
                        st.info("📊 Dados temporais insuficientes para análise de evolução")
            else:
                st.info("📅 Colunas de data ou política não disponíveis")
        except Exception as e:
            st.warning(f"⚠️ Erro na análise temporal: {str(e)}")
            # Fallback simples
            if 'political_spectrum' in df_result.columns:
                pol_simple = df_result['political_spectrum'].value_counts().head(3)
                fig_simple = px.bar(
                    x=pol_simple.index,
                    y=pol_simple.values,
                    title="Distribuição Política (Simplificada)"
                )
                st.plotly_chart(fig_simple, use_container_width=True)

    # === ANÁLISE TÉCNICA ===
    st.header("🔬 Análise Técnica")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Clustering
        if 'cluster_id' in df_result.columns:
            cluster_counts = df_result['cluster_id'].value_counts().head(10)

            fig_cluster = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Distribuição de Clusters",
                labels={'x': 'Cluster ID', 'y': 'Número de Mensagens'}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    with col2:
        # Coordenação
        if 'coordination_score' in df_result.columns:
            coord_bins = pd.cut(df_result['coordination_score'], bins=10)
            coord_counts = coord_bins.value_counts().sort_index()

            fig_coord = px.histogram(
                df_result,
                x='coordination_score',
                title="Distribuição de Coordenação",
                nbins=20,
                labels={'coordination_score': 'Score de Coordenação'}
            )
            st.plotly_chart(fig_coord, use_container_width=True)

    with col3:
        # TF-IDF Scores
        if 'tfidf_max_score' in df_result.columns:
            fig_tfidf = px.histogram(
                df_result,
                x='tfidf_max_score',
                title="Distribuição TF-IDF",
                nbins=20,
                labels={'tfidf_max_score': 'TF-IDF Max Score'}
            )
            st.plotly_chart(fig_tfidf, use_container_width=True)

    # === ANÁLISE DE CONTEÚDO ===
    st.header("📝 Análise de Conteúdo")

    col1, col2 = st.columns(2)

    with col1:
        # Estatísticas de texto
        if 'word_count' in df_result.columns:
            fig_words = px.histogram(
                df_result,
                x='word_count',
                title="Distribuição de Palavras por Mensagem",
                nbins=30,
                labels={'word_count': 'Número de Palavras'}
            )
            st.plotly_chart(fig_words, use_container_width=True)

    with col2:
        # Entidades spaCy
        if 'spacy_entities_count' in df_result.columns:
            fig_entities = px.histogram(
                df_result,
                x='spacy_entities_count',
                title="Distribuição de Entidades NLP",
                nbins=20,
                labels={'spacy_entities_count': 'Número de Entidades'}
            )
            st.plotly_chart(fig_entities, use_container_width=True)

    # === DADOS DETALHADOS ===
    st.header("📋 Dados Detalhados")

    # Filtros para a tabela
    col1, col2, col3 = st.columns(3)

    with col1:
        pol_filter = st.multiselect(
            "Filtrar por Espectro Político:",
            options=df_result['political_spectrum'].unique(),
            default=[]
        )

    with col2:
        if 'cluster_id' in df_result.columns:
            cluster_filter = st.multiselect(
                "Filtrar por Cluster:",
                options=sorted(df_result['cluster_id'].unique()),
                default=[]
            )

    with col3:
        coord_threshold = st.slider(
            "Score de Coordenação mínimo:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )

    # Aplicar filtros
    df_filtered = df_result.copy()

    if pol_filter:
        df_filtered = df_filtered[df_filtered['political_spectrum'].isin(pol_filter)]

    if 'cluster_id' in df_result.columns and cluster_filter:
        df_filtered = df_filtered[df_filtered['cluster_id'].isin(cluster_filter)]

    if 'coordination_score' in df_result.columns:
        df_filtered = df_filtered[df_filtered['coordination_score'] >= coord_threshold]

    # Mostrar tabela
    st.write(f"**Registros filtrados:** {len(df_filtered):,} de {len(df_result):,}")

    # Selecionar colunas importantes para display
    display_cols = ['text', 'political_spectrum', 'word_count']
    if 'cluster_id' in df_result.columns:
        display_cols.append('cluster_id')
    if 'coordination_score' in df_result.columns:
        display_cols.append('coordination_score')
    if 'tfidf_max_score' in df_result.columns:
        display_cols.append('tfidf_max_score')

    available_cols = [col for col in display_cols if col in df_filtered.columns]

    st.dataframe(
        df_filtered[available_cols].head(100),
        use_container_width=True,
        height=400
    )

    # === RESUMO EXECUTIVO ===
    st.header("📊 Resumo Executivo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Principais Achados")

        # Calcular estatísticas principais
        extrema_direita_pct = (df_result['political_spectrum'] == 'extrema-direita').mean() * 100
        high_coord_pct = (df_result['coordination_score'] > 0.7).mean() * 100 if 'coordination_score' in df_result.columns else 0

        st.write(f"• **Conteúdo extremista:** {extrema_direita_pct:.1f}% das mensagens")
        st.write(f"• **Alta coordenação:** {high_coord_pct:.1f}% dos registros")
        st.write(f"• **Clusters identificados:** {len(df_result['cluster_id'].unique()) if 'cluster_id' in df_result.columns else 'N/A'}")
        st.write(f"• **Período analisado:** {df_result['date'].min().strftime('%Y-%m') if 'date' in df_result.columns else 'N/A'} - {df_result['date'].max().strftime('%Y-%m') if 'date' in df_result.columns else 'N/A'}")

    with col2:
        st.subheader("⚙️ Detalhes Técnicos")
        st.write(f"• **Sistema:** digiNEV v.final")
        st.write(f"• **Stages executados:** {stats.get('stages_completed', 14)}/14")
        st.write(f"• **Tempo de processamento:** {analysis_time:.1f}s")
        st.write(f"• **Performance:** {len(df_result)/analysis_time:.1f} registros/segundo")
        st.write(f"• **Features extraídas:** {stats.get('features_extracted', 'N/A')}")

else:
    # Estado inicial
    st.info("👆 **Configure os parâmetros na barra lateral e clique em 'Executar Análise' para começar.**")

    # Informações do sistema
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔬 Sistema digiNEV v.final")
        st.write("""
        - Pipeline de 14 stages científicos
        - Análise política automatizada
        - Processamento linguístico spaCy
        - Clustering e topic modeling
        - Detecção de coordenação
        """)

    with col2:
        st.subheader("📊 Datasets Disponíveis")
        st.write("""
        - **4_2022-2023-elec.csv**: 145k+ mensagens de canais políticos
        - **sample_1000_cases.csv**: Amostra para teste
        - Período: 2022-2023
        - Fonte: Canais públicos Telegram BR
        """)

    with col3:
        st.subheader("📈 Recursos de Análise")
        st.write("""
        - Classificação política brasileira
        - Análise temporal e evolução
        - Visualizações interativas
        - Exportação de resultados
        - Métricas de performance
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    📋 <strong>digiNEV v.final</strong> | Sistema Científico de Análise de Discurso Político Brasileiro<br>
    🔬 Desenvolvido para pesquisa acadêmica em Ciências Sociais | 2025
    </small>
</div>
""", unsafe_allow_html=True)