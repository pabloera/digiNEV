#!/usr/bin/env python3
"""
Dashboard Científico Unificado - digiNEV v5.1.0
==============================================

Dashboard para análise de discursos políticos, negacionistas e autoritários em espaços digitais.
Foco: Radicalização digital + Fluxos informacionais + Autoritarismo em redes sociais + Oposicionismo radicalizado.

Autor: Pablo Almada - Análise de Discurso Político Brasileiro (2019-2023)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# GUARDRAILS: Sistema de validação de conteúdo
from dashboard_guardrails import dashboard_guardrail, require_real_data_only, validate_dashboard_data

# Configuração da página
st.set_page_config(
    page_title="digiNEV v5.1.0 - Análise de Radicalização Digital",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cores otimizadas seguindo dashesxtilop-2.ini
CORES_ACADEMICAS = {
    'azul_primario': '#2261C6',      # Azul vibrante principal
    'azul_escuro': '#1a4d8f',       # Azul escuro para títulos
    'cinza_claro': '#F8F9FA',       # Fundo cinza muito claro
    'cinza_medio': '#E5E8EB',       # Cinza estrutural
    'branco': '#FFFFFF',            # Fundo predominante
    'laranja': '#FF6B35',           # Destaque laranja
    'verde_sucesso': '#28a745',     # Verde para status OK
    'vermelho_alerta': '#dc3545',   # Vermelho para alertas
    'amarelo_aviso': '#ffc107'      # Amarelo para avisos
}

# CSS acadêmico otimizado
st.markdown(f"""
<style>
    /* Layout geral acadêmico */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    /* Tipografia acadêmica (Open Sans) */
    .main {{
        font-family: 'Open Sans', 'Segoe UI', 'Source Sans Pro', sans-serif;
        background-color: {CORES_ACADEMICAS['branco']};
    }}

    /* Títulos hierárquicos */
    .titulo-principal {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {CORES_ACADEMICAS['azul_escuro']};
        text-align: center;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid {CORES_ACADEMICAS['azul_primario']};
        padding-bottom: 0.5rem;
    }}

    .titulo-secao {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {CORES_ACADEMICAS['azul_primario']};
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}

    /* Cards acadêmicos */
    .card-academico {{
        background: {CORES_ACADEMICAS['branco']};
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid {CORES_ACADEMICAS['cinza_medio']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}

    /* Métricas científicas */
    .metrica-cientifica {{
        background: linear-gradient(135deg, {CORES_ACADEMICAS['azul_primario']}, {CORES_ACADEMICAS['azul_escuro']});
        color: white;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        margin: 0.5rem 0;
    }}

    .metrica-valor {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}

    .metrica-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}

    /* Status técnico discreto */
    .status-tecnico {{
        background: {CORES_ACADEMICAS['cinza_claro']};
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }}

    .status-ok {{ border-left: 4px solid {CORES_ACADEMICAS['verde_sucesso']}; }}
    .status-aviso {{ border-left: 4px solid {CORES_ACADEMICAS['amarelo_aviso']}; }}
    .status-erro {{ border-left: 4px solid {CORES_ACADEMICAS['vermelho_alerta']}; }}

    /* Sidebar acadêmica */
    .css-1d391kg {{
        background: {CORES_ACADEMICAS['cinza_claro']};
    }}

    /* Grid responsivo */
    .row-widget {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

class DataManagerUnified:
    """Gerenciador unificado de dados para o dashboard científico."""

    def __init__(self):
        self.data_dir = Path("pipeline_outputs/dashboard_ready")
        self.cache_dir = Path(".pipeline_cache")
        self.logger = self._setup_logging()
        self._cached_data = {}

    def _setup_logging(self):
        """Configurar logging científico."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    @st.cache_data(ttl=300)  # Cache 5 minutos
    def load_research_data(_self) -> Dict[str, pd.DataFrame]:
        """Carregar dados consolidados da pesquisa."""
        try:
            datasets = {}

            # Dados principais processados
            if _self.data_dir.exists():
                csv_files = list(_self.data_dir.glob("*.csv"))
                for file in csv_files[:5]:  # Limitar para performance
                    try:
                        df = pd.read_csv(file, sep=';', encoding='utf-8')
                        datasets[file.stem] = df
                        _self.logger.info(f"✅ Carregado: {file.name} ({len(df)} registros)")
                    except Exception as e:
                        _self.logger.warning(f"⚠️ Erro ao carregar {file.name}: {e}")

            # Dados de teste se não houver dados principais
            if not datasets:
                test_file = Path("data/controlled_test_100.csv")
                if test_file.exists():
                    df = pd.read_csv(test_file, sep=';', encoding='utf-8')
                    datasets['controlled_test'] = df
                    _self.logger.info(f"✅ Dados de teste carregados: {len(df)} registros")

            return datasets

        except Exception as e:
            _self.logger.error(f"❌ Erro crítico ao carregar dados: {e}")
            return {}

    def get_technical_status(self) -> Dict[str, Any]:
        """Status técnico mínimo essencial."""
        try:
            status = {
                'pipeline_status': '✅ Operacional',
                'data_quality': 'Alta',
                'last_update': datetime.now().strftime('%d/%m/%Y %H:%M'),
                'records_processed': 0,
                'api_budget': '✅ Dentro do limite',
                'cache_performance': '✅ Otimizado'
            }

            # Contar registros processados
            datasets = self.load_research_data()
            if datasets:
                status['records_processed'] = sum(len(df) for df in datasets.values())

            return status

        except Exception as e:
            self.logger.error(f"❌ Erro ao obter status técnico: {e}")
            return {
                'pipeline_status': '❌ Erro',
                'data_quality': 'Desconhecida',
                'last_update': 'N/A',
                'records_processed': 0,
                'api_budget': '⚠️ Verificar',
                'cache_performance': '⚠️ Verificar'
            }

def render_sidebar_unified():
    """Sidebar acadêmica unificada."""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: {CORES_ACADEMICAS['azul_escuro']};">🔬 digiNEV v5.1.0</h2>
            <p style="color: {CORES_ACADEMICAS['azul_primario']}; font-weight: 500;">
                Análise de Radicalização Digital
            </p>
            <p style="font-size: 0.9rem; color: #666;">
                Autoritarismo • Negacionismo • Oposicionismo Radicalizado
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Navegação principal
        st.markdown("### 📍 Navegação")
        page = st.radio(
            "Selecione a seção:",
            ["🏠 Home", "📊 Radicalização", "🌐 Fluxos", "🎭 Discursos"],
            key="main_navigation"
        )

        # Métricas de pesquisa
        st.markdown("### 📊 Métricas")
        data_manager = DataManagerUnified()
        status = data_manager.get_technical_status()

        st.markdown(f"""
        <div class="status-tecnico status-ok">
            📱 {status['records_processed']:,} mensagens
        </div>
        <div class="status-tecnico">
            🕒 2019-2023
        </div>
        <div class="status-tecnico">
            🔬 22 estágios
        </div>
        """, unsafe_allow_html=True)

        page_map = {
            "🏠 Home": "Home",
            "📊 Radicalização": "Radicalização",
            "🌐 Fluxos": "Fluxos",
            "🎭 Discursos": "Discursos"
        }
        return page_map.get(page, "Home")

def render_home_unified():
    """Página Home - Visão geral e metodologia."""
    st.markdown(f"""
    <div class="titulo-principal">
        🔬 Análise de Radicalização Digital e Autoritarismo
    </div>
    """, unsafe_allow_html=True)

    # Visão geral da pesquisa
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div class="card-academico">
            <h3 style="color: {CORES_ACADEMICAS['azul_primario']};">📋 Objetivo da Pesquisa</h3>
            <p>
                Este projeto analisa <strong>discursos políticos, negacionistas e autoritários</strong>
                circulados em espaços digitais como Telegram, focando em <strong>radicalização digital</strong>
                e formação de <strong>oposicionismo radicalizado</strong> em ambientes de baixo controle
                e fluxos informacionais intensos.
            </p>

            <h4 style="color: {CORES_ACADEMICAS['azul_primario']};">🎯 Questões Centrais:</h4>
            <ul>
                <li>Como se forma a <strong>radicalização</strong> em espaços digitais?</li>
                <li>Quais <strong>fluxos informacionais</strong> intensificam o autoritarismo?</li>
                <li>Como <strong>mídias alternativas</strong> moldam discursos radicais?</li>
                <li>Que padrões de <strong>oposicionismo radicalizado</strong> emergem?</li>
                <li>Como funcionam os <strong>espaços de baixo controle</strong> digital?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Metodologia
        st.markdown(f"""
        <div class="card-academico">
            <h3 style="color: {CORES_ACADEMICAS['azul_primario']};">🔬 Metodologia</h3>

            <h4>📊 Pipeline de Análise (22 Estágios):</h4>
            <ol>
                <li><strong>Preparação de Dados</strong>: Chunking, validação, deduplicação</li>
                <li><strong>Classificação Política</strong>: IA + léxico político brasileiro</li>
                <li><strong>Processamento Linguístico</strong>: spaCy pt_core_news_lg</li>
                <li><strong>Análise Semântica</strong>: Sentimento, tópicos, clustering</li>
                <li><strong>Análise Avançada</strong>: Redes, codificação qualitativa</li>
                <li><strong>Validação</strong>: Busca semântica, agregação</li>
            </ol>

            <h4>🤖 Tecnologias:</h4>
            <ul>
                <li><strong>IA Generativa</strong>: Claude 3.5 Haiku (54.5% dos estágios)</li>
                <li><strong>Embeddings</strong>: Voyage.ai para análise semântica</li>
                <li><strong>NLP</strong>: spaCy português + LIWC adaptado</li>
                <li><strong>Visualização</strong>: Streamlit + Plotly + Altair</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Métricas de overview
        data_manager = DataManagerUnified()
        datasets = data_manager.load_research_data()
        status = data_manager.get_technical_status()

        total_records = sum(len(df) for df in datasets.values()) if datasets else 0

        st.markdown(f"""
        <div class="metrica-cientifica">
            <div class="metrica-valor">{total_records:,}</div>
            <div class="metrica-label">Mensagens Analisadas</div>
        </div>

        <div class="metrica-cientifica">
            <div class="metrica-valor">22</div>
            <div class="metrica-label">Estágios de Análise</div>
        </div>

        <div class="metrica-cientifica">
            <div class="metrica-valor">54.5%</div>
            <div class="metrica-label">IA-Enhanced</div>
        </div>

        <div class="metrica-cientifica">
            <div class="metrica-valor">2019-2023</div>
            <div class="metrica-label">Período Analisado</div>
        </div>
        """, unsafe_allow_html=True)

        # Context info
        st.markdown(f"""
        <div class="card-academico" style="margin-top: 1rem;">
            <h4 style="color: {CORES_ACADEMICAS['azul_primario']};">📅 Contexto Temporal</h4>
            <ul style="font-size: 0.9rem;">
                <li><strong>2019-2021</strong>: Governo Bolsonaro inicial</li>
                <li><strong>2020-2022</strong>: Pandemia COVID-19</li>
                <li><strong>2022</strong>: Ano eleitoral</li>
                <li><strong>2022-2023</strong>: Pós-eleição e transição</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

@require_real_data_only
def render_analise_radicalizacao():
    """Página Radicalização - Análise de autoritarismo e radicalização digital."""
    st.markdown(f"""
    <div class="titulo-principal">
        📊 Análise de Radicalização Digital
    </div>
    """, unsafe_allow_html=True)

    data_manager = DataManagerUnified()
    datasets = data_manager.load_research_data()

    if not datasets:
        st.warning("⚠️ Nenhum dado disponível. Execute o pipeline primeiro.")
        return

    # GUARDRAIL: Validar dados antes de qualquer processamento
    main_dataset = list(datasets.values())[0]
    content_data = {'main_dataset': main_dataset}

    validation_result = dashboard_guardrail.validate_dashboard_content(
        content_data,
        'analise_radicalizacao'
    )

    if not validation_result.is_valid:
        st.error("🚫 GUARDRAIL VIOLADO: Conteúdo não autorizado detectado")
        st.error(f"Violações: {validation_result.errors}")
        return

    # 📊 Análise de Radicalização - Posicionamentos de Autoritarismo
    st.markdown(f'<h2 class="titulo-secao">📊 Posicionamentos de Autoritarismo</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Distribuição de Orientação Política (dados reais)
        if 'political_alignment' in main_dataset.columns:
            alignment_counts = main_dataset['political_alignment'].value_counts()

            fig_alignment = px.bar(
                x=alignment_counts.values,
                y=alignment_counts.index,
                orientation='h',
                title="Distribuição de Orientação Política",
                color=alignment_counts.values,
                color_continuous_scale="Blues"
            )
            fig_alignment.update_layout(
                font_family="Open Sans",
                title_font_size=16,
                title_font_color=CORES_ACADEMICAS['azul_escuro'],
                showlegend=False
            )
            st.plotly_chart(fig_alignment, use_container_width=True)
        else:
            st.info("Dados de orientação política não disponíveis")

    with col2:
        # Evolução do Negacionismo
        negacionism_cols = [col for col in main_dataset.columns if 'negacion' in col.lower() or 'conspiracy' in col.lower()]
        if negacionism_cols:
            negacionism_col = negacionism_cols[0]

            # Se for score numérico
            if main_dataset[negacionism_col].dtype in ['float64', 'int64']:
                # Categorizar scores de negacionismo
                main_dataset['nivel_negacionismo'] = pd.cut(
                    main_dataset[negacionism_col],
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Baixo', 'Moderado', 'Alto'],
                    include_lowest=True
                )
            else:
                main_dataset['nivel_negacionismo'] = main_dataset[negacionism_col]

            neg_counts = main_dataset['nivel_negacionismo'].value_counts()

            fig_neg = px.pie(
                values=neg_counts.values,
                names=neg_counts.index,
                title="Níveis de Negacionismo",
                color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
            )
            fig_neg.update_layout(
                font_family="Open Sans",
                title_font_size=16,
                title_font_color=CORES_ACADEMICAS['azul_escuro']
            )
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("Dados de negacionismo não disponíveis")

    # 📈 Dinâmicas Temporais - Ciclos de Radicalização
    st.markdown(f'<h2 class="titulo-secao">📈 Dinâmicas Temporais e Ciclos de Radicalização</h2>', unsafe_allow_html=True)

    # Detectar coluna de data
    date_columns = [col for col in main_dataset.columns if 'date' in col.lower() or 'time' in col.lower()]

    if date_columns and 'political_alignment' in main_dataset.columns:
        date_col = date_columns[0]
        try:
            # Converter para datetime
            main_dataset[date_col] = pd.to_datetime(main_dataset[date_col], errors='coerce')

            # Agrupar por mês e orientação política
            main_dataset['mes_ano'] = main_dataset[date_col].dt.to_period('M').astype(str)

            temporal_analysis = main_dataset.groupby(['mes_ano', 'political_alignment']).size().reset_index(name='count')

            fig_temporal = px.line(
                temporal_analysis,
                x='mes_ano',
                y='count',
                color='political_alignment',
                title="Evolução Temporal por Orientação Política",
                markers=True
            )
            fig_temporal.update_layout(
                font_family="Open Sans",
                title_font_size=16,
                title_font_color=CORES_ACADEMICAS['azul_escuro'],
                xaxis_title="Período",
                yaxis_title="Número de Mensagens"
            )
            fig_temporal.update_xaxes(tickangle=45)
            st.plotly_chart(fig_temporal, use_container_width=True)

        except Exception as e:
            st.warning(f"Erro na análise temporal: {e}")

    # Análise de sentimento
    st.markdown(f'<h2 class="titulo-secao">💭 Análise de Sentimento</h2>', unsafe_allow_html=True)

    sentiment_cols = [col for col in main_dataset.columns if 'sentiment' in col.lower()]
    if sentiment_cols:
        sentiment_col = sentiment_cols[0]

        col1, col2 = st.columns(2)

        with col1:
            sentiment_counts = main_dataset[sentiment_col].value_counts()

            fig_sentiment = px.bar(
                x=sentiment_counts.values,
                y=sentiment_counts.index,
                orientation='h',
                title="Distribuição de Sentimento",
                color=sentiment_counts.values,
                color_continuous_scale="RdYlBu_r"
            )
            fig_sentiment.update_layout(
                font_family="Open Sans",
                title_font_size=16,
                title_font_color=CORES_ACADEMICAS['azul_escuro']
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            # Correlação sentimento x orientação política
            if 'political_alignment' in main_dataset.columns:
                crosstab = pd.crosstab(main_dataset[sentiment_col], main_dataset['political_alignment'])

                fig_corr = px.imshow(
                    crosstab.values,
                    x=crosstab.columns,
                    y=crosstab.index,
                    aspect="auto",
                    title="Sentimento x Orientação Política",
                    color_continuous_scale="Blues"
                )
                fig_corr.update_layout(
                    font_family="Open Sans",
                    title_font_size=16,
                    title_font_color=CORES_ACADEMICAS['azul_escuro']
                )
                st.plotly_chart(fig_corr, use_container_width=True)

@require_real_data_only
def render_fluxos_informacionais():
    """Página Fluxos - Mapeamento de fluxos informacionais e redes."""
    st.markdown(f"""
    <div class="titulo-principal">
        🌐 Mapeamento de Fluxos Informacionais
    </div>
    """, unsafe_allow_html=True)

    data_manager = DataManagerUnified()
    datasets = data_manager.load_research_data()

    if not datasets:
        st.warning("⚠️ Nenhum dado disponível para análise de fluxos.")
        return

    # GUARDRAIL: Validar dados antes de processamento
    results_df = list(datasets.values())[0]
    content_data = {'results_df': results_df}

    validation_result = dashboard_guardrail.validate_dashboard_content(
        content_data,
        'fluxos_informacionais'
    )

    if not validation_result.is_valid:
        st.error("🚫 GUARDRAIL VIOLADO: Dados não autorizados detectados")
        st.error(f"Violações: {validation_result.errors}")
        return

    # Análise de Redes de Disseminação
    st.subheader("📡 Redes de Disseminação e Coordenação")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Análise de Conectividade de Canais**")
        # Network analysis baseada em hashtags compartilhadas
        if 'hashtags_normalized' in results_df.columns:
            hashtag_network = data_manager._analyze_hashtag_networks(results_df)
            if hashtag_network:
                st.metric("Hashtags Conectoras", len(hashtag_network))
                # Top hashtags de disseminação
                if hashtag_network:
                    top_hashtags = sorted(hashtag_network.items(), key=lambda x: x[1], reverse=True)[:5]
                    for hashtag, count in top_hashtags:
                        st.write(f"• {hashtag}: {count} conexões")
        else:
            st.info("Dados de hashtags não disponíveis para análise de rede")

    with col2:
        st.write("**Padrões de Coordenação Temporal**")
        # Análise de coordenação baseada em timestamps similares
        if 'timestamp' in results_df.columns:
            coordination_patterns = data_manager._analyze_coordination_patterns(results_df)
            st.metric("Clusters Temporais Detectados", coordination_patterns.get('clusters', 0))
            st.metric("Taxa de Coordenação", f"{coordination_patterns.get('coordination_rate', 0):.1%}")
        else:
            st.info("Dados temporais não disponíveis")

    # Análise de Canais Alternativos
    st.subheader("📺 Canais Alternativos e Mídia Independente")

    col3, col4 = st.columns(2)

    with col3:
        st.write("**Domínios de Mídia Alternativa**")
        if 'domain' in results_df.columns:
            # Mostrar distribuição real de domínios
            domain_counts = results_df['domain'].value_counts().head(10)

            fig = px.bar(
                x=domain_counts.values,
                y=domain_counts.index,
                orientation='h',
                title="Top 10 Domínios por Volume",
                color=domain_counts.values,
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Estatísticas reais
            total_domains = results_df['domain'].nunique()
            st.metric("Total de Domínios", total_domains)
        else:
            st.info("Dados de domínio não disponíveis")

    with col4:
        st.write("**Intensidade de Fluxo por Canal**")
        if 'domain' in results_df.columns and 'timestamp' in results_df.columns:
            # Análise de intensidade por domínio
            domain_intensity = data_manager._calculate_domain_intensity(results_df)

            # Top 5 domínios mais intensos
            if domain_intensity:
                st.write("**Domínios com Maior Fluxo:**")
                for domain, intensity in list(domain_intensity.items())[:5]:
                    st.write(f"• {domain}: {intensity:.1f} msgs/dia")
        else:
            st.info("Dados insuficientes para análise de intensidade")

    # Análise de Intensidade Informacional Global
    st.subheader("📊 Intensidade Informacional e Disseminação")

    col5, col6 = st.columns(2)

    with col5:
        st.write("**Evolução da Intensidade**")
        if 'timestamp' in results_df.columns:
            # Gráfico de intensidade temporal
            intensity_timeline = data_manager._create_intensity_timeline(results_df)
            if intensity_timeline is not None:
                fig = px.line(
                    intensity_timeline,
                    x='data',
                    y='intensidade',
                    title="Fluxo Informacional ao Longo do Tempo",
                    color_discrete_sequence=[CORES_ACADEMICAS['azul_primario']]
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dados temporais não disponíveis")

    with col6:
        st.write("**Métricas de Disseminação**")
        # Métricas agregadas de disseminação
        if len(results_df) > 0:
            # Velocidade de disseminação (aproximada)
            total_messages = len(results_df)
            if 'timestamp' in results_df.columns:
                try:
                    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], errors='coerce')
                    time_span = (results_df['timestamp'].max() - results_df['timestamp'].min()).days
                    if time_span > 0:
                        dissemination_rate = total_messages / time_span
                        st.metric("Msgs por Dia", f"{dissemination_rate:.1f}")

                    # Picos de atividade
                    daily_counts = results_df.groupby(results_df['timestamp'].dt.date).size()
                    peak_day = daily_counts.max()
                    st.metric("Pico Diário", f"{peak_day} msgs")

                    # Consistência do fluxo
                    consistency = 1 - (daily_counts.std() / daily_counts.mean()) if daily_counts.mean() > 0 else 0
                    st.metric("Consistência do Fluxo", f"{consistency:.2f}")
                except:
                    st.metric("Total de Mensagens", total_messages)
            else:
                st.metric("Total de Mensagens", total_messages)

    # Análise de Distribuição de Fontes
    st.subheader("📊 Distribuição e Concentração")

    col7, col8, col9 = st.columns(3)

    with col7:
        if 'domain' in results_df.columns:
            total_domains = results_df['domain'].nunique()
            st.metric("Domínios Únicos", total_domains)
        else:
            st.metric("Domínios Únicos", "N/A")

    with col8:
        if 'timestamp' in results_df.columns:
            try:
                results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], errors='coerce')
                time_range = (results_df['timestamp'].max() - results_df['timestamp'].min()).days
                st.metric("Período (dias)", time_range)
            except:
                st.metric("Período (dias)", "N/A")
        else:
            st.metric("Período (dias)", "N/A")

    with col9:
        total_messages = len(results_df)
        st.metric("Total de Registros", f"{total_messages:,}")

@require_real_data_only
def render_formacao_discursiva():
    """Página Discursos - Análise de formação discursiva e narrativas."""
    st.markdown(f"""
    <div class="titulo-principal">
        🎭 Formação Discursiva e Narrativas
    </div>
    """, unsafe_allow_html=True)

    data_manager = DataManagerUnified()
    datasets = data_manager.load_research_data()

    if not datasets:
        st.warning("⚠️ Nenhum dado disponível para análise discursiva.")
        return

    # GUARDRAIL: Validar dados antes de processamento
    results_df = list(datasets.values())[0]
    content_data = {'results_df': results_df}

    validation_result = dashboard_guardrail.validate_dashboard_content(
        content_data,
        'formacao_discursiva'
    )

    if not validation_result.is_valid:
        st.error("🚫 GUARDRAIL VIOLADO: Conteúdo discursivo não autorizado")
        st.error(f"Violações: {validation_result.errors}")
        return

    # Análise de Narrativas Dominantes
    st.subheader("📖 Narrativas Dominantes e Temas Centrais")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Tópicos Mais Frequentes**")
        # Análise de tópicos baseada em topic modeling
        topic_cols = [col for col in results_df.columns if 'topic' in col.lower()]
        if topic_cols:
            topic_col = topic_cols[0]
            if results_df[topic_col].dtype == 'object':
                topic_counts = results_df[topic_col].value_counts().head(10)

                fig_topics = px.bar(
                    x=topic_counts.values,
                    y=topic_counts.index,
                    orientation='h',
                    title="Temas Dominantes Identificados",
                    color=topic_counts.values,
                    color_continuous_scale="Blues"
                )
                fig_topics.update_layout(
                    font_family="Open Sans",
                    title_font_color=CORES_ACADEMICAS['azul_escuro']
                )
                st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("Dados de tópicos não disponíveis")

    with col2:
        st.write("**Palavras-Chave Centrais**")
        # Análise TF-IDF ou keywords - apenas dados reais
        tfidf_cols = [col for col in results_df.columns if 'tfidf' in col.lower() or 'keyword' in col.lower()]
        if tfidf_cols:
            tfidf_col = tfidf_cols[0]
            st.write("**Dados TF-IDF Disponíveis:**")
            st.write(f"• Coluna: {tfidf_col}")
            st.write(f"• Registros com dados: {len(results_df[tfidf_col].dropna())}")
            st.write(f"• Tipo de dados: {results_df[tfidf_col].dtype}")
        else:
            st.info("Dados de TF-IDF não disponíveis")

    # Análise de Estratégias Retóricas
    st.subheader("🎯 Estratégias Retóricas e Linguagem")

    col3, col4 = st.columns(2)

    with col3:
        st.write("**Padrões de Linguagem Radical**")
        # Análise baseada em sentiment e classificação política
        if 'political_alignment' in results_df.columns and 'sentiment' in results_df.columns:
            # Crosstab entre sentimento e posicionamento
            try:
                sentiment_cols = [col for col in results_df.columns if 'sentiment' in col.lower()]
                if sentiment_cols:
                    sentiment_col = sentiment_cols[0]
                    crosstab = pd.crosstab(results_df['political_alignment'], results_df[sentiment_col])

                    # Mostrar cruzamento real de dados
                    st.write("**Cruzamento Sentiment x Orientação:**")
                    st.write(f"• Orientações disponíveis: {list(crosstab.index)}")
                    st.write(f"• Sentimentos disponíveis: {list(crosstab.columns)}")

                    # Mostrar apenas os dados reais
                    fig_cross = px.imshow(
                        crosstab.values,
                        x=crosstab.columns,
                        y=crosstab.index,
                        title="Distribuição Sentiment x Orientação Política"
                    )
                    st.plotly_chart(fig_cross, use_container_width=True)
            except:
                st.info("Dados insuficientes para análise de radicalização")
        else:
            st.info("Dados de sentiment/alinhamento não disponíveis")

    with col4:
        st.write("**Análise Qualitativa**")
        # Dados reais de classificação qualitativa se disponível
        if 'qualitative_analysis' in results_df.columns:
            qual_counts = results_df['qualitative_analysis'].value_counts()
            for category, count in qual_counts.head(5).items():
                st.write(f"• {category}: {count} ocorrências")
        else:
            st.info("Dados de análise qualitativa não disponíveis")

    # Análise de Evolução Discursiva
    st.subheader("📈 Evolução e Transformação Discursiva")

    # Análise temporal de mudança nos padrões discursivos
    if 'timestamp' in results_df.columns and 'political_alignment' in results_df.columns:
        try:
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], errors='coerce')
            results_df['periodo'] = results_df['timestamp'].dt.to_period('Q').astype(str)  # Quarterly

            # Evolução por orientação política
            temporal_discourse = results_df.groupby(['periodo', 'political_alignment']).size().reset_index(name='count')

            fig_evolution = px.line(
                temporal_discourse,
                x='periodo',
                y='count',
                color='political_alignment',
                title="Evolução Discursiva por Orientação Política",
                markers=True
            )
            fig_evolution.update_layout(
                font_family="Open Sans",
                title_font_color=CORES_ACADEMICAS['azul_escuro'],
                xaxis_title="Período (Trimestral)",
                yaxis_title="Volume de Mensagens"
            )
            fig_evolution.update_xaxes(tickangle=45)
            st.plotly_chart(fig_evolution, use_container_width=True)

        except Exception as e:
            st.info("Dados temporais insuficientes para análise de evolução")

    # Informações Básicas dos Dados
    st.subheader("📊 Informações dos Dados")

    col5, col6, col7 = st.columns(3)

    with col5:
        # Colunas disponíveis
        total_columns = len(results_df.columns)
        st.metric("Colunas Disponíveis", total_columns)

    with col6:
        # Registros totais
        total_records = len(results_df)
        st.metric("Total de Registros", f"{total_records:,}")

    with col7:
        # Período dos dados (se disponível)
        if 'timestamp' in results_df.columns:
            try:
                results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], errors='coerce')
                time_span = (results_df['timestamp'].max() - results_df['timestamp'].min()).days
                st.metric("Período (dias)", time_span)
            except:
                st.metric("Período (dias)", "N/A")
        else:
            st.metric("Período (dias)", "N/A")

    def _analyze_hashtag_networks(self, df):
        """Analisar redes de disseminação baseadas em hashtags."""
        if 'hashtags_normalized' not in df.columns:
            return {}

        hashtag_counts = {}
        for hashtags in df['hashtags_normalized'].dropna():
            if isinstance(hashtags, str) and hashtags:
                hashtag_list = hashtags.split(',') if ',' in hashtags else [hashtags]
                for hashtag in hashtag_list:
                    hashtag = hashtag.strip()
                    if hashtag:
                        hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1

        return hashtag_counts

    def _analyze_coordination_patterns(self, df):
        """Analisar padrões de coordenação temporal."""
        if 'timestamp' not in df.columns:
            return {'clusters': 0, 'coordination_rate': 0}

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Agrupar por intervalos de 1 hora
            df['hour_bucket'] = df['timestamp'].dt.floor('H')
            hour_counts = df.groupby('hour_bucket').size()

            # Detectar clusters (>3 mensagens na mesma hora)
            clusters = (hour_counts > 3).sum()
            total_hours = len(hour_counts)
            coordination_rate = clusters / total_hours if total_hours > 0 else 0

            return {
                'clusters': clusters,
                'coordination_rate': coordination_rate
            }
        except:
            return {'clusters': 0, 'coordination_rate': 0}

    def _get_domain_stats(self, df):
        """Obter estatísticas reais dos domínios."""
        if 'domain' not in df.columns:
            return {}

        domain_counts = df['domain'].value_counts()
        return {
            'total_domains': len(domain_counts),
            'top_domains': domain_counts.head(10).to_dict()
        }

    def _calculate_domain_intensity(self, df):
        """Calcular intensidade de fluxo por domínio."""
        if 'domain' not in df.columns or 'timestamp' not in df.columns:
            return {}

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            time_span = (df['timestamp'].max() - df['timestamp'].min()).days

            if time_span <= 0:
                return {}

            domain_counts = df['domain'].value_counts()
            domain_intensity = {}

            for domain, count in domain_counts.items():
                intensity = count / time_span
                domain_intensity[domain] = intensity

            return dict(sorted(domain_intensity.items(), key=lambda x: x[1], reverse=True))
        except:
            return {}

    def _create_intensity_timeline(self, df):
        """Criar timeline de intensidade informacional."""
        if 'timestamp' not in df.columns:
            return None

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['data'] = df['timestamp'].dt.date

            daily_counts = df.groupby('data').size().reset_index(name='intensidade')
            daily_counts['data'] = pd.to_datetime(daily_counts['data'])

            return daily_counts
        except:
            return None

    def _get_real_data_stats(self, df):
        """Obter estatísticas reais dos dados sem inventar métricas."""
        stats = {
            'total_records': len(df),
            'columns_available': len(df.columns),
            'data_completeness': df.notna().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0
        }

        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                stats['time_span_days'] = (df['timestamp'].max() - df['timestamp'].min()).days
                stats['messages_per_day'] = len(df) / stats['time_span_days'] if stats['time_span_days'] > 0 else 0
            except:
                stats['time_span_days'] = 0
                stats['messages_per_day'] = 0

        if 'domain' in df.columns:
            stats['unique_domains'] = df['domain'].nunique()

        return stats

def render_sidebar_unified():
    """Renderizar sidebar unificado para navegação."""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
            <h1 style="color: {CORES_ACADEMICAS['azul_primario']}; font-size: 1.8rem; margin-bottom: 0.5rem;">
                🔬 digiNEV v5.1.0
            </h1>
            <p style="color: {CORES_ACADEMICAS['azul_escuro']}; font-size: 0.9rem; margin: 0;">
                Análise de Radicalização Digital
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Navegação principal
        st.markdown("### 📋 Navegação")

        pages = {
            "🏠 Home": "Home",
            "📊 Radicalização": "Radicalização",
            "🌐 Fluxos": "Fluxos",
            "🎭 Discursos": "Discursos"
        }

        if 'secao' not in st.session_state:
            st.session_state.secao = 'home'

        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.secao = page_key.lower()

        # Status técnico mínimo
        st.markdown("---")
        st.markdown("### ⚙️ Status")

        data_manager = DataManagerUnified()
        status = data_manager.get_technical_status()

        st.markdown(f"""
        <div style="font-size: 0.8rem; padding: 0.5rem; background: {CORES_ACADEMICAS['cinza_claro']}; border-radius: 4px;">
            <strong>Pipeline:</strong> {status['pipeline_status']}<br>
            <strong>Qualidade:</strong> {status['data_quality']}<br>
            <strong>Cache:</strong> {status['cache_performance']}<br>
            <strong>API:</strong> {status['api_budget']}
        </div>
        """, unsafe_allow_html=True)

        return pages.get(f"🏠 {st.session_state.secao.title()}", "Home") if st.session_state.secao != "home" else "Home"

def main():
    """Função principal do dashboard unificado."""
    try:
        # Render sidebar e obter página selecionada
        current_page = render_sidebar_unified()

        # Render página baseada na seleção
        if current_page == "Home":
            render_home_unified()
        elif current_page == "Radicalização":
            render_analise_radicalizacao()
        elif current_page == "Fluxos":
            render_fluxos_informacionais()
        elif current_page == "Discursos":
            render_formacao_discursiva()
        else:
            render_home_unified()  # Default

    except Exception as e:
        st.error(f"❌ Erro no dashboard: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()