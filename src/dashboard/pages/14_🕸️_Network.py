"""
Stage 14 Network Analysis - Streamlit Page
==========================================

Brazilian Political Discourse Coordination Detection Dashboard
Comprehensive network analysis revealing coordination patterns and influence networks.

TECHNICAL FEATURES:
- Force-directed network visualization with user/channel coordination connections
- Community detection using modularity-based algorithmic clustering
- Centrality analysis with multiple influence metrics (betweenness, closeness, degree, eigenvector)
- Multi-layer network integration combining coordination, sentiment, and topics

ACADEMIC FOCUS:
- Detection of coordinated behavior in Brazilian political messaging
- Identification of influence networks and information brokers
- Analysis of community structures in political discourse
- Multi-dimensional relationship mapping in social media networks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
current_dir = Path(__file__).parent.parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

try:
    from dashboard.stage14_network_dashboard import NetworkAnalysisDashboard
    from dashboard.dashboard_guardrails import DashboardGuardrails
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Stage 14: Network Analysis",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_network_data():
    """
    Load network analysis data from Stage 14 results.

    Returns:
        pd.DataFrame: Network analysis dataset or None if not found
    """
    try:
        data_dir = current_dir / "data" / "dashboard_results"

        # Find most recent Stage 14 network analysis file
        pattern = "*network_analysis*.csv"
        network_files = list(data_dir.glob(pattern))

        if not network_files:
            st.warning("⚠️ Arquivos de análise de rede não encontrados")
            st.info("Execute o pipeline completo para gerar dados de Stage 14")
            return None

        # Get most recent file
        latest_file = max(network_files, key=lambda p: p.stat().st_mtime)

        # Load with proper encoding and separator
        df = pd.read_csv(latest_file, sep=';', encoding='utf-8')

        logger.info(f"Dados de rede carregados: {len(df)} registros de {latest_file.name}")
        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados de rede: {e}")
        logger.error(f"Erro ao carregar dados de rede: {e}")
        return None

def validate_network_columns(df: pd.DataFrame) -> bool:
    """
    Validate required columns for network analysis.

    Args:
        df: Input DataFrame

    Returns:
        bool: True if valid, False otherwise
    """
    required_cols = [
        'user_id', 'channel', 'sender_frequency', 'is_frequent_sender',
        'shared_url_frequency', 'temporal_coordination'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Colunas necessárias ausentes: {missing_cols}")
        st.info("Verifique se o Stage 14 foi executado corretamente")
        return False

    return True

def display_data_overview(df: pd.DataFrame):
    """
    Display overview of network analysis data.

    Args:
        df: Network analysis DataFrame
    """
    st.subheader("📊 Visão Geral dos Dados de Rede")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Registros", len(df))

    with col2:
        unique_users = df['user_id'].nunique()
        st.metric("Usuários Únicos", unique_users)

    with col3:
        unique_channels = df['channel'].nunique()
        st.metric("Canais Únicos", unique_channels)

    with col4:
        frequent_senders = df['is_frequent_sender'].sum()
        st.metric("Remetentes Frequentes", frequent_senders)

    # Additional metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_sender_freq = df['sender_frequency'].mean()
        st.metric("Frequência Média do Remetente", f"{avg_sender_freq:.1f}")

    with col2:
        avg_url_freq = df['shared_url_frequency'].mean()
        st.metric("Frequência Média de URLs", f"{avg_url_freq:.1f}")

    with col3:
        avg_temporal_coord = df['temporal_coordination'].mean()
        st.metric("Coordenação Temporal Média", f"{avg_temporal_coord:.3f}")

def display_network_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display interactive filters for network analysis.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    st.sidebar.subheader("🔍 Filtros de Rede")

    # Minimum sender frequency filter
    min_sender_freq = st.sidebar.slider(
        "Frequência Mínima do Remetente",
        min_value=int(df['sender_frequency'].min()),
        max_value=int(df['sender_frequency'].max()),
        value=int(df['sender_frequency'].quantile(0.25)),
        help="Filtrar usuários por frequência mínima de mensagens"
    )

    # Frequent senders only
    frequent_only = st.sidebar.checkbox(
        "Apenas Remetentes Frequentes",
        value=False,
        help="Mostrar apenas usuários identificados como remetentes frequentes"
    )

    # Minimum URL sharing frequency
    min_url_freq = st.sidebar.slider(
        "Frequência Mínima de URLs Compartilhadas",
        min_value=int(df['shared_url_frequency'].min()),
        max_value=int(df['shared_url_frequency'].max()),
        value=int(df['shared_url_frequency'].min()),
        help="Filtrar por frequência mínima de URLs compartilhadas"
    )

    # Temporal coordination threshold
    min_temporal_coord = st.sidebar.slider(
        "Coordenação Temporal Mínima",
        min_value=float(df['temporal_coordination'].min()),
        max_value=float(df['temporal_coordination'].max()),
        value=float(df['temporal_coordination'].quantile(0.1)),
        format="%.3f",
        help="Filtrar por nível mínimo de coordenação temporal"
    )

    # Channel selection
    selected_channels = st.sidebar.multiselect(
        "Selecionar Canais",
        options=sorted(df['channel'].unique()),
        default=sorted(df['channel'].unique())[:5],  # Default to first 5 channels
        help="Escolher canais específicos para análise"
    )

    # Apply filters
    filtered_df = df[
        (df['sender_frequency'] >= min_sender_freq) &
        (df['shared_url_frequency'] >= min_url_freq) &
        (df['temporal_coordination'] >= min_temporal_coord) &
        (df['channel'].isin(selected_channels))
    ]

    if frequent_only:
        filtered_df = filtered_df[filtered_df['is_frequent_sender'] == True]

    # Display filter results
    st.sidebar.write(f"**Registros após filtros:** {len(filtered_df)}")
    if len(filtered_df) < len(df):
        reduction_pct = (1 - len(filtered_df) / len(df)) * 100
        st.sidebar.write(f"**Redução:** {reduction_pct:.1f}%")

    return filtered_df

def display_network_insights(df: pd.DataFrame):
    """
    Display network analysis insights and statistics.

    Args:
        df: Network analysis DataFrame
    """
    st.subheader("🔍 Insights da Análise de Rede")

    # User activity analysis
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distribuição da Atividade dos Usuários**")

        # Sender frequency distribution
        fig_freq = px.histogram(
            df, x='sender_frequency',
            title="Distribuição da Frequência de Remetentes",
            labels={'sender_frequency': 'Frequência do Remetente', 'count': 'Número de Usuários'},
            color_discrete_sequence=['lightblue']
        )
        fig_freq.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_freq, use_container_width=True)

    with col2:
        st.write("**Coordenação e Compartilhamento**")

        # Temporal coordination vs URL sharing
        fig_coord = px.scatter(
            df, x='temporal_coordination', y='shared_url_frequency',
            title="Coordenação Temporal vs. Compartilhamento de URLs",
            labels={
                'temporal_coordination': 'Coordenação Temporal',
                'shared_url_frequency': 'Frequência de URLs'
            },
            color='is_frequent_sender',
            color_discrete_map={True: 'red', False: 'blue'}
        )
        fig_coord.update_layout(height=300)
        st.plotly_chart(fig_coord, use_container_width=True)

    # Channel analysis
    st.write("**Análise por Canal**")

    channel_stats = df.groupby('channel').agg({
        'user_id': 'nunique',
        'sender_frequency': 'mean',
        'shared_url_frequency': 'mean',
        'temporal_coordination': 'mean',
        'is_frequent_sender': 'sum'
    }).round(3)

    channel_stats.columns = [
        'Usuários Únicos', 'Freq. Média Remetente', 'Freq. Média URLs',
        'Coordenação Temporal Média', 'Remetentes Frequentes'
    ]

    st.dataframe(channel_stats, use_container_width=True)

    # Top users analysis
    st.write("**Usuários Mais Ativos**")

    top_users = df.groupby('user_id').agg({
        'sender_frequency': 'first',
        'shared_url_frequency': 'first',
        'temporal_coordination': 'first',
        'is_frequent_sender': 'first',
        'channel': 'nunique'
    }).round(3)

    top_users.columns = [
        'Frequência', 'URLs Compartilhadas', 'Coordenação Temporal',
        'É Frequente', 'Canais Únicos'
    ]

    top_users = top_users.sort_values('Frequência', ascending=False).head(10)
    st.dataframe(top_users, use_container_width=True)

def main():
    """Main function for Stage 14 Network Analysis page."""

    # Page header
    st.title("🕸️ Stage 14: Análise de Rede")
    st.markdown("**Detecção de coordenação e padrões de influência no discurso político brasileiro**")

    # Initialize guardrails
    guardrails = DashboardGuardrails()

    # Check for data availability
    if not guardrails.check_data_availability():
        st.error("❌ Dados não disponíveis para análise")
        st.info("Execute o pipeline de análise para gerar os dados necessários")
        return

    # Load network data
    with st.spinner("Carregando dados de análise de rede..."):
        df = load_network_data()

    if df is None:
        st.error("❌ Não foi possível carregar os dados de rede")
        return

    # Validate columns
    if not validate_network_columns(df):
        return

    # Display data overview
    display_data_overview(df)

    st.markdown("---")

    # Apply filters
    filtered_df = display_network_filters(df)

    if len(filtered_df) == 0:
        st.warning("⚠️ Nenhum registro encontrado com os filtros aplicados")
        st.info("Ajuste os filtros para incluir mais dados")
        return

    # Network insights
    display_network_insights(filtered_df)

    st.markdown("---")

    # Main network analysis dashboard
    st.subheader("🔬 Análise Avançada de Rede")

    if st.button("🚀 Executar Análise Completa de Rede", type="primary"):
        try:
            # Initialize network analysis dashboard
            dashboard = NetworkAnalysisDashboard()

            # Render complete dashboard
            with st.spinner("Executando análise completa de rede..."):
                dashboard.render_dashboard(filtered_df)

        except Exception as e:
            st.error(f"Erro na análise de rede: {e}")
            logger.error(f"Erro na análise de rede: {e}")

    # Additional analysis options
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Opções Avançadas")

    if st.sidebar.checkbox("Mostrar Dados Brutos", value=False):
        st.subheader("📋 Dados Brutos de Rede")
        st.dataframe(filtered_df.head(100), use_container_width=True)

        # Download option
        csv = filtered_df.to_csv(index=False, sep=';')
        st.download_button(
            label="📥 Baixar Dados Filtrados (CSV)",
            data=csv,
            file_name=f"network_analysis_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Performance monitoring
    if st.sidebar.checkbox("Monitor de Performance", value=False):
        st.sidebar.success(f"✅ {len(filtered_df)} registros carregados")
        st.sidebar.info(f"📊 {filtered_df['user_id'].nunique()} usuários únicos")
        st.sidebar.info(f"📺 {filtered_df['channel'].nunique()} canais únicos")

    # Documentation
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Sobre Stage 14")
    st.sidebar.write("""
    **Análise de Rede para Detecção de Coordenação:**

    • **Force-directed Network**: Visualização interativa das conexões de coordenação
    • **Community Detection**: Identificação algorítmica de grupos coordenados
    • **Centrality Analysis**: Métricas de influência e importância dos nós
    • **Multi-layer Network**: Integração de coordenação, sentimento e tópicos

    **Métricas Calculadas:**
    - Frequência do remetente
    - Frequência de URLs compartilhadas
    - Coordenação temporal
    - Identificação de remetentes frequentes
    """)


if __name__ == "__main__":
    main()