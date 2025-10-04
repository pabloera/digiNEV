"""
Stage 03 Deduplication Analysis Page
===================================

Streamlit page for Brazilian political discourse deduplication analysis.
Part of the digiNEV v.final dashboard system.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from dashboard.stage03_deduplication_dashboard import Stage03DeduplicationDashboard

# Page configuration
st.set_page_config(
    page_title="Stage 03 - Deduplicação Cross-Dataset",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para análise de deduplicação
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
    }

    .dedup-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .stage-info {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔄 Stage 03: Análise de Deduplicação Cross-Dataset</div>', unsafe_allow_html=True)

# Informações sobre o Stage 03
st.markdown("""
<div class="stage-info">
<h4>📋 Sobre o Stage 03 - Cross-Dataset Deduplication</h4>
<p><strong>Objetivo:</strong> Eliminação inteligente de duplicatas entre TODOS os datasets do projeto, com contador de frequência e análise de propagação.</p>
<p><strong>Algoritmo:</strong> Agrupamento por texto idêntico, preservação do registro mais antigo, contagem de duplicatas com <code>dupli_freq</code>.</p>
<p><strong>Redução Esperada:</strong> 40-50% do volume total (300k → 180k registros)</p>
<p><strong>Métricas Geradas:</strong> dupli_freq, channels_found, date_span_days</p>
</div>
""", unsafe_allow_html=True)

# Informações dos datasets
st.markdown("### 📊 Datasets Analisados")
st.markdown("""
| Dataset | Período | Tamanho | Descrição |
|---------|---------|---------|-----------|
| **1_2019-2021-govbolso.csv** | 2019-2021 | 135.9 MB | Período Governo Bolsonaro |
| **2_2021-2022-pandemia.csv** | 2021-2022 | 230.0 MB | Pandemia COVID-19 |
| **3_2022-2023-poseleic.csv** | 2022-2023 | 93.2 MB | Pós-Eleição |
| **4_2022-2023-elec.csv** | 2022-2023 | 54.2 MB | Eleições |
| **5_2022-2023-elec-extra.csv** | 2022-2023 | 25.2 MB | Dados Eleições Extras |
""")

# Sidebar com informações técnicas
with st.sidebar:
    st.markdown("### ℹ️ Informações Técnicas")

    st.markdown("**🔍 Campos de Análise:**")
    st.markdown("""
    - `dupli_freq`: Frequência de duplicação
    - `channels_found`: Canais onde aparece
    - `date_span_days`: Período de propagação
    - `dataset_source`: Dataset de origem
    """)

    st.markdown("**📈 Métricas Calculadas:**")
    st.markdown("""
    - Taxa de deduplicação por dataset
    - Sobreposição entre datasets
    - Velocidade de propagação
    - Alcance por canais
    """)

    st.markdown("**🎯 Indicadores de Qualidade:**")
    st.markdown("""
    - Redução de volume real
    - Preservação de dados temporais
    - Manutenção de metadados
    - Consistência cross-dataset
    """)

# Renderizar dashboard principal
try:
    dashboard = Stage03DeduplicationDashboard()
    dashboard.render_dashboard()

except Exception as e:
    st.error(f"Erro ao carregar dashboard de deduplicação: {e}")

    with st.expander("🔧 Soluções Possíveis"):
        st.markdown("""
        1. **Verificar se o pipeline foi executado**: Execute `python run_pipeline.py`
        2. **Verificar dados processados**: Confirme se existem arquivos em `data/processed/`
        3. **Executar teste**: Execute `python test_clean_analyzer.py`
        4. **Verificar logs**: Consulte logs em `logs/` para erros específicos
        """)

# Footer com informações do projeto
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>digiNEV v.final</strong> - Análise de Discurso Político Brasileiro</p>
<p>Projeto de Pesquisa Acadêmica | Ciências Sociais | 2025</p>
</div>
""", unsafe_allow_html=True)