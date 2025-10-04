"""
Stage 11: Topic Modeling Analysis Page
=====================================

Professional Streamlit page for LDA topic modeling analysis of Brazilian political discourse.
Features Sankey flow diagrams and bubble charts for cross-stage analysis.
Part of the digiNEV v.final dashboard system.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from dashboard.stage11_topic_modeling_dashboard import TopicModelingDashboard

# Page configuration
st.set_page_config(
    page_title="Stage 11: Topic Modeling - digiNEV",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Run the topic modeling analysis page."""
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">
            🏷️ Stage 11: Topic Modeling Analysis
        </h1>
        <p style="color: #e8f4f8; margin: 0.5rem 0 0 0; text-align: center; font-size: 1.1rem;">
            Análise de Tópicos com LDA | Fluxos Cross-Stage | Visualizações Multidimensionais
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Information about the analysis
    with st.expander("ℹ️ Sobre a Análise de Topic Modeling", expanded=False):
        st.markdown("""
        ### Funcionalidades da Análise de Tópicos

        **🎯 Topic Modeling (LDA)**
        - Descoberta automática de tópicos latentes nos textos
        - Modelagem probabilística com Latent Dirichlet Allocation
        - Identificação de palavras-chave por tópico
        - Probabilidades de pertencimento por documento

        **🌊 Análise de Fluxo Sankey**
        - Visualização do fluxo: Tópicos → Clusters → Affordances
        - Mapeamento de relações entre diferentes dimensões analíticas
        - Identificação de padrões de distribuição cross-stage

        **🎈 Análise Multidimensional**
        - Bubble Chart: Tópicos vs Política vs Intensidade Temporal
        - Visualização de múltiplas dimensões simultaneamente
        - Identificação de correlações complexas

        **📊 Estatísticas Avançadas**
        - Distribuições de tópicos e orientações políticas
        - Matrizes de co-ocorrência tópico-cluster
        - Métricas de qualidade e cobertura

        ### Interpretação dos Resultados

        - **Tópicos Dominantes**: Temas mais frequentes no corpus
        - **Fluxos Cruzados**: Como tópicos se distribuem por clusters e affordances
        - **Correlações Políticas**: Associações entre tópicos e orientações
        - **Intensidade Temporal**: Variações dos tópicos no tempo
        """)

    # Create and run dashboard
    dashboard = TopicModelingDashboard()
    dashboard.run()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>digiNEV v.final</strong> | Stage 11: Topic Modeling Analysis<br>
        Sistema de Análise de Discurso Político Brasileiro | Pipeline Científico Consolidado
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
else:
    main()