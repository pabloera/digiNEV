#!/usr/bin/env python3
"""
Stage 04 - Duplication Statistics Dashboard Page
===============================================

Streamlit page for Stage 04 duplication pattern statistics visualization
in the Brazilian political discourse analysis pipeline.
"""

import sys
import streamlit as st
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.append(str(src_dir))
sys.path.append(str(current_dir.parent))

# Import the duplication statistics dashboard
try:
    from stage04_duplication_stats_dashboard import Stage04DuplicationStatsView
    dashboard_available = True
except ImportError as e:
    st.error(f"Erro ao importar dashboard de duplicação: {e}")
    dashboard_available = False

def main():
    """Main page function"""
    st.set_page_config(
        page_title="Stage 04 - Estatísticas de Duplicação",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if dashboard_available:
        try:
            # Initialize and render the duplication statistics dashboard
            dashboard = Stage04DuplicationStatsView()
            dashboard.render_dashboard()
        except Exception as e:
            st.error(f"Erro ao executar dashboard: {e}")
            st.markdown("""
            ### Troubleshooting

            Se você está vendo este erro:

            1. **Verifique se o pipeline foi executado**: `python run_pipeline.py`
            2. **Verifique se há dados do Stage 03**: Os dados de deduplicação são necessários
            3. **Execute o teste**: `python test_stage04_dashboard.py`

            **O que este dashboard mostra:**
            - 📊 Distribuição de frequência de duplicatas
            - 🔄 Análise de ocorrências repetidas
            - 🔗 Estatísticas de sobreposição entre datasets
            - 📈 Resumo estatístico consolidado
            """)
    else:
        st.error("Dashboard de estatísticas de duplicação não disponível")
        st.markdown("""
        ### Sistema não configurado

        O dashboard de estatísticas de duplicação não pôde ser carregado.

        **Para resolver:**
        1. Verifique se todos os arquivos estão no local correto
        2. Execute `python test_stage04_dashboard.py` para validar
        3. Certifique-se de que o ambiente Python está configurado corretamente

        **Estrutura esperada:**
        ```
        src/dashboard/
        ├── stage04_duplication_stats_dashboard.py
        ├── pages/
        │   └── 4_📊_Duplicação.py (este arquivo)
        └── data/dashboard_results/
            └── *03_deduplication*.csv (dados do pipeline)
        ```
        """)

if __name__ == "__main__":
    main()