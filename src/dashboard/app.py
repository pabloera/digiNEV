"""
digiNEV Dashboard de Análise Exploratória: Interface para exploração de resultados do pipeline
Function: Dashboard modular focado na análise acadêmica de dados de discurso político brasileiro
Usage: Interface web para cientistas sociais explorarem padrões, sentimentos e insights
"""

import streamlit as st
import sys
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="digiNEV | Análise de Discurso Digital",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar src ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# CSS customizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .page-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class DigiNEVDashboard:
    """Dashboard principal para análise exploratória dos resultados digiNEV"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        self.project_root = project_root
        
        # Inicializar estado da sessão
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'overview'
        
        # Importar utilitários
        try:
            from dashboard.utils.data_loader import DataLoader
            self.data_loader = DataLoader(self.project_root)
        except ImportError as e:
            st.error(f"Erro ao importar módulos: {e}")
            self.data_loader = None
    
    def run(self):
        """Executa o dashboard principal"""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Renderiza o cabeçalho principal"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 digiNEV | Monitor do Discurso Digital</h1>
            <p>Análise Exploratória de Dados Políticos Brasileiros</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Renderiza a barra lateral com navegação"""
        with st.sidebar:
            st.markdown("## 🧭 Navegação")
            
            # Menu principal
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            pages = {
                'overview': '📋 Visão Geral',
                'political': '🏛️ Análise Política',
                'sentiment': '💭 Análise de Sentimento',
                'topics': '🎨 Modelagem de Tópicos',
                'search': '🔍 Busca Semântica',
                'network': '📊 Análise de Rede',
                'temporal': '⏱️ Análise Temporal',
                'quality': '🔬 Controle de Qualidade'
            }
            
            for page_key, page_name in pages.items():
                if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Informações do sistema
            st.markdown("## ℹ️ Informações")
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            if self.data_loader:
                try:
                    status = self.data_loader.get_data_status()
                    st.metric("Arquivos Disponíveis", status.get('available_files', 0))
                    st.metric("Última Execução", status.get('last_execution', 'N/A'))
                except Exception as e:
                    st.warning(f"Erro ao carregar status: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_main_content(self):
        """Renderiza o conteúdo principal baseado na página selecionada"""
        current_page = st.session_state.current_page
        
        try:
            if current_page == 'overview':
                self._render_overview_page()
            elif current_page == 'political':
                self._render_political_page()
            elif current_page == 'sentiment':
                self._render_sentiment_page()
            elif current_page == 'topics':
                self._render_topics_page()
            elif current_page == 'search':
                self._render_search_page()
            elif current_page == 'network':
                self._render_network_page()
            elif current_page == 'temporal':
                self._render_temporal_page()
            elif current_page == 'quality':
                self._render_quality_page()
            else:
                self._render_overview_page()
                
        except Exception as e:
            st.error(f"Erro ao carregar página: {e}")
            st.info("💡 Execute o pipeline principal para gerar dados de análise.")
    
    def _render_overview_page(self):
        """Renderiza a página de visão geral"""
        try:
            from dashboard.pages.overview import render_overview_page
            render_overview_page(self.data_loader)
        except ImportError:
            self._render_fallback_overview()
    
    def _render_political_page(self):
        """Renderiza a página de análise política"""
        try:
            from dashboard.pages.political_analysis import render_political_page
            render_political_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise Política", "🏛️")
    
    def _render_sentiment_page(self):
        """Renderiza a página de análise de sentimento"""
        try:
            from dashboard.pages.sentiment_analysis import render_sentiment_page
            render_sentiment_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise de Sentimento", "💭")
    
    def _render_topics_page(self):
        """Renderiza a página de modelagem de tópicos"""
        try:
            from dashboard.pages.topic_modeling import render_topics_page
            render_topics_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Modelagem de Tópicos", "🎨")
    
    def _render_search_page(self):
        """Renderiza a página de busca semântica"""
        try:
            from dashboard.pages.semantic_search import render_search_page
            render_search_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Busca Semântica", "🔍")
    
    def _render_network_page(self):
        """Renderiza a página de análise de rede"""
        try:
            from dashboard.pages.network_analysis import render_network_page
            render_network_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise de Rede", "📊")
    
    def _render_temporal_page(self):
        """Renderiza a página de análise temporal"""
        try:
            from dashboard.pages.temporal_analysis import render_temporal_page
            render_temporal_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise Temporal", "⏱️")
    
    def _render_quality_page(self):
        """Renderiza a página de controle de qualidade"""
        try:
            from dashboard.pages.quality_control import render_quality_page
            render_quality_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Controle de Qualidade", "🔬")
    
    def _render_fallback_overview(self):
        """Página de visão geral simplificada como fallback"""
        st.markdown('<div class="page-header"><h2>📋 Visão Geral</h2></div>', unsafe_allow_html=True)
        
        st.info("🔄 Módulos de análise sendo carregados...")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">22</div>
                <div class="metric-label">Etapas do Pipeline</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">5</div>
                <div class="metric-label">Dimensões de Análise</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">8</div>
                <div class="metric-label">Tipos de Visualização</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">✓</div>
                <div class="metric-label">Sistema Operacional</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🚀 Como Usar o Dashboard")
        
        st.markdown("""
        ### Pré-requisitos
        1. **Execute o pipeline principal** para gerar dados:
           ```bash
           poetry run python run_pipeline.py
           ```
        
        2. **Navegue pelas análises** usando o menu lateral:
           - 📋 **Visão Geral**: Métricas e resumo executivo
           - 🏛️ **Análise Política**: Categorização e orientação política
           - 💭 **Análise de Sentimento**: Emoções e polarização
           - 🎨 **Modelagem de Tópicos**: Temas e clusters semânticos
           - 🔍 **Busca Semântica**: Consultas interativas
           - 📊 **Análise de Rede**: Interações e propagação
           - ⏱️ **Análise Temporal**: Evolução e tendências
           - 🔬 **Controle de Qualidade**: Validação e métricas
        
        ### Dados de Entrada
        - **Formato**: Arquivos CSV com mensagens do Telegram
        - **Estrutura**: Texto, timestamp, metadados de origem
        - **Processamento**: 22 etapas de análise automatizada
        """)
    
    def _render_fallback_page(self, title: str, icon: str):
        """Página genérica de fallback"""
        st.markdown(f'<div class="page-header"><h2>{icon} {title}</h2></div>', unsafe_allow_html=True)
        
        st.info(f"🔄 Módulo {title} sendo inicializado...")
        
        st.markdown(f"""
        ### {title} - Em Desenvolvimento
        
        Esta seção apresentará:
        - Visualizações interativas específicas
        - Métricas e estatísticas relevantes
        - Filtros e controles de exploração
        - Exportação de resultados
        
        **Execute o pipeline principal para gerar dados de análise.**
        """)

# Executar aplicação
if __name__ == "__main__":
    dashboard = DigiNEVDashboard()
    dashboard.run()