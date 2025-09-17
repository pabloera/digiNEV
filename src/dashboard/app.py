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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Adicionar src ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# CSS customizado - Paleta profissional minimalista
st.markdown("""
<style>
    :root {
        --primary-blue: #1b365d;      /* Azul escuro */
        --accent-orange: #d85a00;     /* Laranja escuro */
        --success-green: #2d5a27;     /* Verde escuro */
        --neutral-gray: #4a5568;      /* Cinza */
        --light-gray: #f7fafc;        /* Cinza claro */
        --white: #ffffff;             /* Branco */
    }
    
    .main-header {
        background: var(--primary-blue);
        padding: 2rem;
        color: var(--white);
        text-align: left;
        margin-bottom: 2rem;
        border-left: 4px solid var(--accent-orange);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--white);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        color: var(--light-gray);
        opacity: 0.8;
    }
    
    .metric-card {
        background: var(--white);
        padding: 1.2rem;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin: 0.3rem 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--neutral-gray);
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar-section {
        background-color: var(--light-gray);
        border-radius: 4px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--success-green);
    }
    
    .page-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Botões minimalistas */
    .stButton > button {
        background: var(--white);
        color: var(--primary-blue);
        border: 1px solid var(--primary-blue);
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--primary-blue);
        color: var(--white);
    }
    
    /* Sidebar limpa */
    .sidebar .sidebar-content {
        background: var(--white);
    }
    
    /* Remover decorações excessivas */
    .element-container div[data-testid="metric-container"] {
        background: transparent;
        border: none;
        box-shadow: none;
    }
    
    /* Menu horizontal minimalista */
    .horizontal-nav {
        background: var(--white);
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .nav-button {
        background: transparent;
        border: none;
        color: var(--neutral-gray);
        font-weight: 500;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        padding: 0.75rem 1.5rem;
        margin: 0 0.25rem;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .nav-button:hover {
        color: var(--primary-blue);
        border-bottom-color: var(--accent-orange);
    }
    
    .nav-button.active {
        color: var(--primary-blue);
        border-bottom-color: var(--primary-blue);
        font-weight: 600;
    }
    
    /* Fonte sóbria global */
    .main-content {
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
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
        self._render_top_nav()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Renderiza o cabeçalho principal - minimalista e profissional"""
        st.markdown("""
        <div class="main-header">
            <h1>digiNEV | Monitor do Discurso Digital</h1>
            <p>Análise de Dados Políticos Brasileiros</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_top_nav(self):
        """Renderiza menu horizontal minimalista"""
        st.markdown('<div class="horizontal-nav">', unsafe_allow_html=True)
        
        # Menu horizontal com 5 botões
        col1, col2, col3, col4, col5 = st.columns(5)
        
        nav_items = [
            ('overview', 'Visão Geral', col1),
            ('sentiment', 'Análise de Sentimento', col2), 
            ('topics', 'Tópicos', col3),
            ('political', 'Análise Política', col4),
            ('search', 'Busca', col5)
        ]
        
        for page_key, page_name, col in nav_items:
            with col:
                if st.button(page_name, key=f"hnav_{page_key}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Sidebar apenas com status - sem navegação"""
        with st.sidebar:
            st.markdown("### Status do Sistema")
            if self.data_loader:
                try:
                    status = self.data_loader.get_data_status()
                    st.markdown(f"**Arquivos:** {status.get('available_files', 0)}")
                except:
                    st.markdown("**Sistema:** Ativo")
    
    def _render_page_indicator(self, current_page: str):
        """Renderiza indicador minimalista da página atual"""
        page_titles = {
            'overview': 'Visão Geral',
            'sentiment': 'Análise de Sentimento',
            'topics': 'Modelagem de Tópicos', 
            'political': 'Análise Política',
            'search': 'Busca Semântica'
        }
        
        title = page_titles.get(current_page, 'Dashboard')
        st.markdown(f'<div class="page-header"><h2>{title}</h2></div>', unsafe_allow_html=True)
    
    def _render_main_content(self):
        """Renderiza conteúdo principal simplificado"""
        current_page = st.session_state.current_page
        
        # Indicador minimalista da página
        self._render_page_indicator(current_page)
        
        try:
            # Apenas 5 páginas essenciais
            if current_page == 'overview':
                self._render_overview_page()
            elif current_page == 'sentiment':
                self._render_sentiment_page()
            elif current_page == 'topics':
                self._render_topics_page()
            elif current_page == 'political':
                self._render_political_page()
            elif current_page == 'search':
                self._render_search_page()
            else:
                self._render_overview_page()  # Fallback para overview
                
        except Exception as e:
            st.error(f"Erro ao carregar página: {e}")
            st.info("Execute o pipeline para gerar dados de análise.")
    
    def _render_overview_page(self):
        """Renderiza a página de visão geral"""
        try:
            from dashboard.views.overview import render_overview_page
            render_overview_page(self.data_loader)
        except ImportError:
            self._render_fallback_overview()
    
    def _render_political_page(self):
        """Renderiza a página de análise política"""
        try:
            from dashboard.views.political_analysis import render_political_page
            render_political_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise Política", "")
    
    def _render_sentiment_page(self):
        """Renderiza a página de análise de sentimento"""
        try:
            from dashboard.views.sentiment_analysis import render_sentiment_page
            render_sentiment_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise de Sentimento", "")
    
    def _render_topics_page(self):
        """Renderiza a página de modelagem de tópicos"""
        try:
            from dashboard.views.topic_modeling import render_topics_page
            render_topics_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Modelagem de Tópicos", "")
    
    def _render_search_page(self):
        """Renderiza a página de busca semântica"""
        try:
            from dashboard.views.semantic_search import render_search_page
            render_search_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Busca Semântica", "")
    
    # Páginas essenciais mantidas - todas as outras removidas para simplicidade
    
    def _render_fallback_page(self, title: str, icon: str):
        """Página genérica simplificada"""
        st.info(f"Execute o pipeline para gerar dados de {title.lower()}.")

# Executar aplicação
if __name__ == "__main__":
    dashboard = DigiNEVDashboard()
    dashboard.run()