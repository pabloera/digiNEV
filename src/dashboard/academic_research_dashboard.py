"""
Dashboard de Pesquisa Acadêmica - Análise do Discurso Político Brasileiro
=========================================================================

Dashboard acadêmico focado na apresentação da pesquisa sobre o discurso político
brasileiro, com ênfase no conteúdo científico e nos resultados da análise.

Menu: Home, Pesquisa, Dados
- Home: Descrição da pesquisa e metodologia (apenas texto)
- Pesquisa: Resultados e análises acadêmicas
- Dados: Visualização dos dados processados
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import glob
import re
import sys
import os

# Adicionar diretório src ao path para imports
sys.path.append(str(Path(__file__).parent.parent))

# GUARDRAILS: Sistema de validação de conteúdo
try:
    from dashboard_guardrails import dashboard_guardrail, require_real_data_only, validate_dashboard_data
except ImportError:
    # Fallback se guardrails não disponível
    def dashboard_guardrail(f): return f
    def require_real_data_only(f): return f
    def validate_dashboard_data(data): return data

# ScientificAnalyzer - fonte única de dados
try:
    from core.scientific_analyzer import ScientificAnalyzer
    SCIENTIFIC_ANALYZER_AVAILABLE = True
except ImportError:
    SCIENTIFIC_ANALYZER_AVAILABLE = False
    st.warning("⚠️ ScientificAnalyzer não disponível. Algumas funcionalidades serão limitadas.")

# Esquema de cores seguindo dashesxtilop-2.ini
CORES = {
    'primaria': '#2261C6',      # Azul vibrante (especificação exata)
    'secundaria': '#E5E8EB',    # Cinza claro estrutural
    'fundo_branco': '#FFFFFF',  # Branco predominante
    'cinza_claro': '#F8F9FA',   # Cinza muito claro
    'cinza_medio': '#E5E8EB',   # Cinza médio
    'texto_escuro': '#000000',  # Preto para máxima legibilidade
    'texto_claro': '#FFFFFF',   # Texto branco
    'destaque': '#2261C6',      # Azul para CTAs e destaques
    'hover': '#1A4F9E'          # Azul mais escuro para hover
}

# Configuração da página
st.set_page_config(
    page_title="digiNEV - Análise do Discurso Político Brasileiro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado seguindo especificações dashesxtilop-2.ini
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');

    /* Reset e configuração base */
    * {{
        box-sizing: border-box;
    }}

    /* Layout principal - Grid responsivo 12 colunas */
    .main {{
        background-color: {CORES['fundo_branco']};
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {CORES['texto_escuro']};
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        line-height: 1.6;
    }}

    /* Grid system responsivo */
    .grid-container {{
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 2rem;
        margin: 0 auto;
        max-width: 1200px;
    }}

    .col-12 {{ grid-column: span 12; }}
    .col-6 {{ grid-column: span 6; }}
    .col-4 {{ grid-column: span 4; }}
    .col-3 {{ grid-column: span 3; }}

    @media (max-width: 768px) {{
        .col-6, .col-4, .col-3 {{ grid-column: span 12; }}
        .main {{ padding: 1rem; }}
        .grid-container {{ gap: 1rem; }}
    }}

    /* Cabeçalho - Clean e moderno */
    .header-academic {{
        background: {CORES['fundo_branco']};
        color: {CORES['texto_escuro']};
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 3rem;
        border-bottom: 1px solid {CORES['cinza_claro']};
    }}

    .header-academic h1 {{
        color: {CORES['primaria']};
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}

    .header-academic h3 {{
        color: {CORES['texto_escuro']};
        font-size: 1.5rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }}

    .header-academic p {{
        color: {CORES['texto_escuro']};
        font-size: 1.1rem;
        opacity: 0.8;
    }}

    /* Sidebar - Clean e moderna */
    .css-1d391kg {{
        background-color: {CORES['fundo_branco']};
        border-right: 1px solid {CORES['cinza_claro']};
        padding: 2rem 1rem;
    }}

    /* Cartões de conteúdo - Flat design */
    .content-card {{
        background: {CORES['fundo_branco']};
        padding: 3rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid {CORES['cinza_claro']};
        margin-bottom: 2rem;
        line-height: 1.8;
    }}

    /* Tipografia moderna - Open Sans */
    .academic-text {{
        color: {CORES['texto_escuro']};
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 2rem;
        font-weight: 400;
    }}

    /* Títulos - Hierarquia clara */
    .title-main {{
        color: {CORES['primaria']};
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }}

    .title-section {{
        color: {CORES['primaria']};
        font-size: 2rem;
        font-weight: 600;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid {CORES['cinza_claro']};
    }}

    /* Listas - Spacing generoso */
    .academic-list {{
        background: {CORES['cinza_claro']};
        padding: 2rem;
        border-radius: 8px;
        margin: 2rem 0;
        border: 1px solid {CORES['cinza_medio']};
    }}

    .academic-list ul {{
        margin: 1rem 0;
        padding-left: 1.5rem;
    }}

    .academic-list li {{
        margin-bottom: 0.75rem;
        line-height: 1.6;
    }}

    /* Destaque - Flat style azul */
    .highlight-box {{
        background: {CORES['primaria']};
        color: {CORES['texto_claro']};
        padding: 2.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        text-align: center;
        border: none;
    }}

    .highlight-box h3, .highlight-box h4 {{
        color: {CORES['texto_claro']};
        margin-bottom: 1rem;
    }}

    /* Menu sidebar - Clean */
    .sidebar-menu {{
        padding: 1rem 0;
    }}

    /* Botões - Flat style conforme especificação */
    .stButton > button {{
        width: 100%;
        background: {CORES['primaria']};
        color: {CORES['texto_claro']};
        border: none;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        text-transform: none;
        letter-spacing: 0.02em;
    }}

    .stButton > button:hover {{
        background: {CORES['hover']};
        transform: none;
        box-shadow: 0 4px 12px rgba(34, 97, 198, 0.3);
    }}

    /* Métricas - Cards limpos */
    .metric-academic {{
        background: {CORES['fundo_branco']};
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid {CORES['cinza_claro']};
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}

    .metric-value {{
        font-size: 3rem;
        font-weight: 700;
        color: {CORES['primaria']};
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}

    .metric-label {{
        font-size: 1rem;
        color: {CORES['texto_escuro']};
        font-weight: 500;
        opacity: 0.8;
    }}

    /* Citações - Flat style */
    .citation {{
        font-style: italic;
        color: {CORES['texto_escuro']};
        border-left: 4px solid {CORES['primaria']};
        padding: 1.5rem 2rem;
        margin: 2rem 0;
        background: {CORES['cinza_claro']};
        border-radius: 0 8px 8px 0;
        font-size: 1.1rem;
        line-height: 1.8;
    }}

    /* Footer - Clean */
    .footer-academic {{
        text-align: center;
        color: {CORES['texto_escuro']};
        font-size: 0.95rem;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid {CORES['cinza_claro']};
        opacity: 0.8;
    }}

    /* Responsividade aprimorada */
    @media (max-width: 768px) {{
        .header-academic h1 {{ font-size: 2rem; }}
        .title-main {{ font-size: 2rem; }}
        .title-section {{ font-size: 1.5rem; }}
        .content-card {{ padding: 1.5rem; }}
        .metric-value {{ font-size: 2rem; }}
    }}

    /* Acessibilidade - Alto contraste WCAG */
    .stButton > button:focus {{
        outline: 3px solid {CORES['primaria']};
        outline-offset: 2px;
    }}

    /* Elementos visuais flat */
    .stPlotlyChart {{
        border-radius: 8px;
        border: 1px solid {CORES['cinza_claro']};
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)

class ScientificDataManager:
    """Gerenciador de dados científicos usando ScientificAnalyzer"""

    def __init__(self):
        self.analyzer = None
        self.latest_results = None
        self._cached_data = {}

        # Tentar inicializar ScientificAnalyzer
        if SCIENTIFIC_ANALYZER_AVAILABLE:
            try:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                self.analyzer = ScientificAnalyzer(api_key=api_key)
                st.session_state['analyzer_status'] = "✅ ScientificAnalyzer inicializado"
            except Exception as e:
                st.session_state['analyzer_status'] = f"⚠️ Erro na inicialização: {e}"
        else:
            st.session_state['analyzer_status'] = "⚠️ ScientificAnalyzer não disponível - modo demonstração"

    def process_sample_data(self) -> Dict[str, Any]:
        """Processa dados de amostra para demonstração"""
        if not self.analyzer:
            # Modo demonstração sem ScientificAnalyzer
            return self._create_demo_data()

        try:
            # Usar dados de teste se disponível
            test_data_path = "data/controlled_test_100.csv"
            if Path(test_data_path).exists():
                results = self.analyzer.analyze_dataset(test_data_path)
                self.latest_results = results
                return results
            else:
                # Dados sintéticos para demonstração acadêmica
                sample_data = pd.DataFrame({
                    'text': [
                        "Análise do discurso político brasileiro contemporâneo.",
                        "Estudo sobre polarização democrática nas redes sociais.",
                        "Investigação de padrões autoritários na comunicação digital.",
                        "Pesquisa sobre erosão democrática no Brasil.",
                        "Análise temporal do discurso extremista."
                    ],
                    'timestamp': pd.date_range('2019-01-01', periods=5, freq='Y'),
                    'source': ['Telegram', 'Twitter', 'Telegram', 'WhatsApp', 'Telegram']
                })

                results = self.analyzer.analyze_dataset(sample_data)
                self.latest_results = results
                return results

        except Exception as e:
            return {"error": str(e)}

    def _create_demo_data(self) -> Dict[str, Any]:
        """Cria dados de demonstração acadêmica sem ScientificAnalyzer"""
        # Dados sintéticos para demonstração
        demo_data = pd.DataFrame({
            'text': [
                "Análise do discurso político brasileiro contemporâneo.",
                "Estudo sobre polarização democrática nas redes sociais.",
                "Investigação de padrões autoritários na comunicação digital.",
                "Pesquisa sobre erosão democrática no Brasil.",
                "Análise temporal do discurso extremista."
            ],
            'timestamp': pd.date_range('2019-01-01', periods=5, freq='Y'),
            'source': ['Telegram', 'Twitter', 'Telegram', 'WhatsApp', 'Telegram'],
            # Colunas científicas sintéticas para demonstração
            'political_category': ['direita', 'centro', 'esquerda', 'centro-direita', 'extrema-direita'],
            'political_confidence': [0.85, 0.72, 0.91, 0.67, 0.94],
            'sentiment_category': ['positivo', 'neutro', 'negativo', 'positivo', 'negativo'],
            'sentiment_score': [0.7, 0.1, -0.6, 0.5, -0.8],
            'linguistic_complexity': [7.2, 6.5, 8.1, 5.9, 9.3],
            'pos_nouns_count': [12, 8, 15, 7, 18],
            'named_entities_count': [3, 2, 5, 1, 6],
            'topic_main': ['política', 'democracia', 'autoritarismo', 'eleições', 'extremismo'],
            'topic_confidence': [0.88, 0.75, 0.93, 0.69, 0.96],
            'tfidf_top_terms': ['política,brasil', 'democracia,social', 'autoritário,digital', 'erosão,brasil', 'extremista,temporal'],
            'cluster_id': [1, 2, 1, 2, 1],
            'semantic_similarity': [0.82, 0.65, 0.89, 0.71, 0.95]
        })

        self.latest_results = {
            'data': demo_data,
            'stages_completed': 12,
            'enabled_stages': ['political_analysis', 'sentiment_analysis', 'linguistic_processing', 'topic_modeling'],
            'stages_executed': ['political_analysis', 'sentiment_analysis', 'linguistic_processing', 'topic_modeling'],
            'total_records': len(demo_data),
            'total_columns': len(demo_data.columns),
            'text_column_used': 'text',
            'stats': {
                'api_calls': 0,
                'heuristic_calls': 12,
                'errors': 0,
                'processed_records': 5
            }
        }

        return self.latest_results

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da análise científica mais recente"""
        if not self.latest_results:
            return {
                'total_records': 0,
                'stages_completed': 0,
                'total_columns': 0,
                'api_calls': 0,
                'heuristic_calls': 0
            }

        return {
            'total_records': self.latest_results.get('total_records', 0),
            'stages_completed': self.latest_results.get('stages_completed', 0),
            'total_columns': self.latest_results.get('total_columns', 0),
            'enabled_stages': len(self.latest_results.get('enabled_stages', [])),
            'text_column': self.latest_results.get('text_column_used', 'N/A'),
            'stats': self.latest_results.get('stats', {})
        }

    def get_processed_dataframe(self) -> Optional[pd.DataFrame]:
        """Retorna DataFrame processado com todas as colunas científicas"""
        if self.latest_results and 'data' in self.latest_results:
            return self.latest_results['data']
        return None

    def get_scientific_columns(self) -> List[str]:
        """Retorna lista de colunas científicas geradas"""
        df = self.get_processed_dataframe()
        if df is not None:
            # Filtrar colunas científicas (excluir originais)
            original_cols = {'text', 'timestamp', 'source', 'message', 'body', 'content'}
            return [col for col in df.columns if col not in original_cols]
        return []

def render_sidebar():
    """Renderiza o menu lateral acadêmico"""
    st.sidebar.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)

    st.sidebar.markdown("### 📚 Menu de Navegação")

    # Menu de navegação
    if st.sidebar.button("🏠 Home", key="home_btn"):
        st.session_state.page = "home"

    if st.sidebar.button("🔬 Pesquisa", key="research_btn"):
        st.session_state.page = "pesquisa"

    if st.sidebar.button("📊 Dados", key="data_btn"):
        st.session_state.page = "dados"

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Informações do projeto
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Informações")
    st.sidebar.info("""
    **digiNEV v5.1.0**

    Análise do Discurso Político Brasileiro

    Dataset: Mensagens Telegram (2019-2023)

    Foco: Autoritarismo e erosão democrática
    """)

def render_home_page():
    """Renderiza a página inicial - apenas texto descritivo"""

    # Cabeçalho principal
    st.markdown("""
    <div class="header-academic">
        <h1>digiNEV</h1>
        <h3>Análise Digital do Discurso Político Brasileiro</h3>
        <p>Pesquisa Acadêmica sobre Autoritarismo e Erosão Democrática</p>
    </div>
    """, unsafe_allow_html=True)

    # Conteúdo principal - apenas texto
    st.markdown("""
    <div class="content-card">
        <h2 class="title-section">Sobre a Pesquisa</h2>

        <div class="academic-text">
            O projeto <strong>digiNEV</strong> (Digital Network and Electoral Violence) representa uma iniciativa acadêmica
            pioneira na análise computacional do discurso político brasileiro contemporâneo. Desenvolvido para investigar
            os padrões de comunicação política digital e seus impactos na democracia brasileira, o projeto utiliza
            técnicas avançadas de processamento de linguagem natural e análise de redes sociais.
        </div>

        <div class="highlight-box">
            <h3>Objetivo Central</h3>
            <p>Compreender como o discurso político digital influencia processos democráticos
            e identificar padrões de autoritarismo emergente na comunicação política brasileira.</p>
        </div>

        <h3 class="title-section">Metodologia de Pesquisa</h3>

        <div class="academic-text">
            A pesquisa emprega uma abordagem metodológica multidisciplinar, combinando técnicas quantitativas
            e qualitativas para análise de grandes volumes de dados textuais. O corpus de análise compreende
            mensagens coletadas de canais públicos do Telegram entre 2019 e 2023, período que abrange marcos
            significativos da política brasileira contemporânea.
        </div>

        <div class="academic-list">
            <h4><strong>Dimensões de Análise:</strong></h4>
            <ul>
                <li><strong>Análise Política:</strong> Classificação ideológica do discurso em seis categorias (extrema-direita → esquerda)</li>
                <li><strong>Análise Temporal:</strong> Evolução dos padrões discursivos ao longo do período estudado</li>
                <li><strong>Análise de Sentimento:</strong> Identificação de tonalidades emocionais e intensidade do discurso</li>
                <li><strong>Análise de Redes:</strong> Mapeamento de coordenação e influência entre canais</li>
                <li><strong>Análise Semântica:</strong> Extração de tópicos e temas predominantes</li>
            </ul>
        </div>

        <h3 class="title-section">Fundamentação Teórica</h3>

        <div class="academic-text">
            O projeto fundamenta-se em três pilares teóricos principais: a teoria da erosão democrática
            (Levitsky & Ziblatt, 2018), estudos sobre autoritarismo competitivo (Diamond, 2008) e
            análise de discurso político digital (Van Dijk, 2006). Esta base teórica permite uma
            compreensão aprofundada dos fenômenos observados nos dados coletados.
        </div>

        <div class="citation">
            "A análise computacional do discurso político oferece insights únicos sobre como
            as dinâmicas democráticas se manifestam no ambiente digital contemporâneo."
            <br><em>— Fundamentação metodológica do projeto digiNEV</em>
        </div>

        <h3 class="title-section">Relevância Acadêmica</h3>

        <div class="academic-text">
            Esta pesquisa contribui para o campo emergente da ciência política computacional,
            oferecendo uma perspectiva empírica sobre os desafios enfrentados pela democracia
            brasileira na era digital. Os resultados fornecem subsídios para a compreensão
            de fenômenos políticos contemporâneos e para o desenvolvimento de estratégias
            de fortalecimento democrático.
        </div>

        <div class="highlight-box">
            <h4>Período de Análise: 2019-2023</h4>
            <p>Governo Bolsonaro, Pandemia de COVID-19, Eleições 2022</p>
        </div>

        <h3 class="title-section">Considerações Éticas</h3>

        <div class="academic-text">
            Toda a coleta e análise de dados foi conduzida respeitando princípios éticos de pesquisa,
            utilizando exclusivamente dados públicos disponibilizados pelos próprios usuários em
            canais abertos do Telegram. A pesquisa foi desenvolvida com foco acadêmico, visando
            contribuir para o entendimento científico dos fenômenos políticos digitais.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer acadêmico
    st.markdown("""
    <div class="footer-academic">
        <p><strong>digiNEV v5.1.0</strong> | Projeto de Pesquisa Acadêmica |
        Análise do Discurso Político Brasileiro | 2024</p>
    </div>
    """, unsafe_allow_html=True)

def render_research_page():
    """Renderiza a página de pesquisa com resultados acadêmicos"""

    st.markdown('<h1 class="title-main">🔬 Resultados da Pesquisa</h1>', unsafe_allow_html=True)

    # Exibir status do ScientificAnalyzer
    if 'analyzer_status' in st.session_state:
        if "✅" in st.session_state['analyzer_status']:
            st.success(st.session_state['analyzer_status'])
        else:
            st.warning(st.session_state['analyzer_status'])

    data_manager = ScientificDataManager()

    # Botão para processar dados de amostra
    if st.button("🔬 Executar Análise Científica", key="run_analysis"):
        with st.spinner("Executando análise científica..."):
            results = data_manager.process_sample_data()
            if "error" in results:
                st.error(f"Erro na análise: {results['error']}")
            else:
                st.success("Análise científica concluída com sucesso!")

    stats = data_manager.get_analysis_stats()

    # Métricas acadêmicas em grid responsivo
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-academic">
            <div class="metric-value">{stats.get('total_records', 0)}</div>
            <div class="metric-label">Registros Analisados</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-academic">
            <div class="metric-value">{stats.get('stages_completed', 0)}</div>
            <div class="metric-label">Estágios Científicos</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-academic">
            <div class="metric-value">{stats.get('total_columns', 0)}</div>
            <div class="metric-label">Colunas Geradas</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        analyzer_stats = stats.get('stats', {})
        api_calls = analyzer_stats.get('api_calls', 0)
        heuristic_calls = analyzer_stats.get('heuristic_calls', 0)
        total_calls = api_calls + heuristic_calls
        st.markdown(f"""
        <div class="metric-academic">
            <div class="metric-value">{total_calls}</div>
            <div class="metric-label">Operações AI/Heurísticas</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Fechar grid-container

    # Mostrar colunas científicas geradas
    scientific_columns = data_manager.get_scientific_columns()
    if scientific_columns:
        st.markdown('<h2 class="title-section">Colunas Científicas Geradas</h2>', unsafe_allow_html=True)

        # Organizar colunas por categoria
        column_categories = {
            'Análise Política': [col for col in scientific_columns if 'political' in col.lower()],
            'Análise de Sentimento': [col for col in scientific_columns if any(word in col.lower() for word in ['sentiment', 'emotion'])],
            'Análise Linguística': [col for col in scientific_columns if any(word in col.lower() for word in ['linguistic', 'pos_', 'named_entities'])],
            'Análise Semântica': [col for col in scientific_columns if any(word in col.lower() for word in ['topic', 'tfidf', 'cluster', 'semantic'])],
            'Análise Temporal': [col for col in scientific_columns if any(word in col.lower() for word in ['temporal', 'time', 'date'])],
            'Análise de Rede': [col for col in scientific_columns if any(word in col.lower() for word in ['network', 'centrality', 'degree'])],
            'Outras Análises': []
        }

        # Categorizar colunas não categorizadas
        categorized = set()
        for category, columns in column_categories.items():
            categorized.update(columns)

        column_categories['Outras Análises'] = [col for col in scientific_columns if col not in categorized]

        # Exibir categorias com colunas
        for category, columns in column_categories.items():
            if columns:
                with st.expander(f"📊 {category} ({len(columns)} colunas)"):
                    for col in columns:
                        st.write(f"• `{col}`")

    # Conteúdo acadêmico
    st.markdown("""
    <div class="content-card">
        <h2 class="title-section">Principais Achados</h2>

        <div class="academic-text">
            A análise computacional dos dados revelou padrões significativos no discurso político
            brasileiro durante o período estudado. Os resultados indicam uma polarização crescente
            da comunicação política digital, com implicações importantes para a qualidade do
            debate democrático.
        </div>

        <h3>Classificação Política</h3>
        <div class="academic-text">
            O sistema de classificação política desenvolvido identificou seis categorias principais
            de posicionamento ideológico, permitindo mapear a distribuição do espectro político
            nas comunicações analisadas.
        </div>

        <h3>Análise Temporal</h3>
        <div class="academic-text">
            A evolução temporal dos padrões discursivos revela momentos de intensificação
            da polarização, correlacionados com eventos políticos significativos do período.
        </div>

        <h3>Coordenação de Redes</h3>
        <div class="academic-text">
            A análise de redes identificou padrões de coordenação entre diferentes canais,
            sugerindo estratégias organizadas de disseminação de conteúdo político.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_data_page():
    """Renderiza a página de dados com visualizações científicas"""

    st.markdown('<h1 class="title-main">📊 Análise dos Dados Científicos</h1>', unsafe_allow_html=True)

    data_manager = ScientificDataManager()
    df = data_manager.get_processed_dataframe()

    if df is None or df.empty:
        st.info("💡 Execute a análise científica na página 'Pesquisa' para gerar visualizações.")
        return

    st.success(f"✅ Dataset processado: {len(df)} registros, {len(df.columns)} colunas")

    # Tabs para diferentes tipos de análise
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏛️ Análise Política",
        "😊 Análise de Sentimento",
        "📝 Análise Linguística",
        "🔬 Análise Semântica",
        "📈 Métricas Gerais"
    ])

    with tab1:
        render_political_analysis(df)

    with tab2:
        render_sentiment_analysis(df)

    with tab3:
        render_linguistic_analysis(df)

    with tab4:
        render_semantic_analysis(df)

    with tab5:
        render_general_metrics(df)

def render_political_analysis(df: pd.DataFrame):
    """Renderiza visualizações de análise política"""
    st.markdown('<h2 class="title-section">Distribuição Política</h2>', unsafe_allow_html=True)

    # Buscar colunas relacionadas à política
    political_cols = [col for col in df.columns if 'political' in col.lower()]

    if not political_cols:
        st.warning("Nenhuma coluna de análise política encontrada.")
        return

    for col in political_cols:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            # Visualização categórica
            value_counts = df[col].value_counts()

            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribuição: {col}",
                color_discrete_sequence=[CORES['primaria']]
            )
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Frequência",
                template="plotly_white",
                title_font_color=CORES['primaria'],
                title_font_size=16,
                font_family="Open Sans"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_sentiment_analysis(df: pd.DataFrame):
    """Renderiza visualizações de análise de sentimento"""
    st.markdown('<h2 class="title-section">Análise de Sentimento</h2>', unsafe_allow_html=True)

    # Buscar colunas relacionadas ao sentimento
    sentiment_cols = [col for col in df.columns if any(word in col.lower() for word in ['sentiment', 'emotion', 'feeling'])]

    if not sentiment_cols:
        st.warning("Nenhuma coluna de análise de sentimento encontrada.")
        return

    for col in sentiment_cols:
        if df[col].dtype == 'object':
            # Gráfico de pizza para categorias
            value_counts = df[col].value_counts()

            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribuição: {col}",
                color_discrete_sequence=[CORES['primaria'], CORES['cinza_medio'], CORES['hover'], CORES['cinza_claro']]
            )
            fig.update_layout(
                template="plotly_white",
                title_font_color=CORES['primaria'],
                title_font_size=16,
                font_family="Open Sans"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif df[col].dtype in ['float64', 'int64']:
            # Histograma para valores numéricos
            fig = px.histogram(
                df, x=col,
                title=f"Distribuição: {col}",
                color_discrete_sequence=[CORES['primaria']]
            )
            fig.update_layout(
                template="plotly_white",
                title_font_color=CORES['primaria'],
                title_font_size=16,
                font_family="Open Sans"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_linguistic_analysis(df: pd.DataFrame):
    """Renderiza visualizações de análise linguística"""
    st.markdown('<h2 class="title-section">Análise Linguística</h2>', unsafe_allow_html=True)

    # Buscar colunas linguísticas
    linguistic_cols = [col for col in df.columns if any(word in col.lower() for word in ['linguistic', 'pos_', 'named_entities', 'language'])]

    if not linguistic_cols:
        st.warning("Nenhuma coluna de análise linguística encontrada.")
        return

    # Mostrar estatísticas linguísticas
    col1, col2 = st.columns(2)

    with col1:
        for col in linguistic_cols[:len(linguistic_cols)//2]:
            if df[col].dtype in ['float64', 'int64']:
                st.metric(
                    label=col,
                    value=f"{df[col].mean():.2f}",
                    delta=f"±{df[col].std():.2f}"
                )

    with col2:
        for col in linguistic_cols[len(linguistic_cols)//2:]:
            if df[col].dtype in ['float64', 'int64']:
                st.metric(
                    label=col,
                    value=f"{df[col].mean():.2f}",
                    delta=f"±{df[col].std():.2f}"
                )

def render_semantic_analysis(df: pd.DataFrame):
    """Renderiza visualizações de análise semântica"""
    st.markdown('<h2 class="title-section">Análise Semântica</h2>', unsafe_allow_html=True)

    # Buscar colunas semânticas
    semantic_cols = [col for col in df.columns if any(word in col.lower() for word in ['topic', 'tfidf', 'cluster', 'semantic', 'similarity'])]

    if not semantic_cols:
        st.warning("Nenhuma coluna de análise semântica encontrada.")
        return

    for col in semantic_cols:
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            # Gráfico de barras para categorias semânticas
            value_counts = df[col].value_counts().head(10)  # Top 10

            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top 10: {col}",
                color_discrete_sequence=[CORES['primaria']]
            )
            fig.update_layout(
                xaxis_title="Frequência",
                yaxis_title=col,
                template="plotly_white",
                title_font_color=CORES['primaria'],
                title_font_size=16,
                font_family="Open Sans"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_general_metrics(df: pd.DataFrame):
    """Renderiza métricas gerais do dataset"""
    st.markdown('<h2 class="title-section">Métricas Gerais do Dataset</h2>', unsafe_allow_html=True)

    # Grid de métricas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Registros", len(df))

    with col2:
        st.metric("Total de Colunas", len(df.columns))

    with col3:
        # Contagem de colunas científicas
        original_cols = {'text', 'timestamp', 'source', 'message', 'body', 'content'}
        scientific_cols = [col for col in df.columns if col not in original_cols]
        st.metric("Colunas Científicas", len(scientific_cols))

    with col4:
        # Completude média
        completeness = ((df.notna().sum() / len(df)) * 100).mean()
        st.metric("Completude Média", f"{completeness:.1f}%")

    # Heatmap de correlação (apenas colunas numéricas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        st.markdown('<h3 class="title-section">Correlação entre Variáveis Numéricas</h3>', unsafe_allow_html=True)

        correlation_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            correlation_matrix,
            title="Matriz de Correlação",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig.update_layout(
            template="plotly_white",
            title_font_color=CORES['primaria'],
            title_font_size=16,
            font_family="Open Sans"
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Função principal do dashboard"""

    # Inicializar estado da sessão
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # Renderizar sidebar
    render_sidebar()

    # Renderizar página baseada na seleção
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "pesquisa":
        render_research_page()
    elif st.session_state.page == "dados":
        render_data_page()

if __name__ == "__main__":
    main()