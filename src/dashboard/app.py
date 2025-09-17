"""
digiNEV Dashboard de Análise Exploratória: Interface para exploração de resultados do pipeline
Function: Dashboard modular focado na análise acadêmica de dados de discurso político brasileiro
Usage: Interface web para cientistas sociais explorarem padrões, sentimentos e insights
"""

import streamlit as st
import sys
import time
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
        padding: 0.5rem 0;
        color: var(--white);
        text-align: center;
        margin: 0;
        margin-top: -1rem;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        border-bottom: 3px solid var(--accent-orange);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--white);
        font-family: 'Roboto Condensed', 'Arial Narrow', sans-serif;
        letter-spacing: 1px;
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
        border-radius: 6px;
        font-weight: 600;
        font-family: 'Roboto Condensed', sans-serif;
        transition: all 0.2s ease;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background: var(--primary-blue);
        color: var(--white);
    }
    
    /* Compactar o layout geral */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0.5rem;
        max-width: 95%;
    }
    
    .stMarkdown {
        margin-bottom: 0.5rem;
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
    
    /* Menu dropdown principal - estilização melhorada */
    .main-nav {
        background: var(--primary-blue);
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(27, 54, 93, 0.15);
    }
    
    /* Estilizar selectboxes do Streamlit */
    .stSelectbox > div > div {
        background: var(--primary-blue);
        color: var(--white);
        border-radius: 6px;
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: 600;
    }
    
    .stSelectbox > div > div > div {
        color: var(--white);
        font-family: 'Roboto Condensed', sans-serif;
    }
    
    .stSelectbox > div > div[aria-expanded="true"] {
        background: rgba(27, 54, 93, 0.9) !important;
    }
    
    .nav-dropdown {
        background: var(--primary-blue);
        color: var(--white);
        border: none;
        padding: 0.6rem 1.2rem;
        margin: 0 0.3rem;
        border-radius: 6px;
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .nav-dropdown:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .dropdown-content {
        background: rgba(27, 54, 93, 0.8);
        border-radius: 6px;
        margin-top: 0.3rem;
        padding: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .dropdown-item {
        background: transparent;
        color: var(--white);
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
        text-align: left;
        cursor: pointer;
        font-family: 'Roboto Condensed', sans-serif;
        transition: background 0.2s ease;
    }
    
    .dropdown-item:hover {
        background: rgba(255, 255, 255, 0.1);
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
    
    /* Fonte condensada global - menos altura, mais largura */
    .stApp, body, html {
        font-family: 'Roboto Condensed', 'Arial Narrow', 'Liberation Sans Narrow', 'Helvetica Neue Condensed', 'Arial', sans-serif;
        font-stretch: condensed;
    }
    
    /* Hero section na home */
    .hero-section {
        background: linear-gradient(135deg, var(--primary-blue) 0%, rgba(27, 54, 93, 0.8) 100%);
        color: var(--white);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%), 
                    linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%),
                    linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.1) 75%), 
                    linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.1) 75%);
        background-size: 30px 30px;
        background-position: 0 0, 0 15px, 15px -15px, -15px 0px;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .compact-section {
        background: var(--white);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Rodapé */
    .footer {
        background: var(--light-gray);
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        font-size: 0.8rem;
        color: var(--neutral-gray);
        font-family: 'Roboto Condensed', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

class DigiNEVDashboard:
    """Dashboard principal para análise exploratória dos resultados digiNEV"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        self.project_root = project_root
        
        # Inicializar estado da sessão - começar na home
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'geral_home'
        
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
        self._render_main_nav()
        self._render_main_content()
        self._render_footer()
    
    def _render_header(self):
        """Cabeçalho alargado e rente à moldura"""
        st.markdown("""
        <div class="main-header">
            <div class="hero-content">
                <h1>digiNEV | Monitor do Discurso Digital</h1>
                <p>Análise de Dados Políticos Brasileiros</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 📱 INFORMAÇÃO IMPORTANTE DE ACESSO - Resolução do problema do usuário
        st.info("📱 **Como acessar este dashboard**: Use o botão 'Preview' do Replit ou a URL gerada automaticamente. Não tente acessar http://0.0.0.0:5000 diretamente (isso resulta em 'acesso negado').")
    
    def _render_main_nav(self):
        """Menu principal com dropdown corrigido - 5 opções"""
        st.markdown('<div class="main-nav">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 1. GERAL (home/overview) - com navegação corrigida
        with col1:
            geral_options = ["Home", "Informações", "Amostra", "Estrutura"]
            
            # Determinar índice atual baseado na página
            geral_index = 0
            if st.session_state.current_page == 'geral_info':
                geral_index = 1
            elif st.session_state.current_page == 'geral_amostra':
                geral_index = 2
            elif st.session_state.current_page == 'geral_estrutura':
                geral_index = 3
            
            geral_selected = st.selectbox("Geral", geral_options, index=geral_index, key="nav_geral")
            
            # Só atualizar se mudou
            if geral_selected != geral_options[geral_index]:
                if geral_selected == "Home":
                    st.session_state.current_page = 'geral_home'
                elif geral_selected == "Informações":
                    st.session_state.current_page = 'geral_info'
                elif geral_selected == "Amostra":
                    st.session_state.current_page = 'geral_amostra'
                elif geral_selected == "Estrutura":
                    st.session_state.current_page = 'geral_estrutura'
                st.rerun()
        
        # 2. CENÁRIO
        with col2:
            cenario_options = ["Temporal", "Domínios", "Hashtags"]
            cenario_index = 0
            if st.session_state.current_page == 'domains':
                cenario_index = 1
            elif st.session_state.current_page == 'hashtags':
                cenario_index = 2
            
            cenario_selected = st.selectbox("Cenário", cenario_options, index=cenario_index, key="nav_cenario")
            
            if cenario_selected != cenario_options[cenario_index]:
                if cenario_selected == "Temporal":
                    st.session_state.current_page = 'temporal'
                elif cenario_selected == "Domínios":
                    st.session_state.current_page = 'domains'
                elif cenario_selected == "Hashtags":
                    st.session_state.current_page = 'hashtags'
                st.rerun()
        
        # 3. TEMÁTICAS
        with col3:
            tematicas_options = ["Modeling", "TF-IDF", "Clustering"]
            tematicas_index = 0
            if st.session_state.current_page == 'tfidf':
                tematicas_index = 1
            elif st.session_state.current_page == 'clustering':
                tematicas_index = 2
            
            tematicas_selected = st.selectbox("Temáticas", tematicas_options, index=tematicas_index, key="nav_tematicas")
            
            if tematicas_selected != tematicas_options[tematicas_index]:
                if tematicas_selected == "Modeling":
                    st.session_state.current_page = 'modeling'
                elif tematicas_selected == "TF-IDF":
                    st.session_state.current_page = 'tfidf'
                elif tematicas_selected == "Clustering":
                    st.session_state.current_page = 'clustering'
                st.rerun()
        
        # 4. DISCURSOS
        with col4:
            discursos_options = ["Sentimentos", "Política", "Redes"]
            discursos_index = 0
            if st.session_state.current_page == 'political':
                discursos_index = 1
            elif st.session_state.current_page == 'networks':
                discursos_index = 2
            
            discursos_selected = st.selectbox("Discursos", discursos_options, index=discursos_index, key="nav_discursos")
            
            if discursos_selected != discursos_options[discursos_index]:
                if discursos_selected == "Sentimentos":
                    st.session_state.current_page = 'sentiment'
                elif discursos_selected == "Política":
                    st.session_state.current_page = 'political'
                elif discursos_selected == "Redes":
                    st.session_state.current_page = 'networks'
                st.rerun()
        
        # 5. SIGNIFICADOS  
        with col5:
            significados_options = ["Tópicos", "Qualitativa", "Semântica"]
            significados_index = 0
            if st.session_state.current_page == 'qualitative':
                significados_index = 1
            elif st.session_state.current_page == 'semantic':
                significados_index = 2
            
            significados_selected = st.selectbox("Significados", significados_options, index=significados_index, key="nav_significados")
            
            if significados_selected != significados_options[significados_index]:
                if significados_selected == "Tópicos":
                    st.session_state.current_page = 'topics'
                elif significados_selected == "Qualitativa":
                    st.session_state.current_page = 'qualitative'  
                elif significados_selected == "Semântica":
                    st.session_state.current_page = 'semantic'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_footer(self):
        """Rodapé com informações de criação"""
        st.markdown("""
        <div class="footer">
            <p><strong>digiNEV Monitor</strong> | Desenvolvido em 2024 para Análise de Discurso Digital Brasileiro</p>
            <p>Sistema de monitoramento e análise de dados políticos | Universidade Federal do Rio Grande do Sul</p>
        </div>
        """, unsafe_allow_html=True)
    
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
            # GERAL - Páginas do dropdown Geral
            if current_page == 'geral_home':
                self._render_geral_home()
            elif current_page == 'geral_info':
                self._render_geral_info()
            elif current_page == 'geral_amostra':
                self._render_geral_amostra()
            elif current_page == 'geral_estrutura':
                self._render_geral_estrutura()
            
            # CENÁRIO
            elif current_page == 'temporal':
                self._render_analysis_page("Análise Temporal", 14)
            elif current_page == 'domains':
                self._render_analysis_page("Análise de Domínios", 13)
            elif current_page == 'hashtags':
                self._render_analysis_page("Análise de Hashtags", 12)
            
            # TEMÁTICAS  
            elif current_page == 'modeling':
                self._render_analysis_page("Topic Modeling", 9)
            elif current_page == 'tfidf':
                self._render_analysis_page("Análise TF-IDF", 10)
            elif current_page == 'clustering':
                self._render_analysis_page("Análise de Clustering", 11)
            
            # DISCURSOS
            elif current_page == 'sentiment':
                self._render_sentiment_page()
            elif current_page == 'political':
                self._render_political_page()
            elif current_page == 'networks':
                self._render_analysis_page("Análise de Redes", 15)
            
            # SIGNIFICADOS
            elif current_page == 'topics':
                self._render_topics_page()
            elif current_page == 'qualitative':
                self._render_analysis_page("Análise Qualitativa", 16)
            elif current_page == 'semantic':
                self._render_analysis_page("Análise Semântica", 19)
            
            else:
                self._render_geral_home()  # Fallback para home
                
        except Exception as e:
            st.error(f"Erro ao carregar página: {e}")
            st.info("Execute o pipeline para gerar dados de análise.")
    
    def _render_geral_home(self):
        """Home com layout compacto central"""
        
        # Layout compacto central - 3 colunas para centralizar
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            # Caixa de carregamento central
            st.markdown('<div class="compact-section">', unsafe_allow_html=True)
            st.markdown("#### Carregamento de Dataset")
            
            col_upload, col_exec = st.columns([2, 1])
            with col_upload:
                uploaded_file = st.file_uploader(
                    "Selecionar arquivo CSV", 
                    type=['csv'],
                    help="Dados do Telegram com texto, data e metadados",
                    label_visibility="collapsed"
                )
            
            with col_exec:
                # 🚀 CORREÇÃO: Conectar ao sistema real de pipeline
                from src.dashboard.utils.pipeline_runner import get_pipeline_runner
                
                pipeline_runner = get_pipeline_runner()
                is_running = pipeline_runner.is_running()
                
                if not is_running:
                    if st.button("🚀 Executar Pipeline", type="primary", use_container_width=True):
                        if pipeline_runner.start_pipeline():
                            st.success("✅ Pipeline iniciado com sucesso!")
                            st.rerun()
                        else:
                            st.error("❌ Erro ao iniciar pipeline")
                else:
                    if st.button("⏹️ Parar Pipeline", type="secondary", use_container_width=True):
                        if pipeline_runner.stop_pipeline():
                            st.success("Pipeline interrompido")
                            st.rerun()
            
            # Barra de progresso REAL baseada no PipelineRunner
            st.markdown("#### Progresso do Processamento")
            
            progress_obj = pipeline_runner.get_progress()
            completion_pct = pipeline_runner.get_completion_percentage()
            current_stage_name = pipeline_runner.get_current_stage_name()
            
            # Verificar se arquivo foi carregado
            if uploaded_file is not None:
                # Salvar arquivo temporariamente
                temp_path = Path("data/uploads") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
                st.info("👆 Agora clique em 'Executar Pipeline' para processar os dados")
            
            # Barra de progresso real
            progress_bar = st.progress(completion_pct / 100.0)
            st.markdown(f"**{completion_pct:.1f}% concluído** - {current_stage_name}")
            
            # Status baseado na execução real
            if progress_obj.status.value == "running":
                st.info(f"⚡ Executando... Etapa atual: {current_stage_name}")
                # Auto-refresh durante execução
                time.sleep(2)
                st.rerun()
            elif progress_obj.status.value == "completed":
                st.success("🎉 Pipeline concluído com sucesso!")
            elif progress_obj.status.value == "error":
                st.error(f"❌ Erro na execução: {progress_obj.error_message or 'Erro desconhecido'}")
            else:
                st.info("⏳ Aguardando execução do pipeline")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_geral_info(self):
        """Informações gerais do dataset"""
        st.markdown("### Informações do Dataset")
        
        try:
            import pandas as pd
            dataset_path = self.project_root / "data" / "telegram_data.csv"
            
            if dataset_path.exists():
                df = pd.read_csv(dataset_path, nrows=1000)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Mensagens", f"{len(df):,}")
                
                with col2:
                    # Tentar detectar coluna de data
                    date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'data', 'timestamp', 'time'])]
                    if date_cols:
                        try:
                            date_col = date_cols[0]
                            df[date_col] = pd.to_datetime(df[date_col])
                            min_date = df[date_col].min().strftime('%Y-%m-%d')
                            max_date = df[date_col].max().strftime('%Y-%m-%d')
                            st.metric("Período", f"{min_date} a {max_date}")
                        except:
                            st.metric("Período", "Não identificado")
                    else:
                        st.metric("Período", "Sem coluna de data")
                
                with col3:
                    st.metric("Colunas", len(df.columns))
                
                with col4:
                    processed_files = len(list((self.project_root / "pipeline_outputs").glob("*.csv")))
                    st.metric("Etapas Processadas", processed_files)
                    
            else:
                st.warning("Dataset não encontrado. Carregue um arquivo CSV.")
                
        except Exception as e:
            st.error(f"Erro: {e}")
    
    def _render_geral_amostra(self):
        """Amostra dos dados"""
        st.markdown("### Amostra do Dataset")
        
        try:
            import pandas as pd
            dataset_path = self.project_root / "data" / "telegram_data.csv"
            
            if dataset_path.exists():
                df = pd.read_csv(dataset_path, nrows=50)  # Primeiras 50 linhas
                st.dataframe(df, use_container_width=True, height=600)
                
                st.info(f"Exibindo {len(df)} registros do dataset")
            else:
                st.warning("Dataset não encontrado.")
                
        except Exception as e:
            st.error(f"Erro ao carregar amostra: {e}")
    
    def _render_geral_estrutura(self):
        """Estrutura do dataset - colunas e tipos"""
        st.markdown("### Estrutura do Dataset")
        
        try:
            import pandas as pd
            dataset_path = self.project_root / "data" / "telegram_data.csv"
            
            if dataset_path.exists():
                df = pd.read_csv(dataset_path, nrows=1000)
                
                # Criar tabela de informações das colunas
                cols_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    null_count = df[col].isnull().sum()
                    unique_vals = df[col].nunique()
                    
                    cols_info.append({
                        'Coluna': col,
                        'Tipo': dtype,
                        'Não-Nulos': f"{non_null:,}",
                        'Nulos': f"{null_count:,}",
                        'Únicos': f"{unique_vals:,}",
                        'Completude': f"{(non_null/len(df)*100):.1f}%"
                    })
                
                structure_df = pd.DataFrame(cols_info)
                st.dataframe(structure_df, use_container_width=True, height=500)
                
                st.info(f"Dataset com {len(df.columns)} colunas analisadas")
            else:
                st.warning("Dataset não encontrado.")
                
        except Exception as e:
            st.error(f"Erro ao analisar estrutura: {e}")
    
    def _render_overview_page(self):
        """Renderiza página de visão geral reformulada"""
        st.subheader("Carregamento e Análise de Dados")
        
        # 1. CARREGAMENTO DE DADOS
        st.markdown("### 1. Carregamento de Dataset")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Carregar arquivo CSV", 
                type=['csv'],
                help="Formato: mensagens do Telegram com colunas texto, data, etc."
            )
        
        with col2:
            # 🚀 CORREÇÃO: Usar mesmo sistema real de pipeline do home  
            from src.dashboard.utils.pipeline_runner import get_pipeline_runner
            
            pipeline_runner = get_pipeline_runner()
            is_running = pipeline_runner.is_running()
            
            if not is_running:
                if st.button("🔄 Executar Pipeline", type="primary", use_container_width=True):
                    if pipeline_runner.start_pipeline():
                        st.success("✅ Pipeline iniciado com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Erro ao iniciar pipeline")
            else:
                st.button("⏹️ Pipeline Executando...", type="secondary", use_container_width=True, disabled=True)
        
        # 2. INFORMAÇÕES BÁSICAS DO DATASET
        st.markdown("### 2. Informações do Dataset")
        
        try:
            # Tentar carregar dados reais
            import pandas as pd
            dataset_path = self.project_root / "data" / "telegram_data.csv"
            
            if dataset_path.exists():
                df = pd.read_csv(dataset_path, nrows=1000)  # Amostra para performance
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total de Mensagens", f"{len(df):,}")
                
                with col2:
                    if 'timestamp' in df.columns or 'data' in df.columns or 'date' in df.columns:
                        date_col = 'timestamp' if 'timestamp' in df.columns else ('data' if 'data' in df.columns else 'date')
                        try:
                            df[date_col] = pd.to_datetime(df[date_col])
                            min_date = df[date_col].min().strftime('%Y-%m-%d')
                            max_date = df[date_col].max().strftime('%Y-%m-%d')
                            st.metric("📅 Período", f"{min_date} a {max_date}")
                        except:
                            st.metric("📅 Período", "Não identificado")
                    else:
                        st.metric("📅 Período", "Coluna de data não encontrada")
                
                with col3:
                    st.metric("📋 Colunas", len(df.columns))
                
                with col4:
                    processed_files = list((self.project_root / "pipeline_outputs").glob("*.csv"))
                    st.metric("✅ Etapas Processadas", len(processed_files))
                
                # 3. AMOSTRA DOS DADOS
                st.markdown("### 3. Amostra do Dataset")
                st.dataframe(df.head(10), use_container_width=True, height=300)
                
                # 4. LISTA DE COLUNAS
                st.markdown("### 4. Estrutura do Dataset")
                cols_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    cols_info.append({
                        'Coluna': col,
                        'Tipo': dtype,
                        'Valores Não-Nulos': non_null,
                        'Porcentagem': f"{(non_null/len(df)*100):.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(cols_info), use_container_width=True)
                
            else:
                st.warning("Dataset não encontrado. Faça upload de um arquivo CSV.")
                
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
        
        # 5. PROGRESSO DO PIPELINE
        st.markdown("### 5. Progresso do Processamento")
        
        # Lista de etapas esperadas
        expected_steps = [
            "01_dataset_stats", "02_data_cleaning", "03_text_preprocessing", 
            "04_language_detection", "05_political_analysis", "06_emotion_analysis",
            "07_toxicity_detection", "08_sentiment_analysis", "09_topic_modeling",
            "10_tfidf_analysis", "11_clustering_results", "12_hashtag_analysis",
            "13_domain_analysis", "14_temporal_analysis", "15_network_analysis",
            "16_qualitative_analysis", "17_quality_metrics", "18_topic_evolution",
            "19_semantic_analysis", "20_final_insights"
        ]
        
        # Verificar arquivos processados
        pipeline_dir = self.project_root / "pipeline_outputs"
        completed_steps = 0
        if pipeline_dir.exists():
            for step in expected_steps:
                if (pipeline_dir / f"{step}.csv").exists():
                    completed_steps += 1
        
        # Barra de progresso
        progress = completed_steps / len(expected_steps)
        st.progress(progress, text=f"Etapa {completed_steps}/{len(expected_steps)} concluída")
        
        # Estimativa de tempo (exemplo)
        if completed_steps > 0 and completed_steps < len(expected_steps):
            remaining_steps = len(expected_steps) - completed_steps
            estimated_time = remaining_steps * 2  # 2 minutos por etapa (estimativa)
            st.info(f"⏱️ Tempo estimado restante: {estimated_time} minutos")
        elif completed_steps == len(expected_steps):
            st.success("🎉 Pipeline concluído com sucesso!")
        else:
            st.info("💡 Execute o pipeline para iniciar o processamento")
    
    def _render_political_page(self):
        """Renderiza a página de análise política"""
        try:
            from dashboard.views.political_analysis import render_political_page
            render_political_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise Política")
    
    def _render_sentiment_page(self):
        """Renderiza a página de análise de sentimento"""
        try:
            from dashboard.views.sentiment_analysis import render_sentiment_page
            render_sentiment_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Análise de Sentimento")
    
    def _render_topics_page(self):
        """Renderiza a página de modelagem de tópicos"""
        try:
            from dashboard.views.topic_modeling import render_topics_page
            render_topics_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Modelagem de Tópicos")
    
    def _render_search_page(self):
        """Renderiza a página de busca semântica"""
        try:
            from dashboard.views.semantic_search import render_search_page
            render_search_page(self.data_loader)
        except ImportError:
            self._render_fallback_page("Busca Semântica")
    
    # Páginas essenciais mantidas - todas as outras removidas para simplicidade
    
    def _render_analysis_page(self, title: str, step_number: int):
        """Renderiza página de análise genérica baseada no step do pipeline"""
        st.subheader(f"{title} - Etapa {step_number:02d}")
        
        # Verificar se dados da etapa existem
        step_file = self.project_root / "pipeline_outputs" / f"{step_number:02d}_{title.lower().replace(' ', '_').replace('-', '_')}.csv"
        
        if step_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(step_file)
                
                st.success(f"✅ Dados da etapa {step_number} carregados com sucesso!")
                
                # Métricas básicas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Registros", f"{len(df):,}")
                with col2:
                    st.metric("📋 Colunas", len(df.columns))
                with col3:
                    st.metric("💾 Tamanho", f"{step_file.stat().st_size / 1024:.1f} KB")
                
                # Exibir amostra dos dados
                st.markdown("### Amostra dos Resultados")
                st.dataframe(df.head(20), use_container_width=True, height=400)
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {title}",
                    data=csv,
                    file_name=f"{title.lower().replace(' ', '_')}_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Erro ao carregar dados da etapa {step_number}: {e}")
        else:
            st.warning(f"Dados da etapa {step_number} ainda não foram processados.")
            st.info("💡 Execute o pipeline principal para gerar os dados desta análise.")
            
            # Placeholder para quando não há dados
            st.markdown(f"""
            ### {title}
            
            Esta seção apresentará:
            - Resultados específicos da etapa {step_number}
            - Visualizações e métricas relevantes
            - Dados exportáveis em formato CSV
            - Insights e conclusões da análise
            """)
    
    def _render_fallback_page(self, title: str):
        """Página genérica simplificada"""
        st.info(f"Execute o pipeline para gerar dados de {title.lower()}.")

# Executar aplicação
if __name__ == "__main__":
    dashboard = DigiNEVDashboard()
    dashboard.run()