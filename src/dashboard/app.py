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
    
    /* Fonte sóbria global - aplicada corretamente */
    .stApp, body, html {
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Indicador visual do menu ativo */
    .active-nav-indicator {
        color: var(--primary-blue);
        font-weight: 600;
        border-bottom: 2px solid var(--primary-blue);
        padding-bottom: 4px;
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
        """Renderiza menu temático organizado em grupos"""
        
        # Botão de Visão Geral sempre visível
        if st.button("📊 VISÃO GERAL", key="overview_main", type="primary", use_container_width=True):
            st.session_state.current_page = 'overview'
            st.rerun()
        
        st.markdown("---")
        
        # Grupos temáticos organizados
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 🕐 CENÁRIO")
            if st.button("Temporal (14)", key="temporal_14"):
                st.session_state.current_page = 'temporal'
                st.rerun()
            if st.button("Domínios (13)", key="domains_13"):
                st.session_state.current_page = 'domains'
                st.rerun()
            if st.button("Hashtags (12)", key="hashtags_12"):
                st.session_state.current_page = 'hashtags'
                st.rerun()
        
        with col2:
            st.markdown("### 🎯 TEMÁTICAS")
            if st.button("Modeling (09)", key="modeling_09"):
                st.session_state.current_page = 'modeling'
                st.rerun()
            if st.button("TF-IDF (10)", key="tfidf_10"):
                st.session_state.current_page = 'tfidf'
                st.rerun()
            if st.button("Clustering (11)", key="clustering_11"):
                st.session_state.current_page = 'clustering'
                st.rerun()
        
        with col3:
            st.markdown("### 💬 DISCURSOS")
            if st.button("Sentimentos (08)", key="sentiment_08"):
                st.session_state.current_page = 'sentiment'
                st.rerun()
            if st.button("Política (05)", key="political_05"):
                st.session_state.current_page = 'political'
                st.rerun()
            if st.button("Redes (15)", key="networks_15"):
                st.session_state.current_page = 'networks'
                st.rerun()
        
        with col4:
            st.markdown("### 🧠 SIGNIFICADOS")
            if st.button("Tópicos (18)", key="topics_18"):
                st.session_state.current_page = 'topics'
                st.rerun()
            if st.button("Qualitativa (16)", key="qualitative_16"):
                st.session_state.current_page = 'qualitative'
                st.rerun()
            if st.button("Semântica (19)", key="semantic_19"):
                st.session_state.current_page = 'semantic'
                st.rerun()
        
        st.markdown("---")
    
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
            # Página principal
            if current_page == 'overview':
                self._render_overview_page()
            
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
                self._render_overview_page()  # Fallback
                
        except Exception as e:
            st.error(f"Erro ao carregar página: {e}")
            st.info("Execute o pipeline para gerar dados de análise.")
    
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
            if st.button("🔄 Executar Pipeline", type="primary", use_container_width=True):
                st.info("Pipeline iniciado em background...")
        
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