"""
Dashboard de Análise de Dados - Monitor do Discurso Digital v5.0.0
============================================================

Dashboard focado exclusivamente na apresentação dos RESULTADOS das análises
de dados geradas pelos stages do pipeline. Apresenta insights, visualizações
e descobertas sobre o discurso político brasileiro nos dados do Telegram.

🎯 FOCO: Análise dos dados processados e insights gerados
📊 OBJETIVO: Visualizar resultados das análises de conteúdo político
🔍 ESCOPO: Dashboards analíticos, não monitoramento de pipeline
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
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Análise Política - Telegram Brasil",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para análise de dados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-highlight {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .category-political {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    
    .category-sentiment {
        background-color: #f3e5f5;
        border-left: 4px solid #7b1fa2;
    }
    
    .category-discourse {
        background-color: #e8f5e8;
        border-left: 4px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)


class DataAnalysisDashboard:
    """Dashboard principal para análise dos dados processados"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        self.project_root = Path(__file__).parent.parent.parent
        
        # Caminhos para dados processados
        self.data_path = self.project_root / "data/interim/sample_dataset_v495_19_pipeline_validated.csv"
        self.validation_report_path = self.project_root / "logs/pipeline/validation_report_20250611_150026.json"
        
        # Caminhos para dados antes/depois da limpeza
        self.data_original_path = self.project_root / "data/interim/sample_dataset_v495_01_chunked.csv"
        self.data_deduplicated_path = self.project_root / "data/interim/sample_dataset_v495_03_deduplicated.csv"
        self.pre_cleaning_stats_path = self.project_root / "data/interim/sample_dataset_v495_01_chunked_02_encoding_validated_03_deduplicated_04_feature_validated_04b_pre_cleaning_stats.json"
        self.post_cleaning_stats_path = self.project_root / "data/interim/sample_dataset_v495_01_chunked_02_encoding_validated_03_deduplicated_04_feature_validated_05_political_analyzed_06_text_cleaned_06b_post_cleaning_stats.json"
        
        # Carregar dados
        self.df = self._load_dataset()
        self.validation_data = self._load_validation_report()
        
        # Carregar dados antes/depois para comparação
        self.df_original = self._load_dataset_from_path(self.data_original_path)
        self.df_deduplicated = self._load_dataset_from_path(self.data_deduplicated_path)
        self.pre_cleaning_stats = self._load_json_file(self.pre_cleaning_stats_path)
        self.post_cleaning_stats = self._load_json_file(self.post_cleaning_stats_path)
    
    def _load_dataset(self) -> Optional[pd.DataFrame]:
        """Carrega o dataset processado"""
        try:
            if self.data_path.exists():
                df = pd.read_csv(self.data_path, sep=';', quoting=1)
                # Converter datetime
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            return None
        except Exception as e:
            st.error(f"Erro carregando dataset: {e}")
            return None
    
    def _load_validation_report(self) -> Optional[Dict]:
        """Carrega o relatório de validação"""
        try:
            if self.validation_report_path.exists():
                with open(self.validation_report_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Erro carregando relatório: {e}")
            return None
    
    def _load_dataset_from_path(self, path: Path) -> Optional[pd.DataFrame]:
        """Carrega dataset de um caminho específico"""
        try:
            if path.exists():
                df = pd.read_csv(path, sep=';', quoting=1)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            return None
        except Exception as e:
            st.error(f"Erro carregando dataset {path.name}: {e}")
            return None
    
    def _load_json_file(self, path: Path) -> Optional[Dict]:
        """Carrega arquivo JSON"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Erro carregando arquivo JSON {path.name}: {e}")
            return None
    
    def run(self):
        """Executa o dashboard principal"""
        self._render_header()
        self._render_navigation()
        
        if self.df is None:
            self._render_no_data_page()
            return
        
        # Menu de navegação
        page = st.session_state.get('current_page', 'overview')
        
        if page == 'overview':
            self._render_overview_page()
        elif page == 'political_analysis':
            self._render_political_analysis_page()
        elif page == 'sentiment_analysis':
            self._render_sentiment_analysis_page()
        elif page == 'discourse_analysis':
            self._render_discourse_analysis_page()
        elif page == 'temporal_analysis':
            self._render_temporal_analysis_page()
        elif page == 'linguistic_analysis':
            self._render_linguistic_analysis_page()
        elif page == 'clustering_analysis':
            self._render_clustering_analysis_page()
        elif page == 'network_analysis':
            self._render_network_analysis_page()
        elif page == 'comparative_analysis':
            self._render_comparative_analysis_page()
    
    def _render_header(self):
        """Renderiza o cabeçalho"""
        st.markdown('<div class="main-header">🏛️ Análise do Discurso Político Brasileiro</div>', unsafe_allow_html=True)
        st.markdown("### 📊 Pipeline de Limpeza e Transformação de Dados - Telegram (2019-2021)")
        
        # Métricas comparativas principais
        if self.df_original is not None and self.df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                original_count = len(self.df_original)
                st.metric("📥 Dados Originais", f"{original_count:,}")
            
            with col2:
                final_count = len(self.df)
                st.metric("🧹 Dados Finais", f"{final_count:,}")
            
            with col3:
                reduction_pct = ((original_count - final_count) / original_count) * 100 if original_count > 0 else 0
                st.metric("📉 Redução Total", f"{reduction_pct:.1f}%")
            
            with col4:
                stages_executed = 20
                st.metric("⚙️ Stages Executados", f"{stages_executed}")
        
        st.markdown("---")
    
    def _render_navigation(self):
        """Renderiza a navegação lateral"""
        with st.sidebar:
            st.header("🧭 Navegação")
            
            pages = {
                'overview': '📋 Visão Geral',
                'political_analysis': '🏛️ Análise Política',
                'sentiment_analysis': '😊 Análise de Sentimento',
                'discourse_analysis': '💬 Análise do Discurso',
                'temporal_analysis': '📅 Análise Temporal',
                'linguistic_analysis': '🔤 Análise Linguística',
                'clustering_analysis': '🔍 Análise de Agrupamentos',
                'network_analysis': '🌐 Análise de Redes',
                'comparative_analysis': '⚖️ Análise Comparativa'
            }
            
            for page_key, page_name in pages.items():
                if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            # Informações do dataset
            if self.df is not None:
                st.markdown("---")
                st.subheader("📊 Estatísticas")
                st.write(f"**Registros:** {len(self.df):,}")
                st.write(f"**Colunas:** {len(self.df.columns)}")
                
                # Top categorias políticas
                if 'political_category' in self.df.columns:
                    top_categories = self.df['political_category'].value_counts().head(3)
                    st.write("**Top Categorias:**")
                    for cat, count in top_categories.items():
                        st.write(f"• {cat}: {count}")
    
    def _render_no_data_page(self):
        """Página quando não há dados"""
        st.error("📊 Nenhum dado disponível para análise")
        st.markdown("""
        ### Como gerar dados para análise:
        
        1. **Execute o pipeline completo:**
        ```bash
        poetry run python run_pipeline.py
        ```
        
        2. **Ou execute stages específicos:**
        ```bash
        poetry run python src/main.py
        ```
        
        3. **Aguarde o processamento** dos 20 stages do pipeline
        
        4. **Retorne ao dashboard** para visualizar as análises
        """)
    
    def _render_overview_page(self):
        """Página de visão geral com comparação antes/depois da limpeza"""
        st.header("📋 Análise Comparativa: Antes vs Depois da Limpeza de Dados")
        
        # Seção 1: Comparação de Volume de Mensagens
        st.subheader("📊 1. Volume de Mensagens: Original vs Deduplicated")
        
        if self.df_original is not None and self.df_deduplicated is not None:
            col1, col2, col3 = st.columns(3)
            
            original_count = len(self.df_original)
            deduplicated_count = len(self.df_deduplicated)
            reduction_percentage = ((original_count - deduplicated_count) / original_count) * 100 if original_count > 0 else 0
            
            with col1:
                st.metric("📥 Mensagens Originais", f"{original_count:,}")
            
            with col2:
                st.metric("🧹 Após Deduplicação", f"{deduplicated_count:,}")
            
            with col3:
                st.metric("📉 Redução", f"{reduction_percentage:.1f}%", f"-{original_count - deduplicated_count:,}")
            
            # Gráfico de comparação
            comparison_data = {
                'Etapa': ['Original', 'Deduplicated'],
                'Mensagens': [original_count, deduplicated_count],
                'Cor': ['#ff7f7f', '#7fbf7f']
            }
            
            fig_volume = px.bar(
                comparison_data,
                x='Etapa',
                y='Mensagens',
                title="Comparação do Volume de Mensagens",
                color='Etapa',
                color_discrete_map={'Original': '#ff7f7f', 'Deduplicated': '#7fbf7f'}
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Seção 2: Top 10 Hashtags - Antes vs Depois
        st.subheader("🏷️ 2. Top 10 Hashtags: Antes vs Depois")
        self._render_hashtags_comparison()
        
        # Seção 3: Top 10 Menções - Antes vs Depois  
        st.subheader("👥 3. Top 10 Menções: Antes vs Depois")
        self._render_mentions_comparison()
        
        # Seção 4: Top 10 Domínios - Antes vs Depois
        st.subheader("🌐 4. Top 10 Domínios: Antes vs Depois")
        self._render_domains_comparison()
        
        # Seção 5: Resumo das Transformações
        st.subheader("🔄 5. Resumo das Transformações do Pipeline")
        self._render_transformation_summary()
    
    def _generate_main_insights(self) -> List[str]:
        """Gera insights principais dos dados"""
        insights = []
        
        if 'political_category' in self.df.columns:
            top_category = self.df['political_category'].value_counts().index[0]
            top_percentage = (self.df['political_category'].value_counts().iloc[0] / len(self.df)) * 100
            insights.append(f"🏛️ **Categoria política dominante:** {top_category} ({top_percentage:.1f}% das mensagens)")
        
        if 'sentiment' in self.df.columns:
            sentiment_counts = self.df['sentiment'].value_counts()
            dominant_sentiment = sentiment_counts.index[0]
            sentiment_percentage = (sentiment_counts.iloc[0] / len(self.df)) * 100
            insights.append(f"😊 **Sentimento predominante:** {dominant_sentiment} ({sentiment_percentage:.1f}% das mensagens)")
        
        if 'discourse_type' in self.df.columns:
            discourse_counts = self.df['discourse_type'].value_counts()
            main_discourse = discourse_counts.index[0]
            discourse_percentage = (discourse_counts.iloc[0] / len(self.df)) * 100
            insights.append(f"💬 **Tipo de discurso principal:** {main_discourse} ({discourse_percentage:.1f}% das mensagens)")
        
        if 'text_length' in self.df.columns:
            avg_length = self.df['text_length'].mean()
            insights.append(f"📝 **Comprimento médio das mensagens:** {avg_length:.0f} caracteres")
        
        if 'cluster_name' in self.df.columns:
            unique_clusters = self.df['cluster_name'].nunique()
            insights.append(f"🔍 **Grupos temáticos identificados:** {unique_clusters} clusters semânticos distintos")
        
        return insights
    
    def _render_hashtags_comparison(self):
        """Comparação de hashtags antes vs depois"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏷️ Antes da Limpeza (Dados Originais)**")
                
                if self.pre_cleaning_stats and 'hashtag_analysis' in self.pre_cleaning_stats:
                    top_hashtags_before = self.pre_cleaning_stats['hashtag_analysis'].get('top_hashtags', {})
                    
                    if top_hashtags_before:
                        hashtags_df_before = pd.DataFrame([
                            {'Hashtag': k, 'Frequência': v} 
                            for k, v in list(top_hashtags_before.items())[:10]
                        ])
                        
                        fig_before = px.bar(
                            hashtags_df_before,
                            y='Hashtag',
                            x='Frequência',
                            orientation='h',
                            title="Top 10 Hashtags - Dados Originais",
                            color_discrete_sequence=['#ff7f7f']
                        )
                        st.plotly_chart(fig_before, use_container_width=True)
                    else:
                        st.info("Dados de hashtags originais não disponíveis")
                else:
                    st.info("Estatísticas de pré-limpeza não disponíveis")
            
            with col2:
                st.write("**🧹 Depois da Limpeza (Dados Processados)**")
                
                if self.df is not None and 'hashtag' in self.df.columns:
                    # Extrair hashtags dos dados processados
                    hashtags_after = []
                    for hashtag_field in self.df['hashtag'].dropna():
                        if hashtag_field and hashtag_field.strip():
                            hashtags_after.extend([h.strip() for h in str(hashtag_field).split(',') if h.strip()])
                    
                    if hashtags_after:
                        hashtag_counts_after = pd.Series(hashtags_after).value_counts().head(10)
                        
                        hashtags_df_after = pd.DataFrame({
                            'Hashtag': hashtag_counts_after.index,
                            'Frequência': hashtag_counts_after.values
                        })
                        
                        fig_after = px.bar(
                            hashtags_df_after,
                            y='Hashtag',
                            x='Frequência',
                            orientation='h',
                            title="Top 10 Hashtags - Dados Processados",
                            color_discrete_sequence=['#7fbf7f']
                        )
                        st.plotly_chart(fig_after, use_container_width=True)
                    else:
                        st.info("Nenhuma hashtag encontrada nos dados processados")
                else:
                    st.info("Coluna de hashtags não disponível nos dados processados")
                    
        except Exception as e:
            st.error(f"Erro na comparação de hashtags: {e}")
    
    def _render_mentions_comparison(self):
        """Comparação de menções antes vs depois"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**👥 Antes da Limpeza (Dados Originais)**")
                
                if self.df_original is not None and 'mentions' in self.df_original.columns:
                    mentions_before = []
                    for mention_field in self.df_original['mentions'].dropna():
                        if mention_field and mention_field.strip():
                            mentions_before.extend([m.strip() for m in str(mention_field).split(',') if m.strip()])
                    
                    if mentions_before:
                        mention_counts_before = pd.Series(mentions_before).value_counts().head(10)
                        
                        mentions_df_before = pd.DataFrame({
                            'Menção': mention_counts_before.index,
                            'Frequência': mention_counts_before.values
                        })
                        
                        fig_before = px.bar(
                            mentions_df_before,
                            y='Menção',
                            x='Frequência', 
                            orientation='h',
                            title="Top 10 Menções - Dados Originais",
                            color_discrete_sequence=['#ff7f7f']
                        )
                        st.plotly_chart(fig_before, use_container_width=True)
                    else:
                        st.info("Nenhuma menção encontrada nos dados originais")
                else:
                    st.info("Dados originais não disponíveis")
            
            with col2:
                st.write("**🧹 Depois da Limpeza (Dados Processados)**")
                
                if self.df is not None and 'mentions' in self.df.columns:
                    mentions_after = []
                    for mention_field in self.df['mentions'].dropna():
                        if mention_field and mention_field.strip():
                            mentions_after.extend([m.strip() for m in str(mention_field).split(',') if m.strip()])
                    
                    if mentions_after:
                        mention_counts_after = pd.Series(mentions_after).value_counts().head(10)
                        
                        mentions_df_after = pd.DataFrame({
                            'Menção': mention_counts_after.index,
                            'Frequência': mention_counts_after.values
                        })
                        
                        fig_after = px.bar(
                            mentions_df_after,
                            y='Menção',
                            x='Frequência',
                            orientation='h',
                            title="Top 10 Menções - Dados Processados",
                            color_discrete_sequence=['#7fbf7f']
                        )
                        st.plotly_chart(fig_after, use_container_width=True)
                    else:
                        st.info("Nenhuma menção encontrada nos dados processados")
                else:
                    st.info("Coluna de menções não disponível nos dados processados")
                    
        except Exception as e:
            st.error(f"Erro na comparação de menções: {e}")
    
    def _render_domains_comparison(self):
        """Comparação de domínios antes vs depois"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🌐 Antes da Limpeza (Dados Originais)**")
                
                if self.pre_cleaning_stats and 'url_analysis' in self.pre_cleaning_stats:
                    top_domains_before = self.pre_cleaning_stats['url_analysis'].get('top_domains', {})
                    
                    if top_domains_before:
                        domains_df_before = pd.DataFrame([
                            {'Domínio': k, 'Frequência': v} 
                            for k, v in list(top_domains_before.items())[:10]
                        ])
                        
                        fig_before = px.bar(
                            domains_df_before,
                            y='Domínio',
                            x='Frequência',
                            orientation='h',
                            title="Top 10 Domínios - Dados Originais",
                            color_discrete_sequence=['#ff7f7f']
                        )
                        st.plotly_chart(fig_before, use_container_width=True)
                    else:
                        st.info("Dados de domínios originais não disponíveis")
                else:
                    st.info("Estatísticas de URL originais não disponíveis")
            
            with col2:
                st.write("**🧹 Depois da Limpeza (Dados Processados)**")
                
                if self.df is not None and 'domain' in self.df.columns:
                    domain_counts_after = self.df['domain'].value_counts().head(10)
                    
                    if len(domain_counts_after) > 0:
                        domains_df_after = pd.DataFrame({
                            'Domínio': domain_counts_after.index,
                            'Frequência': domain_counts_after.values
                        })
                        
                        fig_after = px.bar(
                            domains_df_after,
                            y='Domínio',
                            x='Frequência',
                            orientation='h',
                            title="Top 10 Domínios - Dados Processados",
                            color_discrete_sequence=['#7fbf7f']
                        )
                        st.plotly_chart(fig_after, use_container_width=True)
                    else:
                        st.info("Nenhum domínio encontrado nos dados processados")
                else:
                    st.info("Coluna de domínios não disponível nos dados processados")
                    
        except Exception as e:
            st.error(f"Erro na comparação de domínios: {e}")
    
    def _render_transformation_summary(self):
        """Resumo das transformações aplicadas pelo pipeline"""
        try:
            st.write("### 📝 Principais Transformações Aplicadas:")
            
            transformations = [
                "🔍 **Stage 01-02**: Validação de encoding e estrutura dos dados",
                "🧹 **Stage 03**: Deduplicação inteligente com múltiplas estratégias",
                "📊 **Stage 04**: Validação e enriquecimento de features básicas",
                "🏛️ **Stage 05**: Análise política com classificação automática via IA",
                "✨ **Stage 06**: Limpeza inteligente de texto preservando contexto",
                "🔤 **Stage 07**: Processamento linguístico avançado com spaCy",
                "😊 **Stage 08**: Análise de sentimento contextualizada",
                "🎯 **Stage 09-11**: Modelagem de tópicos e clustering semântico",
                "🏷️ **Stage 12**: Normalização de hashtags políticas",
                "🌐 **Stage 13-15**: Análise de domínios, temporal e redes sociais",
                "🔬 **Stage 16-20**: Análise qualitativa e validação final do pipeline"
            ]
            
            for transformation in transformations:
                st.markdown(transformation)
            
            # Estatísticas finais de transformação
            if self.pre_cleaning_stats and self.post_cleaning_stats:
                st.write("### 📈 Estatísticas de Transformação:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    pre_chars = self.pre_cleaning_stats.get('text_statistics', {}).get('total_characters', 0)
                    post_chars = self.post_cleaning_stats.get('text_statistics', {}).get('total_characters', 0)
                    char_reduction = ((int(pre_chars) - int(post_chars)) / int(pre_chars)) * 100 if int(pre_chars) > 0 else 0
                    st.metric("Redução de Caracteres", f"{char_reduction:.1f}%")
                
                with col2:
                    pre_words = self.pre_cleaning_stats.get('text_statistics', {}).get('total_words', 0)
                    post_words = self.post_cleaning_stats.get('text_statistics', {}).get('total_words', 0)
                    word_reduction = ((int(pre_words) - int(post_words)) / int(pre_words)) * 100 if int(pre_words) > 0 else 0
                    st.metric("Redução de Palavras", f"{word_reduction:.1f}%")
                
                with col3:
                    if self.df is not None:
                        final_columns = len(self.df.columns)
                        original_columns = len(self.df_original.columns) if self.df_original is not None else 0
                        column_increase = final_columns - original_columns
                        st.metric("Colunas Adicionadas", f"+{column_increase}")
                        
        except Exception as e:
            st.error(f"Erro no resumo de transformações: {e}")
    
    def _render_temporal_overview(self):
        """Renderiza overview temporal"""
        if 'datetime' in self.df.columns:
            # Mensagens por mês
            df_monthly = self.df.set_index('datetime').resample('M').size().reset_index()
            df_monthly.columns = ['Mês', 'Mensagens']
            
            fig_temporal = px.line(
                df_monthly,
                x='Mês',
                y='Mensagens',
                title="Evolução do Volume de Mensagens",
                markers=True
            )
            st.plotly_chart(fig_temporal, use_container_width=True)
    
    def _render_political_analysis_page(self):
        """🏛️ Análise Política Hierárquica Completa - 4 Níveis"""
        st.header("🏛️ Análise Política Hierárquica - Taxonomia Brasileira (4 Níveis)")
        
        # Verificar disponibilidade de dados políticos
        political_columns = ['political_category', 'political_alignment', 'discourse_type', 'radicalization_level']
        available_columns = [col for col in political_columns if col in self.df.columns]
        
        if not available_columns:
            st.warning("⚠️ Dados de análise política não disponíveis")
            return
        
        st.info(f"📊 **Dataset:** {len(self.df):,} mensagens analisadas com taxonomia política hierárquica")
        
        # ============================
        # SEÇÃO 1: VISÃO GERAL HIERÁRQUICA
        # ============================
        st.subheader("📋 1. Visão Geral da Taxonomia Política Hierárquica")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'political_alignment' in self.df.columns:
                alignment_counts = self.df['political_alignment'].value_counts()
                dominant_alignment = alignment_counts.index[0]
                alignment_pct = (alignment_counts.iloc[0] / len(self.df)) * 100
                st.metric(
                    "🎯 Nível 2: Alinhamento Dominante", 
                    dominant_alignment.title(), 
                    f"{alignment_pct:.1f}%"
                )
        
        with col2:
            if 'political_category' in self.df.columns:
                category_counts = self.df['political_category'].value_counts()
                dominant_category = category_counts.index[0]
                category_pct = (category_counts.iloc[0] / len(self.df)) * 100
                st.metric(
                    "🏷️ Categoria Dominante", 
                    dominant_category.title(), 
                    f"{category_pct:.1f}%"
                )
        
        with col3:
            if 'discourse_type' in self.df.columns:
                discourse_counts = self.df['discourse_type'].value_counts()
                dominant_discourse = discourse_counts.index[0]
                discourse_pct = (discourse_counts.iloc[0] / len(self.df)) * 100
                st.metric(
                    "💬 Tipo de Discurso Principal", 
                    dominant_discourse.title(), 
                    f"{discourse_pct:.1f}%"
                )
        
        with col4:
            if 'radicalization_level' in self.df.columns:
                radical_counts = self.df['radicalization_level'].value_counts()
                radical_high_pct = (radical_counts.get('alto', 0) / len(self.df)) * 100
                st.metric(
                    "🔥 Taxa de Radicalização Alta", 
                    f"{radical_high_pct:.1f}%", 
                    f"{radical_counts.get('alto', 0)} mensagens"
                )
        
        # ============================
        # SEÇÃO 2: DISTRIBUIÇÕES POR NÍVEL HIERÁRQUICO
        # ============================
        st.subheader("📊 2. Distribuições por Nível Hierárquico")
        
        # Nível 2: Alinhamento Político
        if 'political_alignment' in self.df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Nível 2: Alinhamento Político**")
                alignment_counts = self.df['political_alignment'].value_counts()
                
                # Mapear cores específicas para alinhamentos políticos
                color_map_alignment = {
                    'direita': '#FF6B6B',      # Vermelho suave
                    'esquerda': '#4ECDC4',     # Verde azulado
                    'neutro': '#95A5A6',       # Cinza
                    'indefinido': '#F39C12'    # Laranja
                }
                
                colors_alignment = [color_map_alignment.get(align, '#95A5A6') for align in alignment_counts.index]
                
                fig_alignment = px.pie(
                    values=alignment_counts.values,
                    names=alignment_counts.index,
                    title="Distribuição do Alinhamento Político",
                    color_discrete_sequence=colors_alignment,
                    hole=0.3
                )
                fig_alignment.update_traces(textinfo='percent+label', textfont_size=12)
                st.plotly_chart(fig_alignment, use_container_width=True)
            
            with col2:
                st.write("**🏷️ Categoria Política Específica**")
                if 'political_category' in self.df.columns:
                    category_counts = self.df['political_category'].value_counts()
                    
                    # Cores para categorias políticas
                    color_map_category = {
                        'bolsonarista': '#E74C3C',    # Vermelho forte
                        'petista': '#E67E22',         # Laranja forte
                        'geral': '#3498DB',           # Azul
                        'neutro': '#95A5A6'           # Cinza
                    }
                    
                    colors_category = [color_map_category.get(cat, '#95A5A6') for cat in category_counts.index]
                    
                    fig_category = px.bar(
                        y=category_counts.index,
                        x=category_counts.values,
                        orientation='h',
                        title="Mensagens por Categoria Política",
                        color=category_counts.index,
                        color_discrete_sequence=colors_category
                    )
                    fig_category.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_category, use_container_width=True)
        
        # Nível 3: Tipo de Discurso e Radicalização
        col1, col2 = st.columns(2)
        
        with col1:
            if 'discourse_type' in self.df.columns:
                st.write("**💬 Nível 3: Tipo de Discurso**")
                discourse_counts = self.df['discourse_type'].value_counts()
                
                color_map_discourse = {
                    'informativo': '#2ECC71',     # Verde
                    'agressivo': '#E74C3C',       # Vermelho
                    'mobilizador': '#F39C12',     # Laranja
                    'conspiratório': '#8E44AD'    # Roxo
                }
                
                colors_discourse = [color_map_discourse.get(disc, '#95A5A6') for disc in discourse_counts.index]
                
                fig_discourse = px.pie(
                    values=discourse_counts.values,
                    names=discourse_counts.index,
                    title="Tipos de Discurso Identificados",
                    color_discrete_sequence=colors_discourse,
                    hole=0.3
                )
                st.plotly_chart(fig_discourse, use_container_width=True)
        
        with col2:
            if 'radicalization_level' in self.df.columns:
                st.write("**🔥 Nível de Radicalização**")
                radical_counts = self.df['radicalization_level'].value_counts()
                
                color_map_radical = {
                    'baixo': '#2ECC71',      # Verde
                    'médio': '#F39C12',      # Laranja
                    'alto': '#E74C3C'        # Vermelho
                }
                
                colors_radical = [color_map_radical.get(level, '#95A5A6') for level in radical_counts.index]
                
                fig_radical = px.bar(
                    x=radical_counts.index,
                    y=radical_counts.values,
                    title="Distribuição de Níveis de Radicalização",
                    color=radical_counts.index,
                    color_discrete_sequence=colors_radical
                )
                fig_radical.update_layout(showlegend=False)
                st.plotly_chart(fig_radical, use_container_width=True)
        
        # ============================
        # SEÇÃO 3: CORRELAÇÕES ENTRE NÍVEIS HIERÁRQUICOS
        # ============================
        st.subheader("🔗 3. Correlações entre Níveis Hierárquicos")
        
        if 'political_alignment' in self.df.columns and 'discourse_type' in self.df.columns:
            # Matriz de correlação política × discurso
            correlation_table = pd.crosstab(
                self.df['political_alignment'], 
                self.df['discourse_type'],
                normalize='index'
            ) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Correlação: Alinhamento × Tipo de Discurso (%)**")
                
                fig_correlation = px.imshow(
                    correlation_table.values,
                    x=correlation_table.columns,
                    y=correlation_table.index,
                    title="Heatmap: Alinhamento Político × Tipo de Discurso",
                    color_continuous_scale="RdYlBu_r",
                    aspect="auto"
                )
                fig_correlation.update_layout(height=400)
                st.plotly_chart(fig_correlation, use_container_width=True)
            
            with col2:
                # Tabela de correlação numérica
                st.write("**📋 Tabela de Correlação (%)**")
                st.dataframe(correlation_table.round(1), use_container_width=True)
                
                # Insights automáticos
                st.write("**💡 Insights Automáticos:**")
                for alignment in correlation_table.index:
                    max_discourse = correlation_table.loc[alignment].idxmax()
                    max_percentage = correlation_table.loc[alignment].max()
                    st.write(f"• **{alignment.title()}**: {max_percentage:.1f}% → {max_discourse}")
        
        # ============================
        # SEÇÃO 4: ANÁLISE TEMPORAL HIERÁRQUICA
        # ============================
        st.subheader("📅 4. Evolução Temporal das Categorias Políticas")
        self._render_political_temporal_hierarchy()
        
        # ============================
        # SEÇÃO 5: TÓPICOS E AGRUPAMENTOS ESPECÍFICOS (NÍVEL 4)
        # ============================
        self._render_level4_analysis()
        
        # ============================
        # SEÇÃO 6: ANÁLISE COMPARATIVA AVANÇADA
        # ============================
        self._render_advanced_political_comparison()
    
    def _render_political_temporal_hierarchy(self):
        """Análise temporal hierárquica das categorias políticas"""
        
        if 'datetime' not in self.df.columns:
            st.warning("⚠️ Dados temporais não disponíveis para análise hierárquica")
            return
        
        df_temp = self.df.copy()
        df_temp['month'] = df_temp['datetime'].dt.to_period('M').astype(str)
        df_temp['year'] = df_temp['datetime'].dt.year
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'political_alignment' in self.df.columns:
                st.write("**📊 Evolução do Alinhamento Político**")
                monthly_alignment = df_temp.groupby(['month', 'political_alignment']).size().unstack(fill_value=0)
                
                if not monthly_alignment.empty:
                    fig_temporal_alignment = px.area(
                        monthly_alignment,
                        title="Evolução Temporal do Alinhamento Político",
                        labels={'value': 'Mensagens', 'index': 'Período'},
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95A5A6', '#F39C12']
                    )
                    st.plotly_chart(fig_temporal_alignment, use_container_width=True)
        
        with col2:
            if 'discourse_type' in self.df.columns:
                st.write("**💬 Evolução do Tipo de Discurso**")
                monthly_discourse = df_temp.groupby(['month', 'discourse_type']).size().unstack(fill_value=0)
                
                if not monthly_discourse.empty:
                    fig_temporal_discourse = px.line(
                        monthly_discourse,
                        title="Evolução Temporal do Tipo de Discurso",
                        labels={'value': 'Mensagens', 'index': 'Período'},
                        markers=True
                    )
                    st.plotly_chart(fig_temporal_discourse, use_container_width=True)
        
        # Análise anual
        if len(df_temp['year'].unique()) > 1:
            st.write("**📅 Distribuição Anual por Categoria**")
            try:
                yearly_analysis = df_temp.groupby(['year', 'political_category']).size().unstack(fill_value=0)
                
                if not yearly_analysis.empty:
                    fig_yearly = px.bar(
                        yearly_analysis,
                        title="Distribuição Anual das Categorias Políticas",
                        labels={'value': 'Número de Mensagens', 'index': 'Ano'},
                        barmode='stack'
                    )
                    st.plotly_chart(fig_yearly, use_container_width=True)
                else:
                    st.info("ℹ️ Dados insuficientes para análise anual")
            except Exception as e:
                st.warning(f"⚠️ Erro na análise anual: {str(e)}")
                # Fallback: mostrar apenas contagem por ano
                yearly_simple = df_temp['year'].value_counts().sort_index()
                if len(yearly_simple) > 0:
                    fig_yearly_simple = px.bar(
                        x=yearly_simple.index,
                        y=yearly_simple.values,
                        title="Distribuição Anual Total de Mensagens",
                        labels={'x': 'Ano', 'y': 'Número de Mensagens'}
                    )
                    st.plotly_chart(fig_yearly_simple, use_container_width=True)

    def _render_level4_analysis(self):
        """Análise do Nível 4: Tópicos e Agrupamentos Específicos"""
        st.subheader("🎯 5. Análise de Nível 4: Tópicos e Agrupamentos Específicos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Análise de topic_name (Nível 4)
            if 'topic_name' in self.df.columns:
                st.write("**🏷️ Tópicos Específicos Identificados**")
                topic_counts = self.df['topic_name'].value_counts()
                
                # Filtrar apenas tópicos classificados (não "Não Classificado")
                classified_topics = topic_counts[topic_counts.index != 'Não Classificado']
                
                if len(classified_topics) > 0:
                    st.write(f"**📊 {len(classified_topics)} tópicos específicos identificados:**")
                    
                    fig_topics = px.bar(
                        y=classified_topics.index,
                        x=classified_topics.values,
                        orientation='h',
                        title="Tópicos Políticos Específicos",
                        color=classified_topics.values,
                        color_continuous_scale="Viridis"
                    )
                    fig_topics.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_topics, use_container_width=True)
                    
                    # Tabela detalhada
                    st.write("**📋 Detalhamento dos Tópicos:**")
                    topics_df = pd.DataFrame({
                        'Tópico': classified_topics.index,
                        'Mensagens': classified_topics.values,
                        'Percentual': (classified_topics.values / len(self.df) * 100).round(1)
                    })
                    st.dataframe(topics_df, use_container_width=True)
                else:
                    st.info("ℹ️ Maioria das mensagens não foi classificada em tópicos específicos")
            else:
                st.info("ℹ️ Dados de tópicos específicos não disponíveis")
        
        with col2:
            # Análise de cluster_name (Agrupamentos Semânticos)
            if 'cluster_name' in self.df.columns:
                st.write("**🔍 Agrupamentos Semânticos (Clusters)**")
                cluster_counts = self.df['cluster_name'].value_counts()
                
                if len(cluster_counts) > 0:
                    # Gráfico de pizza para clusters
                    color_map_clusters = {
                        'Cultura Bolsonarista Digital': '#3498DB',
                        'Narrativa Bolsonarista Antipetista': '#E74C3C',
                        'Mobilização Conservadora': '#F39C12',
                        'Discurso Institucional': '#2ECC71'
                    }
                    
                    colors_clusters = [color_map_clusters.get(cluster, '#95A5A6') for cluster in cluster_counts.index]
                    
                    fig_clusters = px.pie(
                        values=cluster_counts.values,
                        names=cluster_counts.index,
                        title="Distribuição dos Agrupamentos Semânticos",
                        color_discrete_sequence=colors_clusters,
                        hole=0.3
                    )
                    fig_clusters.update_traces(textinfo='percent+label', textfont_size=10)
                    st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    # Tabela detalhada de clusters
                    st.write("**📊 Análise dos Clusters:**")
                    clusters_df = pd.DataFrame({
                        'Cluster': cluster_counts.index,
                        'Mensagens': cluster_counts.values,
                        'Percentual': (cluster_counts.values / len(self.df) * 100).round(1)
                    })
                    st.dataframe(clusters_df, use_container_width=True)
            else:
                st.info("ℹ️ Dados de clustering semântico não disponíveis")
        
        # Análise de qualidade semântica
        if 'semantic_quality' in self.df.columns:
            st.write("**⭐ Qualidade Semântica dos Agrupamentos**")
            
            col1, col2, col3 = st.columns(3)
            
            quality_scores = pd.to_numeric(self.df['semantic_quality'], errors='coerce').dropna()
            
            if len(quality_scores) > 0:
                with col1:
                    avg_quality = quality_scores.mean()
                    st.metric("Qualidade Média", f"{avg_quality:.3f}")
                
                with col2:
                    high_quality_count = (quality_scores >= 0.7).sum()
                    high_quality_pct = (high_quality_count / len(quality_scores)) * 100
                    st.metric("Alta Qualidade (≥0.7)", f"{high_quality_pct:.1f}%")
                
                with col3:
                    max_quality = quality_scores.max()
                    st.metric("Qualidade Máxima", f"{max_quality:.3f}")

    def _render_advanced_political_comparison(self):
        """Análise Comparativa Avançada entre Dimensões Políticas"""
        st.subheader("🔬 6. Análise Comparativa Avançada")
        
        # Análise multidimensional
        if all(col in self.df.columns for col in ['political_alignment', 'political_category', 'discourse_type']):
            st.write("**🎭 Análise Multidimensional: Alinhamento × Categoria × Discurso**")
            
            # Criar análise tridimensional
            multi_analysis = self.df.groupby(['political_alignment', 'political_category', 'discourse_type']).size().reset_index(name='count')
            
            if not multi_analysis.empty:
                # Sunburst chart para visualização hierárquica
                fig_sunburst = px.sunburst(
                    multi_analysis,
                    path=['political_alignment', 'political_category', 'discourse_type'],
                    values='count',
                    title="Hierarquia Política: Alinhamento → Categoria → Discurso",
                    color='count',
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Comparação de características textuais por categoria política
        if all(col in self.df.columns for col in ['political_category', 'text_length', 'word_count']):
            st.write("**📏 Características Textuais por Categoria Política**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot de comprimento de texto por categoria
                fig_length_box = px.box(
                    self.df,
                    x='political_category',
                    y='text_length',
                    title="Distribuição do Comprimento de Texto por Categoria",
                    color='political_category',
                    color_discrete_sequence=['#3498DB', '#E74C3C', '#F39C12']
                )
                st.plotly_chart(fig_length_box, use_container_width=True)
            
            with col2:
                # Estatísticas descritivas
                text_stats = self.df.groupby('political_category')[['text_length', 'word_count']].describe()
                st.write("**📊 Estatísticas Descritivas:**")
                st.dataframe(text_stats.round(1), use_container_width=True)
        
        # Análise de densidade de entidades políticas
        if all(col in self.df.columns for col in ['political_category', 'political_entity_density']):
            st.write("**🎯 Densidade de Entidades Políticas por Categoria**")
            
            entity_density = self.df.groupby('political_category')['political_entity_density'].agg(['mean', 'median', 'std']).round(4)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_density = px.bar(
                    x=entity_density.index,
                    y=entity_density['mean'],
                    title="Densidade Média de Entidades Políticas",
                    color=entity_density['mean'],
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_density, use_container_width=True)
            
            with col2:
                st.write("**📋 Estatísticas de Densidade:**")
                st.dataframe(entity_density, use_container_width=True)
    
    def _render_political_temporal_analysis(self):
        """Análise temporal das categorias políticas (compatibilidade)"""
        self._render_political_temporal_hierarchy()
    
    def _render_sentiment_analysis_page(self):
        """Análise de sentimento detalhada"""
        st.header("😊 Análise de Sentimento Detalhada")
        
        if 'sentiment' not in self.df.columns:
            st.warning("Dados de análise de sentimento não disponíveis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição geral de sentimentos
            sentiment_counts = self.df['sentiment'].value_counts()
            
            colors = {
                'positive': '#2E8B57',
                'neutral': '#4682B4', 
                'negative': '#DC143C'
            }
            
            sentiment_colors = [colors.get(sent.lower(), '#808080') for sent in sentiment_counts.index]
            
            fig_sentiment = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Distribuição Geral de Sentimentos",
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Sentimento por score
            if 'sentiment_score' in self.df.columns:
                fig_score = px.histogram(
                    self.df,
                    x='sentiment_score',
                    title="Distribuição dos Scores de Sentimento",
                    nbins=20,
                    color_discrete_sequence=['#4CAF50']
                )
                st.plotly_chart(fig_score, use_container_width=True)
        
        # Sentimento por categoria política
        if 'political_category' in self.df.columns:
            st.subheader("🏛️ Sentimento por Categoria Política")
            
            sentiment_political = pd.crosstab(
                self.df['political_category'], 
                self.df['sentiment'], 
                normalize='index'
            ) * 100
            
            fig_sentiment_political = px.bar(
                sentiment_political,
                title="Distribuição de Sentimentos por Categoria Política (%)",
                labels={'value': 'Percentual (%)', 'index': 'Categoria Política'}
            )
            st.plotly_chart(fig_sentiment_political, use_container_width=True)
        
        # Evolução temporal do sentimento
        self._render_sentiment_temporal_analysis()
    
    def _render_sentiment_temporal_analysis(self):
        """Análise temporal dos sentimentos"""
        st.subheader("📈 Evolução Temporal dos Sentimentos")
        
        if 'datetime' in self.df.columns:
            df_temp = self.df.copy()
            df_temp['month'] = df_temp['datetime'].dt.to_period('M').astype(str)
            
            monthly_sentiment = df_temp.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            
            fig_temporal_sentiment = px.area(
                monthly_sentiment,
                title="Evolução dos Sentimentos ao Longo do Tempo",
                labels={'value': 'Número de Mensagens', 'index': 'Período'}
            )
            st.plotly_chart(fig_temporal_sentiment, use_container_width=True)
    
    def _render_discourse_analysis_page(self):
        """Análise do tipo de discurso"""
        st.header("💬 Análise do Discurso")
        
        if 'discourse_type' not in self.df.columns:
            st.warning("Dados de análise de discurso não disponíveis")
            return
        
        # Distribuição dos tipos de discurso
        discourse_counts = self.df['discourse_type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_discourse = px.pie(
                values=discourse_counts.values,
                names=discourse_counts.index,
                title="Tipos de Discurso Identificados",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_discourse, use_container_width=True)
        
        with col2:
            # Comprimento médio por tipo de discurso
            if 'text_length' in self.df.columns:
                avg_length_discourse = self.df.groupby('discourse_type')['text_length'].mean().sort_values(ascending=False)
                
                fig_length_discourse = px.bar(
                    x=avg_length_discourse.values,
                    y=avg_length_discourse.index,
                    orientation='h',
                    title="Comprimento Médio por Tipo de Discurso",
                    color=avg_length_discourse.values,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_length_discourse, use_container_width=True)
    
    def _render_temporal_analysis_page(self):
        """Análise temporal detalhada"""
        st.header("📅 Análise Temporal Detalhada")
        
        if 'datetime' not in self.df.columns:
            st.warning("Dados temporais não disponíveis")
            return
        
        # Atividade por hora do dia
        col1, col2 = st.columns(2)
        
        with col1:
            df_temp = self.df.copy()
            df_temp['hour'] = df_temp['datetime'].dt.hour
            hourly_activity = df_temp['hour'].value_counts().sort_index()
            
            fig_hourly = px.bar(
                x=hourly_activity.index,
                y=hourly_activity.values,
                title="Atividade por Hora do Dia",
                labels={'x': 'Hora', 'y': 'Número de Mensagens'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Atividade por dia da semana
            df_temp['weekday'] = df_temp['datetime'].dt.day_name()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_activity = df_temp['weekday'].value_counts().reindex(weekday_order)
            
            fig_weekday = px.bar(
                x=weekday_activity.index,
                y=weekday_activity.values,
                title="Atividade por Dia da Semana",
                labels={'x': 'Dia da Semana', 'y': 'Número de Mensagens'}
            )
            st.plotly_chart(fig_weekday, use_container_width=True)
    
    def _render_linguistic_analysis_page(self):
        """Análise linguística dos dados"""
        st.header("🔤 Análise Linguística")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de comprimento de texto
            if 'text_length' in self.df.columns:
                fig_length = px.histogram(
                    self.df,
                    x='text_length',
                    title="Distribuição do Comprimento das Mensagens",
                    nbins=30,
                    color_discrete_sequence=['#FF6B6B']
                )
                st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            # Distribuição de contagem de palavras
            if 'word_count' in self.df.columns:
                fig_words = px.histogram(
                    self.df,
                    x='word_count',
                    title="Distribuição do Número de Palavras",
                    nbins=30,
                    color_discrete_sequence=['#4ECDC4']
                )
                st.plotly_chart(fig_words, use_container_width=True)
        
        # Análise de complexidade linguística
        if 'spacy_linguistic_complexity' in self.df.columns:
            st.subheader("🧠 Complexidade Linguística")
            
            complexity_stats = self.df['spacy_linguistic_complexity'].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Média", f"{complexity_stats['mean']:.3f}")
            with col2:
                st.metric("Mediana", f"{complexity_stats['50%']:.3f}")
            with col3:
                st.metric("Mínimo", f"{complexity_stats['min']:.3f}")
            with col4:
                st.metric("Máximo", f"{complexity_stats['max']:.3f}")
    
    def _render_clustering_analysis_page(self):
        """Análise de agrupamentos semânticos"""
        st.header("🔍 Análise de Agrupamentos Semânticos")
        
        if 'cluster_name' not in self.df.columns:
            st.warning("Dados de clustering não disponíveis")
            return
        
        # Distribuição dos clusters
        cluster_counts = self.df['cluster_name'].value_counts()
        
        fig_clusters = px.bar(
            x=cluster_counts.values,
            y=cluster_counts.index,
            orientation='h',
            title="Distribuição de Mensagens por Cluster Semântico",
            color=cluster_counts.values,
            color_continuous_scale="Plasma"
        )
        fig_clusters.update_layout(height=500)
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # Análise de qualidade semântica
        if 'semantic_quality' in self.df.columns:
            st.subheader("🎯 Qualidade Semântica")
            
            avg_quality = self.df['semantic_quality'].mean()
            quality_dist = self.df['semantic_quality'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Qualidade Semântica Média", f"{avg_quality:.3f}")
                
                fig_quality = px.histogram(
                    self.df,
                    x='semantic_quality',
                    title="Distribuição da Qualidade Semântica",
                    nbins=20
                )
                st.plotly_chart(fig_quality, use_container_width=True)
    
    def _render_network_analysis_page(self):
        """Análise de redes e interações"""
        st.header("🌐 Análise de Redes e Interações")
        
        # Análise de menções
        if 'mentions' in self.df.columns:
            mentions_data = self.df[self.df['mentions'].notna() & (self.df['mentions'] != '')]
            
            if len(mentions_data) > 0:
                st.subheader("📢 Análise de Menções")
                st.metric("Mensagens com Menções", f"{len(mentions_data):,}")
                percentage_mentions = (len(mentions_data) / len(self.df)) * 100
                st.metric("Percentual com Menções", f"{percentage_mentions:.1f}%")
        
        # Análise de hashtags
        if 'hashtag' in self.df.columns:
            hashtag_data = self.df[self.df['hashtag'].notna() & (self.df['hashtag'] != '')]
            
            if len(hashtag_data) > 0:
                st.subheader("# Análise de Hashtags")
                st.metric("Mensagens com Hashtags", f"{len(hashtag_data):,}")
                percentage_hashtags = (len(hashtag_data) / len(self.df)) * 100
                st.metric("Percentual com Hashtags", f"{percentage_hashtags:.1f}%")
        
        # Análise de URLs
        if 'url' in self.df.columns:
            url_data = self.df[self.df['url'].notna() & (self.df['url'] != '')]
            
            if len(url_data) > 0:
                st.subheader("🔗 Análise de URLs")
                st.metric("Mensagens com URLs", f"{len(url_data):,}")
                percentage_urls = (len(url_data) / len(self.df)) * 100
                st.metric("Percentual com URLs", f"{percentage_urls:.1f}%")
    
    def _render_comparative_analysis_page(self):
        """Análise comparativa entre diferentes dimensões"""
        st.header("⚖️ Análise Comparativa")
        
        # Comparação entre categorias políticas e sentimentos
        if 'political_category' in self.df.columns and 'sentiment' in self.df.columns:
            st.subheader("🏛️ vs 😊 Política × Sentimento")
            
            comparison_table = pd.crosstab(
                self.df['political_category'], 
                self.df['sentiment'],
                normalize='index'
            ) * 100
            
            fig_comparison = px.imshow(
                comparison_table.values,
                x=comparison_table.columns,
                y=comparison_table.index,
                title="Heatmap: Sentimento por Categoria Política (%)",
                aspect="auto",
                color_continuous_scale="RdYlBu_r"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Comparação temporal
        st.subheader("📊 Estatísticas Comparativas")
        
        if 'text_length' in self.df.columns and 'political_category' in self.df.columns:
            length_by_category = self.df.groupby('political_category')['text_length'].agg(['mean', 'median', 'std'])
            st.dataframe(length_by_category, use_container_width=True)


def main():
    """Função principal"""
    dashboard = DataAnalysisDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()