"""
Page: 9 - Análise TF-IDF (Stage 09)
Vetorização TF-IDF para análise de termos relevantes

Visualizações:
1. Bar chart dos top 20 termos mais relevantes
2. Treemap hierárquico por importância (até 50 termos)
3. Análise de diferenças temporais (únicos vs compartilhados)
4. Evolução do ranking dos termos importantes
"""

import streamlit as st
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dashboard.stage09_tfidf_dashboard import main_dashboard, TFIDFAnalyzer
from dashboard.data_analysis_dashboard import load_processed_data, get_available_datasets

# Configuração da página
st.set_page_config(
    page_title="Análise TF-IDF - Stage 09",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Função principal da página."""
    st.title("📊 Stage 09 - Análise TF-IDF Vectorization")
    st.markdown("""
    **Análise de Termos Mais Relevantes via TF-IDF (Term Frequency-Inverse Document Frequency)**

    Esta análise identifica os termos mais importantes no discurso político brasileiro através da técnica TF-IDF,
    que combina a frequência local dos termos com sua raridade no corpus completo:

    - **📊 Top Termos**: Os 20 termos mais relevantes com scores TF-IDF
    - **🗺️ Hierarquia**: Visualização hierárquica dos 50 termos por categoria temática
    - **⏰ Evolução Temporal**: Comparação de termos únicos vs compartilhados entre períodos
    - **📈 Ranking**: Mudanças na posição dos termos mais importantes ao longo do tempo
    """)

    # Verificar se há dados processados disponíveis
    try:
        available_datasets = get_available_datasets()

        if not available_datasets:
            st.error("📁 Nenhum dataset processado encontrado")
            st.info("""
            Para usar esta análise TF-IDF:
            1. Execute o pipeline completo: `python run_pipeline.py`
            2. Ou carregue dados através da página Home
            3. Certifique-se de que o Stage 09 foi executado
            """)
            return

        # Sidebar para seleção de dataset
        st.sidebar.header("🔧 Configurações")

        selected_dataset = st.sidebar.selectbox(
            "📊 Selecionar Dataset:",
            options=list(available_datasets.keys()),
            format_func=lambda x: f"{x} ({available_datasets[x]['size']})"
        )

        if st.sidebar.button("🔄 Recarregar Dados"):
            st.cache_data.clear()
            st.experimental_rerun()

        # Carregar dados selecionados
        with st.spinner(f"Carregando {selected_dataset}..."):
            df = load_processed_data(selected_dataset)

        if df is None or df.empty:
            st.error(f"❌ Erro ao carregar dataset: {selected_dataset}")
            return

        # Verificar se há dados TF-IDF processados
        tfidf_columns = ['tfidf_score_mean', 'tfidf_score_max', 'tfidf_top_terms']
        missing_columns = [col for col in tfidf_columns if col not in df.columns]

        if missing_columns:
            st.warning(f"""
            ⚠️ **Dados TF-IDF não encontrados**

            Colunas faltantes: {', '.join(missing_columns)}

            Este dataset não possui processamento TF-IDF (Stage 09).
            """)

            # Verificar se há dados de texto para processar
            text_columns = [col for col in df.columns if col in ['body', 'normalized_text', 'text']]

            if text_columns and len(df) <= 1000:  # Só para datasets pequenos
                if st.button("🔄 Processar TF-IDF Agora"):
                    analyzer = TFIDFAnalyzer()

                    # Simular processamento TF-IDF básico
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    import numpy as np

                    try:
                        with st.spinner("Executando vetorização TF-IDF..."):
                            # Usar coluna de texto disponível
                            text_col = text_columns[0]
                            texts = df[text_col].fillna('').astype(str).tolist()

                            # Vetorização TF-IDF
                            vectorizer = TfidfVectorizer(
                                max_features=50,
                                min_df=1,
                                stop_words=None,
                                ngram_range=(1, 1)
                            )

                            tfidf_matrix = vectorizer.fit_transform(texts)
                            feature_names = vectorizer.get_feature_names_out()

                            # Calcular scores
                            tfidf_dense = tfidf_matrix.toarray()
                            df['tfidf_score_mean'] = np.mean(tfidf_dense, axis=1)
                            df['tfidf_score_max'] = np.max(tfidf_dense, axis=1)

                            # Top terms por documento
                            df['tfidf_top_terms'] = [
                                [feature_names[i] for i in row.argsort()[::-1][:5] if row[i] > 0]
                                for row in tfidf_dense
                            ]

                        st.success(f"✅ TF-IDF processado para {len(df)} documentos")

                        # Executar dashboard com dados processados
                        main_dashboard(df)

                    except Exception as e:
                        st.error(f"❌ Erro no processamento TF-IDF: {str(e)}")
                        st.info("Instale scikit-learn: `pip install scikit-learn`")

            else:
                st.info("""
                💡 **Sugestões:**
                - Execute o pipeline completo para gerar dados TF-IDF
                - Use um dataset menor (≤ 1000 registros) para processamento em tempo real
                - Verifique se o Stage 09 está incluído na configuração do pipeline
                """)

            return

        # Informações do dataset
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Informações do Dataset")
        st.sidebar.write(f"**Registros:** {len(df):,}")
        st.sidebar.write(f"**Colunas:** {len(df.columns)}")

        # Verificar qualidade dos dados TF-IDF
        valid_tfidf = df[
            (df['tfidf_top_terms'].notna()) &
            (df['tfidf_top_terms'] != '[]') &
            (df['tfidf_top_terms'] != '')
        ]

        st.sidebar.write(f"**Documentos com TF-IDF:** {len(valid_tfidf):,}")

        if len(valid_tfidf) > 0:
            avg_score = valid_tfidf['tfidf_score_mean'].mean()
            max_score = valid_tfidf['tfidf_score_max'].max()

            st.sidebar.write(f"**Score Médio:** {avg_score:.3f}")
            st.sidebar.write(f"**Score Máximo:** {max_score:.3f}")

            # Estimar vocabulário
            analyzer = TFIDFAnalyzer()
            term_scores = analyzer.extract_terms_from_data(valid_tfidf.head(100))  # Amostra para performance
            st.sidebar.write(f"**Vocabulário (amostra):** {len(term_scores):,}")

        if len(valid_tfidf) == 0:
            st.warning("""
            ⚠️ **Nenhum dado TF-IDF válido encontrado**

            Isso pode indicar:
            - Stage 09 não executado completamente
            - Textos muito curtos ou vazios
            - Falha na vetorização TF-IDF
            """)

            # Mostrar amostra dos dados TF-IDF
            st.subheader("🔍 Amostra dos Dados TF-IDF")
            if 'tfidf_top_terms' in df.columns:
                sample_df = df[['tfidf_score_mean', 'tfidf_score_max', 'tfidf_top_terms']].head(10)
                st.dataframe(sample_df, use_container_width=True)

            return

        # Controles de filtragem
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Controles de Análise")

        # Filtro por score mínimo
        min_score = st.sidebar.slider(
            "Score TF-IDF Mínimo",
            min_value=0.0,
            max_value=float(valid_tfidf['tfidf_score_mean'].max()),
            value=0.0,
            step=0.01,
            help="Filtrar documentos com score TF-IDF mínimo"
        )

        # Filtro temporal (se disponível)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            valid_dates = df.dropna(subset=['datetime'])

            if len(valid_dates) > 0:
                date_range = st.sidebar.date_input(
                    "📅 Período de Análise",
                    value=(
                        valid_dates['datetime'].min().date(),
                        valid_dates['datetime'].max().date()
                    ),
                    min_value=valid_dates['datetime'].min().date(),
                    max_value=valid_dates['datetime'].max().date(),
                    help="Selecionar período para análise temporal"
                )

                if len(date_range) == 2:
                    start_date, end_date = date_range
                    valid_tfidf = valid_tfidf[
                        (valid_tfidf['datetime'].dt.date >= start_date) &
                        (valid_tfidf['datetime'].dt.date <= end_date)
                    ]

        # Aplicar filtro de score
        if min_score > 0:
            valid_tfidf = valid_tfidf[valid_tfidf['tfidf_score_mean'] >= min_score]
            st.sidebar.write(f"**Após filtros:** {len(valid_tfidf):,} docs")

        if len(valid_tfidf) == 0:
            st.warning("⚠️ Nenhum documento atende aos critérios de filtragem")
            return

        # Executar dashboard principal
        main_dashboard(valid_tfidf)

        # Seção de análise adicional
        st.markdown("---")
        st.subheader("📈 Análise Avançada")

        with st.expander("🔍 Explorar Dados TF-IDF"):
            # Estatísticas detalhadas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Documentos Válidos",
                    f"{len(valid_tfidf):,}",
                    f"{(len(valid_tfidf)/len(df)*100):.1f}% do total"
                )

            with col2:
                st.metric(
                    "Score Médio TF-IDF",
                    f"{valid_tfidf['tfidf_score_mean'].mean():.3f}",
                    f"Desvio: {valid_tfidf['tfidf_score_mean'].std():.3f}"
                )

            with col3:
                st.metric(
                    "Diversidade Vocabulário",
                    f"{len(term_scores):,}",
                    "Termos únicos identificados"
                )

            # Distribuição de scores
            st.subheader("📊 Distribuição dos Scores TF-IDF")

            import plotly.express as px

            fig_dist = px.histogram(
                valid_tfidf,
                x='tfidf_score_mean',
                nbins=30,
                title='Distribuição dos Scores Médios TF-IDF',
                labels={'tfidf_score_mean': 'Score Médio TF-IDF', 'count': 'Frequência'}
            )
            fig_dist.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Correlação com outras métricas (se disponíveis)
            if 'word_count' in df.columns:
                st.subheader("🔗 Correlação com Tamanho do Texto")

                fig_scatter = px.scatter(
                    valid_tfidf,
                    x='word_count',
                    y='tfidf_score_mean',
                    title='TF-IDF vs. Número de Palavras',
                    labels={'word_count': 'Número de Palavras', 'tfidf_score_mean': 'Score TF-IDF Médio'}
                )
                fig_scatter.update_layout(template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Informações técnicas
        with st.expander("ℹ️ Informações Técnicas"):
            st.markdown(f"""
            **Metodologia TF-IDF:**
            - **TF (Term Frequency)**: Frequência do termo no documento
            - **IDF (Inverse Document Frequency)**: Log inverso da frequência nos documentos
            - **Score Final**: TF × IDF, normalizado

            **Configuração Atual:**
            - Documentos processados: {len(valid_tfidf):,}
            - Vocabulário extraído: {len(term_scores):,} termos
            - Score médio: {valid_tfidf['tfidf_score_mean'].mean():.4f}
            - Range de scores: {valid_tfidf['tfidf_score_mean'].min():.3f} - {valid_tfidf['tfidf_score_mean'].max():.3f}

            **Filtragem:**
            - Remoção de stopwords em português
            - Termos com mínimo 3 caracteres
            - Filtros de qualidade aplicados

            **Categorização:**
            - Política Institucional: governo, presidente, política, etc.
            - Economia: economia, trabalho, desenvolvimento, etc.
            - Políticas Sociais: educação, saúde, segurança, etc.
            - Estado e Nação: brasil, país, estado, nacional, etc.

            **Análise Temporal:**
            - Divisão automática em períodos
            - Identificação de termos únicos vs compartilhados
            - Tracking de mudanças no ranking
            """)

    except Exception as e:
        st.error(f"❌ Erro ao carregar página: {str(e)}")
        st.info("Tente recarregar a página ou verificar os logs do sistema")

        # Debug info
        with st.expander("🐛 Informações de Debug"):
            st.code(f"""
            Erro: {str(e)}
            Tipo: {type(e).__name__}

            Verifique:
            - Dados TF-IDF estão disponíveis
            - Colunas necessárias existem
            - Formato dos dados está correto
            """)

if __name__ == "__main__":
    main()