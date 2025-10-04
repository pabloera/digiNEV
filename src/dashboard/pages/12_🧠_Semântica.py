"""
Page: 12 - Análise Semântica (Stage 12)
Análise de sentimento e padrões semânticos no discurso político brasileiro

Visualizações:
1. Gauge charts para distribuição de sentimentos (positivo, negativo, neutro)
2. Timeline de evolução temporal do sentimento
"""

import streamlit as st
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dashboard.stage12_semantic_dashboard import main_dashboard, SemanticAnalyzer
from dashboard.data_analysis_dashboard import load_processed_data, get_available_datasets

# Configuração da página
st.set_page_config(
    page_title="Análise Semântica - Stage 12",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Função principal da página."""
    st.title("🧠 Stage 12 - Análise Semântica")
    st.markdown("""
    **Análise de sentimento e padrões semânticos no discurso político brasileiro**

    Esta análise examina características semânticas e emocionais dos textos através de:

    - **Análise de Sentimento**: Classificação de polaridade (positivo, negativo, neutro)
    - **Intensidade Emocional**: Medição de intensidade emocional baseada em marcadores
    - **Linguagem Agressiva**: Detecção de padrões linguísticos agressivos
    - **Diversidade Semântica**: Análise da variedade vocabular
    """)

    # Verificar se há dados processados disponíveis
    try:
        available_datasets = get_available_datasets()

        if not available_datasets:
            st.error("📁 Nenhum dataset processado encontrado")
            st.info("""
            Para usar esta análise:
            1. Execute o pipeline de processamento: `python run_pipeline.py`
            2. Ou carregue dados através da página Home
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

        # Verificar se há dados semânticos processados
        analyzer = SemanticAnalyzer()
        validation = analyzer.validate_semantic_columns(df)
        missing_columns = [col for col, exists in validation.items() if not exists]

        # Informações do dataset
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Informações do Dataset")
        st.sidebar.write(f"**Registros:** {len(df):,}")
        st.sidebar.write(f"**Colunas:** {len(df.columns)}")

        # Verificar processamento semântico
        semantic_columns = [col for col in df.columns if any(sem_col in col for sem_col in
                           ['sentiment', 'emotion', 'semantic', 'aggressive'])]

        if semantic_columns:
            st.sidebar.write("**Análise Semântica:**")
            for col in semantic_columns:
                if col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        avg_value = df[col].mean()
                        st.sidebar.write(f"- {col}: {avg_value:.3f} média")
                    elif df[col].dtype == 'bool':
                        pct_true = (df[col] == True).mean() * 100
                        st.sidebar.write(f"- {col}: {pct_true:.1f}% True")
                    else:
                        unique_count = df[col].nunique()
                        st.sidebar.write(f"- {col}: {unique_count} categorias")

        if missing_columns:
            st.warning(f"""
            ⚠️ **Processamento semântico incompleto**

            Colunas ausentes: {', '.join(missing_columns)}

            Este dataset não possui processamento semântico completo (Stage 12).
            """)

            # Opção para executar análise básica
            if st.button("🔄 Executar Análise Semântica Básica"):
                if 'body' in df.columns or 'normalized_text' in df.columns:
                    text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'

                    # Análise básica para amostra
                    sample_size = min(500, len(df))
                    df_sample = df.sample(n=sample_size, random_state=42)

                    with st.spinner("Executando análise semântica..."):
                        # Simular análise semântica básica
                        import re
                        from collections import Counter

                        def basic_sentiment_analysis(text):
                            if pd.isna(text):
                                return 0.0, 'neutral'

                            positive_words = ['bom', 'ótimo', 'excelente', 'amor', 'feliz', 'sucesso', 'vitória']
                            negative_words = ['ruim', 'péssimo', 'ódio', 'raiva', 'erro', 'fracasso', 'problema']

                            text_lower = str(text).lower()
                            pos_count = sum(1 for word in positive_words if word in text_lower)
                            neg_count = sum(1 for word in negative_words if word in text_lower)

                            polarity = (pos_count - neg_count) / max(len(text_lower.split()), 1)

                            if polarity > 0.05:
                                label = 'positive'
                            elif polarity < -0.05:
                                label = 'negative'
                            else:
                                label = 'neutral'

                            return polarity, label

                        def basic_emotion_intensity(text):
                            if pd.isna(text):
                                return 0.0

                            markers = text.count('!') + text.count('?') + text.count('...')
                            caps = sum(1 for word in str(text).split() if word.isupper() and len(word) > 2)
                            return min((markers + caps) / 10.0, 1.0)

                        def detect_aggressive(text):
                            if pd.isna(text):
                                return False

                            aggressive = ['ódio', 'matar', 'destruir', 'burro', 'idiota']
                            return any(word in str(text).lower() for word in aggressive)

                        def semantic_diversity(text):
                            if pd.isna(text):
                                return 0.0

                            words = str(text).split()
                            if len(words) == 0:
                                return 0.0
                            return len(set(words)) / len(words)

                        # Aplicar análises
                        results = df_sample[text_column].apply(basic_sentiment_analysis)
                        df_sample['sentiment_polarity'] = [r[0] for r in results]
                        df_sample['sentiment_label'] = [r[1] for r in results]
                        df_sample['emotion_intensity'] = df_sample[text_column].apply(basic_emotion_intensity)
                        df_sample['has_aggressive_language'] = df_sample[text_column].apply(detect_aggressive)
                        df_sample['semantic_diversity'] = df_sample[text_column].apply(semantic_diversity)

                    st.success(f"✅ Análise semântica básica executada para {len(df_sample)} mensagens")

                    # Executar dashboard com dados processados
                    main_dashboard(df_sample)

                else:
                    st.error("❌ Nenhuma coluna de texto encontrada para análise")

            return

        # Verificar qualidade dos dados semânticos
        quality_metrics = {}

        if 'sentiment_label' in df.columns:
            sentiment_dist = df['sentiment_label'].value_counts()
            total_sentiment = sentiment_dist.sum()
            quality_metrics['total_with_sentiment'] = total_sentiment
            st.sidebar.write(f"**Sentimentos:** {total_sentiment:,} analisados")

        if 'emotion_intensity' in df.columns:
            high_emotion = (df['emotion_intensity'] > 0.3).sum()
            quality_metrics['high_emotion_count'] = high_emotion
            st.sidebar.write(f"**Alta Emoção:** {high_emotion:,} mensagens")

        if 'has_aggressive_language' in df.columns:
            aggressive_count = df['has_aggressive_language'].sum()
            quality_metrics['aggressive_count'] = aggressive_count
            st.sidebar.write(f"**Linguagem Agressiva:** {aggressive_count:,} casos")

        # Executar dashboard principal
        main_dashboard(df)

        # Seção de análise avançada
        st.markdown("---")
        st.subheader("📈 Análise Estatística Avançada")

        with st.expander("🔍 Estatísticas Detalhadas"):
            if 'sentiment_polarity' in df.columns:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Polaridade de Sentimento:**")
                    polarity_stats = df['sentiment_polarity'].describe()
                    for stat, value in polarity_stats.items():
                        st.write(f"- {stat}: {value:.4f}")

                with col2:
                    if 'emotion_intensity' in df.columns:
                        st.markdown("**Intensidade Emocional:**")
                        emotion_stats = df['emotion_intensity'].describe()
                        for stat, value in emotion_stats.items():
                            st.write(f"- {stat}: {value:.4f}")

                with col3:
                    if 'semantic_diversity' in df.columns:
                        st.markdown("**Diversidade Semântica:**")
                        diversity_stats = df['semantic_diversity'].describe()
                        for stat, value in diversity_stats.items():
                            st.write(f"- {stat}: {value:.4f}")

            # Correlações entre variáveis semânticas
            if all(col in df.columns for col in ['sentiment_polarity', 'emotion_intensity', 'semantic_diversity']):
                st.markdown("**Matriz de Correlação:**")
                semantic_cols = ['sentiment_polarity', 'emotion_intensity', 'semantic_diversity']
                correlation_matrix = df[semantic_cols].corr()

                import plotly.express as px
                fig_corr = px.imshow(
                    correlation_matrix,
                    title="Correlações entre Variáveis Semânticas",
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # Informações técnicas
        with st.expander("ℹ️ Informações Técnicas"):
            st.markdown("""
            **Metodologia de Análise Semântica (Stage 12):**

            **1. Análise de Sentimento:**
            - Polaridade calculada com base em léxico de palavras positivas e negativas
            - Classificação categórica: positive (>0.1), negative (<-0.1), neutral
            - Foco em contexto político brasileiro

            **2. Intensidade Emocional:**
            - Baseada em marcadores textuais: pontuação (!, ?, ...), palavras em maiúsculas
            - Escala normalizada 0-1
            - Indicador de engagement emocional

            **3. Detecção de Linguagem Agressiva:**
            - Léxico específico de termos agressivos e ofensivos
            - Classificação binária (True/False)
            - Relevante para análise de radicalização

            **4. Diversidade Semântica:**
            - Razão entre palavras únicas e total de palavras
            - Indicador de complexidade vocabular
            - Correlacionado com sofisticação discursiva

            **Limitações:**
            - Análise baseada em léxico (não contextual profunda)
            - Específico para português brasileiro
            - Sensível à qualidade do pré-processamento
            """)

    except Exception as e:
        st.error(f"❌ Erro ao carregar página: {str(e)}")
        st.info("Tente recarregar a página ou verificar os logs do sistema")

if __name__ == "__main__":
    main()