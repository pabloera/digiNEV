"""
Page: 7 - Análise Linguística (Stage 07)
Named Entity Recognition (NER) para discurso político brasileiro

Visualizações:
1. Word cloud de entidades por tipo (PERSON, ORG, GPE)
2. Rede de conexões entre entidades políticas
"""

import streamlit as st
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dashboard.stage07_linguistic_dashboard import main_dashboard, LinguisticAnalyzer
from dashboard.data_analysis_dashboard import load_processed_data, get_available_datasets

# Configuração da página
st.set_page_config(
    page_title="Análise Linguística - Stage 07",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Função principal da página."""
    st.title("🔤 Stage 07 - Análise Linguística")
    st.markdown("""
    **Processamento linguístico com Named Entity Recognition (NER)**

    Esta análise utiliza o modelo spaCy português para extrair e visualizar entidades nomeadas
    relevantes para o discurso político brasileiro:

    - **Pessoas (PERSON)**: Políticos, líderes, figuras públicas
    - **Organizações (ORG)**: Partidos políticos, instituições, empresas
    - **Locais Geopolíticos (GPE)**: Estados, cidades, países
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

        # Verificar se há dados linguísticos processados
        linguistic_columns = [col for col in df.columns if 'spacy' in col.lower() or 'entities' in col.lower()]

        if not linguistic_columns:
            st.warning("""
            ⚠️ **Dados linguísticos não encontrados**

            Este dataset não possui processamento linguístico (Stage 07).
            """)

            # Opção para processar entidades em tempo real
            if st.button("🔄 Processar Entidades Agora"):
                analyzer = LinguisticAnalyzer()

                # Verificar se spaCy está disponível
                if not analyzer.nlp:
                    st.error("❌ Modelo spaCy português não disponível")
                    st.info("Instale com: `python -m spacy download pt_core_news_lg`")
                    return

                # Processar amostra pequena
                sample_size = min(100, len(df))
                df_sample = df.sample(n=sample_size, random_state=42)

                with st.spinner("Extraindo entidades..."):
                    df_processed = analyzer.extract_entities_from_dataframe(df_sample)

                st.success(f"✅ Entidades extraídas para {len(df_sample)} mensagens")

                # Executar dashboard com dados processados
                main_dashboard(df_processed)

            return

        # Informações do dataset
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Informações do Dataset")
        st.sidebar.write(f"**Registros:** {len(df):,}")
        st.sidebar.write(f"**Colunas:** {len(df.columns)}")

        # Verificar colunas linguísticas disponíveis
        if linguistic_columns:
            st.sidebar.write("**Processamento Linguístico:**")
            for col in linguistic_columns:
                if 'count' in col:
                    avg_value = df[col].mean() if df[col].dtype in ['int64', 'float64'] else 0
                    st.sidebar.write(f"- {col}: {avg_value:.1f} média")

        # Verificar qualidade dos dados linguísticos
        has_entities = False
        if 'spacy_entities_count' in df.columns:
            total_entities = df['spacy_entities_count'].sum()
            has_entities = total_entities > 0
            st.sidebar.write(f"**Total Entidades:** {total_entities:,}")

        if not has_entities:
            st.warning("""
            ⚠️ **Poucas entidades encontradas no dataset**

            Isso pode indicar:
            - Textos muito curtos
            - Baixa qualidade de texto
            - Necessidade de reprocessamento
            """)

            # Mostrar amostra de dados
            st.subheader("🔍 Amostra dos Dados")
            text_columns = [col for col in df.columns if col in ['body', 'normalized_text', 'text']]
            if text_columns:
                sample_df = df[text_columns + linguistic_columns].head(10)
                st.dataframe(sample_df, use_container_width=True)

        # Executar dashboard principal
        main_dashboard(df)

        # Seção de análise adicional
        st.markdown("---")
        st.subheader("📈 Análise Avançada")

        with st.expander("🔍 Explorar Dados Linguísticos"):
            # Estatísticas detalhadas
            if 'spacy_tokens_count' in df.columns:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Tokens Médios",
                        f"{df['spacy_tokens_count'].mean():.1f}",
                        f"σ: {df['spacy_tokens_count'].std():.1f}"
                    )

                with col2:
                    if 'spacy_entities_count' in df.columns:
                        st.metric(
                            "Entidades Médias",
                            f"{df['spacy_entities_count'].mean():.1f}",
                            f"Max: {df['spacy_entities_count'].max()}"
                        )

                with col3:
                    if 'lemmatized_text' in df.columns:
                        non_empty_lemmas = df['lemmatized_text'].str.len() > 0
                        st.metric(
                            "Textos Lemmatizados",
                            f"{non_empty_lemmas.sum():,}",
                            f"{(non_empty_lemmas.mean()*100):.1f}%"
                        )

            # Distribuição de tokens
            if 'spacy_tokens_count' in df.columns:
                st.subheader("📊 Distribuição de Tokens")

                import plotly.express as px

                fig_hist = px.histogram(
                    df,
                    x='spacy_tokens_count',
                    nbins=30,
                    title='Distribuição do Número de Tokens por Mensagem',
                    labels={'spacy_tokens_count': 'Número de Tokens', 'count': 'Frequência'}
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

        # Informações técnicas
        with st.expander("ℹ️ Informações Técnicas"):
            st.markdown("""
            **Modelo de Processamento:**
            - spaCy com modelo português (pt_core_news_lg/sm)
            - Extração de entidades nomeadas (NER)
            - Lemmatização e tokenização

            **Tipos de Entidades:**
            - **PERSON**: Pessoas (políticos, líderes)
            - **ORG**: Organizações (partidos, instituições)
            - **GPE**: Entidades geopolíticas (estados, cidades)

            **Filtragem:**
            - Entidades politicamente relevantes para contexto brasileiro
            - Filtragem por frequência mínima
            - Co-ocorrência para análise de rede
            """)

    except Exception as e:
        st.error(f"❌ Erro ao carregar página: {str(e)}")
        st.info("Tente recarregar a página ou verificar os logs do sistema")

if __name__ == "__main__":
    main()