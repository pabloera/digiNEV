"""
Page: 13 - Análise Temporal (Stage 13)
Análise temporal de padrões de coordenação no discurso político brasileiro

Visualizações:
1. Line chart: Volume de mensagens ao longo do tempo
2. Event correlation: Picos de atividade vs eventos políticos
3. Heatmap: Coordenação temporal entre usuários/canais
4. Network graph: Clusters de atividade sincronizada
5. Timeline: Períodos de alta coordenação identificados
6. Sankey: Fluxo temporal → sentimento → affordances
"""

import streamlit as st
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dashboard.stage13_temporal_dashboard import main_dashboard, TemporalAnalyzer

# Simplified data loading functions
def get_available_datasets():
    """Get available processed datasets."""
    import os
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
        return []

    files = []
    for file in os.listdir(processed_dir):
        if file.endswith('.csv'):
            files.append(file)
    return files

def load_processed_data(filename):
    """Load processed dataset."""
    import pandas as pd
    try:
        if not filename.endswith('.csv'):
            filename += '.csv'
        filepath = f"data/processed/{filename}"
        return pd.read_csv(filepath, sep=';', encoding='utf-8')
    except Exception as e:
        st.error(f"Erro ao carregar dataset: {e}")
        return None

# Configuração da página
st.set_page_config(
    page_title="Análise Temporal - Stage 13",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Função principal da página."""
    st.title("⏰ Stage 13 - Análise Temporal")
    st.markdown("""
    **Análise temporal de padrões de coordenação no discurso político brasileiro**

    Esta análise examina características temporais e de coordenação através de:

    - **Volume Temporal**: Evolução da atividade de mensagens ao longo do tempo
    - **Correlação de Eventos**: Picos de atividade correlacionados com eventos políticos brasileiros
    - **Coordenação Temporal**: Padrões de sincronização entre usuários e canais
    - **Redes de Atividade**: Clusters de atividade sincronizada e coordenada
    - **Timeline de Coordenação**: Identificação de períodos de alta coordenação
    - **Fluxos Integrados**: Análise de fluxo temporal → sentimento → affordances
    """)

    # Verificar se há dados processados disponíveis
    try:
        available_datasets = get_available_datasets()

        if not available_datasets:
            st.error("📁 Nenhum dataset processado encontrado")
            st.info("""
            Para usar esta análise:
            1. Execute o pipeline de processamento: `python run_pipeline.py`
            2. Ou carregue dados já processados na pasta `/data/processed/`
            3. Certifique-se de que o Stage 13 (Análise Temporal) foi executado
            """)
            return

        # Sidebar para seleção de dataset
        st.sidebar.header("⚙️ Configurações")

        selected_dataset = st.sidebar.selectbox(
            "Selecionar Dataset:",
            available_datasets,
            help="Escolha o dataset processado para análise temporal"
        )

        # Opções de filtro temporal
        st.sidebar.subheader("🔍 Filtros Temporais")

        # Filtro por período
        time_filter = st.sidebar.selectbox(
            "Filtrar por período:",
            ["Todos os períodos", "Horário comercial", "Fora do horário", "Fins de semana", "Dias úteis"],
            help="Filtrar dados por períodos específicos"
        )

        # Filtro por coordenação
        coord_filter = st.sidebar.slider(
            "Coordenação mínima:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filtrar mensagens por nível mínimo de coordenação temporal"
        )

        # Carregamento dos dados
        with st.spinner(f"Carregando dataset: {selected_dataset}"):
            df = load_processed_data(selected_dataset)

        if df is None or df.empty:
            st.error(f"❌ Erro ao carregar dataset: {selected_dataset}")
            return

        # Verificar se contém dados do Stage 13
        temporal_columns = [
            'hour', 'day_of_week', 'month', 'year', 'day_of_year',
            'sender_frequency', 'is_frequent_sender', 'temporal_coordination',
            'is_weekend', 'is_business_hours'
        ]

        available_temporal_cols = [col for col in temporal_columns if col in df.columns]

        if len(available_temporal_cols) < 3:
            st.warning(f"""
            ⚠️ **Dados de Stage 13 incompletos**

            Colunas temporais encontradas: {len(available_temporal_cols)}/10

            Colunas disponíveis: {', '.join(available_temporal_cols) if available_temporal_cols else 'Nenhuma'}

            Execute o pipeline completo para obter dados de análise temporal.
            """)

        # Aplicar filtros
        df_filtered = df.copy()

        # Aplicar filtro temporal
        if time_filter != "Todos os períodos":
            if time_filter == "Horário comercial" and 'is_business_hours' in df.columns:
                df_filtered = df_filtered[df_filtered['is_business_hours'] == True]
            elif time_filter == "Fora do horário" and 'is_business_hours' in df.columns:
                df_filtered = df_filtered[df_filtered['is_business_hours'] == False]
            elif time_filter == "Fins de semana" and 'is_weekend' in df.columns:
                df_filtered = df_filtered[df_filtered['is_weekend'] == True]
            elif time_filter == "Dias úteis" and 'is_weekend' in df.columns:
                df_filtered = df_filtered[df_filtered['is_weekend'] == False]

        # Aplicar filtro de coordenação
        if coord_filter > 0.0 and 'temporal_coordination' in df.columns:
            df_filtered = df_filtered[df_filtered['temporal_coordination'] >= coord_filter]

        # Mostrar informações do dataset filtrado
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Registros totais", f"{len(df):,}")

        with col2:
            st.metric("🔍 Registros filtrados", f"{len(df_filtered):,}")

        with col3:
            if len(df) > 0:
                filter_percentage = (len(df_filtered) / len(df)) * 100
                st.metric("📈 % Mantido após filtros", f"{filter_percentage:.1f}%")
            else:
                st.metric("📈 % Mantido após filtros", "0%")

        if len(df_filtered) == 0:
            st.warning("⚠️ Nenhum registro restante após aplicar filtros. Ajuste os critérios de filtro.")
            return

        # Análise temporal principal
        st.markdown("---")

        # Informações do dataset
        with st.expander("ℹ️ Informações do Dataset"):
            st.write(f"**Dataset:** {selected_dataset}")
            st.write(f"**Registros:** {len(df_filtered):,}")
            st.write(f"**Colunas:** {len(df_filtered.columns)}")

            if 'datetime' in df_filtered.columns:
                try:
                    df_temp = df_filtered.copy()
                    df_temp['parsed_datetime'] = pd.to_datetime(df_temp['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    valid_dates = df_temp['parsed_datetime'].dropna()

                    if len(valid_dates) > 0:
                        st.write(f"**Período:** {valid_dates.min().strftime('%Y-%m-%d')} a {valid_dates.max().strftime('%Y-%m-%d')}")
                        st.write(f"**Duração:** {(valid_dates.max() - valid_dates.min()).days} dias")
                except:
                    st.write("**Período:** Não disponível")

            # Mostrar colunas temporais disponíveis
            st.write("**Colunas temporais disponíveis:**")
            for col in temporal_columns:
                status = "✅" if col in df_filtered.columns else "❌"
                st.write(f"{status} {col}")

        # Executar dashboard principal
        main_dashboard(df_filtered)

        # Seção de análise adicional
        st.markdown("---")
        st.subheader("📈 Análise Adicional")

        # Duas colunas para análises extras
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Distribuição Temporal**")
            if 'hour' in df_filtered.columns:
                hour_dist = df_filtered['hour'].value_counts().sort_index()
                st.bar_chart(hour_dist)
            else:
                st.info("Dados de hora não disponíveis")

        with col2:
            st.write("**Coordenação por Período**")
            if 'temporal_coordination' in df_filtered.columns and 'is_business_hours' in df_filtered.columns:
                coord_by_period = df_filtered.groupby('is_business_hours')['temporal_coordination'].mean()
                coord_df = pd.DataFrame({
                    'Período': ['Fora do horário', 'Horário comercial'],
                    'Coordenação': [coord_by_period.get(False, 0), coord_by_period.get(True, 0)]
                })
                st.bar_chart(coord_df.set_index('Período'))
            else:
                st.info("Dados de coordenação por período não disponíveis")

        # Tabela de resumo estatístico
        st.subheader("📊 Resumo Estatístico Temporal")

        temporal_stats = {}

        if 'temporal_coordination' in df_filtered.columns:
            temporal_stats['Coordenação Temporal'] = {
                'Média': df_filtered['temporal_coordination'].mean(),
                'Mediana': df_filtered['temporal_coordination'].median(),
                'Desvio Padrão': df_filtered['temporal_coordination'].std(),
                'Mínimo': df_filtered['temporal_coordination'].min(),
                'Máximo': df_filtered['temporal_coordination'].max()
            }

        if 'sender_frequency' in df_filtered.columns:
            temporal_stats['Frequência do Remetente'] = {
                'Média': df_filtered['sender_frequency'].mean(),
                'Mediana': df_filtered['sender_frequency'].median(),
                'Desvio Padrão': df_filtered['sender_frequency'].std(),
                'Mínimo': df_filtered['sender_frequency'].min(),
                'Máximo': df_filtered['sender_frequency'].max()
            }

        if temporal_stats:
            stats_df = pd.DataFrame(temporal_stats).round(4)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Estatísticas temporais não disponíveis")

        # Footer informativo
        st.markdown("---")
        st.markdown("""
        **Sobre a Análise Temporal:**

        - **Coordenação Temporal**: Mede a sincronização de atividade entre usuários
        - **Eventos Políticos**: Correlaciona picos de atividade com eventos do cenário político brasileiro
        - **Redes de Atividade**: Identifica clusters de usuários com atividade coordenada
        - **Fluxos Integrados**: Conecta padrões temporais com análises de sentimento e affordances

        *Esta análise é parte do Stage 13 do pipeline de análise de discurso político brasileiro.*
        """)

    except Exception as e:
        st.error(f"❌ Erro ao carregar análise temporal: {str(e)}")
        st.info("Verifique se o pipeline foi executado corretamente e se os dados estão disponíveis.")

if __name__ == "__main__":
    main()