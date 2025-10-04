"""
Stage 03 Deduplication Visualizations Dashboard
===============================================

Professional dashboard component focused on cross-dataset deduplication analysis
for Brazilian political discourse research (digiNEV v.final). Visualizes patterns
of duplicate content propagation, frequency distributions, and temporal dynamics
across the five political datasets.

🎯 FOCUS: Academic-quality visualizations of deduplication results
📊 SCOPE: Stage 03 cross-dataset deduplication analysis
🔍 DATA: Real deduplication metrics from pipeline processing
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from collections import Counter

# Simple validation functions
def validate_data(df):
    """Simple validation for dashboard data"""
    if df is None or len(df) == 0:
        return None
    return df

class Stage03DeduplicationDashboard:
    """Dashboard para análise de deduplicação cross-dataset"""

    def __init__(self):
        """Inicializa o dashboard de deduplicação"""
        self.project_root = Path(__file__).parent.parent.parent
        self.dashboard_results_path = self.project_root / "src" / "dashboard" / "data" / "dashboard_results"

        # Mapear datasets para nomes descritivos
        self.dataset_names = {
            '1_2019-2021-govbolso.csv': 'Governo Bolsonaro\n(2019-2021)',
            '2_2021-2022-pandemia.csv': 'Pandemia\n(2021-2022)',
            '3_2022-2023-poseleic.csv': 'Pós-Eleição\n(2022-2023)',
            '4_2022-2023-elec.csv': 'Eleições\n(2022-2023)',
            '5_2022-2023-elec-extra.csv': 'Eleições Extra\n(2022-2023)'
        }

        # Cores consistentes para datasets
        self.dataset_colors = {
            'Governo Bolsonaro\n(2019-2021)': '#1f77b4',
            'Pandemia\n(2021-2022)': '#ff7f0e',
            'Pós-Eleição\n(2022-2023)': '#2ca02c',
            'Eleições\n(2022-2023)': '#d62728',
            'Eleições Extra\n(2022-2023)': '#9467bd'
        }

    def load_deduplication_data(self) -> Optional[pd.DataFrame]:
        """Carrega dados reais de deduplicação processados"""
        try:
            # Buscar arquivos de dados processados com deduplicação
            data_files = [
                self.project_root / "data" / "processed" / "processed_1_2019-2021-govbolso.csv",
                self.project_root / "data" / "processed" / "processed_2_2021-2022-pandemia.csv"
            ]

            combined_data = []

            for file_path in data_files:
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path, sep=';', quoting=1, low_memory=False)

                        # Verificar se tem colunas de deduplicação
                        if 'dupli_freq' in df.columns:
                            # Extrair nome do dataset do arquivo
                            dataset_name = file_path.name.replace('processed_', '').replace('.csv', '.csv')
                            df['dataset_source'] = self.dataset_names.get(dataset_name, dataset_name)

                            # Converter datetime se presente
                            if 'datetime_parsed' in df.columns:
                                df['datetime_parsed'] = pd.to_datetime(df['datetime_parsed'], errors='coerce')
                            elif 'datetime' in df.columns:
                                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                                df['datetime_parsed'] = df['datetime']

                            combined_data.append(df)

                    except Exception as e:
                        st.warning(f"Erro carregando {file_path.name}: {e}")
                        continue

            if combined_data:
                final_df = pd.concat(combined_data, ignore_index=True)
                return validate_data(final_df)

            return None

        except Exception as e:
            st.error(f"Erro carregando dados de deduplicação: {e}")
            return None

    def create_duplicate_frequency_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Cria heatmap de frequência de duplicatas entre datasets"""

        # Preparar dados para heatmap
        duplicate_matrix = []
        dataset_names = sorted(df['dataset_source'].unique())

        # Calcular sobreposição entre datasets
        for dataset1 in dataset_names:
            row = []
            for dataset2 in dataset_names:
                if dataset1 == dataset2:
                    # Duplicatas internas do dataset
                    internal_dups = df[
                        (df['dataset_source'] == dataset1) & (df['dupli_freq'] > 1)
                    ]['dupli_freq'].sum()
                    row.append(internal_dups)
                else:
                    # Textos compartilhados entre datasets
                    texts_d1 = set(df[df['dataset_source'] == dataset1]['normalized_text'].dropna())
                    texts_d2 = set(df[df['dataset_source'] == dataset2]['normalized_text'].dropna())
                    shared_count = len(texts_d1.intersection(texts_d2))
                    row.append(shared_count)
            duplicate_matrix.append(row)

        # Criar heatmap
        fig = go.Figure(data=go.Heatmap(
            z=duplicate_matrix,
            x=dataset_names,
            y=dataset_names,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Duplicatas: %{z}<extra></extra>'
        ))

        # Adicionar anotações com valores
        for i in range(len(dataset_names)):
            for j in range(len(dataset_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(duplicate_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="white" if duplicate_matrix[i][j] > max(max(row) for row in duplicate_matrix) * 0.5 else "black")
                )

        fig.update_layout(
            title={
                'text': "Matriz de Frequência de Duplicatas entre Datasets<br><sub>Diagonal: duplicatas internas | Off-diagonal: conteúdo compartilhado</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Dataset Destino",
            yaxis_title="Dataset Origem",
            font=dict(family="Arial", size=12),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_duplicate_content_clustering(self, df: pd.DataFrame) -> go.Figure:
        """Visualiza clustering de conteúdo duplicado"""

        # Analisar padrões de duplicação
        dup_analysis = df[df['dupli_freq'] > 1].copy()

        if len(dup_analysis) == 0:
            # Figura vazia se não há duplicatas
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhuma duplicata encontrada nos dados carregados",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Clustering de Conteúdo Duplicado",
                height=400
            )
            return fig

        # Agrupar por frequência de duplicação
        dup_groups = dup_analysis.groupby('dupli_freq').agg({
            'normalized_text': 'count',
            'channels_found': 'mean',
            'date_span_days': 'mean',
            'dataset_source': lambda x: len(set(x))
        }).reset_index()

        dup_groups.columns = ['freq_duplicatas', 'qtd_textos', 'canais_medio', 'dias_medio', 'datasets_envolvidos']

        # Criar scatter plot 3D
        fig = go.Figure(data=go.Scatter3d(
            x=dup_groups['freq_duplicatas'],
            y=dup_groups['canais_medio'],
            z=dup_groups['dias_medio'],
            mode='markers+text',
            marker=dict(
                size=dup_groups['qtd_textos'] * 2,
                color=dup_groups['datasets_envolvidos'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Datasets<br>Envolvidos"),
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=dup_groups['qtd_textos'],
            textposition="middle center",
            hovertemplate=(
                '<b>Padrão de Duplicação</b><br>'
                'Frequência: %{x}<br>'
                'Canais (média): %{y:.1f}<br>'
                'Período (dias): %{z:.1f}<br>'
                'Quantidade de textos: %{text}<br>'
                '<extra></extra>'
            )
        ))

        fig.update_layout(
            title={
                'text': "Clustering de Padrões de Duplicação<br><sub>Tamanho = quantidade de textos | Cor = datasets envolvidos</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title="Frequência de Duplicação",
                yaxis_title="Canais (média)",
                zaxis_title="Período de Propagação (dias)",
                bgcolor='white'
            ),
            font=dict(family="Arial", size=12),
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_temporal_duplicate_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Cria visualização da distribuição temporal de duplicatas"""

        if 'datetime_parsed' not in df.columns:
            # Figura vazia se não há dados temporais
            fig = go.Figure()
            fig.add_annotation(
                text="Dados temporais não disponíveis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Distribuição Temporal de Duplicatas",
                height=400
            )
            return fig

        # Filtrar dados com datetime válido
        temporal_data = df[df['datetime_parsed'].notna()].copy()

        if len(temporal_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhum dado temporal válido encontrado",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Distribuição Temporal de Duplicatas", height=400)
            return fig

        # Criar bins temporais mensais
        temporal_data['year_month'] = temporal_data['datetime_parsed'].dt.to_period('M')

        # Agrupar por período e dataset
        temporal_agg = temporal_data.groupby(['year_month', 'dataset_source']).agg({
            'dupli_freq': ['count', 'sum', 'mean']
        }).reset_index()

        temporal_agg.columns = ['periodo', 'dataset', 'qtd_registros', 'total_duplicatas', 'freq_media']
        temporal_agg['periodo_str'] = temporal_agg['periodo'].astype(str)

        # Criar subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Quantidade de Registros por Período', 'Frequência Média de Duplicação'),
            vertical_spacing=0.1
        )

        # Gráfico 1: Quantidade de registros
        for dataset in temporal_agg['dataset'].unique():
            data_subset = temporal_agg[temporal_agg['dataset'] == dataset]
            fig.add_trace(
                go.Scatter(
                    x=data_subset['periodo_str'],
                    y=data_subset['qtd_registros'],
                    mode='lines+markers',
                    name=dataset,
                    line=dict(color=self.dataset_colors.get(dataset, '#636EFA'), width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Registros: %{y}<extra></extra>'
                ),
                row=1, col=1
            )

        # Gráfico 2: Frequência média
        for dataset in temporal_agg['dataset'].unique():
            data_subset = temporal_agg[temporal_agg['dataset'] == dataset]
            fig.add_trace(
                go.Scatter(
                    x=data_subset['periodo_str'],
                    y=data_subset['freq_media'],
                    mode='lines+markers',
                    name=dataset,
                    line=dict(color=self.dataset_colors.get(dataset, '#636EFA'), width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Freq. Média: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )

        fig.update_layout(
            title={
                'text': "Evolução Temporal de Duplicatas por Dataset",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=700,
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )

        # Atualizar eixos
        fig.update_xaxes(title_text="Período (Ano-Mês)", row=2, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Frequência Média", row=2, col=1)

        return fig

    def create_shared_content_flow(self, df: pd.DataFrame) -> go.Figure:
        """Cria diagrama de fluxo de conteúdo compartilhado"""

        # Analisar fluxo entre datasets
        shared_content = df[df['dupli_freq'] > 1].copy()

        if len(shared_content) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhum conteúdo compartilhado encontrado",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Fluxo de Conteúdo Compartilhado", height=400)
            return fig

        # Preparar dados para Sankey
        datasets = list(df['dataset_source'].unique())

        # Calcular fluxos entre datasets baseado em duplicatas
        flows = []

        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets):
                if i != j:  # Excluir autofluxos
                    # Contar textos compartilhados
                    texts_d1 = set(df[df['dataset_source'] == dataset1]['normalized_text'].dropna())
                    texts_d2 = set(df[df['dataset_source'] == dataset2]['normalized_text'].dropna())
                    shared = len(texts_d1.intersection(texts_d2))

                    if shared > 0:
                        flows.append({
                            'source': i,
                            'target': j + len(datasets),  # Offset para targets
                            'value': shared,
                            'source_name': dataset1,
                            'target_name': dataset2
                        })

        if not flows:
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhum fluxo detectado entre datasets",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Fluxo de Conteúdo Compartilhado", height=400)
            return fig

        # Criar labels para Sankey (sources + targets)
        labels = datasets + [f"{ds} (recebe)" for ds in datasets]

        # Cores para nós
        node_colors = []
        for dataset in datasets:
            node_colors.append(self.dataset_colors.get(dataset, '#636EFA'))
        for dataset in datasets:
            node_colors.append(self.dataset_colors.get(dataset, '#636EFA'))

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=[flow['source'] for flow in flows],
                target=[flow['target'] for flow in flows],
                value=[flow['value'] for flow in flows],
                color='rgba(135, 206, 235, 0.6)'
            )
        )])

        fig.update_layout(
            title={
                'text': "Fluxo de Conteúdo Compartilhado entre Datasets<br><sub>Largura das conexões = quantidade de textos compartilhados</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(family="Arial", size=12),
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_duplicate_propagation_patterns(self, df: pd.DataFrame) -> go.Figure:
        """Analisa padrões de propagação de duplicatas"""

        # Analisar velocidade de propagação
        propagation_data = df[df['dupli_freq'] > 1].copy()

        if len(propagation_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhum padrão de propagação detectado",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Padrões de Propagação de Duplicatas", height=400)
            return fig

        # Categorizar velocidade de propagação
        propagation_data['propagation_speed'] = pd.cut(
            propagation_data['date_span_days'],
            bins=[-1, 0, 1, 7, 30, float('inf')],
            labels=['Instantânea', 'Mesmo Dia', '1 Semana', '1 Mês', 'Longo Prazo']
        )

        # Categorizar alcance por canais
        propagation_data['reach_category'] = pd.cut(
            propagation_data['channels_found'],
            bins=[-1, 1, 3, 10, float('inf')],
            labels=['Canal Único', 'Poucos Canais', 'Vários Canais', 'Ampla Distribuição']
        )

        # Criar matriz de propagação
        prop_matrix = propagation_data.groupby(['propagation_speed', 'reach_category']).agg({
            'dupli_freq': ['count', 'mean']
        }).reset_index()

        prop_matrix.columns = ['velocidade', 'alcance', 'quantidade', 'freq_media']

        # Pivot para heatmap
        heatmap_data = prop_matrix.pivot(index='alcance', columns='velocidade', values='quantidade').fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Reds',
            showscale=True,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Velocidade: %{x}<br>Quantidade: %{z}<extra></extra>'
        ))

        # Adicionar anotações
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(int(heatmap_data.iloc[i, j])),
                    showarrow=False,
                    font=dict(
                        color="white" if heatmap_data.iloc[i, j] > heatmap_data.values.max() * 0.5 else "black",
                        size=12
                    )
                )

        fig.update_layout(
            title={
                'text': "Padrões de Propagação: Velocidade vs Alcance<br><sub>Valores = quantidade de casos observados</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Velocidade de Propagação",
            yaxis_title="Alcance por Canais",
            font=dict(family="Arial", size=12),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def render_dashboard(self):
        """Renderiza o dashboard completo de deduplicação Stage 03"""

        st.markdown("### 🔄 Stage 03: Análise de Deduplicação Cross-Dataset")
        st.markdown("---")

        # Carregar dados
        with st.spinner("Carregando dados de deduplicação..."):
            df = self.load_deduplication_data()

        if df is None or len(df) == 0:
            st.warning("⚠️ Nenhum dado de deduplicação encontrado. Execute o pipeline primeiro.")
            return

        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_records = len(df)
            st.metric("Total de Registros", f"{total_records:,}")

        with col2:
            duplicates = len(df[df['dupli_freq'] > 1])
            dup_rate = (duplicates / total_records * 100) if total_records > 0 else 0
            st.metric("Registros com Duplicatas", f"{duplicates:,}", f"{dup_rate:.1f}%")

        with col3:
            total_dup_freq = df['dupli_freq'].sum()
            reduction = ((total_dup_freq - total_records) / total_dup_freq * 100) if total_dup_freq > 0 else 0
            st.metric("Redução por Deduplicação", f"{reduction:.1f}%")

        with col4:
            datasets_count = df['dataset_source'].nunique()
            st.metric("Datasets Processados", datasets_count)

        st.markdown("---")

        # Visualizações
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔥 Heatmap de Frequência",
            "🎯 Clustering de Conteúdo",
            "📅 Distribuição Temporal",
            "🔗 Fluxo de Conteúdo",
            "⚡ Padrões de Propagação"
        ])

        with tab1:
            st.plotly_chart(
                self.create_duplicate_frequency_heatmap(df),
                use_container_width=True
            )

            with st.expander("ℹ️ Interpretação do Heatmap"):
                st.markdown("""
                - **Diagonal**: Duplicatas internas de cada dataset
                - **Off-diagonal**: Conteúdo compartilhado entre datasets
                - **Cores mais escuras**: Maior sobreposição de conteúdo
                - **Padrão esperado**: Datasets temporalmente próximos têm mais sobreposição
                """)

        with tab2:
            st.plotly_chart(
                self.create_duplicate_content_clustering(df),
                use_container_width=True
            )

            with st.expander("ℹ️ Interpretação do Clustering"):
                st.markdown("""
                - **Eixo X**: Frequência de duplicação (quantas vezes o texto aparece)
                - **Eixo Y**: Número médio de canais onde aparece
                - **Eixo Z**: Período de propagação em dias
                - **Tamanho**: Quantidade de textos com esse padrão
                - **Cor**: Número de datasets envolvidos
                """)

        with tab3:
            st.plotly_chart(
                self.create_temporal_duplicate_distribution(df),
                use_container_width=True
            )

            with st.expander("ℹ️ Interpretação Temporal"):
                st.markdown("""
                - **Gráfico Superior**: Volume de registros por período
                - **Gráfico Inferior**: Frequência média de duplicação
                - **Tendências**: Identificar picos de atividade e propagação
                - **Comparação**: Diferenças entre datasets por período
                """)

        with tab4:
            st.plotly_chart(
                self.create_shared_content_flow(df),
                use_container_width=True
            )

            with st.expander("ℹ️ Interpretação do Fluxo"):
                st.markdown("""
                - **Largura das conexões**: Quantidade de conteúdo compartilhado
                - **Direção**: Fluxo de conteúdo entre datasets
                - **Padrões**: Identificar datasets que são fontes vs receptores
                - **Coordenação**: Evidências de distribuição coordenada
                """)

        with tab5:
            st.plotly_chart(
                self.create_duplicate_propagation_patterns(df),
                use_container_width=True
            )

            with st.expander("ℹ️ Interpretação da Propagação"):
                st.markdown("""
                - **Eixo X**: Velocidade de propagação (instantânea até longo prazo)
                - **Eixo Y**: Alcance por canais (único até ampla distribuição)
                - **Valores**: Quantidade de casos observados
                - **Padrões**: Identificar tipos de propagação predominantes
                """)

def main():
    """Função principal para execução standalone"""
    st.set_page_config(
        page_title="Stage 03 - Deduplicação Cross-Dataset",
        page_icon="🔄",
        layout="wide"
    )

    dashboard = Stage03DeduplicationDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()