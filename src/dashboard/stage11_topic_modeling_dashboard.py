"""
Stage 11 Topic Modeling Dashboard

Professional dashboard for LDA topic modeling analysis of Brazilian political discourse.
Features Sankey flow diagrams and bubble charts for cross-stage analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import os
import glob
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TopicModelingDashboard:
    """Professional topic modeling dashboard for Brazilian political discourse."""

    def __init__(self):
        self.data = None
        self.topic_data = None
        self.cluster_data = None
        self.affordance_data = None

    def load_topic_modeling_data(self) -> Optional[pd.DataFrame]:
        """Load topic modeling results from pipeline outputs."""
        try:
            # Look for topic modeling results
            results_dir = "/Users/pabloalmada/development/project/dataanalysis-bolsonarismo/src/dashboard/data/dashboard_results"

            # Try to find Stage 11 clustering results (which include topic data)
            pattern = os.path.join(results_dir, "*_11_clustering_*.csv")
            topic_files = glob.glob(pattern)

            if not topic_files:
                # Fallback to topic modeling files
                pattern = os.path.join(results_dir, "*_09_topic_modeling_*.csv")
                topic_files = glob.glob(pattern)

            if not topic_files:
                st.error("Nenhum resultado de topic modeling encontrado.")
                return None

            # Use most recent file
            latest_file = max(topic_files, key=os.path.getctime)
            st.info(f"Carregando: {os.path.basename(latest_file)}")

            df = pd.read_csv(latest_file, sep=';', encoding='utf-8')

            # Validate required columns for topic modeling
            required_columns = ['text_content']
            topic_columns = [col for col in df.columns if 'topic' in col.lower()]

            if not topic_columns:
                st.warning("Colunas de tópicos não encontradas. Dados podem estar incompletos.")

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Colunas necessárias ausentes: {missing_columns}")
                return None

            return df

        except Exception as e:
            st.error(f"Erro ao carregar dados de topic modeling: {e}")
            return None

    def prepare_cross_stage_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare data for cross-stage analysis."""
        try:
            cross_data = {}

            # Extract topic data
            if 'dominant_topic' in df.columns or 'topic_id' in df.columns:
                topic_col = 'dominant_topic' if 'dominant_topic' in df.columns else 'topic_id'

                # Build topic data columns dynamically
                topic_columns = [topic_col, 'text_content']

                # Add optional topic columns if they exist
                optional_cols = {
                    'topic_probability': 'topic_probability',
                    'topic_keywords': 'topic_keywords',
                    'topic_label': 'topic_label',
                    'topic_theme': 'topic_theme',
                    'political_alignment': 'political_alignment',
                    'cluster_id': 'cluster_id',
                    'date': 'date'
                }

                for col_key, col_name in optional_cols.items():
                    if col_name in df.columns:
                        topic_columns.append(col_name)

                topic_data = df[topic_columns].dropna(subset=[topic_col])
                cross_data['topics'] = topic_data

            # Extract cluster data
            if 'cluster_id' in df.columns:
                cluster_data = df[['cluster_id', 'cluster_name' if 'cluster_name' in df.columns else 'cluster_id', 'text_content']].dropna(subset=['cluster_id'])
                cross_data['clusters'] = cluster_data

            # Extract affordance data (from Stage 06)
            affordance_cols = [col for col in df.columns if 'affordance' in col.lower() or 'discourse_type' in col.lower()]
            if affordance_cols:
                affordance_data = df[['text_content'] + affordance_cols].dropna()
                cross_data['affordances'] = affordance_data

            # Extract political data
            if 'political_alignment' in df.columns:
                political_data = df[['political_alignment', 'alignment_confidence' if 'alignment_confidence' in df.columns else 'political_alignment', 'text_content']].dropna(subset=['political_alignment'])
                cross_data['politics'] = political_data

            # Extract temporal data
            if 'date' in df.columns:
                temporal_data = df[['date', 'text_content']].dropna(subset=['date'])
                if not temporal_data.empty:
                    temporal_data['date'] = pd.to_datetime(temporal_data['date'], errors='coerce')
                    temporal_data = temporal_data.dropna(subset=['date'])
                    temporal_data['year_month'] = temporal_data['date'].dt.to_period('M')
                    cross_data['temporal'] = temporal_data

            return cross_data

        except Exception as e:
            st.error(f"Erro na preparação de dados cross-stage: {e}")
            return {}

    def create_sankey_diagram(self, cross_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create Sankey diagram showing topic → cluster → affordance flow."""
        try:
            # Prepare data for Sankey
            if 'topics' not in cross_data or 'clusters' not in cross_data:
                st.warning("Dados insuficientes para diagrama Sankey (necessário: tópicos e clusters)")
                return go.Figure()

            topics_df = cross_data['topics']

            # Get topic column name
            topic_col = 'dominant_topic' if 'dominant_topic' in topics_df.columns else 'topic_id'

            # Create flow data
            flows = []

            # Topic → Cluster flows
            if 'cluster_id' in topics_df.columns:
                topic_cluster_flows = topics_df.groupby([topic_col, 'cluster_id']).size().reset_index(name='count')

                for _, row in topic_cluster_flows.iterrows():
                    flows.append({
                        'source': f"Tópico {row[topic_col]}",
                        'target': f"Cluster {row['cluster_id']}",
                        'value': row['count']
                    })

            # Cluster → Affordance flows (if available)
            if 'affordances' in cross_data and 'cluster_id' in topics_df.columns:
                # Merge cluster data with affordance data
                cluster_affordance_data = topics_df.merge(
                    cross_data['affordances'],
                    on='text_content',
                    how='inner'
                )

                affordance_cols = [col for col in cluster_affordance_data.columns if 'discourse_type' in col.lower()]
                if affordance_cols:
                    affordance_col = affordance_cols[0]
                    cluster_affordance_flows = cluster_affordance_data.groupby(['cluster_id', affordance_col]).size().reset_index(name='count')

                    for _, row in cluster_affordance_flows.iterrows():
                        flows.append({
                            'source': f"Cluster {row['cluster_id']}",
                            'target': f"Affordance: {row[affordance_col]}",
                            'value': row['count']
                        })

            if not flows:
                st.warning("Nenhum fluxo identificado para o diagrama Sankey")
                return go.Figure()

            # Create nodes and links
            all_nodes = list(set([flow['source'] for flow in flows] + [flow['target'] for flow in flows]))
            node_dict = {node: i for i, node in enumerate(all_nodes)}

            # Define colors for different node types
            node_colors = []
            for node in all_nodes:
                if 'Tópico' in node:
                    node_colors.append('#3498db')  # Blue for topics
                elif 'Cluster' in node:
                    node_colors.append('#e74c3c')  # Red for clusters
                elif 'Affordance' in node:
                    node_colors.append('#2ecc71')  # Green for affordances
                else:
                    node_colors.append('#95a5a6')  # Gray for others

            # Create Sankey figure
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors
                ),
                link=dict(
                    source=[node_dict[flow['source']] for flow in flows],
                    target=[node_dict[flow['target']] for flow in flows],
                    value=[flow['value'] for flow in flows],
                    color='rgba(128, 128, 128, 0.3)'
                )
            )])

            fig.update_layout(
                title="Fluxo de Tópicos → Clusters → Affordances",
                title_x=0.5,
                font_size=12,
                height=600,
                margin=dict(t=60, l=50, r=50, b=50)
            )

            return fig

        except Exception as e:
            st.error(f"Erro na criação do diagrama Sankey: {e}")
            return go.Figure()

    def create_bubble_chart(self, cross_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create bubble chart for topics vs politics vs temporal intensity."""
        try:
            if 'topics' not in cross_data:
                st.warning("Dados de tópicos não disponíveis para bubble chart")
                return go.Figure()

            topics_df = cross_data['topics']
            topic_col = 'dominant_topic' if 'dominant_topic' in topics_df.columns else 'topic_id'

            # Prepare bubble chart data
            bubble_data = []


            # Group by topic and political alignment
            if 'political_alignment' in topics_df.columns:
                # Prepare aggregation dictionary
                agg_dict = {'text_content': 'count'}
                if 'topic_probability' in topics_df.columns:
                    agg_dict['topic_probability'] = ['mean', 'count']
                # Note: Don't add topic_col to aggregation since it's already in groupby

                grouped = topics_df.groupby([topic_col, 'political_alignment']).agg(agg_dict).reset_index()

                # Flatten column names - handle MultiIndex properly
                if isinstance(grouped.columns, pd.MultiIndex):
                    grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in grouped.columns.values]

                for _, row in grouped.iterrows():
                    # Get topic label if available
                    topic_label = f"Tópico {row[topic_col]}"
                    if 'topic_label' in topics_df.columns:
                        topic_labels = topics_df[topics_df[topic_col] == row[topic_col]]['topic_label'].iloc[0] if len(topics_df[topics_df[topic_col] == row[topic_col]]) > 0 else f"Tópico {row[topic_col]}"
                        topic_label = topic_labels

                    bubble_data.append({
                        'topic': topic_label,
                        'political_alignment': row['political_alignment'],
                        'topic_intensity': row.get('topic_probability_mean', row['text_content']),
                        'frequency': row['text_content'],
                        'avg_probability': row.get('topic_probability_mean', 0.5)
                    })
            else:
                # Fallback: just topic distribution
                # Prepare aggregation dictionary
                agg_dict = {'text_content': 'count'}
                if 'topic_probability' in topics_df.columns:
                    agg_dict['topic_probability'] = ['mean', 'count']
                # Note: Don't add topic_col to aggregation since it's already in groupby

                grouped = topics_df.groupby(topic_col).agg(agg_dict).reset_index()

                # Flatten column names - handle MultiIndex properly
                if isinstance(grouped.columns, pd.MultiIndex):
                    grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in grouped.columns.values]

                for _, row in grouped.iterrows():
                    # Get topic label if available
                    topic_label = f"Tópico {row[topic_col]}"
                    if 'topic_label' in topics_df.columns:
                        topic_labels = topics_df[topics_df[topic_col] == row[topic_col]]['topic_label'].iloc[0] if len(topics_df[topics_df[topic_col] == row[topic_col]]) > 0 else f"Tópico {row[topic_col]}"
                        topic_label = topic_labels

                    bubble_data.append({
                        'topic': topic_label,
                        'political_alignment': 'Indefinido',
                        'topic_intensity': row.get('topic_probability_mean', row['text_content']),
                        'frequency': row['text_content'],
                        'avg_probability': row.get('topic_probability_mean', 0.5)
                    })

            if not bubble_data:
                st.warning("Nenhum dado disponível para bubble chart")
                return go.Figure()

            bubble_df = pd.DataFrame(bubble_data)

            # Create bubble chart
            fig = px.scatter(
                bubble_df,
                x='political_alignment',
                y='topic_intensity',
                size='frequency',
                color='topic',
                hover_data=['avg_probability', 'frequency'],
                title="Tópicos vs Orientação Política vs Intensidade Temporal"
            )

            fig.update_layout(
                xaxis_title="Orientação Política",
                yaxis_title="Intensidade do Tópico",
                height=600,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )

            # Update hover template
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                             "Orientação Política: %{x}<br>" +
                             "Intensidade: %{y:.3f}<br>" +
                             "Frequência: %{marker.size}<br>" +
                             "Probabilidade Média: %{customdata[0]:.3f}<br>" +
                             "<extra></extra>"
            )

            return fig

        except Exception as e:
            st.error(f"Erro na criação do bubble chart: {e}")
            return go.Figure()

    def create_topic_summary_stats(self, cross_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create statistical summary of topic modeling results."""
        try:
            if 'topics' not in cross_data:
                return {}

            topics_df = cross_data['topics']
            topic_col = 'dominant_topic' if 'dominant_topic' in topics_df.columns else 'topic_id'

            stats = {
                'total_documents': len(topics_df),
                'unique_topics': topics_df[topic_col].nunique(),
                'topic_distribution': topics_df[topic_col].value_counts().to_dict()
            }

            if 'topic_probability' in topics_df.columns:
                stats['avg_topic_probability'] = topics_df['topic_probability'].mean()
                stats['topic_probability_std'] = topics_df['topic_probability'].std()

            if 'political_alignment' in topics_df.columns:
                stats['political_distribution'] = topics_df['political_alignment'].value_counts().to_dict()

            if 'cluster_id' in topics_df.columns:
                stats['unique_clusters'] = topics_df['cluster_id'].nunique()
                stats['topic_cluster_matrix'] = pd.crosstab(topics_df[topic_col], topics_df['cluster_id']).to_dict()

            return stats

        except Exception as e:
            st.error(f"Erro no cálculo de estatísticas: {e}")
            return {}

    def run(self):
        """Run the topic modeling dashboard."""
        st.title("🏷️ Stage 11: Topic Modeling Analysis")
        st.markdown("---")

        # Load data
        with st.spinner("Carregando dados de topic modeling..."):
            self.data = self.load_topic_modeling_data()

        if self.data is None:
            st.stop()

        # Prepare cross-stage data
        with st.spinner("Preparando dados para análise cross-stage..."):
            cross_data = self.prepare_cross_stage_data(self.data)

        if not cross_data:
            st.error("Não foi possível preparar dados para análise cross-stage")
            st.stop()

        # Sidebar controls
        st.sidebar.header("Controles de Análise")

        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Tipo de Análise",
            ["Visão Geral", "Fluxo Sankey", "Análise Multidimensional", "Estatísticas Detalhadas"]
        )

        # Main content based on analysis type
        if analysis_type == "Visão Geral":
            self.show_overview(cross_data)
        elif analysis_type == "Fluxo Sankey":
            self.show_sankey_analysis(cross_data)
        elif analysis_type == "Análise Multidimensional":
            self.show_bubble_analysis(cross_data)
        elif analysis_type == "Estatísticas Detalhadas":
            self.show_detailed_statistics(cross_data)

    def show_overview(self, cross_data: Dict[str, pd.DataFrame]):
        """Show overview of topic modeling results."""
        st.subheader("📊 Visão Geral do Topic Modeling")

        # Calculate basic statistics
        stats = self.create_topic_summary_stats(cross_data)

        if stats:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total de Documentos", stats.get('total_documents', 0))
            with col2:
                st.metric("Tópicos Únicos", stats.get('unique_topics', 0))
            with col3:
                if 'avg_topic_probability' in stats:
                    st.metric("Probabilidade Média", f"{stats['avg_topic_probability']:.3f}")

        # Topic distribution
        if 'topics' in cross_data:
            topics_df = cross_data['topics']
            topic_col = 'dominant_topic' if 'dominant_topic' in topics_df.columns else 'topic_id'

            fig = px.histogram(
                topics_df,
                x=topic_col,
                title="Distribuição de Tópicos",
                labels={topic_col: "Tópico", "count": "Frequência"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def show_sankey_analysis(self, cross_data: Dict[str, pd.DataFrame]):
        """Show Sankey flow analysis."""
        st.subheader("🌊 Análise de Fluxo: Tópicos → Clusters → Affordances")

        fig = self.create_sankey_diagram(cross_data)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretação do Diagrama Sankey:**
            - **Tópicos (Azul)**: Tópicos descobertos pelo LDA
            - **Clusters (Vermelho)**: Grupos semânticos de documentos
            - **Affordances (Verde)**: Tipos de affordances discursivas

            A espessura das conexões indica o volume de documentos que fluem entre as categorias.
            """)
        else:
            st.info("Diagrama Sankey não disponível com os dados atuais")

    def show_bubble_analysis(self, cross_data: Dict[str, pd.DataFrame]):
        """Show bubble chart analysis."""
        st.subheader("🎈 Análise Multidimensional: Tópicos vs Política vs Intensidade")

        fig = self.create_bubble_chart(cross_data)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretação do Bubble Chart:**
            - **Eixo X**: Orientação política identificada
            - **Eixo Y**: Intensidade/probabilidade do tópico
            - **Tamanho**: Frequência de documentos
            - **Cor**: Diferentes tópicos descobertos

            Bubbles maiores indicam tópicos mais frequentes; posição vertical indica a intensidade do tópico.
            """)
        else:
            st.info("Bubble chart não disponível com os dados atuais")

    def show_detailed_statistics(self, cross_data: Dict[str, pd.DataFrame]):
        """Show detailed statistics."""
        st.subheader("📈 Estatísticas Detalhadas")

        stats = self.create_topic_summary_stats(cross_data)

        if stats:
            # Topic distribution table
            if 'topic_distribution' in stats:
                st.subheader("Distribuição de Tópicos")
                topic_dist_df = pd.DataFrame(
                    list(stats['topic_distribution'].items()),
                    columns=['Tópico', 'Frequência']
                )
                topic_dist_df['Percentual'] = (topic_dist_df['Frequência'] / topic_dist_df['Frequência'].sum() * 100).round(2)
                st.dataframe(topic_dist_df, use_container_width=True)

            # Political distribution
            if 'political_distribution' in stats:
                st.subheader("Distribuição Política")
                pol_dist_df = pd.DataFrame(
                    list(stats['political_distribution'].items()),
                    columns=['Orientação Política', 'Frequência']
                )
                pol_dist_df['Percentual'] = (pol_dist_df['Frequência'] / pol_dist_df['Frequência'].sum() * 100).round(2)
                st.dataframe(pol_dist_df, use_container_width=True)

            # Topic-Cluster cross-tabulation
            if 'topic_cluster_matrix' in stats:
                st.subheader("Matriz Tópico-Cluster")
                matrix_df = pd.DataFrame(stats['topic_cluster_matrix'])
                if not matrix_df.empty:
                    st.dataframe(matrix_df, use_container_width=True)

        # Data preview
        if 'topics' in cross_data:
            st.subheader("Prévia dos Dados")
            preview_df = cross_data['topics'].head(10)
            st.dataframe(preview_df, use_container_width=True)