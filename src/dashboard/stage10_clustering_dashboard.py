"""
Stage 10 Clustering Analysis Dashboard

Professional dashboard for K-Means clustering analysis of Brazilian political discourse.
Features 2D projection visualization, interactive cluster selection, and profile analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import glob
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalysisDashboard:
    """Professional clustering analysis dashboard for Brazilian political discourse."""

    def __init__(self):
        self.data = None
        self.tfidf_matrix = None
        self.cluster_profiles = None

    def load_clustering_data(self) -> Optional[pd.DataFrame]:
        """Load clustering analysis results from pipeline outputs."""
        try:
            # Look for clustering results
            results_dir = "/Users/pabloalmada/development/project/dataanalysis-bolsonarismo/src/dashboard/data/dashboard_results"
            pattern = os.path.join(results_dir, "*_11_clustering_*.csv")
            clustering_files = glob.glob(pattern)

            if not clustering_files:
                st.error("Nenhum resultado de clustering encontrado.")
                return None

            # Use most recent file
            latest_file = max(clustering_files, key=os.path.getctime)

            df = pd.read_csv(latest_file, sep=';', encoding='utf-8')

            # Validate required columns
            required_columns = ['cluster_id', 'text_content', 'political_orientation']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Colunas necessárias ausentes: {missing_columns}")
                return None

            return df

        except Exception as e:
            st.error(f"Erro ao carregar dados de clustering: {e}")
            return None

    def prepare_tfidf_matrix(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare TF-IDF matrix for dimension reduction."""
        try:
            # Filter valid text content
            valid_texts = df['text_content'].dropna()
            if len(valid_texts) < 10:
                st.warning("Dados de texto insuficientes para análise TF-IDF")
                return None

            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            return tfidf_matrix.toarray()

        except Exception as e:
            st.error(f"Erro na preparação TF-IDF: {e}")
            return None

    def perform_dimension_reduction(self, tfidf_matrix: np.ndarray, method: str = 'PCA') -> Optional[np.ndarray]:
        """Perform dimension reduction for 2D visualization."""
        try:
            if method == 'PCA':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 't-SNE':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tfidf_matrix)-1))
            else:
                raise ValueError(f"Método não suportado: {method}")

            # Apply dimension reduction
            coords_2d = reducer.fit_transform(tfidf_matrix)
            return coords_2d

        except Exception as e:
            st.error(f"Erro na redução dimensional ({method}): {e}")
            return None

    def create_2d_scatter_plot(self, df: pd.DataFrame, coords_2d: np.ndarray, method: str) -> go.Figure:
        """Create interactive 2D scatter plot of document clusters."""
        try:
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1],
                'cluster_id': df['cluster_id'].iloc[:len(coords_2d)],
                'political_orientation': df['political_orientation'].iloc[:len(coords_2d)],
                'text_preview': df['text_content'].iloc[:len(coords_2d)].str[:100] + '...',
                'channel': df.get('channel', 'N/A').iloc[:len(coords_2d)]
            })

            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color='cluster_id',
                hover_data=['political_orientation', 'channel'],
                hover_name='text_preview',
                title=f'Projeção 2D dos Documentos - {method}',
                labels={
                    'x': f'{method} Componente 1',
                    'y': f'{method} Componente 2',
                    'cluster_id': 'Cluster ID'
                }
            )

            # Update layout for professional appearance
            fig.update_layout(
                template='plotly_white',
                height=600,
                font=dict(size=12),
                title_font_size=16,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )

            # Update markers
            fig.update_traces(
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            )

            return fig

        except Exception as e:
            st.error(f"Erro na criação do scatter plot: {e}")
            return go.Figure()

    def calculate_cluster_profiles(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive cluster profiles."""
        try:
            profiles = {}

            # Group by cluster
            for cluster_id in df['cluster_id'].unique():
                cluster_data = df[df['cluster_id'] == cluster_id]

                profile = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100
                }

                # Political alignment distribution
                if 'political_orientation' in df.columns:
                    political_dist = cluster_data['political_orientation'].value_counts(normalize=True) * 100
                    profile['political_distribution'] = political_dist.to_dict()

                # Content characteristics
                if 'spacy_tokens_count' in df.columns:
                    profile['avg_tokens'] = cluster_data['spacy_tokens_count'].mean()

                if 'spacy_linguistic_complexity' in df.columns:
                    profile['avg_complexity'] = cluster_data['spacy_linguistic_complexity'].mean()

                if 'sentiment_score' in df.columns:
                    profile['avg_sentiment'] = cluster_data['sentiment_score'].mean()

                # Discourse characteristics
                if 'discourse_type' in df.columns:
                    discourse_dist = cluster_data['discourse_type'].value_counts(normalize=True) * 100
                    profile['discourse_distribution'] = discourse_dist.to_dict()

                # Channel distribution
                if 'channel' in df.columns:
                    channel_dist = cluster_data['channel'].value_counts(normalize=True) * 100
                    profile['channel_distribution'] = channel_dist.to_dict()

                profiles[cluster_id] = profile

            return profiles

        except Exception as e:
            st.error(f"Erro no cálculo dos perfis de cluster: {e}")
            return {}

    def create_radar_chart(self, profiles: Dict, selected_clusters: List[int]) -> go.Figure:
        """Create radar chart comparing cluster profiles."""
        try:
            if not selected_clusters:
                return go.Figure().add_annotation(
                    text="Selecione clusters para comparação",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )

            fig = go.Figure()

            # Define categories for radar chart
            categories = [
                'Tamanho (%)',
                'Complexidade Média',
                'Tokens Médios',
                'Sentimento Médio',
                'Diversidade Política',
                'Intensidade Discursiva'
            ]

            colors = px.colors.qualitative.Set1

            for i, cluster_id in enumerate(selected_clusters):
                if cluster_id not in profiles:
                    continue

                profile = profiles[cluster_id]

                # Normalize values for radar chart (0-100 scale)
                values = []

                # Size percentage (already 0-100)
                values.append(profile.get('percentage', 0))

                # Complexity (0-1 scale, convert to 0-100)
                complexity = profile.get('avg_complexity', 0) * 100
                values.append(min(100, max(0, complexity)))

                # Tokens (normalize to 0-100 based on typical range)
                tokens = profile.get('avg_tokens', 0)
                normalized_tokens = min(100, (tokens / 50) * 100)  # Assume 50 tokens as midpoint
                values.append(normalized_tokens)

                # Sentiment (convert -1 to 1 scale to 0-100)
                sentiment = profile.get('avg_sentiment', 0)
                normalized_sentiment = ((sentiment + 1) / 2) * 100
                values.append(normalized_sentiment)

                # Political diversity (based on entropy of distribution)
                political_dist = profile.get('political_distribution', {})
                if political_dist:
                    probs = list(political_dist.values())
                    entropy = -sum(p/100 * np.log(p/100 + 1e-10) for p in probs if p > 0)
                    max_entropy = np.log(len(probs)) if len(probs) > 1 else 1
                    diversity = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
                else:
                    diversity = 0
                values.append(diversity)

                # Discourse intensity (placeholder - could be based on various factors)
                discourse_dist = profile.get('discourse_distribution', {})
                intensity = len(discourse_dist) * 20  # Simple measure
                values.append(min(100, intensity))

                # Close the radar chart
                values_closed = values + [values[0]]
                categories_closed = categories + [categories[0]]

                fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill='toself',
                    name=f'Cluster {cluster_id}',
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)],
                    opacity=0.3
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        ticksuffix='%'
                    )
                ),
                showlegend=True,
                title="Perfil Comparativo dos Clusters",
                template='plotly_white',
                height=500,
                font=dict(size=12)
            )

            return fig

        except Exception as e:
            st.error(f"Erro na criação do radar chart: {e}")
            return go.Figure()

    def display_cluster_statistics(self, profiles: Dict):
        """Display comprehensive cluster statistics."""
        try:
            st.subheader("📊 Estatísticas dos Clusters")

            # Create summary table
            summary_data = []
            for cluster_id, profile in profiles.items():
                summary_data.append({
                    'Cluster': f"Cluster {cluster_id}",
                    'Tamanho': profile['size'],
                    'Porcentagem': f"{profile['percentage']:.1f}%",
                    'Tokens Médios': f"{profile.get('avg_tokens', 0):.1f}",
                    'Complexidade': f"{profile.get('avg_complexity', 0):.3f}",
                    'Sentimento': f"{profile.get('avg_sentiment', 0):.3f}"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Detailed breakdown for selected cluster
            selected_cluster = st.selectbox(
                "Selecione um cluster para análise detalhada:",
                options=list(profiles.keys()),
                format_func=lambda x: f"Cluster {x}"
            )

            if selected_cluster is not None:
                profile = profiles[selected_cluster]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Distribuição Política:**")
                    political_dist = profile.get('political_distribution', {})
                    if political_dist:
                        for alignment, percentage in political_dist.items():
                            st.write(f"- {alignment}: {percentage:.1f}%")
                    else:
                        st.write("Dados não disponíveis")

                with col2:
                    st.write("**Distribuição por Canal:**")
                    channel_dist = profile.get('channel_distribution', {})
                    if channel_dist:
                        # Show top 5 channels
                        top_channels = dict(list(channel_dist.items())[:5])
                        for channel, percentage in top_channels.items():
                            st.write(f"- {channel}: {percentage:.1f}%")
                    else:
                        st.write("Dados não disponíveis")

        except Exception as e:
            st.error(f"Erro na exibição das estatísticas: {e}")

    def run(self):
        """Main dashboard execution."""
        st.title("🎯 Stage 10: Análise de Clustering")
        st.markdown("---")

        # Load data
        df = self.load_clustering_data()
        if df is None:
            return

        # Display data overview
        st.subheader("📋 Visão Geral dos Dados")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de Documentos", len(df))
        with col2:
            n_clusters = df['cluster_id'].nunique()
            st.metric("Número de Clusters", n_clusters)
        with col3:
            avg_cluster_size = len(df) / n_clusters
            st.metric("Tamanho Médio do Cluster", f"{avg_cluster_size:.1f}")
        with col4:
            # Most common political alignment
            if 'political_orientation' in df.columns:
                most_common = df['political_orientation'].mode()[0] if not df['political_orientation'].mode().empty else "N/A"
                st.metric("Alinhamento Mais Comum", most_common)

        st.markdown("---")

        # Dimension reduction method selection
        st.subheader("🔬 Configurações de Análise")
        reduction_method = st.selectbox(
            "Selecione o método de redução dimensional:",
            ['PCA', 't-SNE'],
            help="PCA preserva variância global, t-SNE preserva estrutura local"
        )

        # Prepare TF-IDF matrix
        with st.spinner("Preparando matriz TF-IDF..."):
            tfidf_matrix = self.prepare_tfidf_matrix(df)

        if tfidf_matrix is None:
            return

        # Perform dimension reduction
        with st.spinner(f"Executando {reduction_method}..."):
            coords_2d = self.perform_dimension_reduction(tfidf_matrix, reduction_method)

        if coords_2d is None:
            return

        # Create 2D scatter plot
        st.subheader("📍 Projeção 2D dos Documentos")
        scatter_fig = self.create_2d_scatter_plot(df, coords_2d, reduction_method)
        st.plotly_chart(scatter_fig, use_container_width=True)

        # Calculate cluster profiles
        profiles = self.calculate_cluster_profiles(df)

        # Cluster selection for radar chart
        st.subheader("🕸️ Comparação de Perfis dos Clusters")
        all_clusters = list(profiles.keys())
        selected_clusters = st.multiselect(
            "Selecione clusters para comparação:",
            all_clusters,
            default=all_clusters[:3] if len(all_clusters) >= 3 else all_clusters,
            format_func=lambda x: f"Cluster {x}"
        )

        # Create and display radar chart
        if selected_clusters:
            radar_fig = self.create_radar_chart(profiles, selected_clusters)
            st.plotly_chart(radar_fig, use_container_width=True)

        # Display cluster statistics
        self.display_cluster_statistics(profiles)

def main():
    """Run the clustering analysis dashboard."""
    dashboard = ClusteringAnalysisDashboard()
    dashboard.run()

if __name__ == "__main__":
    # Only set page config when running standalone
    st.set_page_config(
        page_title="Stage 10: Clustering Analysis",
        page_icon="🎯",
        layout="wide"
    )
    main()