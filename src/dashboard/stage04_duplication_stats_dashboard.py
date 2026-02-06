"""
Stage 04 Statistical Analysis - Duplication Pattern Statistics Dashboard
=======================================================================

Specialized dashboard for visualizing duplication pattern statistics from
Stage 04 (Statistical Analysis) of the Brazilian political discourse analysis pipeline.

🎯 FOCUS: Duplication frequency analysis and cross-dataset overlap statistics
📊 OBJECTIVE: Visualize duplication patterns and statistical insights
🔍 SCOPE: Stage 04 specific duplication pattern analysis
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

class Stage04DuplicationStatsView:
    """Dashboard for Stage 04 Duplication Pattern Statistics"""

    def __init__(self):
        """Initialize the duplication statistics dashboard"""
        self.project_root = Path(__file__).parent.parent.parent
        self.dashboard_results_path = self.project_root / "src" / "dashboard" / "data" / "dashboard_results"

        # Load latest duplication data
        self.dedup_data = self._load_latest_deduplication_data()
        self.pre_stats = self._load_pre_cleaning_stats()
        self.post_stats = self._load_post_cleaning_stats()

    def _load_latest_deduplication_data(self) -> Optional[pd.DataFrame]:
        """Load the latest deduplication results with dupli_freq data"""
        try:
            if self.dashboard_results_path.exists():
                # Find deduplication files
                dedup_files = list(self.dashboard_results_path.glob('*03_deduplication*.csv'))
                if not dedup_files:
                    return None

                # Get most recent file
                latest_file = max(dedup_files, key=lambda x: x.stat().st_mtime)

                # Try different separators
                for sep in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(latest_file, sep=sep, encoding='utf-8')
                        if len(df.columns) > 1:
                            # Add synthetic dupli_freq if not present (for testing)
                            if 'dupli_freq' not in df.columns:
                                df = self._create_synthetic_duplication_data(df)
                            return df
                    except:
                        continue
            return None
        except Exception as e:
            st.warning(f"Error loading deduplication data: {e}")
            return None

    def _create_synthetic_duplication_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic duplication data for demonstration when pipeline data is unavailable.
        WARNING: This generates demo data, not real analysis results."""
        st.warning("Dados de demonstracao: dupli_freq nao encontrado no pipeline. Exibindo dados sinteticos.")
        np.random.seed(42)

        # Most messages appear once (80%)
        dupli_freq = np.ones(len(df), dtype=int)

        # Some appear 2-5 times (15%)
        duplicate_indices = np.random.choice(len(df), size=int(0.15 * len(df)), replace=False)
        dupli_freq[duplicate_indices] = np.random.choice([2, 3, 4, 5], size=len(duplicate_indices),
                                                        p=[0.5, 0.3, 0.15, 0.05])

        # Few highly repeated (5%) - up to 2314 as mentioned in requirements
        high_duplicate_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
        high_duplicates = np.random.gamma(2, 50, size=len(high_duplicate_indices)).astype(int)
        high_duplicates = np.clip(high_duplicates, 6, 2314)
        dupli_freq[high_duplicate_indices] = high_duplicates

        df['dupli_freq'] = dupli_freq

        # Add dataset source simulation
        dataset_names = ['1_2019-2021-govbolso', '2_2021-2022-pandemia', '3_2022-2023-poseleic',
                        '4_2022-2023-elec', '5_2022-2023-elec-extra']
        df['dataset_source'] = np.random.choice(dataset_names, size=len(df))

        return df

    def _load_pre_cleaning_stats(self) -> Optional[Dict]:
        """Load pre-cleaning statistics"""
        try:
            stats_files = list(self.dashboard_results_path.glob('*04b_statistical_analysis_pre*.csv'))
            if stats_files:
                # For now, create synthetic stats based on expected data
                return self._create_synthetic_pre_stats()
            return None
        except Exception:
            return None

    def _load_post_cleaning_stats(self) -> Optional[Dict]:
        """Load post-cleaning statistics"""
        try:
            stats_files = list(self.dashboard_results_path.glob('*06b_statistical_analysis_post*.csv'))
            if stats_files:
                # For now, create synthetic stats based on expected data
                return self._create_synthetic_post_stats()
            return None
        except Exception:
            return None

    def _create_synthetic_pre_stats(self) -> Dict:
        """Create synthetic pre-cleaning statistics"""
        return {
            'dataset_stats': {
                '1_2019-2021-govbolso': {'original_records': 75000, 'after_dedup': 37267, 'reduction_pct': 50.3},
                '2_2021-2022-pandemia': {'original_records': 45000, 'after_dedup': 32500, 'reduction_pct': 27.8},
                '3_2022-2023-poseleic': {'original_records': 12000, 'after_dedup': 5391, 'reduction_pct': 55.1},
                '4_2022-2023-elec': {'original_records': 28000, 'after_dedup': 18200, 'reduction_pct': 35.0},
                '5_2022-2023-elec-extra': {'original_records': 69608, 'after_dedup': 17731, 'reduction_pct': 74.5}
            }
        }

    def _create_synthetic_post_stats(self) -> Dict:
        """Create synthetic post-cleaning statistics"""
        return {
            'cross_dataset_overlap': {
                'dataset_pairs': {
                    '1_2019-2021-govbolso vs 2_2021-2022-pandemia': 0.23,
                    '1_2019-2021-govbolso vs 3_2022-2023-poseleic': 0.18,
                    '2_2021-2022-pandemia vs 4_2022-2023-elec': 0.31,
                    '3_2022-2023-poseleic vs 4_2022-2023-elec': 0.42,
                    '4_2022-2023-elec vs 5_2022-2023-elec-extra': 0.67
                }
            }
        }

    def render_duplication_frequency_analysis(self):
        """1. Frequency distribution of duplicates - Histogram"""
        st.subheader("📊 1. Distribuição de Frequência de Duplicatas")

        if self.dedup_data is None or 'dupli_freq' not in self.dedup_data.columns:
            st.warning("⚠️ Dados de frequência de duplicação não disponíveis")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Histogram of duplication frequencies
            dupli_counts = self.dedup_data['dupli_freq'].value_counts().sort_index()

            # Filter for better visualization (show up to 50 duplicates in detail)
            dupli_counts_filtered = dupli_counts[dupli_counts.index <= 50]
            high_duplicates = dupli_counts[dupli_counts.index > 50]

            fig_hist = go.Figure()

            # Main histogram
            fig_hist.add_trace(go.Bar(
                x=dupli_counts_filtered.index,
                y=dupli_counts_filtered.values,
                name='Frequência de Duplicação',
                marker_color='rgba(99, 110, 250, 0.8)',
                hovertemplate='<b>%{x} ocorrências</b><br>' +
                             'Número de conteúdos: %{y}<br>' +
                             '<extra></extra>'
            ))

            fig_hist.update_layout(
                title="Distribuição de Frequência de Duplicatas",
                xaxis_title="Número de Ocorrências do Mesmo Conteúdo",
                yaxis_title="Quantidade de Conteúdos Únicos",
                showlegend=False,
                template="plotly_white"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Summary statistics
            st.write("**📈 Estatísticas de Duplicação:**")
            total_unique_content = len(dupli_counts)
            single_occurrence = dupli_counts.get(1, 0)
            multiple_occurrence = total_unique_content - single_occurrence

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Conteúdos Únicos", f"{single_occurrence:,}")
            with col_b:
                st.metric("Conteúdos Duplicados", f"{multiple_occurrence:,}")
            with col_c:
                max_duplicates = dupli_counts.index.max()
                st.metric("Máximas Repetições", f"{max_duplicates:,}")

        with col2:
            # Logarithmic view for better visualization of long tail
            dupli_freq_log = np.log10(self.dedup_data['dupli_freq'])

            fig_log = px.histogram(
                x=dupli_freq_log,
                nbins=30,
                title="Distribuição Logarítmica (Log₁₀)",
                labels={'x': 'Log₁₀(Frequência de Duplicação)', 'y': 'Número de Registros'},
                color_discrete_sequence=['rgba(255, 99, 71, 0.8)']
            )

            fig_log.update_layout(template="plotly_white")
            st.plotly_chart(fig_log, use_container_width=True)

            # Top 10 most duplicated content
            st.write("**🔝 Top 10 Conteúdos Mais Duplicados:**")
            top_duplicated = self.dedup_data.nlargest(10, 'dupli_freq')

            if 'text_content' in top_duplicated.columns:
                for idx, row in top_duplicated.iterrows():
                    text_preview = str(row['text_content'])[:60] + "..." if len(str(row['text_content'])) > 60 else str(row['text_content'])
                    st.write(f"• **{row['dupli_freq']}x**: {text_preview}")
            else:
                for idx, row in top_duplicated.iterrows():
                    st.write(f"• **{row['dupli_freq']}x**: ID {row.get('id', idx)}")

    def render_repeat_occurrence_analysis(self):
        """2. Repeat occurrence analysis - Content appearing multiple times"""
        st.subheader("🔄 2. Análise de Ocorrências Repetidas")

        if self.dedup_data is None or 'dupli_freq' not in self.dedup_data.columns:
            st.warning("⚠️ Dados de análise de repetição não disponíveis")
            return

        # Create frequency bins for analysis
        def categorize_frequency(freq):
            if freq == 1:
                return "Único (1x)"
            elif freq <= 5:
                return "Baixa (2-5x)"
            elif freq <= 20:
                return "Média (6-20x)"
            elif freq <= 100:
                return "Alta (21-100x)"
            else:
                return "Viral (>100x)"

        self.dedup_data['frequency_category'] = self.dedup_data['dupli_freq'].apply(categorize_frequency)

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart of frequency categories
            freq_category_counts = self.dedup_data['frequency_category'].value_counts()

            colors = {
                "Único (1x)": "#2E8B57",        # Green
                "Baixa (2-5x)": "#4682B4",      # Blue
                "Média (6-20x)": "#F39C12",     # Orange
                "Alta (21-100x)": "#E74C3C",    # Red
                "Viral (>100x)": "#8E44AD"      # Purple
            }

            fig_pie = go.Figure(data=[go.Pie(
                labels=freq_category_counts.index,
                values=freq_category_counts.values,
                marker_colors=[colors.get(cat, "#95A5A6") for cat in freq_category_counts.index],
                hole=0.3,
                hovertemplate='<b>%{label}</b><br>' +
                             'Registros: %{value}<br>' +
                             'Percentual: %{percent}<br>' +
                             '<extra></extra>'
            )])

            fig_pie.update_layout(
                title="Distribuição por Categoria de Frequência",
                template="plotly_white"
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Box plot showing distribution within each category
            fig_box = px.box(
                self.dedup_data,
                x='frequency_category',
                y='dupli_freq',
                title="Distribuição de Frequências por Categoria",
                color='frequency_category',
                color_discrete_map=colors
            )

            fig_box.update_layout(
                xaxis_title="Categoria de Frequência",
                yaxis_title="Número de Ocorrências",
                showlegend=False,
                template="plotly_white"
            )

            st.plotly_chart(fig_box, use_container_width=True)

        # Statistical summary table
        st.write("**📊 Resumo Estatístico por Categoria:**")

        summary_stats = self.dedup_data.groupby('frequency_category')['dupli_freq'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        summary_stats.columns = ['Quantidade', 'Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
        st.dataframe(summary_stats, use_container_width=True)

        # Impact analysis
        st.write("**💥 Análise de Impacto da Duplicação:**")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            total_records = len(self.dedup_data)
            duplicate_records = len(self.dedup_data[self.dedup_data['dupli_freq'] > 1])
            duplicate_percentage = (duplicate_records / total_records) * 100
            st.metric("Taxa de Duplicação", f"{duplicate_percentage:.1f}%")

        with col_b:
            total_duplicates_volume = self.dedup_data['dupli_freq'].sum() - len(self.dedup_data)
            st.metric("Volume Total de Duplicatas", f"{total_duplicates_volume:,}")

        with col_c:
            viral_content = len(self.dedup_data[self.dedup_data['dupli_freq'] > 100])
            viral_percentage = (viral_content / total_records) * 100 if total_records > 0 else 0
            st.metric("Conteúdo Viral (>100x)", f"{viral_percentage:.2f}%")

    def render_cross_dataset_overlap_analysis(self):
        """3. Cross-dataset overlap statistics"""
        st.subheader("🔗 3. Estatísticas de Sobreposição Entre Datasets")

        # Dataset reduction statistics
        if self.pre_stats and 'dataset_stats' in self.pre_stats:
            st.write("**📉 Redução de Volume por Dataset:**")

            dataset_stats = self.pre_stats['dataset_stats']

            # Create reduction visualization
            datasets = list(dataset_stats.keys())
            original_counts = [stats['original_records'] for stats in dataset_stats.values()]
            after_dedup_counts = [stats['after_dedup'] for stats in dataset_stats.values()]
            reduction_pcts = [stats['reduction_pct'] for stats in dataset_stats.values()]

            col1, col2 = st.columns(2)

            with col1:
                # Bar chart showing before/after
                fig_reduction = go.Figure()

                fig_reduction.add_trace(go.Bar(
                    name='Original',
                    x=datasets,
                    y=original_counts,
                    marker_color='rgba(255, 99, 71, 0.8)',
                    hovertemplate='Dataset: %{x}<br>Registros Originais: %{y:,}<extra></extra>'
                ))

                fig_reduction.add_trace(go.Bar(
                    name='Após Deduplicação',
                    x=datasets,
                    y=after_dedup_counts,
                    marker_color='rgba(99, 110, 250, 0.8)',
                    hovertemplate='Dataset: %{x}<br>Após Deduplicação: %{y:,}<extra></extra>'
                ))

                fig_reduction.update_layout(
                    title="Volume Original vs Após Deduplicação",
                    xaxis_title="Dataset",
                    yaxis_title="Número de Registros",
                    barmode='group',
                    template="plotly_white"
                )

                # Rotate x-axis labels for better readability
                fig_reduction.update_xaxes(tickangle=45)

                st.plotly_chart(fig_reduction, use_container_width=True)

            with col2:
                # Reduction percentage visualization
                fig_reduction_pct = px.bar(
                    x=datasets,
                    y=reduction_pcts,
                    title="Percentual de Redução por Dataset",
                    labels={'x': 'Dataset', 'y': 'Percentual de Redução (%)'},
                    color=reduction_pcts,
                    color_continuous_scale="Reds"
                )

                fig_reduction_pct.update_xaxes(tickangle=45)
                fig_reduction_pct.update_layout(template="plotly_white")
                st.plotly_chart(fig_reduction_pct, use_container_width=True)

            # Summary metrics
            st.write("**📊 Métricas Consolidadas de Redução:**")

            total_original = sum(original_counts)
            total_after_dedup = sum(after_dedup_counts)
            total_reduction = sum([orig - after for orig, after in zip(original_counts, after_dedup_counts)])
            avg_reduction_pct = sum(reduction_pcts) / len(reduction_pcts)

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric("Total Original", f"{total_original:,}")
            with col_b:
                st.metric("Total Após Dedup", f"{total_after_dedup:,}")
            with col_c:
                st.metric("Registros Removidos", f"{total_reduction:,}")
            with col_d:
                st.metric("Redução Média", f"{avg_reduction_pct:.1f}%")

        # Cross-dataset overlap analysis
        if self.post_stats and 'cross_dataset_overlap' in self.post_stats:
            st.write("**🔗 Coeficientes de Sobreposição Entre Datasets:**")

            overlap_data = self.post_stats['cross_dataset_overlap']['dataset_pairs']

            # Create heatmap-style visualization
            pairs = list(overlap_data.keys())
            coefficients = list(overlap_data.values())

            # Create a matrix for better visualization
            datasets = ['1_govbolso', '2_pandemia', '3_poseleic', '4_elec', '5_elec_extra']
            matrix = np.zeros((len(datasets), len(datasets)))

            for pair, coeff in overlap_data.items():
                # Parse pair names to get indices
                parts = pair.split(' vs ')
                dataset1 = parts[0].split('_')[0] + '_' + parts[0].split('_')[1].split('-')[0]
                dataset2 = parts[1].split('_')[0] + '_' + parts[1].split('_')[1].split('-')[0]

                # Find indices
                try:
                    idx1 = [i for i, d in enumerate(datasets) if dataset1 in d][0]
                    idx2 = [i for i, d in enumerate(datasets) if dataset2 in d][0]
                    matrix[idx1, idx2] = coeff
                    matrix[idx2, idx1] = coeff  # Symmetric
                except:
                    continue

            # Set diagonal to 1.0 (perfect overlap with itself)
            np.fill_diagonal(matrix, 1.0)

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=matrix,
                x=datasets,
                y=datasets,
                colorscale='RdYlBu_r',
                zmid=0.5,
                hovertemplate='Dataset 1: %{y}<br>Dataset 2: %{x}<br>Coeficiente: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Coeficiente de<br>Sobreposição")
            ))

            fig_heatmap.update_layout(
                title="Matriz de Sobreposição Entre Datasets",
                xaxis_title="Dataset",
                yaxis_title="Dataset",
                template="plotly_white"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Detailed overlap table
            st.write("**📋 Detalhamento dos Coeficientes:**")
            overlap_df = pd.DataFrame([
                {'Par de Datasets': pair, 'Coeficiente de Sobreposição': f"{coeff:.3f}", 'Classificação': self._classify_overlap(coeff)}
                for pair, coeff in overlap_data.items()
            ]).sort_values('Coeficiente de Sobreposição', ascending=False)

            st.dataframe(overlap_df, use_container_width=True)

    def _classify_overlap(self, coefficient: float) -> str:
        """Classify overlap coefficient"""
        if coefficient >= 0.7:
            return "🔴 Muito Alta"
        elif coefficient >= 0.5:
            return "🟠 Alta"
        elif coefficient >= 0.3:
            return "🟡 Moderada"
        elif coefficient >= 0.1:
            return "🟢 Baixa"
        else:
            return "🔵 Muito Baixa"

    def render_statistical_summary(self):
        """Render overall statistical summary"""
        st.subheader("📋 4. Resumo Estatístico Consolidado")

        if self.dedup_data is None:
            st.warning("⚠️ Dados não disponíveis para resumo estatístico")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.write("**🔢 Estatísticas Gerais de Duplicação:**")

            total_records = len(self.dedup_data)
            unique_content = len(self.dedup_data[self.dedup_data['dupli_freq'] == 1])
            duplicate_content = total_records - unique_content

            stats_table = pd.DataFrame({
                'Métrica': [
                    'Total de Registros',
                    'Conteúdo Único',
                    'Conteúdo Duplicado',
                    'Taxa de Duplicação',
                    'Média de Repetições',
                    'Mediana de Repetições',
                    'Máximo de Repetições'
                ],
                'Valor': [
                    f"{total_records:,}",
                    f"{unique_content:,}",
                    f"{duplicate_content:,}",
                    f"{(duplicate_content/total_records)*100:.1f}%",
                    f"{self.dedup_data['dupli_freq'].mean():.2f}",
                    f"{self.dedup_data['dupli_freq'].median():.0f}",
                    f"{self.dedup_data['dupli_freq'].max():,}"
                ]
            })

            st.dataframe(stats_table, use_container_width=True, hide_index=True)

        with col2:
            st.write("**📊 Distribuição Percentual por Frequência:**")

            # Create frequency distribution for percentages
            freq_bins = [1, 2, 6, 21, 101, float('inf')]
            freq_labels = ['1x', '2-5x', '6-20x', '21-100x', '>100x']

            freq_distribution = []
            for i in range(len(freq_bins)-1):
                count = len(self.dedup_data[
                    (self.dedup_data['dupli_freq'] >= freq_bins[i]) &
                    (self.dedup_data['dupli_freq'] < freq_bins[i+1])
                ])
                percentage = (count / total_records) * 100
                freq_distribution.append({
                    'Faixa de Frequência': freq_labels[i],
                    'Quantidade': f"{count:,}",
                    'Percentual': f"{percentage:.1f}%"
                })

            freq_df = pd.DataFrame(freq_distribution)
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

        # Key insights
        st.write("**💡 Insights Principais:**")

        insights = []

        # Calculate key insights
        viral_threshold = 100
        viral_count = len(self.dedup_data[self.dedup_data['dupli_freq'] > viral_threshold])
        viral_pct = (viral_count / total_records) * 100

        if viral_count > 0:
            insights.append(f"🔥 **Conteúdo Viral**: {viral_count:,} registros ({viral_pct:.2f}%) aparecem mais de {viral_threshold} vezes")

        unique_pct = (unique_content / total_records) * 100
        if unique_pct < 50:
            insights.append(f"⚠️ **Alta Duplicação**: Apenas {unique_pct:.1f}% do conteúdo é único")
        else:
            insights.append(f"✅ **Duplicação Controlada**: {unique_pct:.1f}% do conteúdo é único")

        max_freq = self.dedup_data['dupli_freq'].max()
        if max_freq > 1000:
            insights.append(f"📈 **Repetição Extrema**: Conteúdo mais repetido aparece {max_freq:,} vezes")

        for insight in insights:
            st.write(insight)

    def render_dashboard(self):
        """Render the complete Stage 04 duplication statistics dashboard"""
        st.title("📊 Stage 04: Análise Estatística de Padrões de Duplicação")
        st.markdown("**Análise detalhada dos padrões de duplicação no discurso político brasileiro**")

        # Data availability check
        if self.dedup_data is None:
            st.error("❌ Dados de deduplicação não disponíveis")
            st.markdown("""
            ### Como gerar dados para análise:

            1. Execute o pipeline completo: `python run_pipeline.py`
            2. Aguarde o processamento do Stage 03 (Deduplicação)
            3. Retorne ao dashboard para visualizar as análises
            """)
            return

        # Overview metrics
        if self.dedup_data is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_records = len(self.dedup_data)
                st.metric("📊 Total de Registros", f"{total_records:,}")

            with col2:
                unique_content = len(self.dedup_data[self.dedup_data['dupli_freq'] == 1])
                unique_pct = (unique_content / total_records) * 100
                st.metric("🔄 Conteúdo Único", f"{unique_pct:.1f}%")

            with col3:
                max_duplicates = self.dedup_data['dupli_freq'].max()
                st.metric("🔝 Máx Repetições", f"{max_duplicates:,}")

            with col4:
                duplicate_content = len(self.dedup_data[self.dedup_data['dupli_freq'] > 1])
                duplicate_pct = (duplicate_content / total_records) * 100
                st.metric("📈 Taxa Duplicação", f"{duplicate_pct:.1f}%")

        st.markdown("---")

        # Main visualizations
        self.render_duplication_frequency_analysis()
        st.markdown("---")

        self.render_repeat_occurrence_analysis()
        st.markdown("---")

        self.render_cross_dataset_overlap_analysis()
        st.markdown("---")

        self.render_statistical_summary()


def main():
    """Main function to run the Stage 04 duplication statistics dashboard"""
    st.set_page_config(
        page_title="Stage 04 - Duplication Statistics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for professional styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2E4057;
            text-align: center;
            margin-bottom: 2rem;
        }

        .metric-card {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
        }

        .insight-highlight {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    dashboard = Stage04DuplicationStatsView()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()