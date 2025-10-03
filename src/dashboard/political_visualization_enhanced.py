"""
Enhanced Political Visualization Dashboard - digiNEV v5.1.0
===========================================================

Streamlit dashboard specifically designed for Brazilian political discourse analysis.
Focuses on visualizing political classification, sentiment analysis, and temporal patterns.

Features:
- Political alignment distribution
- Sentiment analysis by political category
- Temporal evolution of discourse
- Cluster analysis visualization
- Network patterns in political messaging

Author: digiNEV Research Team
Date: 2025-10-01
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPoliticalVisualizationDashboard:
    """Enhanced dashboard for political discourse visualization"""

    def __init__(self):
        self.data_path = Path("pipeline_outputs/dashboard_ready")
        self.df = None
        self.load_latest_data()

    def load_latest_data(self) -> Optional[pd.DataFrame]:
        """Load the most recent pipeline output data"""
        try:
            # Find the most recent output file
            csv_files = list(self.data_path.glob("*_15_network_analysis_*.csv"))
            if not csv_files:
                logger.warning("No network analysis output files found")
                return None

            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from: {latest_file}")

            # Load with proper separator detection
            self.df = pd.read_csv(latest_file, sep=';', encoding='utf-8')

            # Data validation and cleaning
            if self.df is not None:
                self.df = self._clean_and_validate_data(self.df)
                logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")

            return self.df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the loaded data"""
        # Remove completely empty rows
        df = df.dropna(how='all')

        # Fix date parsing issues
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Clean sentiment scores
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')

        # Ensure political categories are strings
        if 'political_category' in df.columns:
            df['political_category'] = df['political_category'].fillna('indefinido').astype(str)

        if 'political_alignment' in df.columns:
            df['political_alignment'] = df['political_alignment'].fillna('indefinido').astype(str)

        # Clean text length and word count
        if 'text_length' in df.columns:
            df['text_length'] = pd.to_numeric(df['text_length'], errors='coerce').fillna(0)

        if 'word_count' in df.columns:
            df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce').fillna(0)

        return df

    def render_main_dashboard(self):
        """Render the main political visualization dashboard"""
        st.set_page_config(
            page_title="Political Analysis - Brazil Telegram",
            page_icon="🏛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🏛️ Brazilian Political Discourse Analysis")
        st.markdown("**digiNEV v5.1.0** - Telegram Political Messages Analysis Dashboard")

        if self.df is None or len(self.df) == 0:
            st.error("No data available for visualization. Please run the pipeline first.")
            st.info("Run: `python run_pipeline.py --dataset data/controlled_test_100.csv`")
            return

        # Sidebar filters
        self._render_sidebar_filters()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🎯 Political Analysis",
            "💭 Sentiment Analysis",
            "📅 Temporal Patterns",
            "🔗 Network Analysis"
        ])

        with tab1:
            self._render_overview_tab()

        with tab2:
            self._render_political_analysis_tab()

        with tab3:
            self._render_sentiment_analysis_tab()

        with tab4:
            self._render_temporal_analysis_tab()

        with tab5:
            self._render_network_analysis_tab()

    def _render_sidebar_filters(self):
        """Render sidebar with data filters"""
        st.sidebar.header("📋 Dataset Information")
        st.sidebar.metric("Total Messages", len(self.df))
        st.sidebar.metric("Date Range", f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
        st.sidebar.metric("Unique Channels", self.df['channel'].nunique())

        st.sidebar.header("🔍 Filters")

        # Political category filter
        if 'political_category' in self.df.columns:
            categories = ['All'] + sorted(self.df['political_category'].unique().tolist())
            selected_category = st.sidebar.selectbox("Political Category", categories)
            if selected_category != 'All':
                self.df = self.df[self.df['political_category'] == selected_category]

        # Date range filter
        if 'date' in self.df.columns:
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(self.df['date'].min(), self.df['date'].max()),
                min_value=self.df['date'].min(),
                max_value=self.df['date'].max()
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                self.df = self.df[
                    (self.df['date'] >= pd.Timestamp(start_date)) &
                    (self.df['date'] <= pd.Timestamp(end_date))
                ]

    def _render_overview_tab(self):
        """Render overview tab with key metrics"""
        st.header("📊 Dataset Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", len(self.df))

        with col2:
            if 'political_category' in self.df.columns:
                political_count = self.df['political_category'].value_counts().iloc[0]
                dominant_category = self.df['political_category'].value_counts().index[0]
                st.metric("Dominant Category", f"{dominant_category} ({political_count})")

        with col3:
            if 'sentiment' in self.df.columns:
                sentiment_count = self.df['sentiment'].value_counts().iloc[0]
                dominant_sentiment = self.df['sentiment'].value_counts().index[0]
                st.metric("Dominant Sentiment", f"{dominant_sentiment} ({sentiment_count})")

        with col4:
            avg_length = self.df['text_length'].mean()
            st.metric("Avg Message Length", f"{avg_length:.0f} chars")

        # Political category distribution
        if 'political_category' in self.df.columns:
            st.subheader("Political Category Distribution")
            category_counts = self.df['political_category'].value_counts()

            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribution of Political Categories",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    def _render_political_analysis_tab(self):
        """Render detailed political analysis"""
        st.header("🎯 Political Analysis")

        if 'political_category' not in self.df.columns or 'political_alignment' not in self.df.columns:
            st.warning("Political analysis columns not found in dataset")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Political Category Distribution")
            category_counts = self.df['political_category'].value_counts()

            # Color mapping for political categories
            color_map = {
                'bolsonarista': '#FF6B6B',
                'lulista': '#4ECDC4',
                'neutro': '#95E1D3',
                'geral': '#45B7D1',
                'anti-bolsonaro': '#96CEB4',
                'indefinido': '#FCEA2B'
            }

            colors = [color_map.get(cat, '#CCCCCC') for cat in category_counts.index]

            fig_category = go.Figure(data=[
                go.Bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    marker_color=colors,
                    text=category_counts.values,
                    textposition='auto'
                )
            ])
            fig_category.update_layout(
                title="Political Categories",
                xaxis_title="Category",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_category, use_container_width=True)

        with col2:
            st.subheader("Political Alignment Distribution")
            alignment_counts = self.df['political_alignment'].value_counts()

            # Color mapping for political alignment
            alignment_colors = {
                'extrema-direita': '#8B0000',
                'direita': '#FF6B6B',
                'centro-direita': '#FFA07A',
                'centro': '#87CEEB',
                'centro-esquerda': '#90EE90',
                'esquerda': '#32CD32',
                'indefinido': '#CCCCCC'
            }

            colors_align = [alignment_colors.get(align, '#CCCCCC') for align in alignment_counts.index]

            fig_alignment = go.Figure(data=[
                go.Bar(
                    x=alignment_counts.index,
                    y=alignment_counts.values,
                    marker_color=colors_align,
                    text=alignment_counts.values,
                    textposition='auto'
                )
            ])
            fig_alignment.update_layout(
                title="Political Alignment",
                xaxis_title="Alignment",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_alignment, use_container_width=True)

        # Political vs Sentiment Analysis
        if 'sentiment' in self.df.columns:
            st.subheader("Political Category vs Sentiment")

            cross_tab = pd.crosstab(self.df['political_category'], self.df['sentiment'])

            fig_heatmap = px.imshow(
                cross_tab.values,
                x=cross_tab.columns,
                y=cross_tab.index,
                aspect="auto",
                color_continuous_scale="RdYlBu",
                title="Political Category vs Sentiment Heatmap"
            )
            fig_heatmap.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Political Category"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    def _render_sentiment_analysis_tab(self):
        """Render sentiment analysis visualizations"""
        st.header("💭 Sentiment Analysis")

        if 'sentiment' not in self.df.columns:
            st.warning("Sentiment analysis columns not found in dataset")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = self.df['sentiment'].value_counts()

            # Color mapping for sentiments
            sentiment_colors = {
                'positivo': '#2ECC71',
                'negativo': '#E74C3C',
                'neutro': '#95A5A6'
            }

            colors = [sentiment_colors.get(sent, '#CCCCCC') for sent in sentiment_counts.index]

            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map=sentiment_colors
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            if 'sentiment_score' in self.df.columns:
                st.subheader("Sentiment Score Distribution")

                fig_scores = px.histogram(
                    self.df,
                    x='sentiment_score',
                    title="Distribution of Sentiment Scores",
                    nbins=30,
                    color_discrete_sequence=['#3498DB']
                )
                fig_scores.update_layout(
                    xaxis_title="Sentiment Score",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_scores, use_container_width=True)

        # Sentiment by Political Category
        if 'political_category' in self.df.columns:
            st.subheader("Sentiment by Political Category")

            sentiment_political = self.df.groupby(['political_category', 'sentiment']).size().unstack(fill_value=0)

            fig_stacked = px.bar(
                sentiment_political.T,
                title="Sentiment Distribution by Political Category",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_stacked.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Count",
                legend_title="Political Category"
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

    def _render_temporal_analysis_tab(self):
        """Render temporal pattern analysis"""
        st.header("📅 Temporal Patterns")

        if 'date' not in self.df.columns:
            st.warning("Date column not found for temporal analysis")
            return

        # Monthly activity
        st.subheader("Monthly Activity")
        monthly_data = self.df.groupby(self.df['date'].dt.to_period('M')).size()

        fig_monthly = px.line(
            x=monthly_data.index.astype(str),
            y=monthly_data.values,
            title="Monthly Message Volume",
            labels={'x': 'Month', 'y': 'Message Count'}
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Political evolution over time
        if 'political_category' in self.df.columns:
            st.subheader("Political Category Evolution")

            temporal_political = self.df.groupby([
                self.df['date'].dt.to_period('M'),
                'political_category'
            ]).size().unstack(fill_value=0)

            fig_evolution = px.area(
                temporal_political,
                title="Evolution of Political Categories Over Time",
                labels={'index': 'Month', 'value': 'Message Count'}
            )
            st.plotly_chart(fig_evolution, use_container_width=True)

        # Hourly patterns
        if 'hour' in self.df.columns:
            st.subheader("Hourly Activity Patterns")
            hourly_data = self.df.groupby('hour').size()

            fig_hourly = px.bar(
                x=hourly_data.index,
                y=hourly_data.values,
                title="Message Activity by Hour of Day",
                labels={'x': 'Hour', 'y': 'Message Count'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

    def _render_network_analysis_tab(self):
        """Render network analysis visualizations"""
        st.header("🔗 Network Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Channel Activity")
            if 'channel' in self.df.columns:
                channel_counts = self.df['channel'].value_counts().head(10)

                fig_channels = px.bar(
                    x=channel_counts.values,
                    y=channel_counts.index,
                    orientation='h',
                    title="Top 10 Most Active Channels",
                    labels={'x': 'Message Count', 'y': 'Channel'}
                )
                st.plotly_chart(fig_channels, use_container_width=True)

        with col2:
            st.subheader("User Activity")
            if 'user_id' in self.df.columns:
                user_counts = self.df['user_id'].value_counts().head(10)

                fig_users = px.bar(
                    x=user_counts.values,
                    y=user_counts.index,
                    orientation='h',
                    title="Top 10 Most Active Users",
                    labels={'x': 'Message Count', 'y': 'User ID'}
                )
                st.plotly_chart(fig_users, use_container_width=True)

        # Cluster analysis
        if 'cluster_name' in self.df.columns:
            st.subheader("Cluster Analysis")
            cluster_counts = self.df['cluster_name'].value_counts()

            fig_clusters = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                title="Message Clusters",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

        # Topic analysis
        if 'topic_name' in self.df.columns:
            st.subheader("Topic Distribution")
            topic_counts = self.df['topic_name'].value_counts().head(10)

            fig_topics = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                title="Top 10 Topics",
                labels={'x': 'Topic', 'y': 'Message Count'}
            )
            fig_topics.update_xaxes(tickangle=45)
            st.plotly_chart(fig_topics, use_container_width=True)

def main():
    """Main function to run the enhanced political visualization dashboard"""
    dashboard = EnhancedPoliticalVisualizationDashboard()
    dashboard.render_main_dashboard()

if __name__ == "__main__":
    main()