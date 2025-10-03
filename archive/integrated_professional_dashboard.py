"""
Professional Integrated Dashboard - Brazilian Political Discourse Analysis
========================================================================

Professional dashboard with dark blue (#1B365D) and dark orange (#FF8C42) theme
implementing single-page layout with left sidebar and 3-part content division.

Design Specifications:
- Color Scheme: Dark blue (#1B365D) primary, Dark orange (#FF8C42) secondary
- Layout: Light sidebar left, extended colors top, 3-part content division
- Charts: Big main charts with small side charts
- Style: Minimalist with clean traces and professional appearance
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
import glob
import re

# GUARDRAILS: Sistema de validação de conteúdo
from dashboard_guardrails import dashboard_guardrail, require_real_data_only, validate_dashboard_data

# Professional color scheme
COLORS = {
    'primary': '#1B365D',      # Dark blue
    'secondary': '#FF8C42',     # Dark orange
    'accent_light': '#F5F5F5',  # Light gray
    'accent_dark': '#E0E0E0',   # Medium gray
    'text_dark': '#2C2C2C',     # Dark text
    'text_light': '#FFFFFF',    # White text
    'success': '#28A745',       # Green
    'warning': '#FFC107',       # Yellow
    'danger': '#DC3545'         # Red
}

# Page configuration
st.set_page_config(
    page_title="Brazilian Political Discourse Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown(f"""
<style>
    /* Main layout */
    .main {{
        background-color: {COLORS['accent_light']};
    }}

    /* Top header */
    .top-header {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: {COLORS['text_light']};
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }}

    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {COLORS['accent_light']};
        border-right: 2px solid {COLORS['primary']};
    }}

    /* Content cards */
    .content-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {COLORS['primary']};
        margin-bottom: 1rem;
    }}

    /* Metrics cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: {COLORS['text_light']};
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}

    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}

    /* Chart containers */
    .chart-container {{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}

    /* Section headers */
    .section-header {{
        color: {COLORS['primary']};
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['secondary']};
    }}

    /* Small chart headers */
    .small-chart-header {{
        color: {COLORS['primary']};
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

class DataLoader:
    """Loads and manages pipeline output data"""

    def __init__(self):
        self.data_path = Path("pipeline_outputs/dashboard_ready")
        self.cache = {}

    def get_latest_files(self, pattern: str) -> List[Path]:
        """Get latest files matching pattern"""
        files = list(self.data_path.glob(f"*{pattern}*.csv"))
        if not files:
            return []

        # Sort by timestamp in filename
        files.sort(key=lambda x: self._extract_timestamp(x.name), reverse=True)
        return files

    def _extract_timestamp(self, filename: str) -> str:
        """Extract timestamp from filename"""
        match = re.search(r'(\d{8}_\d{6})', filename)
        return match.group(1) if match else "0"

    def load_political_data(self) -> Optional[pd.DataFrame]:
        """Load political analysis data"""
        files = self.get_latest_files("05_political_analysis")
        if not files:
            return None

        try:
            df = pd.read_csv(files[0], sep=';')
            return df
        except Exception as e:
            st.error(f"Error loading political data: {e}")
            return None

    def load_sentiment_data(self) -> Optional[pd.DataFrame]:
        """Load sentiment analysis data"""
        files = self.get_latest_files("08_sentiment_analysis")
        if not files:
            return None

        try:
            df = pd.read_csv(files[0], sep=';')
            return df
        except Exception as e:
            st.error(f"Error loading sentiment data: {e}")
            return None

    def load_temporal_data(self) -> Optional[pd.DataFrame]:
        """Load temporal analysis data"""
        files = self.get_latest_files("14_temporal_analysis")
        if not files:
            return None

        try:
            df = pd.read_csv(files[0], sep=';')
            return df
        except Exception as e:
            st.error(f"Error loading temporal data: {e}")
            return None

    def load_network_data(self) -> Optional[pd.DataFrame]:
        """Load network analysis data"""
        files = self.get_latest_files("15_network_analysis")
        if not files:
            return None

        try:
            df = pd.read_csv(files[0], sep=';')
            return df
        except Exception as e:
            st.error(f"Error loading network data: {e}")
            return None

    def load_summary_data(self) -> Optional[Dict]:
        """Load pipeline summary data"""
        files = list(self.data_path.glob("*summary*.json"))
        if not files:
            return None

        files.sort(key=lambda x: self._extract_timestamp(x.name), reverse=True)

        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading summary data: {e}")
            return None

class ChartCreator:
    """Creates professional charts with consistent styling"""

    def __init__(self):
        self.color_discrete_map = {
            'extrema-direita': COLORS['danger'],
            'direita': '#FF6B6B',
            'centro-direita': '#FFE066',
            'centro': '#A8E6CF',
            'centro-esquerda': '#88D8C0',
            'esquerda': COLORS['success'],
            'bolsonarista': COLORS['danger'],
            'oposicionista': COLORS['success'],
            'neutro': COLORS['accent_dark'],
            'geral': COLORS['primary']
        }

    def create_political_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create main political distribution chart"""
        if 'political_alignment' not in df.columns:
            return go.Figure()

        # Count political alignments
        alignment_counts = df['political_alignment'].value_counts()

        fig = go.Figure(data=[
            go.Bar(
                x=alignment_counts.index,
                y=alignment_counts.values,
                marker_color=[self.color_discrete_map.get(x, COLORS['primary']) for x in alignment_counts.index],
                marker_line=dict(width=1, color=COLORS['text_dark']),
                text=alignment_counts.values,
                textposition='auto',
                textfont=dict(color=COLORS['text_light'], size=12, family="Arial Black")
            )
        ])

        fig.update_layout(
            title=dict(
                text="Political Alignment Distribution",
                font=dict(size=20, color=COLORS['primary'], family="Arial Black"),
                x=0.5
            ),
            xaxis=dict(
                title="Political Alignment",
                titlefont=dict(color=COLORS['primary'], size=14),
                tickfont=dict(color=COLORS['text_dark'], size=12),
                gridcolor=COLORS['accent_dark']
            ),
            yaxis=dict(
                title="Message Count",
                titlefont=dict(color=COLORS['primary'], size=14),
                tickfont=dict(color=COLORS['text_dark'], size=12),
                gridcolor=COLORS['accent_dark']
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )

        return fig

    def create_sentiment_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create sentiment timeline chart"""
        if 'date' not in df.columns or 'sentiment_score' not in df.columns:
            return go.Figure()

        try:
            df['date'] = pd.to_datetime(df['date'])
            daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['sentiment_score'],
                mode='lines+markers',
                line=dict(color=COLORS['secondary'], width=2),
                marker=dict(color=COLORS['primary'], size=4),
                name='Average Sentiment'
            ))

            fig.update_layout(
                title=dict(
                    text="Sentiment Evolution",
                    font=dict(size=14, color=COLORS['primary']),
                    x=0.5
                ),
                xaxis=dict(
                    title="Date",
                    titlefont=dict(color=COLORS['primary'], size=12),
                    tickfont=dict(color=COLORS['text_dark'], size=10)
                ),
                yaxis=dict(
                    title="Sentiment Score",
                    titlefont=dict(color=COLORS['primary'], size=12),
                    tickfont=dict(color=COLORS['text_dark'], size=10)
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=250,
                showlegend=False
            )

        except Exception:
            fig = go.Figure()

        return fig

    def create_category_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create small political category pie chart"""
        if 'political_category' not in df.columns:
            return go.Figure()

        category_counts = df['political_category'].value_counts()

        fig = go.Figure(data=[
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                marker_colors=[self.color_discrete_map.get(x, COLORS['primary']) for x in category_counts.index],
                textinfo='label+percent',
                textfont=dict(size=10),
                hole=0.4
            )
        ])

        fig.update_layout(
            title=dict(
                text="Content Categories",
                font=dict(size=14, color=COLORS['primary']),
                x=0.5
            ),
            height=250,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_temporal_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create temporal activity heatmap"""
        if 'date' not in df.columns:
            return go.Figure()

        try:
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.day_name()

            # Create hour x day of week heatmap
            heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)

            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order)

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=[[0, COLORS['accent_light']], [1, COLORS['primary']]],
                showscale=False
            ))

            fig.update_layout(
                title=dict(
                    text="Activity Heatmap",
                    font=dict(size=20, color=COLORS['primary']),
                    x=0.5
                ),
                xaxis=dict(
                    title="Hour of Day",
                    titlefont=dict(color=COLORS['primary'], size=14)
                ),
                yaxis=dict(
                    title="Day of Week",
                    titlefont=dict(color=COLORS['primary'], size=14)
                ),
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

        except Exception:
            fig = go.Figure()

        return fig

    def create_network_summary_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create network analysis summary"""
        if 'user_id' not in df.columns:
            return go.Figure()

        user_activity = df['user_id'].value_counts().head(10)

        fig = go.Figure(data=[
            go.Bar(
                x=user_activity.values,
                y=user_activity.index,
                orientation='h',
                marker_color=COLORS['secondary'],
                marker_line=dict(width=1, color=COLORS['primary'])
            )
        ])

        fig.update_layout(
            title=dict(
                text="Top Active Users",
                font=dict(size=14, color=COLORS['primary']),
                x=0.5
            ),
            xaxis=dict(
                title="Messages",
                titlefont=dict(color=COLORS['primary'], size=12)
            ),
            yaxis=dict(
                title="User ID",
                titlefont=dict(color=COLORS['primary'], size=12)
            ),
            height=250,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

def create_metrics_cards(summary_data: Dict, political_df: pd.DataFrame) -> None:
    """Create metrics display cards"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = summary_data.get('stage_summary', {}).get('01_chunk_processing', {}).get('records_processed', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-label">Total Messages</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        stages_completed = summary_data.get('stages_completed', 0)
        total_stages = summary_data.get('total_stages', 22)
        completion_rate = (stages_completed / total_stages * 100) if total_stages > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{completion_rate:.1f}%</div>
            <div class="metric-label">Pipeline Completion</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if political_df is not None and 'political_alignment' in political_df.columns:
            political_categories = political_df['political_alignment'].nunique()
        else:
            political_categories = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{political_categories}</div>
            <div class="metric-label">Political Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        processing_date = summary_data.get('timestamp', 'Unknown')
        if processing_date != 'Unknown':
            try:
                date_obj = datetime.strptime(processing_date, '%Y%m%d_%H%M%S')
                processing_date = date_obj.strftime('%Y-%m-%d')
            except:
                pass

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{processing_date}</div>
            <div class="metric-label">Last Analysis</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""

    # Top header
    st.markdown("""
    <div class="top-header">
        <h1 style="margin: 0; text-align: center;">Brazilian Political Discourse Analysis Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;">
            Professional Analysis of Telegram Political Content (2019-2023)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize data loader and chart creator
    data_loader = DataLoader()
    chart_creator = ChartCreator()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem; background: {COLORS['primary']}; color: white; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0; text-align: center;">Navigation</h3>
        </div>
        """, unsafe_allow_html=True)

        view_option = st.selectbox(
            "Select Analysis View",
            ["Overview", "Political Analysis", "Sentiment Analysis", "Temporal Analysis", "Network Analysis"],
            key="view_selector"
        )

        st.markdown("---")

        # Data refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            data_loader.cache.clear()
            st.rerun()

        # Data status
        st.markdown("### Data Status")
        available_files = len(list(data_loader.data_path.glob("*.csv")))
        st.metric("Available Files", available_files)

        summary_data = data_loader.load_summary_data()
        if summary_data:
            st.metric("Pipeline Stages", f"{summary_data.get('stages_completed', 0)}/{summary_data.get('total_stages', 22)}")

    # Load data
    with st.spinner("Loading analysis data..."):
        political_df = data_loader.load_political_data()
        sentiment_df = data_loader.load_sentiment_data()
        temporal_df = data_loader.load_temporal_data()
        network_df = data_loader.load_network_data()
        summary_data = data_loader.load_summary_data() or {}

    # Display metrics cards
    if summary_data:
        create_metrics_cards(summary_data, political_df)

    st.markdown("---")

    # Main content based on selected view
    if view_option == "Overview":
        # 3-part layout: Main area + 2 side areas
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown('<div class="section-header">Political Distribution Analysis</div>', unsafe_allow_html=True)
            if political_df is not None:
                fig = chart_creator.create_political_distribution_chart(political_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Political analysis data not available")

        with col2:
            st.markdown('<div class="small-chart-header">Content Categories</div>', unsafe_allow_html=True)
            if political_df is not None:
                fig = chart_creator.create_category_pie_chart(political_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Category data not available")

        with col3:
            st.markdown('<div class="small-chart-header">Sentiment Trends</div>', unsafe_allow_html=True)
            if sentiment_df is not None:
                fig = chart_creator.create_sentiment_timeline(sentiment_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sentiment data not available")

        # Second row: Main temporal heatmap + side network summary
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('<div class="section-header">Activity Temporal Patterns</div>', unsafe_allow_html=True)
            if temporal_df is not None:
                fig = chart_creator.create_temporal_heatmap(temporal_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Temporal analysis data not available")

        with col2:
            st.markdown('<div class="small-chart-header">Network Activity</div>', unsafe_allow_html=True)
            if network_df is not None:
                fig = chart_creator.create_network_summary_chart(network_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Network data not available")

    elif view_option == "Political Analysis":
        if political_df is not None:
            st.markdown('<div class="section-header">Detailed Political Analysis</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                fig = chart_creator.create_political_distribution_chart(political_df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = chart_creator.create_category_pie_chart(political_df)
                st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.markdown('<div class="section-header">Sample Political Classifications</div>', unsafe_allow_html=True)
            display_cols = ['text_content', 'political_category', 'political_alignment']
            available_cols = [col for col in display_cols if col in political_df.columns]
            if available_cols:
                st.dataframe(
                    political_df[available_cols].head(10),
                    use_container_width=True
                )
        else:
            st.error("Political analysis data not available")

    elif view_option == "Sentiment Analysis":
        if sentiment_df is not None:
            st.markdown('<div class="section-header">Sentiment Analysis Results</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = chart_creator.create_sentiment_timeline(sentiment_df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'sentiment_category' in sentiment_df.columns:
                    sentiment_counts = sentiment_df['sentiment_category'].value_counts()
                    fig = go.Figure(data=[go.Pie(
                        labels=sentiment_counts.index,
                        values=sentiment_counts.values,
                        marker_colors=[COLORS['success'], COLORS['warning'], COLORS['danger']]
                    )])
                    fig.update_layout(title="Sentiment Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Sentiment analysis data not available")

    elif view_option == "Temporal Analysis":
        if temporal_df is not None:
            st.markdown('<div class="section-header">Temporal Activity Patterns</div>', unsafe_allow_html=True)
            fig = chart_creator.create_temporal_heatmap(temporal_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Temporal analysis data not available")

    elif view_option == "Network Analysis":
        if network_df is not None:
            st.markdown('<div class="section-header">Network Analysis Results</div>', unsafe_allow_html=True)
            fig = chart_creator.create_network_summary_chart(network_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Network analysis data not available")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {COLORS['text_dark']}; padding: 1rem;">
        <small>
            Brazilian Political Discourse Analysis Dashboard v5.1.0 |
            Academic Research Tool |
            Data processed through 22-stage pipeline
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()