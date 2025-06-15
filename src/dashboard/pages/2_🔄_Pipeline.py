"""
digiNEV Dashboard - Pipeline Monitor: Real-time monitoring of 22-stage Brazilian political discourse analysis pipeline
Function: Live pipeline execution tracking with stage-by-stage progress, error detection, and performance metrics
Usage: Social scientists monitor analysis execution, identify bottlenecks, and track processing status in real-time
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="digiNEV - Pipeline Monitor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Professional styling
st.markdown("""
<style>
    .pipeline-header {
        background: linear-gradient(90deg, #2196F3 0%, #21CBF3 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stage-group {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stage-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .stage-item:last-child {
        border-bottom: none;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-failed {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-running {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-pending {
        color: #6c757d;
        font-weight: bold;
    }
    
    .progress-bar {
        width: 100%;
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        transition: width 0.3s ease;
    }
    
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_pipeline_results() -> Optional[Dict]:
    """Load most recent pipeline results"""
    try:
        results_dir = project_root / 'src' / 'dashboard' / 'data' / 'dashboard_results'
        json_files = list(results_dir.glob('*.json'))
        
        if not json_files:
            return None
            
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def get_stage_groups() -> Dict[str, List[str]]:
    """Define logical groupings of pipeline stages"""
    return {
        "🔧 Data Foundation": [
            "01_chunk_processing",
            "02_encoding_validation", 
            "03_deduplication",
            "04_feature_validation",
            "04b_statistical_analysis_pre"
        ],
        "🤖 AI & Semantic Analysis": [
            "05_political_analysis",
            "08_sentiment_analysis",
            "09_topic_modeling",
            "10_tfidf_extraction",
            "11_clustering"
        ],
        "📝 Content & Context Analysis": [
            "06_text_cleaning",
            "07_linguistic_processing",
            "12_hashtag_normalization",
            "13_domain_analysis",
            "14_temporal_analysis",
            "15_network_analysis",
            "16_qualitative_analysis"
        ],
        "✅ Validation & Quality": [
            "17_smart_pipeline_review",
            "18_topic_interpretation", 
            "19_semantic_search",
            "20_pipeline_validation",
            "academic_performance_summary"
        ]
    }

def create_pipeline_header():
    """Create pipeline monitoring header"""
    st.markdown("""
    <div class="pipeline-header">
        <h1>🔄 Pipeline Monitor</h1>
        <h3>Real-time Execution Tracking & Performance Analytics</h3>
        <p>Monitor 22-Stage Brazilian Political Discourse Analysis Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

def create_overall_progress():
    """Create overall pipeline progress visualization"""
    st.markdown("## 📊 Overall Pipeline Progress")
    
    results = load_pipeline_results()
    if not results:
        st.warning("No pipeline execution data available. Run the pipeline to see progress.")
        return
    
    # Calculate overall progress
    stages_completed = results.get('stages_completed', {})
    total_stages = len(stages_completed)
    successful_stages = sum(1 for stage_data in stages_completed.values() 
                           if stage_data and any(s.get('success', False) for s in stage_data))
    
    progress_percentage = (successful_stages / total_stages * 100) if total_stages > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{successful_stages}/{total_stages}</h3>
            <p>Stages Completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{progress_percentage:.1f}%</h3>
            <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        execution_time = results.get('execution_time', 0)
        st.markdown(f"""
        <div class="metric-box">
            <h3>{execution_time:.2f}s</h3>
            <p>Execution Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_records = results.get('total_records_processed', 0)
        st.markdown(f"""
        <div class="metric-box">
            <h3>{total_records:,}</h3>
            <p>Records Processed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_percentage}%"></div>
    </div>
    <p style="text-align: center; margin-top: 0.5rem;">
        Pipeline Progress: {progress_percentage:.1f}%
    </p>
    """, unsafe_allow_html=True)

def create_stage_group_monitoring():
    """Create detailed stage group monitoring"""
    st.markdown("## 🔍 Stage Group Details")
    
    results = load_pipeline_results()
    if not results:
        return
    
    stage_groups = get_stage_groups()
    stages_completed = results.get('stages_completed', {})
    
    # Create tabs for each stage group
    tabs = st.tabs(list(stage_groups.keys()))
    
    for tab, (group_name, stage_list) in zip(tabs, stage_groups.items()):
        with tab:
            st.markdown(f"### {group_name}")
            
            # Group statistics
            group_success = 0
            group_total = len(stage_list)
            group_records = 0
            
            stage_details = []
            
            for stage_name in stage_list:
                stage_data = stages_completed.get(stage_name, [])
                if stage_data and len(stage_data) > 0:
                    success = stage_data[0].get('success', False)
                    records = stage_data[0].get('records', 0)
                    
                    if success:
                        group_success += 1
                        group_records += records
                    
                    status_icon = "✅" if success else "❌"
                    status_class = "status-success" if success else "status-failed"
                    
                    stage_details.append({
                        'Stage': stage_name.replace('_', ' ').title(),
                        'Status': f"{status_icon} {'Success' if success else 'Failed'}",
                        'Records': f"{records:,}",
                        'Status_Class': status_class
                    })
                else:
                    stage_details.append({
                        'Stage': stage_name.replace('_', ' ').title(),
                        'Status': "⏳ Pending",
                        'Records': "0",
                        'Status_Class': "status-pending"
                    })
            
            # Group summary
            group_percentage = (group_success / group_total * 100) if group_total > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stages Completed", f"{group_success}/{group_total}")
            with col2:
                st.metric("Success Rate", f"{group_percentage:.1f}%")
            with col3:
                st.metric("Records Processed", f"{group_records:,}")
            
            # Stage details table
            if stage_details:
                df = pd.DataFrame(stage_details)
                st.dataframe(
                    df[['Stage', 'Status', 'Records']], 
                    use_container_width=True,
                    hide_index=True
                )

def create_timeline_visualization():
    """Create pipeline execution timeline"""
    st.markdown("## ⏱️ Execution Timeline")
    
    results = load_pipeline_results()
    if not results:
        return
    
    stages_completed = results.get('stages_completed', {})
    
    # Create timeline data
    timeline_data = []
    start_time = datetime.fromisoformat(results.get('start_time', '').replace('Z', '+00:00')) if results.get('start_time') else datetime.now()
    
    current_time = start_time
    for stage_name, stage_data in stages_completed.items():
        if stage_data and len(stage_data) > 0:
            success = stage_data[0].get('success', False)
            records = stage_data[0].get('records', 0)
            
            # Estimate duration (simplified)
            duration = 0.01 if success else 0.05  # Successful stages are faster
            
            timeline_data.append({
                'Stage': stage_name.replace('_', ' ').title(),
                'Start': current_time,
                'End': current_time + timedelta(seconds=duration),
                'Success': success,
                'Records': records
            })
            
            current_time += timedelta(seconds=duration)
    
    if timeline_data:
        # Create Gantt chart
        fig = go.Figure()
        
        for i, item in enumerate(timeline_data):
            color = '#28a745' if item['Success'] else '#dc3545'
            
            fig.add_trace(go.Bar(
                x=[item['End'] - item['Start']],
                y=[item['Stage']],
                orientation='h',
                name=item['Stage'],
                marker=dict(color=color),
                text=f"Records: {item['Records']}",
                textposition='inside',
                hovertemplate=f"<b>{item['Stage']}</b><br>Records: {item['Records']}<br>Status: {'Success' if item['Success'] else 'Failed'}<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Pipeline Execution Timeline",
            xaxis_title="Duration",
            yaxis_title="Pipeline Stages",
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_performance_metrics():
    """Create performance metrics visualization"""
    st.markdown("## 📈 Performance Metrics")
    
    results = load_pipeline_results()
    if not results:
        return
    
    # Load recent results for comparison
    results_dir = project_root / 'src' / 'dashboard' / 'data' / 'dashboard_results'
    json_files = list(results_dir.glob('*.json'))
    
    if len(json_files) < 2:
        st.info("Need multiple execution results for performance comparison.")
        return
    
    # Get recent files for trend analysis
    recent_files = sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    
    trend_data = []
    for file in recent_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            stages_success = sum(1 for stage_data in data.get('stages_completed', {}).values() 
                               if stage_data and any(s.get('success', False) for s in stage_data))
            
            trend_data.append({
                'Timestamp': data.get('end_time', ''),
                'Success_Rate': (stages_success / len(data.get('stages_completed', {})) * 100) if data.get('stages_completed') else 0,
                'Execution_Time': data.get('execution_time', 0),
                'Records_Processed': data.get('total_records_processed', 0)
            })
        except Exception:
            continue
    
    if trend_data:
        df = pd.DataFrame(trend_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate trend
            fig1 = px.line(df, x='Timestamp', y='Success_Rate', 
                          title='Success Rate Trend',
                          labels={'Success_Rate': 'Success Rate (%)'})
            fig1.update_traces(line_color='#28a745')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Execution time trend
            fig2 = px.line(df, x='Timestamp', y='Execution_Time',
                          title='Execution Time Trend',
                          labels={'Execution_Time': 'Time (seconds)'})
            fig2.update_traces(line_color='#007bff')
            st.plotly_chart(fig2, use_container_width=True)

def create_monitoring_controls():
    """Create monitoring and control buttons"""
    st.markdown("## 🎛️ Pipeline Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.switch_page("pages/3_📊_Analytics.py")
    
    with col3:
        if st.button("🔧 Quality Control", use_container_width=True):
            st.switch_page("pages/4_🔧_Quality.py")
    
    with col4:
        auto_refresh = st.checkbox("Auto Refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()

def main():
    """Main pipeline monitoring function"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.markdown("**Current Page:** 🔄 Pipeline Monitor")
        
        st.markdown("### 📋 Monitoring Options")
        
        # Real-time toggle
        real_time = st.toggle("Real-time Monitoring")
        if real_time:
            st.info("🔴 Live monitoring active")
        
        # Refresh interval
        refresh_interval = st.selectbox(
            "Refresh Interval",
            [5, 10, 30, 60],
            index=2
        )
        
        st.markdown("---")
        st.markdown("### 📊 Pipeline Info")
        results = load_pipeline_results()
        if results:
            st.markdown(f"**Last Run:** {results.get('end_time', 'Unknown')}")
            st.markdown(f"**Records:** {results.get('total_records_processed', 0):,}")
            st.markdown(f"**Duration:** {results.get('execution_time', 0):.2f}s")
    
    # Main content
    create_pipeline_header()
    create_overall_progress()
    create_stage_group_monitoring()
    create_timeline_visualization()
    create_performance_metrics()
    create_monitoring_controls()

if __name__ == "__main__":
    main()