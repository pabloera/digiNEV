"""
digiNEV Dashboard - System Monitor: Consolidated monitoring for pipeline execution, quality control, costs, and performance
Function: Unified monitoring interface combining pipeline status, quality metrics, cost tracking, and performance analytics
Usage: Social scientists monitor all system aspects in one centralized view while focusing on research analysis
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="digiNEV - System Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Consolidated monitoring styling
st.markdown("""
<style>
    .monitor-header {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .monitor-section {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .status-excellent {
        border-left: 4px solid #28a745;
        background: #f8fff9;
    }
    
    .status-good {
        border-left: 4px solid #17a2b8;
        background: #f0f9ff;
    }
    
    .status-warning {
        border-left: 4px solid #ffc107;
        background: #fffcf0;
    }
    
    .status-critical {
        border-left: 4px solid #dc3545;
        background: #fff5f5;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .compact-chart {
        height: 300px;
        margin: 0.5rem 0;
    }
    
    .system-alert {
        border: 2px solid;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-success {
        border-color: #28a745;
        background: #f8fff9;
        color: #155724;
    }
    
    .alert-warning {
        border-color: #ffc107;
        background: #fffcf0;
        color: #856404;
    }
    
    .alert-danger {
        border-color: #dc3545;
        background: #fff5f5;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def load_pipeline_results() -> List[Dict]:
    """Load pipeline results for comprehensive monitoring"""
    try:
        results_dir = project_root / 'src' / 'dashboard' / 'data' / 'dashboard_results'
        json_files = list(results_dir.glob('*.json'))
        
        if not json_files:
            return []
        
        results = []
        for file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
            try:
                with open(file, 'r') as f:
                    results.append(json.load(f))
            except Exception:
                continue
        
        return results
    except Exception:
        return []

def calculate_system_overview(results: List[Dict]) -> Dict:
    """Calculate comprehensive system overview metrics"""
    if not results:
        return {
            'pipeline': {'status': 'No Data', 'success_rate': 0, 'avg_time': 0},
            'quality': {'score': 0, 'variability': 0},
            'cost': {'monthly': 0, 'per_execution': 0},
            'performance': {'throughput': 0, 'efficiency': 0}
        }
    
    # Pipeline metrics
    success_rates = []
    execution_times = []
    records_processed = []
    
    for result in results:
        stages_completed = result.get('stages_completed', {})
        if stages_completed:
            successful_stages = sum(1 for stage_data in stages_completed.values() 
                                   if stage_data and any(s.get('success', False) for s in stage_data))
            total_stages = len(stages_completed)
            success_rates.append((successful_stages / total_stages) * 100 if total_stages > 0 else 0)
        
        execution_times.append(result.get('execution_time', 0))
        records_processed.append(result.get('total_records_processed', 0))
    
    # Calculate overview metrics
    avg_success_rate = np.mean(success_rates) if success_rates else 0
    avg_execution_time = np.mean(execution_times) if execution_times else 0
    total_records = sum(records_processed)
    
    # Quality metrics
    quality_score = avg_success_rate
    variability = np.std(success_rates) if len(success_rates) > 1 else 0
    
    # Cost estimation (simplified)
    COST_PER_MESSAGE = 0.0001  # Estimated cost per message
    estimated_monthly_cost = total_records * COST_PER_MESSAGE * 4  # Assuming monthly execution
    cost_per_execution = estimated_monthly_cost / len(results) if results else 0
    
    # Performance metrics
    throughput = total_records / sum(execution_times) if sum(execution_times) > 0 else 0
    efficiency = (avg_success_rate / 100) * throughput if throughput > 0 else 0
    
    # Determine status
    if avg_success_rate >= 95 and variability <= 5:
        pipeline_status = 'Excellent'
    elif avg_success_rate >= 85 and variability <= 10:
        pipeline_status = 'Good'
    elif avg_success_rate >= 70:
        pipeline_status = 'Warning'
    else:
        pipeline_status = 'Critical'
    
    return {
        'pipeline': {
            'status': pipeline_status,
            'success_rate': avg_success_rate,
            'avg_time': avg_execution_time,
            'executions': len(results)
        },
        'quality': {
            'score': quality_score,
            'variability': variability,
            'status': 'Excellent' if variability <= 5 else 'Warning' if variability <= 15 else 'Critical'
        },
        'cost': {
            'monthly': estimated_monthly_cost,
            'per_execution': cost_per_execution,
            'status': 'Good' if estimated_monthly_cost <= 50 else 'Warning' if estimated_monthly_cost <= 75 else 'Critical'
        },
        'performance': {
            'throughput': throughput,
            'efficiency': efficiency,
            'total_records': total_records,
            'status': 'Good' if efficiency > 10 else 'Warning' if efficiency > 5 else 'Critical'
        }
    }

def create_monitor_header():
    """Create consolidated monitoring header"""
    st.markdown("""
    <div class="monitor-header">
        <h1>🔧 System Monitor</h1>
        <h3>Consolidated Pipeline | Quality | Cost | Performance Monitoring</h3>
        <p>Unified view of all system aspects while focusing on research analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_system_overview(overview: Dict):
    """Create comprehensive system overview dashboard"""
    st.markdown("## 📊 System Overview")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pipeline_status = overview['pipeline']['status']
        status_class = f"status-{pipeline_status.lower()}" if pipeline_status != 'Excellent' else "status-excellent"
        
        st.markdown(f"""
        <div class="monitor-section {status_class}">
            <h4>🔄 Pipeline Status</h4>
            <h3>{pipeline_status}</h3>
            <p>{overview['pipeline']['success_rate']:.1f}% Success Rate</p>
            <small>{overview['pipeline']['executions']} executions</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_status = overview['quality']['status']
        status_class = f"status-{quality_status.lower()}" if quality_status != 'Excellent' else "status-excellent"
        
        st.markdown(f"""
        <div class="monitor-section {status_class}">
            <h4>🔧 Quality Score</h4>
            <h3>{overview['quality']['score']:.1f}%</h3>
            <p>±{overview['quality']['variability']:.1f}% variability</p>
            <small>{quality_status} quality</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cost_status = overview['cost']['status']
        status_class = f"status-{cost_status.lower()}"
        
        st.markdown(f"""
        <div class="monitor-section {status_class}">
            <h4>💰 Monthly Cost</h4>
            <h3>${overview['cost']['monthly']:.2f}</h3>
            <p>${overview['cost']['per_execution']:.4f} per run</p>
            <small>{cost_status} budget</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        perf_status = overview['performance']['status']
        status_class = f"status-{perf_status.lower()}"
        
        st.markdown(f"""
        <div class="monitor-section {status_class}">
            <h4>⚡ Performance</h4>
            <h3>{overview['performance']['efficiency']:.1f}</h3>
            <p>{overview['performance']['throughput']:.1f} rec/sec</p>
            <small>{perf_status} efficiency</small>
        </div>
        """, unsafe_allow_html=True)

def create_pipeline_monitoring_compact(results: List[Dict]):
    """Create compact pipeline monitoring section"""
    st.markdown("## 🔄 Pipeline Monitoring")
    
    if not results:
        st.warning("No pipeline execution data available.")
        return
    
    # Recent executions summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Recent Execution Trends")
        
        # Prepare trend data
        trend_data = []
        for i, result in enumerate(results[:10]):  # Last 10 executions
            stages_completed = result.get('stages_completed', {})
            success_count = sum(1 for stage_data in stages_completed.values() 
                               if stage_data and any(s.get('success', False) for s in stage_data))
            
            trend_data.append({
                'Execution': f"Run {i+1}",
                'Success_Rate': (success_count / len(stages_completed) * 100) if stages_completed else 0,
                'Execution_Time': result.get('execution_time', 0),
                'Records': result.get('total_records_processed', 0)
            })
        
        if trend_data:
            df_trend = pd.DataFrame(trend_data)
            
            fig = px.line(df_trend, x='Execution', y='Success_Rate',
                         title='Success Rate Trend (Last 10 Runs)',
                         markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Stage Success Analysis")
        
        # Aggregate stage success rates
        stage_success = {}
        for result in results:
            stages_completed = result.get('stages_completed', {})
            for stage_name, stage_data in stages_completed.items():
                if stage_name not in stage_success:
                    stage_success[stage_name] = {'success': 0, 'total': 0}
                
                if stage_data and len(stage_data) > 0:
                    stage_success[stage_name]['total'] += 1
                    if stage_data[0].get('success', False):
                        stage_success[stage_name]['success'] += 1
        
        # Calculate success rates and create chart
        if stage_success:
            stage_names = []
            success_rates = []
            
            for stage, stats in stage_success.items():
                if stats['total'] > 0:
                    stage_names.append(stage.replace('_', ' ').title()[:20] + '...' if len(stage) > 20 else stage.replace('_', ' ').title())
                    success_rates.append((stats['success'] / stats['total']) * 100)
            
            if stage_names:
                fig = px.bar(x=success_rates, y=stage_names, orientation='h',
                           title='Stage Success Rates (%)',
                           color=success_rates,
                           color_continuous_scale='RdYlGn')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def create_quality_monitoring_compact(results: List[Dict]):
    """Create compact quality monitoring section"""
    st.markdown("## 🔧 Quality Control")
    
    if not results:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Quality Metrics")
        
        # Calculate quality statistics
        success_rates = []
        execution_times = []
        
        for result in results:
            stages_completed = result.get('stages_completed', {})
            if stages_completed:
                successful_stages = sum(1 for stage_data in stages_completed.values() 
                                       if stage_data and any(s.get('success', False) for s in stage_data))
                total_stages = len(stages_completed)
                success_rates.append((successful_stages / total_stages) * 100 if total_stages > 0 else 0)
            
            execution_times.append(result.get('execution_time', 0))
        
        if success_rates:
            # Quality metrics
            mean_success = np.mean(success_rates)
            std_success = np.std(success_rates)
            
            st.metric("Average Success Rate", f"{mean_success:.1f}%")
            st.metric("Quality Variability", f"±{std_success:.1f}%")
            st.metric("Quality Status", 
                     "✅ Excellent" if std_success <= 5 else 
                     "⚠️ Warning" if std_success <= 15 else 
                     "❌ Critical")
        
        # Academic standards check
        st.markdown("#### 🎓 Academic Standards")
        if success_rates:
            academic_compliance = mean_success >= 95 and std_success <= 5
            st.markdown(f"**Publication Ready:** {'✅ Yes' if academic_compliance else '❌ No'}")
            st.markdown(f"**Reproducibility:** {'✅ High' if std_success <= 5 else '⚠️ Moderate' if std_success <= 15 else '❌ Low'}")
    
    with col2:
        st.markdown("### 📈 Control Chart")
        
        if success_rates and len(success_rates) > 1:
            # Simple control chart
            mean_line = np.mean(success_rates)
            std_line = np.std(success_rates)
            ucl = mean_line + 2 * std_line
            lcl = max(0, mean_line - 2 * std_line)
            
            fig = go.Figure()
            
            # Data points
            fig.add_trace(go.Scatter(
                x=list(range(len(success_rates))),
                y=success_rates,
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='blue')
            ))
            
            # Control limits
            fig.add_hline(y=mean_line, line_dash="solid", line_color="green", 
                         annotation_text=f"Mean: {mean_line:.1f}%")
            fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                         annotation_text=f"UCL: {ucl:.1f}%")
            fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                         annotation_text=f"LCL: {lcl:.1f}%")
            
            fig.update_layout(
                title="Quality Control Chart",
                xaxis_title="Execution",
                yaxis_title="Success Rate (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_cost_performance_monitoring(overview: Dict):
    """Create compact cost and performance monitoring"""
    st.markdown("## 💰 Cost & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Cost Analysis")
        
        # Budget status
        monthly_cost = overview['cost']['monthly']
        budget_limit = 50.0  # Academic budget
        usage_percent = (monthly_cost / budget_limit) * 100
        
        # Cost breakdown visualization
        cost_data = {
            'Category': ['Used Budget', 'Remaining Budget'],
            'Amount': [monthly_cost, max(0, budget_limit - monthly_cost)],
            'Color': ['#dc3545' if usage_percent > 80 else '#ffc107' if usage_percent > 60 else '#28a745', '#e9ecef']
        }
        
        fig = px.pie(
            values=cost_data['Amount'],
            names=cost_data['Category'],
            title=f"Budget Usage: {usage_percent:.1f}%",
            color_discrete_sequence=cost_data['Color']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost alerts
        if usage_percent > 90:
            alert_class = "alert-danger"
            alert_message = f"🚨 Budget exceeded! ${monthly_cost:.2f} / ${budget_limit:.2f}"
        elif usage_percent > 75:
            alert_class = "alert-warning"
            alert_message = f"⚠️ High usage: ${monthly_cost:.2f} / ${budget_limit:.2f}"
        else:
            alert_class = "alert-success"
            alert_message = f"✅ Budget healthy: ${monthly_cost:.2f} / ${budget_limit:.2f}"
        
        st.markdown(f"""
        <div class="system-alert {alert_class}">
            <strong>{alert_message}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚡ Performance Metrics")
        
        # Performance indicators
        throughput = overview['performance']['throughput']
        efficiency = overview['performance']['efficiency']
        total_records = overview['performance']['total_records']
        
        st.metric("Throughput", f"{throughput:.1f} records/sec")
        st.metric("Efficiency Score", f"{efficiency:.1f}")
        st.metric("Total Records Processed", f"{total_records:,}")
        
        # Performance trend (simulated)
        performance_data = {
            'Metric': ['Throughput', 'Efficiency', 'Success Rate'],
            'Value': [throughput, efficiency, overview['pipeline']['success_rate']],
            'Target': [20, 15, 95]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.bar(df_perf, x='Metric', y=['Value', 'Target'],
                    title='Performance vs Targets',
                    barmode='group')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def create_system_alerts(overview: Dict):
    """Create system alerts and recommendations"""
    st.markdown("## 🚨 System Alerts & Recommendations")
    
    alerts = []
    
    # Pipeline alerts
    pipeline_success = overview['pipeline']['success_rate']
    if pipeline_success < 85:
        alerts.append({
            'type': 'danger',
            'title': 'Pipeline Performance Critical',
            'message': f'Success rate ({pipeline_success:.1f}%) below academic standards (≥95%)',
            'action': 'Review API configurations and stage implementations'
        })
    elif pipeline_success < 95:
        alerts.append({
            'type': 'warning',
            'title': 'Pipeline Performance Warning',
            'message': f'Success rate ({pipeline_success:.1f}%) below optimal for research',
            'action': 'Consider optimizing error handling and stage robustness'
        })
    
    # Quality alerts
    quality_variability = overview['quality']['variability']
    if quality_variability > 15:
        alerts.append({
            'type': 'warning',
            'title': 'Quality Variability High',
            'message': f'Variability ({quality_variability:.1f}%) affects reproducibility',
            'action': 'Implement more consistent processing conditions'
        })
    
    # Cost alerts
    monthly_cost = overview['cost']['monthly']
    if monthly_cost > 50:
        alerts.append({
            'type': 'danger',
            'title': 'Budget Exceeded',
            'message': f'Monthly cost (${monthly_cost:.2f}) exceeds academic budget ($50)',
            'action': 'Enable cost optimization features (caching, sampling)'
        })
    elif monthly_cost > 37.5:  # 75% of budget
        alerts.append({
            'type': 'warning',
            'title': 'Budget Warning',
            'message': f'Monthly cost (${monthly_cost:.2f}) approaching limit',
            'action': 'Monitor usage and consider optimization'
        })
    
    # Performance alerts
    efficiency = overview['performance']['efficiency']
    if efficiency < 5:
        alerts.append({
            'type': 'warning',
            'title': 'Performance Below Optimal',
            'message': f'System efficiency ({efficiency:.1f}) can be improved',
            'action': 'Enable parallelization and streaming optimizations'
        })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            alert_class = f"alert-{alert['type']}"
            icon = '🚨' if alert['type'] == 'danger' else '⚠️'
            
            st.markdown(f"""
            <div class="system-alert {alert_class}">
                <h5>{icon} {alert['title']}</h5>
                <p><strong>Issue:</strong> {alert['message']}</p>
                <p><strong>Recommended Action:</strong> {alert['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="system-alert alert-success">
            <h5>✅ All Systems Operational</h5>
            <p>No critical alerts. System is operating within academic research parameters.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main consolidated monitoring function"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.markdown("**Current Page:** 🔧 System Monitor")
        
        st.markdown("### 🔄 Monitoring Options")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)")
        
        # Monitoring focus
        monitor_focus = st.selectbox(
            "Monitoring Focus",
            ["Complete Overview", "Pipeline Focus", "Quality Focus", "Cost Focus", "Performance Focus"]
        )
        
        # Time range
        time_range = st.selectbox(
            "Time Range",
            ["Last 10 executions", "Last 20 executions", "All available data"]
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.button("🏠 Back to Research Hub"):
            st.switch_page("pages/1_🏠_Home.py")
        
        if st.button("📊 Execute Pipeline"):
            st.info("Run: `poetry run python run_pipeline.py`")
        
        st.markdown("---")
        st.markdown("### 📋 System Info")
        st.markdown("**Focus:** Unified Monitoring")
        st.markdown("**Purpose:** Support Research Analysis")
        st.markdown("**Academic:** Budget & Quality Optimized")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    # Main content
    create_monitor_header()
    
    # Load data and calculate overview
    results = load_pipeline_results()
    overview = calculate_system_overview(results)
    
    # Display monitoring sections based on focus
    if monitor_focus == "Complete Overview":
        create_system_overview(overview)
        create_pipeline_monitoring_compact(results)
        create_quality_monitoring_compact(results)
        create_cost_performance_monitoring(overview)
        create_system_alerts(overview)
    elif monitor_focus == "Pipeline Focus":
        create_system_overview(overview)
        create_pipeline_monitoring_compact(results)
    elif monitor_focus == "Quality Focus":
        create_system_overview(overview)
        create_quality_monitoring_compact(results)
    elif monitor_focus == "Cost Focus":
        create_system_overview(overview)
        create_cost_performance_monitoring(overview)
    elif monitor_focus == "Performance Focus":
        create_system_overview(overview)
        create_cost_performance_monitoring(overview)
    
    # Always show alerts if there are any
    if monitor_focus == "Complete Overview":
        pass  # Already shown above
    else:
        create_system_alerts(overview)
    
    # Quick stats footer
    if results:
        st.markdown("---")
        st.markdown(f"""
        **System Summary:** {len(results)} recent executions | 
        {overview['pipeline']['success_rate']:.1f}% avg success | 
        ${overview['cost']['monthly']:.2f} monthly cost | 
        {overview['performance']['total_records']:,} records processed
        """)
    else:
        st.warning("No execution data available. Run the pipeline to generate monitoring data.")

if __name__ == "__main__":
    main()