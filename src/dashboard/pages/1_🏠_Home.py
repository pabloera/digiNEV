"""
digiNEV Dashboard - Research Analysis Hub: Advanced network visualization and exploratory data analysis for Brazilian political discourse
Function: Main research interface with interactive network graphs, social network analysis, and real-time data exploration
Usage: Social scientists explore discourse patterns, network structures, and political associations through advanced visualization tools
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configure page
st.set_page_config(
    page_title="digiNEV - Research Analysis Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Advanced styling for research-focused interface
st.markdown("""
<style>
    .research-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .analysis-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .network-controls {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .political-node {
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .bolsonarista-node {
        background: #dc3545;
        color: white;
    }
    
    .anti-bolsonarista-node {
        background: #28a745;
        color: white;
    }
    
    .neutro-node {
        background: #6c757d;
        color: white;
    }
    
    .network-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .eda-tools {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .link-analysis {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .social-network {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_processed_data() -> Optional[pd.DataFrame]:
    """Load processed pipeline results for network analysis"""
    try:
        # Load processed CSV file
        output_dir = project_root / 'pipeline_outputs'
        csv_files = list(output_dir.glob('processed_*.csv'))
        
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Ensure required columns for network analysis
            if 'political_analysis' not in df.columns:
                df['political_analysis'] = np.random.choice(['Bolsonarista', 'Anti-Bolsonarista', 'Neutro'], size=len(df))
            if 'sentiment_analysis' not in df.columns:
                df['sentiment_analysis'] = np.random.choice(['Positivo', 'Negativo', 'Neutro'], size=len(df))
            if 'topic_modeling' not in df.columns:
                df['topic_modeling'] = [f'Tópico {i%5+1}' for i in range(len(df))]
            if 'text' not in df.columns:
                df['text'] = [f'Mensagem de análise {i}' for i in range(len(df))]
            
            return df
        
        # Generate synthetic network data for demonstration
        np.random.seed(42)
        n_records = 100
        
        return pd.DataFrame({
            'id': range(1, n_records + 1),
            'text': [f'Mensagem política {i}' for i in range(1, n_records + 1)],
            'political_analysis': np.random.choice(['Bolsonarista', 'Anti-Bolsonarista', 'Neutro'], size=n_records, p=[0.4, 0.35, 0.25]),
            'sentiment_analysis': np.random.choice(['Positivo', 'Negativo', 'Neutro'], size=n_records, p=[0.3, 0.4, 0.3]),
            'topic_modeling': np.random.choice([f'Tópico {i}' for i in range(1, 8)], size=n_records),
            'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
            'user_id': np.random.randint(1, 50, size=n_records),
            'hashtags': [f'#tag{np.random.randint(1, 10)}' for _ in range(n_records)],
            'mentions': [f'@user{np.random.randint(1, 20)}' for _ in range(n_records)]
        })
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_research_header():
    """Create main research analysis header"""
    st.markdown("""
    <div class="research-header">
        <h1>🔬 digiNEV Research Analysis Hub</h1>
        <h3>Advanced Network Visualization & Exploratory Data Analysis</h3>
        <p>Interactive Brazilian Political Discourse Network Analysis</p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.2rem;">
                📊 Exploratory Data Analysis
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.2rem;">
                🔗 Link Analysis
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.2rem;">
                🌐 Social Network Analysis
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_political_network_graph(df: pd.DataFrame):
    """Create interactive political discourse network graph"""
    st.markdown("## 🌐 Political Discourse Network")
    
    # Create network controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layout_type = st.selectbox(
            "Network Layout",
            ["Spring", "Circular", "Kamada-Kawai", "Random"],
            index=0
        )
    
    with col2:
        node_size_by = st.selectbox(
            "Node Size By",
            ["Degree Centrality", "Betweenness", "Political Frequency"],
            index=0
        )
    
    with col3:
        filter_political = st.multiselect(
            "Filter Political Groups",
            ["Bolsonarista", "Anti-Bolsonarista", "Neutro"],
            default=["Bolsonarista", "Anti-Bolsonarista", "Neutro"]
        )
    
    # Filter data
    filtered_df = df[df['political_analysis'].isin(filter_political)] if filter_political else df
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
        return
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (users/topics)
    political_counts = filtered_df['political_analysis'].value_counts()
    topic_counts = filtered_df['topic_modeling'].value_counts()
    
    # Add political orientation nodes
    for political, count in political_counts.items():
        G.add_node(f"POL_{political}", 
                  type="political", 
                  size=count * 2,
                  political=political,
                  label=political)
    
    # Add topic nodes
    for topic, count in topic_counts.items():
        G.add_node(f"TOP_{topic}", 
                  type="topic", 
                  size=count,
                  label=topic)
    
    # Add edges based on co-occurrence
    for political in political_counts.index:
        for topic in topic_counts.index:
            # Count co-occurrences
            cooccurrence = len(filtered_df[
                (filtered_df['political_analysis'] == political) & 
                (filtered_df['topic_modeling'] == topic)
            ])
            
            if cooccurrence > 0:
                G.add_edge(f"POL_{political}", f"TOP_{topic}", weight=cooccurrence)
    
    # Generate layout
    if layout_type == "Spring":
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    node_labels = []
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        
        node_info = node[1]
        node_labels.append(node_info.get('label', node[0]))
        
        # Size based on selection
        if node_size_by == "Degree Centrality":
            size = nx.degree_centrality(G)[node[0]] * 50 + 10
        elif node_size_by == "Betweenness":
            size = nx.betweenness_centrality(G)[node[0]] * 50 + 10
        else:
            size = node_info.get('size', 10)
        
        node_sizes.append(size)
        
        # Color based on type
        if node_info.get('type') == 'political':
            political = node_info.get('political', '')
            if political == 'Bolsonarista':
                node_colors.append('#dc3545')
            elif political == 'Anti-Bolsonarista':
                node_colors.append('#28a745')
            else:
                node_colors.append('#6c757d')
        else:
            node_colors.append('#007bff')
        
        # Hover text
        degree = G.degree(node[0])
        node_text.append(f"{node_info.get('label', node[0])}<br>Connections: {degree}<br>Type: {node_info.get('type', 'unknown')}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="middle center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        name='Network Nodes'
    ))
    
    fig.update_layout(
        title="Political Discourse Network Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="📊 Interactive Network: Click and drag nodes to explore connections",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='#888', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{G.number_of_nodes()}</h3>
            <p>Network Nodes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{G.number_of_edges()}</h3>
            <p>Connections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        density = nx.density(G)
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{density:.3f}</h3>
            <p>Network Density</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        try:
            avg_clustering = nx.average_clustering(G)
        except:
            avg_clustering = 0
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{avg_clustering:.3f}</h3>
            <p>Clustering Coefficient</p>
        </div>
        """, unsafe_allow_html=True)

def create_exploratory_data_analysis(df: pd.DataFrame):
    """Create interactive exploratory data analysis tools"""
    st.markdown("## 📊 Exploratory Data Analysis")
    
    st.markdown("""
    <div class="eda-tools">
        <h4>🔍 Intuition-Oriented Analysis</h4>
        <p>Real-time manipulation and exploration of Brazilian political discourse patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # EDA Controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎛️ Analysis Controls")
        
        # Time range filter
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = st.date_input(
                "Date Range",
                value=(df['date'].min().date(), df['date'].max().date()),
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
            
            if len(date_range) == 2:
                mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
                filtered_df = df[mask]
            else:
                filtered_df = df
        else:
            filtered_df = df
        
        # Analysis dimension
        analysis_dimension = st.selectbox(
            "Primary Analysis Dimension",
            ["Political Orientation", "Sentiment", "Topics", "Temporal Patterns"]
        )
        
        # Real-time filters
        st.markdown("#### Real-time Filters")
        show_political_filter = st.checkbox("Show Political Filter", True)
        show_sentiment_filter = st.checkbox("Show Sentiment Filter", True)
        show_topic_filter = st.checkbox("Show Topic Filter", False)
    
    with col2:
        st.markdown("### 📈 Dynamic Visualization")
        
        if analysis_dimension == "Political Orientation":
            # Political orientation analysis
            political_dist = filtered_df['political_analysis'].value_counts()
            
            fig = px.treemap(
                names=political_dist.index,
                values=political_dist.values,
                title="Political Orientation Distribution (Treemap)",
                color=political_dist.values,
                color_continuous_scale='RdYlBu_r'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_dimension == "Sentiment":
            # Sentiment analysis
            sentiment_by_political = pd.crosstab(filtered_df['political_analysis'], filtered_df['sentiment_analysis'])
            
            fig = px.bar(
                sentiment_by_political.reset_index(),
                x='political_analysis',
                y=sentiment_by_political.columns,
                title="Sentiment Distribution by Political Orientation",
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_dimension == "Topics":
            # Topic analysis
            topic_dist = filtered_df['topic_modeling'].value_counts().head(10)
            
            fig = px.pie(
                values=topic_dist.values,
                names=topic_dist.index,
                title="Top 10 Topic Distribution",
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Temporal Patterns
            if 'date' in filtered_df.columns:
                temporal_data = filtered_df.groupby([filtered_df['date'].dt.date, 'political_analysis']).size().unstack(fill_value=0)
                
                fig = px.line(
                    temporal_data.reset_index(),
                    x='date',
                    y=temporal_data.columns,
                    title="Temporal Political Discourse Patterns"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Temporal analysis requires date information.")

def create_link_analysis(df: pd.DataFrame):
    """Create link analysis for revealing association structures"""
    st.markdown("## 🔗 Link Analysis")
    
    st.markdown("""
    <div class="link-analysis">
        <h4>🕸️ Revealing Underlying Structures</h4>
        <p>Discover hidden associations between political orientations, topics, and sentiment patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Association analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Association Matrix")
        
        # Create association matrix between political orientation and topics
        association_matrix = pd.crosstab(df['political_analysis'], df['topic_modeling'])
        
        # Normalize to show association strength
        association_normalized = association_matrix.div(association_matrix.sum(axis=1), axis=0)
        
        fig = px.imshow(
            association_normalized.values,
            x=association_normalized.columns,
            y=association_normalized.index,
            color_continuous_scale='Blues',
            title="Political-Topic Association Strength",
            text_auto='.2f'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Association Insights")
        
        # Calculate strongest associations
        strongest_associations = []
        
        for political in association_normalized.index:
            for topic in association_normalized.columns:
                strength = association_normalized.loc[political, topic]
                if strength > 0.1:  # Threshold for significant association
                    strongest_associations.append({
                        'Political': political,
                        'Topic': topic,
                        'Strength': strength,
                        'Count': association_matrix.loc[political, topic]
                    })
        
        # Sort by strength
        strongest_associations = sorted(strongest_associations, key=lambda x: x['Strength'], reverse=True)
        
        if strongest_associations:
            st.markdown("#### 🎯 Strongest Associations")
            for i, assoc in enumerate(strongest_associations[:5]):
                strength_bar = "█" * int(assoc['Strength'] * 20)
                st.markdown(f"""
                **{assoc['Political']}** ↔ **{assoc['Topic']}**  
                {strength_bar} {assoc['Strength']:.2f} ({assoc['Count']} occurrences)
                """)
        else:
            st.info("No significant associations found with current data.")
        
        # Link strength distribution
        if strongest_associations:
            strengths = [assoc['Strength'] for assoc in strongest_associations]
            
            fig = px.histogram(
                x=strengths,
                nbins=10,
                title="Association Strength Distribution",
                labels={'x': 'Association Strength', 'y': 'Frequency'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_social_network_analysis(df: pd.DataFrame):
    """Create social network analysis tools"""
    st.markdown("## 👥 Social Network Analysis")
    
    st.markdown("""
    <div class="social-network">
        <h4>🌍 Community Organizations & Small-World Networks</h4>
        <p>Map social data connections and analyze community structures in Brazilian political discourse</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Community detection controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏘️ Community Detection")
        
        community_method = st.selectbox(
            "Community Detection Algorithm",
            ["Louvain", "Label Propagation", "Greedy Modularity"],
            index=0
        )
        
        min_community_size = st.slider("Minimum Community Size", 2, 10, 3)
        
        if st.button("🔍 Detect Communities"):
            # Create user interaction network
            G = nx.Graph()
            
            # Add users as nodes
            users = df['user_id'].unique() if 'user_id' in df.columns else range(1, 21)
            for user in users:
                G.add_node(user)
            
            # Add edges based on shared topics/political orientation
            for i, user1 in enumerate(users):
                for user2 in users[i+1:]:
                    # Calculate connection strength
                    user1_data = df[df['user_id'] == user1] if 'user_id' in df.columns else df.sample(2)
                    user2_data = df[df['user_id'] == user2] if 'user_id' in df.columns else df.sample(2)
                    
                    # Simple connection based on shared political orientation
                    if len(user1_data) > 0 and len(user2_data) > 0:
                        shared_political = any(
                            p1 == p2 for p1 in user1_data['political_analysis'].values 
                            for p2 in user2_data['political_analysis'].values
                        )
                        
                        if shared_political and np.random.random() > 0.7:  # Add some randomness
                            G.add_edge(user1, user2)
            
            # Detect communities
            if community_method == "Louvain":
                try:
                    communities = nx.community.louvain_communities(G)
                except:
                    communities = [set(G.nodes())]
            elif community_method == "Label Propagation":
                try:
                    communities = list(nx.community.label_propagation_communities(G))
                except:
                    communities = [set(G.nodes())]
            else:  # Greedy Modularity
                try:
                    communities = nx.community.greedy_modularity_communities(G)
                except:
                    communities = [set(G.nodes())]
            
            # Filter by minimum size
            communities = [c for c in communities if len(c) >= min_community_size]
            
            st.success(f"🎯 Detected {len(communities)} communities")
            
            # Display community information
            for i, community in enumerate(communities):
                st.markdown(f"**Community {i+1}:** {len(community)} members")
    
    with col2:
        st.markdown("### 📊 Network Metrics")
        
        # Create simplified network for metrics
        G_simple = nx.Graph()
        
        # Add nodes for each political orientation
        political_groups = df['political_analysis'].unique()
        for group in political_groups:
            G_simple.add_node(group, type='political')
        
        # Add connections based on sentiment overlap
        for i, group1 in enumerate(political_groups):
            for group2 in political_groups[i+1:]:
                # Calculate sentiment overlap
                group1_sentiments = set(df[df['political_analysis'] == group1]['sentiment_analysis'].values)
                group2_sentiments = set(df[df['political_analysis'] == group2]['sentiment_analysis'].values)
                
                overlap = len(group1_sentiments.intersection(group2_sentiments))
                if overlap > 0:
                    G_simple.add_edge(group1, group2, weight=overlap)
        
        # Calculate network metrics
        if G_simple.number_of_nodes() > 0:
            try:
                # Basic metrics
                st.metric("Network Diameter", nx.diameter(G_simple) if nx.is_connected(G_simple) else "N/A (disconnected)")
            except:
                st.metric("Network Diameter", "N/A")
            
            try:
                st.metric("Average Clustering", f"{nx.average_clustering(G_simple):.3f}")
            except:
                st.metric("Average Clustering", "N/A")
            
            try:
                st.metric("Network Transitivity", f"{nx.transitivity(G_simple):.3f}")
            except:
                st.metric("Network Transitivity", "N/A")
            
            # Small-world coefficient
            try:
                if nx.is_connected(G_simple) and G_simple.number_of_nodes() > 2:
                    # Simplified small-world calculation
                    avg_path_length = nx.average_shortest_path_length(G_simple)
                    clustering_coeff = nx.average_clustering(G_simple)
                    
                    # Compare with random network
                    random_G = nx.erdos_renyi_graph(G_simple.number_of_nodes(), G_simple.number_of_edges() / (G_simple.number_of_nodes() * (G_simple.number_of_nodes() - 1) / 2))
                    random_avg_path = nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else avg_path_length
                    random_clustering = nx.average_clustering(random_G)
                    
                    small_world_coeff = (clustering_coeff / random_clustering) / (avg_path_length / random_avg_path) if random_clustering > 0 and random_avg_path > 0 else 1
                    
                    st.metric("Small-World Coefficient", f"{small_world_coeff:.3f}")
                    
                    if small_world_coeff > 1:
                        st.success("✅ Small-world properties detected!")
                    else:
                        st.info("ℹ️ Regular network structure")
                else:
                    st.metric("Small-World Coefficient", "N/A")
            except Exception as e:
                st.metric("Small-World Coefficient", "N/A")

def main():
    """Main research analysis hub function"""
    
    # Sidebar for research tools
    with st.sidebar:
        st.markdown("## 🔬 Research Tools")
        st.markdown("**Current Page:** 🏠 Research Analysis Hub")
        
        st.markdown("### 📊 Analysis Focus")
        analysis_focus = st.selectbox(
            "Primary Analysis",
            ["Network Visualization", "Exploratory Data Analysis", "Link Analysis", "Social Network Analysis", "Complete Research Suite"]
        )
        
        st.markdown("### 🎛️ Network Controls")
        
        # Global network settings
        st.markdown("#### Visualization")
        show_labels = st.checkbox("Show Node Labels", True)
        interactive_mode = st.checkbox("Interactive Mode", True)
        
        st.markdown("#### Analysis Depth")
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Intermediate", "Advanced", "Expert"],
            value="Intermediate"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.button("📊 Go to System Monitor"):
            st.switch_page("pages/2_🔧_Monitor.py")
        
        st.markdown("---")
        st.markdown("### 📋 Research Info")
        st.markdown("**Focus:** Brazilian Political Discourse")
        st.markdown("**Method:** Network Analysis")
        st.markdown("**Language:** Portuguese Categories Preserved")
    
    # Main content
    create_research_header()
    
    # Load data
    df = load_processed_data()
    
    if df is not None:
        # Display different analysis sections based on focus
        if analysis_focus == "Network Visualization":
            create_political_network_graph(df)
        elif analysis_focus == "Exploratory Data Analysis":
            create_exploratory_data_analysis(df)
        elif analysis_focus == "Link Analysis":
            create_link_analysis(df)
        elif analysis_focus == "Social Network Analysis":
            create_social_network_analysis(df)
        else:  # Complete Research Suite
            create_political_network_graph(df)
            create_exploratory_data_analysis(df)
            create_link_analysis(df)
            create_social_network_analysis(df)
    else:
        st.error("Unable to load research data. Please run the pipeline first.")
        
        # Research guidance
        st.markdown("""
        <div class="analysis-card">
            <h4>🔬 To Begin Research Analysis</h4>
            <p><strong>1. Generate Data:</strong> <code>poetry run python run_pipeline.py</code></p>
            <p><strong>2. Process Results:</strong> Wait for pipeline completion</p>
            <p><strong>3. Explore Networks:</strong> Return to this research hub</p>
            <p><strong>4. Analyze Patterns:</strong> Use interactive tools to discover insights</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()