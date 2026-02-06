"""
Stage 14 Network Analysis Dashboard
===================================

Comprehensive network analysis dashboard for Brazilian political discourse coordination detection.
Implements 4 core network visualizations:
1. Force-directed network: User/channel coordination connections
2. Community detection: Algorithmic identification of coordinated groups
3. Centrality analysis: Influence metrics (betweenness, closeness, degree, eigenvector)
4. Multi-layer network: Coordination + sentiment + topics integration

Academic Focus: Revealing coordination patterns and influence networks in political messaging.
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as colors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
import itertools
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkAnalysisDashboard:
    """
    Comprehensive network analysis dashboard for coordination detection.
    """

    def __init__(self):
        """Initialize dashboard components."""
        self.network_graph = None
        self.community_data = None
        self.centrality_metrics = None
        self.multi_layer_data = None

    def load_data(self, df: pd.DataFrame) -> bool:
        """
        Load and validate network analysis data.

        Args:
            df: DataFrame with network analysis results from Stage 14

        Returns:
            bool: Success status
        """
        try:
            # Required columns for network analysis
            required_cols = ['user_id', 'channel', 'sender_frequency', 'is_frequent_sender',
                           'shared_url_frequency', 'temporal_coordination']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Colunas necessárias ausentes: {missing_cols}")
                return False

            self.df = df.copy()
            logger.info(f"Dados carregados: {len(self.df)} registros para análise de rede")
            return True

        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            logger.error(f"Erro ao carregar dados: {e}")
            return False

    def build_coordination_network(self) -> nx.Graph:
        """
        Build network graph based on coordination patterns.

        Returns:
            nx.Graph: Network graph with coordination edges
        """
        try:
            G = nx.Graph()

            # Add nodes (users and channels)
            users = self.df['user_id'].unique()
            channels = self.df['channel'].unique()

            # Add user nodes
            for user in users:
                user_data = self.df[self.df['user_id'] == user].iloc[0]
                G.add_node(f"user_{user}",
                          node_type='user',
                          frequency=user_data['sender_frequency'],
                          is_frequent=user_data['is_frequent_sender'],
                          label=f"User {user}")

            # Add channel nodes
            for channel in channels:
                channel_msgs = self.df[self.df['channel'] == channel]
                G.add_node(f"channel_{channel}",
                          node_type='channel',
                          message_count=len(channel_msgs),
                          label=f"Canal {channel}")

            # Add edges based on coordination patterns
            self._add_coordination_edges(G)

            logger.info(f"Rede construída: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
            return G

        except Exception as e:
            logger.error(f"Erro ao construir rede: {e}")
            return nx.Graph()

    def _add_coordination_edges(self, G: nx.Graph):
        """Add edges based on coordination patterns."""

        # User-Channel connections
        for _, row in self.df.iterrows():
            user_node = f"user_{row['user_id']}"
            channel_node = f"channel_{row['channel']}"

            if G.has_node(user_node) and G.has_node(channel_node):
                weight = row['sender_frequency'] * row['temporal_coordination']
                G.add_edge(user_node, channel_node,
                          weight=weight,
                          edge_type='participation')

        # User-User coordination based on shared URLs and temporal patterns
        users_data = self.df.groupby('user_id').agg({
            'shared_url_frequency': 'mean',
            'temporal_coordination': 'mean',
            'hour': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).reset_index()

        for i, user1_data in users_data.iterrows():
            for j, user2_data in users_data.iterrows():
                if i >= j:
                    continue

                # Calculate coordination strength
                url_similarity = min(user1_data['shared_url_frequency'],
                                   user2_data['shared_url_frequency'])
                temporal_similarity = abs(user1_data['temporal_coordination'] -
                                        user2_data['temporal_coordination'])

                coordination_strength = url_similarity * (1 - temporal_similarity)

                if coordination_strength > 0.1:  # Threshold for coordination
                    user1_node = f"user_{user1_data['user_id']}"
                    user2_node = f"user_{user2_data['user_id']}"

                    if G.has_node(user1_node) and G.has_node(user2_node):
                        G.add_edge(user1_node, user2_node,
                                  weight=coordination_strength,
                                  edge_type='coordination')

    def detect_communities(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Detect communities using modularity-based clustering.

        Args:
            G: Network graph

        Returns:
            Dict: Community detection results
        """
        try:
            if G.number_of_nodes() == 0:
                return {}

            # Use Louvain algorithm for community detection
            communities = nx.community.louvain_communities(G, seed=42)

            # Create community mapping
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i

            # Calculate modularity
            modularity = nx.community.modularity(G, communities)

            # Community statistics
            community_stats = {}
            for i, community in enumerate(communities):
                user_count = sum(1 for node in community if node.startswith('user_'))
                channel_count = sum(1 for node in community if node.startswith('channel_'))

                # Internal edges (within community)
                subgraph = G.subgraph(community)
                internal_edges = subgraph.number_of_edges()

                community_stats[f"Community_{i}"] = {
                    'size': len(community),
                    'users': user_count,
                    'channels': channel_count,
                    'internal_edges': internal_edges,
                    'nodes': list(community)
                }

            return {
                'communities': communities,
                'community_map': community_map,
                'modularity': modularity,
                'stats': community_stats
            }

        except Exception as e:
            logger.error(f"Erro na detecção de comunidades: {e}")
            return {}

    def calculate_centrality_metrics(self, G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Calculate multiple centrality metrics.

        Args:
            G: Network graph

        Returns:
            Dict: Centrality metrics for each node
        """
        try:
            if G.number_of_nodes() == 0:
                return {}

            centrality_metrics = {}

            # Degree centrality
            degree_centrality = nx.degree_centrality(G)

            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G)

            # Closeness centrality (only for connected components)
            if nx.is_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
            else:
                closeness_centrality = {}
                for component in nx.connected_components(G):
                    subgraph = G.subgraph(component)
                    if len(component) > 1:
                        closeness_centrality.update(nx.closeness_centrality(subgraph))
                    else:
                        closeness_centrality.update({node: 0.0 for node in component})

            # Eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                eigenvector_centrality = {node: 0.0 for node in G.nodes()}

            # Combine all metrics
            for node in G.nodes():
                centrality_metrics[node] = {
                    'degree': degree_centrality.get(node, 0),
                    'betweenness': betweenness_centrality.get(node, 0),
                    'closeness': closeness_centrality.get(node, 0),
                    'eigenvector': eigenvector_centrality.get(node, 0)
                }

            logger.info(f"Métricas de centralidade calculadas para {len(centrality_metrics)} nós")
            return centrality_metrics

        except Exception as e:
            logger.error(f"Erro no cálculo de centralidade: {e}")
            return {}

    def create_multi_layer_data(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Create multi-layer network data combining coordination, sentiment, and topics.

        Args:
            G: Base network graph

        Returns:
            Dict: Multi-layer network data
        """
        try:
            multi_layer = {
                'coordination_layer': {},
                'sentiment_layer': {},
                'topic_layer': {}
            }

            # Coordination layer (base network)
            for node in G.nodes():
                multi_layer['coordination_layer'][node] = {
                    'connections': list(G.neighbors(node)),
                    'weight': G.degree(node, weight='weight')
                }

            # Sentiment layer
            if 'sentiment_polarity' in self.df.columns:
                sentiment_groups = self.df.groupby('user_id')['sentiment_polarity'].mean()
                for user_id, sentiment in sentiment_groups.items():
                    node = f"user_{user_id}"
                    if node in G.nodes():
                        multi_layer['sentiment_layer'][node] = {
                            'sentiment': sentiment,
                            'sentiment_category': 'positive' if sentiment > 0.1 else 'negative' if sentiment < -0.1 else 'neutral'
                        }

            # Topic layer
            if 'dominant_topic' in self.df.columns:
                topic_groups = self.df.groupby('user_id')['dominant_topic'].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)
                for user_id, topic in topic_groups.items():
                    node = f"user_{user_id}"
                    if node in G.nodes():
                        multi_layer['topic_layer'][node] = {
                            'topic_id': topic,
                            'topic_label': f"Topic_{topic}"
                        }

            logger.info("Dados multi-layer criados")
            return multi_layer

        except Exception as e:
            logger.error(f"Erro na criação de dados multi-layer: {e}")
            return {}

    def render_force_directed_network(self, G: nx.Graph, communities: Dict = None):
        """
        Render interactive force-directed network visualization.

        Args:
            G: Network graph
            communities: Community detection results
        """
        st.subheader("1. Rede de Coordenação Force-Directed")
        st.write("**Visualização interativa das conexões de coordenação entre usuários e canais**")

        if G.number_of_nodes() == 0:
            st.warning("Rede vazia - dados insuficientes para análise")
            return

        # Interactive controls
        col1, col2, col3 = st.columns(3)

        with col1:
            layout_algorithm = st.selectbox(
                "Algoritmo de Layout",
                ["spring", "circular", "kamada_kawai", "random"],
                index=0,
                help="Escolha o algoritmo para posicionamento dos nós"
            )

        with col2:
            node_size_metric = st.selectbox(
                "Tamanho do Nó por",
                ["frequency", "degree", "betweenness", "uniform"],
                index=0,
                help="Métrica para determinar o tamanho dos nós"
            )

        with col3:
            edge_weight_threshold = st.slider(
                "Limiar de Peso das Arestas",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Mostrar apenas arestas acima deste peso"
            )

        # Filter edges by weight
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                          if d.get('weight', 0) < edge_weight_threshold]
        G_filtered = G.copy()
        G_filtered.remove_edges_from(edges_to_remove)

        # Calculate layout based on selection
        if layout_algorithm == "spring":
            try:
                pos = nx.spring_layout(G_filtered, k=3, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G_filtered, seed=42)
        elif layout_algorithm == "circular":
            pos = nx.circular_layout(G_filtered)
        elif layout_algorithm == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G_filtered)
            except:
                pos = nx.spring_layout(G_filtered, seed=42)
        else:
            pos = nx.random_layout(G_filtered, seed=42)

        # Calculate centrality for node sizing
        if node_size_metric == "degree":
            centrality = nx.degree_centrality(G_filtered)
        elif node_size_metric == "betweenness":
            centrality = nx.betweenness_centrality(G_filtered)
        else:
            centrality = {node: G_filtered.nodes[node].get('frequency', 1) for node in G_filtered.nodes()}

        # Prepare node traces
        node_trace = self._create_node_trace_interactive(G_filtered, pos, communities, centrality, node_size_metric)
        edge_trace = self._create_edge_trace_weighted(G_filtered, pos)

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f"Rede de Coordenação - Layout: {layout_algorithm.title()}",
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"Nós: {G_filtered.number_of_nodes()} | Arestas: {G_filtered.number_of_edges()}<br>" +
                                   f"Filtro de peso: ≥{edge_weight_threshold}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           height=700
                       ))

        st.plotly_chart(fig, use_container_width=True)

        # Network statistics
        self._display_network_stats_enhanced(G_filtered, edge_weight_threshold)

        # Node selection and detailed info
        self._render_node_explorer(G_filtered, centrality)

    def _create_node_trace(self, G: nx.Graph, pos: Dict, communities: Dict = None):
        """Create node trace for plotly."""
        node_x = []
        node_y = []
        node_info = []
        node_color = []
        node_size = []
        node_symbol = []

        # Color palette for communities
        color_palette = px.colors.qualitative.Set3

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node attributes
            node_data = G.nodes[node]
            node_type = node_data.get('node_type', 'unknown')

            # Node info for hover
            if node_type == 'user':
                frequency = node_data.get('frequency', 0)
                is_frequent = node_data.get('is_frequent', False)
                info = f"User: {node}<br>Frequency: {frequency}<br>Frequent: {is_frequent}"
                symbol = 'circle'
                size = 15 + frequency * 2
            else:  # channel
                msg_count = node_data.get('message_count', 0)
                info = f"Channel: {node}<br>Messages: {msg_count}"
                symbol = 'diamond'
                size = 20 + msg_count * 0.5

            node_info.append(info)
            node_symbol.append(symbol)
            node_size.append(min(size, 40))  # Cap size

            # Community coloring
            if communities and 'community_map' in communities:
                community_id = communities['community_map'].get(node, 0)
                color = color_palette[community_id % len(color_palette)]
            else:
                color = 'lightblue' if node_type == 'user' else 'lightcoral'

            node_color.append(color)

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_info,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='black'),
                symbol=node_symbol
            )
        )

    def _create_edge_trace(self, G: nx.Graph, pos: Dict):
        """Create edge trace for plotly."""
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

    def _create_node_trace_interactive(self, G: nx.Graph, pos: Dict, communities: Dict,
                                     centrality: Dict, size_metric: str):
        """Create enhanced interactive node trace."""
        node_x = []
        node_y = []
        node_info = []
        node_color = []
        node_size = []
        node_symbol = []

        # Color palette for communities
        color_palette = px.colors.qualitative.Set3

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node attributes
            node_data = G.nodes[node]
            node_type = node_data.get('node_type', 'unknown')

            # Calculate size based on selected metric
            if size_metric == "uniform":
                size = 15
            else:
                size_value = centrality.get(node, 0)
                size = 10 + size_value * 30  # Scale to reasonable size

            # Node info for hover
            if node_type == 'user':
                frequency = node_data.get('frequency', 0)
                is_frequent = node_data.get('is_frequent', False)
                degree = G.degree(node)
                info = (f"User: {node}<br>"
                       f"Frequency: {frequency}<br>"
                       f"Frequent: {is_frequent}<br>"
                       f"Degree: {degree}<br>"
                       f"{size_metric.title()}: {centrality.get(node, 0):.3f}")
                symbol = 'circle'
            else:  # channel
                msg_count = node_data.get('message_count', 0)
                degree = G.degree(node)
                info = (f"Channel: {node}<br>"
                       f"Messages: {msg_count}<br>"
                       f"Degree: {degree}<br>"
                       f"{size_metric.title()}: {centrality.get(node, 0):.3f}")
                symbol = 'diamond'

            node_info.append(info)
            node_symbol.append(symbol)
            node_size.append(min(max(size, 8), 50))  # Cap size between 8-50

            # Community coloring
            if communities and 'community_map' in communities:
                community_id = communities['community_map'].get(node, 0)
                color = color_palette[community_id % len(color_palette)]
            else:
                color = 'lightblue' if node_type == 'user' else 'lightcoral'

            node_color.append(color)

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            name='Nós da Rede',
            hoverinfo='text',
            text=node_info,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='black'),
                symbol=node_symbol
            )
        )

    def _create_edge_trace_weighted(self, G: nx.Graph, pos: Dict):
        """Create weighted edge trace for plotly."""
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            weight = edge[2].get('weight', 0.1)
            edge_weights.append(weight)

        # Normalize weights for line width
        if edge_weights:
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            if max_weight > min_weight:
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 3 + 0.5
                                    for w in edge_weights]
            else:
                normalized_weights = [1.0] * len(edge_weights)
        else:
            normalized_weights = [1.0]

        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='rgba(128,128,128,0.6)'),
            hoverinfo='none',
            mode='lines',
            name='Conexões'
        )

    def _display_network_stats_enhanced(self, G: nx.Graph, weight_threshold: float):
        """Display enhanced network statistics."""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Nós", G.number_of_nodes())
        with col2:
            st.metric("Arestas", G.number_of_edges())
        with col3:
            density = nx.density(G)
            st.metric("Densidade", f"{density:.3f}")
        with col4:
            components = nx.number_connected_components(G)
            st.metric("Componentes", components)
        with col5:
            if G.number_of_edges() > 0:
                avg_clustering = nx.average_clustering(G)
                st.metric("Clustering Médio", f"{avg_clustering:.3f}")
            else:
                st.metric("Clustering Médio", "0.000")

        # Additional statistics
        if G.number_of_nodes() > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                # Degree distribution
                degrees = [d for n, d in G.degree()]
                if degrees:
                    avg_degree = np.mean(degrees)
                    max_degree = max(degrees)
                    st.write(f"**Grau médio:** {avg_degree:.2f}")
                    st.write(f"**Grau máximo:** {max_degree}")

            with col2:
                # Edge weight statistics
                if G.number_of_edges() > 0:
                    weights = [d.get('weight', 0) for u, v, d in G.edges(data=True)]
                    if weights:
                        avg_weight = np.mean(weights)
                        max_weight = max(weights)
                        st.write(f"**Peso médio das arestas:** {avg_weight:.3f}")
                        st.write(f"**Peso máximo:** {max_weight:.3f}")
                        st.write(f"**Limiar aplicado:** ≥{weight_threshold}")

            with col3:
                # Network type identification
                user_nodes = [n for n in G.nodes() if n.startswith('user_')]
                channel_nodes = [n for n in G.nodes() if n.startswith('channel_')]
                st.write(f"**Usuários:** {len(user_nodes)}")
                st.write(f"**Canais:** {len(channel_nodes)}")

    def _render_node_explorer(self, G: nx.Graph, centrality: Dict):
        """Render node explorer for detailed analysis."""
        st.subheader("🔍 Explorador de Nós")

        if G.number_of_nodes() == 0:
            st.warning("Nenhum nó disponível para exploração")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            # Node selection
            selected_node = st.selectbox(
                "Selecionar Nó para Análise",
                options=sorted(G.nodes()),
                help="Escolha um nó para análise detalhada"
            )

        with col2:
            if selected_node:
                # Node details
                node_data = G.nodes[selected_node]
                neighbors = list(G.neighbors(selected_node))
                degree = G.degree(selected_node)

                st.write(f"**Nó Selecionado:** {selected_node}")
                st.write(f"**Tipo:** {node_data.get('node_type', 'Unknown')}")
                st.write(f"**Grau:** {degree}")
                st.write(f"**Vizinhos:** {len(neighbors)}")
                st.write(f"**Centralidade:** {centrality.get(selected_node, 0):.3f}")

                if neighbors:
                    st.write("**Conectado a:**")
                    neighbor_list = ", ".join(neighbors[:10])
                    if len(neighbors) > 10:
                        neighbor_list += f" ... (+{len(neighbors) - 10} mais)"
                    st.write(neighbor_list)

        # Ego network visualization
        if selected_node and st.checkbox("Mostrar Rede Ego", help="Visualizar apenas o nó selecionado e seus vizinhos"):
            ego_graph = nx.ego_graph(G, selected_node, radius=1)

            if ego_graph.number_of_nodes() > 1:
                st.write(f"**Rede Ego de {selected_node}**")

                # Simple ego network visualization
                pos = nx.spring_layout(ego_graph, seed=42)

                ego_fig = go.Figure()

                # Edges
                for edge in ego_graph.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    ego_fig.add_trace(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='gray'),
                        hoverinfo='none',
                        showlegend=False
                    ))

                # Nodes
                node_x = [pos[node][0] for node in ego_graph.nodes()]
                node_y = [pos[node][1] for node in ego_graph.nodes()]
                node_colors = ['red' if node == selected_node else 'lightblue' for node in ego_graph.nodes()]
                node_sizes = [25 if node == selected_node else 15 for node in ego_graph.nodes()]

                ego_fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=list(ego_graph.nodes()),
                    textposition="middle center",
                    marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='black')),
                    hoverinfo='text',
                    name='Nós',
                    showlegend=False
                ))

                ego_fig.update_layout(
                    title=f"Rede Ego: {selected_node}",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400,
                    plot_bgcolor='white'
                )

                st.plotly_chart(ego_fig, use_container_width=True)
            else:
                st.info("Nó isolado - sem conexões para mostrar")

    def _display_network_stats(self, G: nx.Graph):
        """Display network statistics."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nós", G.number_of_nodes())
        with col2:
            st.metric("Arestas", G.number_of_edges())
        with col3:
            density = nx.density(G)
            st.metric("Densidade", f"{density:.3f}")
        with col4:
            components = nx.number_connected_components(G)
            st.metric("Componentes", components)

    def render_community_detection(self, G: nx.Graph, communities: Dict):
        """
        Render community detection visualization.

        Args:
            G: Network graph
            communities: Community detection results
        """
        st.subheader("2. Detecção de Comunidades")
        st.write("**Identificação algorítmica de grupos coordenados usando modularidade**")

        if not communities or 'stats' not in communities:
            st.warning("Dados de comunidades não disponíveis")
            return

        # Community overview
        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Número de Comunidades", len(communities['stats']))
            st.metric("Modularidade", f"{communities.get('modularity', 0):.3f}")

        with col2:
            # Community size distribution
            sizes = [data['size'] for data in communities['stats'].values()]
            fig_sizes = px.bar(
                x=list(communities['stats'].keys()),
                y=sizes,
                title="Distribuição do Tamanho das Comunidades",
                labels={'x': 'Comunidade', 'y': 'Número de Nós'}
            )
            fig_sizes.update_layout(height=300)
            st.plotly_chart(fig_sizes, use_container_width=True)

        # Detailed community analysis
        st.subheader("Análise Detalhada das Comunidades")

        for comm_name, comm_data in communities['stats'].items():
            with st.expander(f"{comm_name} ({comm_data['size']} nós)"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Usuários", comm_data['users'])
                with col2:
                    st.metric("Canais", comm_data['channels'])
                with col3:
                    st.metric("Conexões Internas", comm_data['internal_edges'])

                # Node list
                if comm_data['nodes']:
                    user_nodes = [n for n in comm_data['nodes'] if n.startswith('user_')]
                    channel_nodes = [n for n in comm_data['nodes'] if n.startswith('channel_')]

                    if user_nodes:
                        st.write(f"**Usuários:** {', '.join(user_nodes[:10])}")
                        if len(user_nodes) > 10:
                            st.write(f"... e mais {len(user_nodes) - 10} usuários")

                    if channel_nodes:
                        st.write(f"**Canais:** {', '.join(channel_nodes)}")

    def render_centrality_analysis(self, G: nx.Graph, centrality_metrics: Dict):
        """
        Render centrality analysis visualization.

        Args:
            G: Network graph
            centrality_metrics: Calculated centrality metrics
        """
        st.subheader("3. Análise de Centralidade")
        st.write("**Identificação de nós influentes através de múltiplas métricas de centralidade**")

        if not centrality_metrics:
            st.warning("Métricas de centralidade não disponíveis")
            return

        # Prepare data for visualization
        nodes = list(centrality_metrics.keys())
        degree_vals = [centrality_metrics[node]['degree'] for node in nodes]
        betweenness_vals = [centrality_metrics[node]['betweenness'] for node in nodes]
        closeness_vals = [centrality_metrics[node]['closeness'] for node in nodes]
        eigenvector_vals = [centrality_metrics[node]['eigenvector'] for node in nodes]

        # Create subplots for different centrality measures
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Centralidade de Grau', 'Centralidade de Intermediação',
                          'Centralidade de Proximidade', 'Centralidade de Autovetor'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Degree centrality
        fig.add_trace(
            go.Bar(x=nodes[:10], y=degree_vals[:10], name="Grau", marker_color='lightblue'),
            row=1, col=1
        )

        # Betweenness centrality
        fig.add_trace(
            go.Bar(x=nodes[:10], y=betweenness_vals[:10], name="Intermediação", marker_color='lightcoral'),
            row=1, col=2
        )

        # Closeness centrality
        fig.add_trace(
            go.Bar(x=nodes[:10], y=closeness_vals[:10], name="Proximidade", marker_color='lightgreen'),
            row=2, col=1
        )

        # Eigenvector centrality
        fig.add_trace(
            go.Bar(x=nodes[:10], y=eigenvector_vals[:10], name="Autovetor", marker_color='lightyellow'),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False, title_text="Métricas de Centralidade")
        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

        # Top influential nodes
        self._display_top_influential_nodes(centrality_metrics)

    def _display_top_influential_nodes(self, centrality_metrics: Dict):
        """Display top influential nodes by different metrics."""
        st.subheader("Nós Mais Influentes")

        # Calculate composite influence score
        composite_scores = {}
        for node, metrics in centrality_metrics.items():
            # Weighted combination of centrality measures
            composite = (metrics['degree'] * 0.3 +
                        metrics['betweenness'] * 0.3 +
                        metrics['closeness'] * 0.2 +
                        metrics['eigenvector'] * 0.2)
            composite_scores[node] = composite

        # Sort by composite score
        top_nodes = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Display in table format
        table_data = []
        for node, score in top_nodes:
            metrics = centrality_metrics[node]
            table_data.append({
                'Nó': node,
                'Score Composto': f"{score:.3f}",
                'Grau': f"{metrics['degree']:.3f}",
                'Intermediação': f"{metrics['betweenness']:.3f}",
                'Proximidade': f"{metrics['closeness']:.3f}",
                'Autovetor': f"{metrics['eigenvector']:.3f}"
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)

    def render_multi_layer_network(self, G: nx.Graph, multi_layer_data: Dict):
        """
        Render multi-layer network visualization.

        Args:
            G: Network graph
            multi_layer_data: Multi-layer network data
        """
        st.subheader("4. Rede Multi-Layer")
        st.write("**Integração de coordenação, sentimento e tópicos em visualização unificada**")

        if not multi_layer_data:
            st.warning("Dados multi-layer não disponíveis")
            return

        # Layer selection
        selected_layers = st.multiselect(
            "Selecione as camadas para visualização:",
            ['Coordenação', 'Sentimento', 'Tópicos'],
            default=['Coordenação', 'Sentimento']
        )

        if not selected_layers:
            st.warning("Selecione pelo menos uma camada")
            return

        # Create multi-layer visualization
        fig = self._create_multi_layer_plot(G, multi_layer_data, selected_layers)
        st.plotly_chart(fig, use_container_width=True)

        # Layer statistics
        self._display_layer_statistics(multi_layer_data, selected_layers)

    def _create_multi_layer_plot(self, G: nx.Graph, multi_layer_data: Dict, selected_layers: List[str]):
        """Create multi-layer network plot."""
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)

        fig = go.Figure()

        # Base coordination layer (always shown)
        edge_trace = self._create_edge_trace(G, pos)
        fig.add_trace(edge_trace)

        # Node traces for different layers
        if 'Coordenação' in selected_layers:
            coord_trace = self._create_coordination_layer_trace(G, pos, multi_layer_data)
            fig.add_trace(coord_trace)

        if 'Sentimento' in selected_layers:
            sentiment_trace = self._create_sentiment_layer_trace(G, pos, multi_layer_data)
            if sentiment_trace:
                fig.add_trace(sentiment_trace)

        if 'Tópicos' in selected_layers:
            topic_trace = self._create_topic_layer_trace(G, pos, multi_layer_data)
            if topic_trace:
                fig.add_trace(topic_trace)

        fig.update_layout(
            title="Rede Multi-Layer: Coordenação + Sentimento + Tópicos",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )

        return fig

    def _create_coordination_layer_trace(self, G: nx.Graph, pos: Dict, multi_layer_data: Dict):
        """Create coordination layer trace."""
        node_x = []
        node_y = []
        node_info = []

        for node in G.nodes():
            if node in multi_layer_data.get('coordination_layer', {}):
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                coord_data = multi_layer_data['coordination_layer'][node]
                info = f"{node}<br>Conexões: {len(coord_data['connections'])}<br>Peso: {coord_data['weight']:.2f}"
                node_info.append(info)

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            name='Coordenação',
            hoverinfo='text',
            text=node_info,
            marker=dict(size=12, color='blue', symbol='circle')
        )

    def _create_sentiment_layer_trace(self, G: nx.Graph, pos: Dict, multi_layer_data: Dict):
        """Create sentiment layer trace."""
        sentiment_layer = multi_layer_data.get('sentiment_layer', {})
        if not sentiment_layer:
            return None

        node_x = []
        node_y = []
        node_info = []
        node_colors = []

        sentiment_color_map = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }

        for node in sentiment_layer:
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                sentiment_data = sentiment_layer[node]
                sentiment_cat = sentiment_data['sentiment_category']
                sentiment_val = sentiment_data['sentiment']

                info = f"{node}<br>Sentimento: {sentiment_cat}<br>Valor: {sentiment_val:.3f}"
                node_info.append(info)
                node_colors.append(sentiment_color_map[sentiment_cat])

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            name='Sentimento',
            hoverinfo='text',
            text=node_info,
            marker=dict(size=10, color=node_colors, symbol='square')
        )

    def _create_topic_layer_trace(self, G: nx.Graph, pos: Dict, multi_layer_data: Dict):
        """Create topic layer trace."""
        topic_layer = multi_layer_data.get('topic_layer', {})
        if not topic_layer:
            return None

        node_x = []
        node_y = []
        node_info = []
        node_colors = []

        # Color palette for topics
        topic_colors = px.colors.qualitative.Set1

        for node in topic_layer:
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                topic_data = topic_layer[node]
                topic_id = topic_data['topic_id']
                topic_label = topic_data['topic_label']

                info = f"{node}<br>Tópico: {topic_label}<br>ID: {topic_id}"
                node_info.append(info)

                color_idx = topic_id % len(topic_colors)
                node_colors.append(topic_colors[color_idx])

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            name='Tópicos',
            hoverinfo='text',
            text=node_info,
            marker=dict(size=8, color=node_colors, symbol='diamond')
        )

    def _display_layer_statistics(self, multi_layer_data: Dict, selected_layers: List[str]):
        """Display statistics for selected layers."""
        st.subheader("Estatísticas das Camadas")

        cols = st.columns(len(selected_layers))

        for i, layer in enumerate(selected_layers):
            with cols[i]:
                if layer == 'Coordenação':
                    coord_layer = multi_layer_data.get('coordination_layer', {})
                    st.metric(f"Nós {layer}", len(coord_layer))
                    if coord_layer:
                        avg_weight = np.mean([data['weight'] for data in coord_layer.values()])
                        st.metric("Peso Médio", f"{avg_weight:.2f}")

                elif layer == 'Sentimento':
                    sentiment_layer = multi_layer_data.get('sentiment_layer', {})
                    st.metric(f"Nós {layer}", len(sentiment_layer))
                    if sentiment_layer:
                        sentiments = [data['sentiment_category'] for data in sentiment_layer.values()]
                        sentiment_counts = Counter(sentiments)
                        st.write("Distribuição:")
                        for sent, count in sentiment_counts.items():
                            st.write(f"• {sent}: {count}")

                elif layer == 'Tópicos':
                    topic_layer = multi_layer_data.get('topic_layer', {})
                    st.metric(f"Nós {layer}", len(topic_layer))
                    if topic_layer:
                        topics = [data['topic_id'] for data in topic_layer.values()]
                        unique_topics = len(set(topics))
                        st.metric("Tópicos Únicos", unique_topics)

    def render_dashboard(self, df: pd.DataFrame):
        """
        Render complete network analysis dashboard.

        Args:
            df: DataFrame with network analysis results
        """
        st.title("🕸️ Stage 14: Análise de Rede")
        st.markdown("**Detecção de coordenação e padrões de influência no discurso político brasileiro**")

        if not self.load_data(df):
            return

        # Build network components
        with st.spinner("Construindo rede de coordenação..."):
            G = self.build_coordination_network()

        with st.spinner("Detectando comunidades..."):
            communities = self.detect_communities(G)

        with st.spinner("Calculando métricas de centralidade..."):
            centrality_metrics = self.calculate_centrality_metrics(G)

        with st.spinner("Criando dados multi-layer..."):
            multi_layer_data = self.create_multi_layer_data(G)

        # Render visualizations
        self.render_force_directed_network(G, communities)

        st.markdown("---")
        self.render_community_detection(G, communities)

        st.markdown("---")
        self.render_centrality_analysis(G, centrality_metrics)

        st.markdown("---")
        self.render_multi_layer_network(G, multi_layer_data)

        # Summary insights
        self._render_summary_insights(G, communities, centrality_metrics, multi_layer_data)

    def _render_summary_insights(self, G: nx.Graph, communities: Dict,
                                centrality_metrics: Dict, multi_layer_data: Dict):
        """Render summary insights section."""
        st.markdown("---")
        st.subheader("📊 Insights da Análise de Rede")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Estrutura da Rede:**")
            density = nx.density(G)
            if density > 0.3:
                st.success(f"Rede altamente conectada (densidade: {density:.3f})")
            elif density > 0.1:
                st.warning(f"Rede moderadamente conectada (densidade: {density:.3f})")
            else:
                st.info(f"Rede esparsamente conectada (densidade: {density:.3f})")

            if communities and 'modularity' in communities:
                modularity = communities['modularity']
                if modularity > 0.3:
                    st.success(f"Comunidades bem definidas (modularidade: {modularity:.3f})")
                else:
                    st.info(f"Comunidades fracas (modularidade: {modularity:.3f})")

        with col2:
            st.write("**Padrões de Influência:**")
            if centrality_metrics:
                # Find most influential nodes
                composite_scores = {}
                for node, metrics in centrality_metrics.items():
                    composite = (metrics['degree'] * 0.3 + metrics['betweenness'] * 0.3 +
                               metrics['closeness'] * 0.2 + metrics['eigenvector'] * 0.2)
                    composite_scores[node] = composite

                top_node = max(composite_scores.items(), key=lambda x: x[1])
                st.info(f"Nó mais influente: {top_node[0]} (score: {top_node[1]:.3f})")

                # Check for centralization
                degree_values = [metrics['degree'] for metrics in centrality_metrics.values()]
                degree_std = np.std(degree_values)
                if degree_std > 0.2:
                    st.warning("Rede altamente centralizada")
                else:
                    st.info("Rede descentralizada")


def main():
    """Main function for standalone testing."""
    st.set_page_config(
        page_title="Stage 14: Network Analysis",
        page_icon="🕸️",
        layout="wide"
    )

    # Initialize dashboard
    dashboard = NetworkAnalysisDashboard()

    # Sample data for standalone testing only (not used in production dashboard)
    sample_data = {
        'user_id': ['user1', 'user2', 'user3', 'user4', 'user5'] * 20,
        'channel': ['canal_a', 'canal_b', 'canal_c'] * 33 + ['canal_a'],
        'sender_frequency': np.random.randint(1, 10, 100),
        'is_frequent_sender': np.random.choice([True, False], 100),
        'shared_url_frequency': np.random.randint(0, 5, 100),
        'temporal_coordination': np.random.uniform(0, 1, 100),
        'sentiment_polarity': np.random.uniform(-1, 1, 100),
        'dominant_topic': np.random.randint(0, 5, 100),
        'hour': np.random.randint(0, 24, 100)
    }

    df_sample = pd.DataFrame(sample_data)

    # Render dashboard
    dashboard.render_dashboard(df_sample)


if __name__ == "__main__":
    main()