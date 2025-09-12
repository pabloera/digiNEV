"""
Página de Análise de Rede do Dashboard digiNEV
Análise de interações, propagação e estrutura de rede
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional

def render_network_page(data_loader):
    """Renderiza a página de análise de rede"""
    
    st.markdown('<div class="page-header"><h2>📊 Análise de Rede</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados de rede
    network_data = data_loader.load_data('network_metrics')
    
    if network_data is None:
        st.warning("📊 Dados de análise de rede não disponíveis")
        st.info("Execute o pipeline principal para gerar a análise de rede (Stage 15)")
        return
    
    # Filtros interativos
    st.subheader("🔍 Filtros de Análise de Rede")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtro por tipo de nó
        if 'node_type' in network_data.columns:
            node_types = ['Todos'] + list(network_data['node_type'].unique())
            selected_node_type = st.selectbox("Tipo de Nó", node_types)
        else:
            selected_node_type = 'Todos'
    
    with col2:
        # Filtro por centralidade
        if 'centrality' in network_data.columns:
            min_centrality = st.slider(
                "Centralidade Mínima",
                min_value=float(network_data['centrality'].min()),
                max_value=float(network_data['centrality'].max()),
                value=float(network_data['centrality'].min()),
                step=0.001
            )
        else:
            min_centrality = 0.0
    
    with col3:
        # Filtro por comunidade
        if 'community' in network_data.columns:
            communities = ['Todas'] + sorted(list(network_data['community'].unique()))
            selected_community = st.selectbox("Comunidade", communities)
        else:
            selected_community = 'Todas'
    
    # Aplicar filtros
    filtered_data = network_data.copy()
    
    if selected_node_type != 'Todos' and 'node_type' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['node_type'] == selected_node_type]
    
    if 'centrality' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['centrality'] >= min_centrality]
    
    if selected_community != 'Todas' and 'community' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['community'] == selected_community]
    
    # Métricas principais
    st.subheader("📊 Métricas de Rede")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_nodes = len(filtered_data)
        st.metric("Total de Nós", f"{total_nodes:,}")
    
    with col2:
        if 'degree' in filtered_data.columns:
            avg_degree = filtered_data['degree'].mean()
            st.metric("Grau Médio", f"{avg_degree:.2f}")
        else:
            st.metric("Grau Médio", "N/A")
    
    with col3:
        if 'community' in filtered_data.columns:
            num_communities = filtered_data['community'].nunique()
            st.metric("Comunidades", num_communities)
        else:
            st.metric("Comunidades", "N/A")
    
    with col4:
        if 'centrality' in filtered_data.columns:
            max_centrality = filtered_data['centrality'].max()
            st.metric("Centralidade Máxima", f"{max_centrality:.3f}")
        else:
            st.metric("Centralidade", "N/A")
    
    # Visualizações principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 Distribuição de Graus")
        
        if 'degree' in filtered_data.columns:
            fig_degree = px.histogram(
                filtered_data,
                x='degree',
                nbins=20,
                title="Distribuição de Graus dos Nós",
                labels={'degree': 'Grau do Nó', 'count': 'Frequência'}
            )
            
            # Adicionar linha da média
            mean_degree = filtered_data['degree'].mean()
            fig_degree.add_vline(
                x=mean_degree,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {mean_degree:.1f}"
            )
            
            fig_degree.update_layout(height=400)
            st.plotly_chart(fig_degree, use_container_width=True)
        else:
            st.info("Dados de grau não disponíveis")
    
    with col2:
        st.subheader("🎯 Análise de Centralidade")
        
        if 'centrality' in filtered_data.columns:
            # Top 10 nós por centralidade
            top_nodes = filtered_data.nlargest(10, 'centrality')
            
            fig_centrality = go.Figure()
            fig_centrality.add_trace(go.Bar(
                x=top_nodes['centrality'],
                y=top_nodes.index if 'node_id' not in top_nodes.columns else top_nodes['node_id'],
                orientation='h',
                marker_color='lightcoral',
                text=top_nodes['centrality'].round(3),
                textposition='auto'
            ))
            
            fig_centrality.update_layout(
                title="Top 10 Nós por Centralidade",
                xaxis_title="Centralidade",
                yaxis_title="Nó",
                height=400
            )
            
            st.plotly_chart(fig_centrality, use_container_width=True)
        else:
            st.info("Dados de centralidade não disponíveis")
    
    # Análise de comunidades
    if 'community' in filtered_data.columns:
        st.subheader("🏘️ Análise de Comunidades")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de tamanhos de comunidades
            community_sizes = filtered_data['community'].value_counts()
            
            fig_communities = go.Figure()
            fig_communities.add_trace(go.Pie(
                labels=[f"Comunidade {c}" for c in community_sizes.index],
                values=community_sizes.values,
                hole=0.3,
                textinfo='label+percent'
            ))
            
            fig_communities.update_layout(
                title="Distribuição de Comunidades",
                height=400
            )
            
            st.plotly_chart(fig_communities, use_container_width=True)
        
        with col2:
            # Métricas por comunidade
            community_stats = filtered_data.groupby('community').agg({
                'degree': ['mean', 'max'],
                'centrality': ['mean', 'max'] if 'centrality' in filtered_data.columns else ['count']
            }).round(3)
            
            community_stats.columns = ['Grau Médio', 'Grau Máximo', 'Centralidade Média', 'Centralidade Máxima'] if 'centrality' in filtered_data.columns else ['Grau Médio', 'Grau Máximo', 'Contagem', 'Contagem']
            
            st.write("**Estatísticas por Comunidade:**")
            st.dataframe(community_stats, use_container_width=True)
    
    # Visualização de rede interativa
    st.subheader("🕸️ Visualização da Rede")
    
    # Configurações de visualização
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layout_type = st.selectbox("Tipo de Layout", ["spring", "circular", "hierarchical"], key="network_layout")
    
    with col2:
        max_nodes = st.slider("Máximo de Nós", 10, 100, 50, key="max_nodes_viz")
    
    with col3:
        node_size_metric = st.selectbox("Tamanho do Nó", ["degree", "centrality"] if 'centrality' in filtered_data.columns else ["degree"], key="node_size")
    
    # Criar visualização de rede
    if len(filtered_data) > 0:
        # Pegar amostra dos nós mais importantes
        if node_size_metric in filtered_data.columns:
            top_nodes = filtered_data.nlargest(max_nodes, node_size_metric)
        else:
            top_nodes = filtered_data.head(max_nodes)
        
        # Simular conexões (em produção viria dos dados reais de edge)
        network_viz = create_network_visualization(top_nodes, layout_type, node_size_metric)
        st.plotly_chart(network_viz, use_container_width=True)
    else:
        st.info("Nenhum nó disponível para visualização com os filtros atuais")
    
    # Análise de propagação
    st.subheader("📡 Análise de Propagação")
    
    if 'influence_score' in filtered_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de influência
            fig_influence = px.histogram(
                filtered_data,
                x='influence_score',
                nbins=20,
                title="Distribuição de Scores de Influência",
                labels={'influence_score': 'Score de Influência', 'count': 'Frequência'}
            )
            
            fig_influence.update_layout(height=350)
            st.plotly_chart(fig_influence, use_container_width=True)
        
        with col2:
            # Correlação influência vs centralidade
            if 'centrality' in filtered_data.columns:
                fig_correlation = px.scatter(
                    filtered_data.sample(min(500, len(filtered_data))),
                    x='centrality',
                    y='influence_score',
                    color='community' if 'community' in filtered_data.columns else None,
                    title="Centralidade vs Influência",
                    labels={
                        'centrality': 'Centralidade',
                        'influence_score': 'Score de Influência'
                    },
                    opacity=0.6
                )
                
                fig_correlation.update_layout(height=350)
                st.plotly_chart(fig_correlation, use_container_width=True)
            else:
                st.info("Dados de centralidade não disponíveis para correlação")
    else:
        st.info("Dados de influência não disponíveis")
    
    # Análise temporal da rede
    if 'timestamp' in filtered_data.columns or 'date' in filtered_data.columns:
        st.subheader("⏱️ Evolução Temporal da Rede")
        
        date_col = 'date' if 'date' in filtered_data.columns else 'timestamp'
        
        try:
            filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
            
            # Métricas ao longo do tempo
            period = st.selectbox("Período de Análise", ["Diário", "Semanal", "Mensal"], index=1, key="network_temporal_period")
            
            if period == "Diário":
                freq = 'D'
            elif period == "Semanal":
                freq = 'W'
            else:
                freq = 'M'
            
            # Agrupar métricas por período
            temporal_metrics = filtered_data.groupby(filtered_data[date_col].dt.to_period(freq)).agg({
                'degree': 'mean',
                'centrality': 'mean' if 'centrality' in filtered_data.columns else 'count',
                'community': 'nunique'
            }).round(3)
            
            temporal_metrics.columns = ['Grau Médio', 'Centralidade Média', 'Número de Comunidades']
            temporal_metrics = temporal_metrics.reset_index()
            temporal_metrics[date_col] = temporal_metrics[date_col].dt.to_timestamp()
            
            # Gráfico temporal
            fig_temporal = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Métricas de Conectividade', 'Estrutura da Rede'),
                vertical_spacing=0.15
            )
            
            # Grau médio
            fig_temporal.add_trace(
                go.Scatter(
                    x=temporal_metrics[date_col],
                    y=temporal_metrics['Grau Médio'],
                    mode='lines+markers',
                    name='Grau Médio',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Centralidade média
            fig_temporal.add_trace(
                go.Scatter(
                    x=temporal_metrics[date_col],
                    y=temporal_metrics['Centralidade Média'],
                    mode='lines+markers',
                    name='Centralidade Média',
                    line=dict(color='red'),
                    yaxis='y2'
                ),
                row=1, col=1
            )
            
            # Número de comunidades
            fig_temporal.add_trace(
                go.Scatter(
                    x=temporal_metrics[date_col],
                    y=temporal_metrics['Número de Comunidades'],
                    mode='lines+markers',
                    name='Comunidades',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            fig_temporal.update_layout(
                height=600,
                title=f"Evolução {period} das Métricas de Rede"
            )
            
            st.plotly_chart(fig_temporal, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Erro na análise temporal: {e}")
    
    # Tabela de nós importantes
    st.subheader("👑 Nós Mais Importantes")
    
    # Seletor de métrica para ranking
    ranking_metric = st.selectbox(
        "Rankear por:",
        ['degree', 'centrality', 'influence_score'] if all(col in filtered_data.columns for col in ['centrality', 'influence_score']) else ['degree'],
        key="ranking_metric"
    )
    
    if ranking_metric in filtered_data.columns:
        top_important = filtered_data.nlargest(10, ranking_metric)
        
        # Preparar colunas para exibição
        display_columns = []
        for col in ['node_id', 'degree', 'centrality', 'community', 'influence_score', 'node_type']:
            if col in top_important.columns:
                display_columns.append(col)
        
        if display_columns:
            st.dataframe(
                top_important[display_columns],
                use_container_width=True
            )
    
    # Ferramentas de análise
    st.subheader("🛠️ Ferramentas de Análise de Rede")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Exportar Métricas", key="export_network"):
            export_path = data_loader.export_data('network_metrics', 'csv')
            if export_path:
                st.success(f"Métricas exportadas: {export_path.name}")
    
    with col2:
        if st.button("🔍 Detectar Comunidades", key="detect_communities"):
            if 'community' in filtered_data.columns:
                num_communities = filtered_data['community'].nunique()
                avg_size = len(filtered_data) / num_communities
                st.info(f"🏘️ {num_communities} comunidades detectadas (tamanho médio: {avg_size:.1f} nós)")
            else:
                st.warning("Dados de comunidade não disponíveis")
    
    with col3:
        if st.button("📊 Calcular Centralidade", key="calculate_centrality"):
            if 'centrality' in filtered_data.columns:
                avg_centrality = filtered_data['centrality'].mean()
                max_centrality_node = filtered_data.loc[filtered_data['centrality'].idxmax()]
                st.info(f"📈 Centralidade média: {avg_centrality:.3f}")
                st.info(f"🎯 Nó mais central: {max_centrality_node.name}")
            else:
                st.warning("Dados de centralidade não disponíveis")
    
    with col4:
        if st.button("🔄 Atualizar Cache", key="refresh_network_cache"):
            data_loader.clear_cache()
            st.rerun()
    
    # Insights da análise de rede
    st.subheader("💡 Insights da Análise de Rede")
    
    insights = []
    
    if 'degree' in filtered_data.columns:
        avg_degree = filtered_data['degree'].mean()
        max_degree = filtered_data['degree'].max()
        insights.append(f"• **Conectividade**: Grau médio de {avg_degree:.1f}, máximo de {max_degree}")
    
    if 'community' in filtered_data.columns:
        num_communities = filtered_data['community'].nunique()
        largest_community_size = filtered_data['community'].value_counts().iloc[0]
        insights.append(f"• **Estrutura**: {num_communities} comunidades identificadas, maior com {largest_community_size} nós")
    
    if 'centrality' in filtered_data.columns:
        high_centrality_nodes = (filtered_data['centrality'] > filtered_data['centrality'].quantile(0.9)).sum()
        insights.append(f"• **Influência**: {high_centrality_nodes} nós altamente centrais (top 10%)")
    
    if 'influence_score' in filtered_data.columns:
        avg_influence = filtered_data['influence_score'].mean()
        insights.append(f"• **Propagação**: Score médio de influência de {avg_influence:.3f}")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Execute análise de rede completa para gerar insights automáticos")

def create_network_visualization(nodes_data: pd.DataFrame, layout_type: str, size_metric: str) -> go.Figure:
    """Cria visualização interativa da rede"""
    
    if len(nodes_data) == 0:
        return go.Figure()
    
    # Gerar posições dos nós baseado no layout
    if layout_type == "circular":
        angles = np.linspace(0, 2*np.pi, len(nodes_data), endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
    elif layout_type == "hierarchical":
        # Layout hierárquico simples por grau
        if 'degree' in nodes_data.columns:
            sorted_nodes = nodes_data.sort_values('degree', ascending=False)
            levels = np.ceil(np.arange(len(sorted_nodes)) / 10)  # 10 nós por nível
            x_pos = np.arange(len(sorted_nodes)) % 10
            y_pos = -levels  # Níveis crescem para baixo
        else:
            x_pos = np.arange(len(nodes_data))
            y_pos = np.zeros(len(nodes_data))
    else:  # spring layout simulado
        np.random.seed(42)  # Para reproducibilidade
        x_pos = np.random.uniform(-1, 1, len(nodes_data))
        y_pos = np.random.uniform(-1, 1, len(nodes_data))
    
    # Tamanhos dos nós
    if size_metric in nodes_data.columns:
        sizes = nodes_data[size_metric]
        # Normalizar para range 10-50
        sizes_norm = 10 + 40 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
    else:
        sizes_norm = [20] * len(nodes_data)
    
    # Cores dos nós (por comunidade se disponível)
    if 'community' in nodes_data.columns:
        colors = nodes_data['community']
        color_scale = 'Viridis'
    else:
        colors = ['blue'] * len(nodes_data)
        color_scale = None
    
    # Texto dos nós
    if 'node_id' in nodes_data.columns:
        hover_text = [f"Nó: {nid}<br>Grau: {deg}<br>Centralidade: {cent:.3f}" 
                     for nid, deg, cent in zip(
                         nodes_data['node_id'], 
                         nodes_data['degree'], 
                         nodes_data.get('centrality', [0]*len(nodes_data))
                     )]
    else:
        hover_text = [f"Nó: {i}<br>Grau: {deg}" 
                     for i, deg in enumerate(nodes_data['degree'])]
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar edges (conexões simuladas entre nós próximos)
    edge_x = []
    edge_y = []
    
    for i in range(len(nodes_data)):
        for j in range(i+1, min(i+3, len(nodes_data))):  # Conectar com próximos 2 nós
            edge_x.extend([x_pos[i], x_pos[j], None])
            edge_y.extend([y_pos[i], y_pos[j], None])
    
    # Adicionar edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='lightgray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Adicionar nós
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers',
        marker=dict(
            size=sizes_norm,
            color=colors,
            colorscale=color_scale,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Layout da figura
    fig.update_layout(
        title=f"Visualização da Rede ({layout_type.title()} Layout)",
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig