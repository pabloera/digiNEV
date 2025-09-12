"""
Página de Modelagem de Tópicos do Dashboard digiNEV
Análise de temas semânticos e clustering de conteúdo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional

def render_topics_page(data_loader):
    """Renderiza a página de modelagem de tópicos"""
    
    st.markdown('<div class="page-header"><h2>🎨 Modelagem de Tópicos</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados de tópicos e clustering
    topic_data = data_loader.load_data('topic_modeling')
    clustering_data = data_loader.load_data('clustering_results')
    
    if topic_data is None:
        st.warning("📊 Dados de modelagem de tópicos não disponíveis")
        st.info("Execute o pipeline principal para gerar a modelagem de tópicos (Stage 09)")
        return
    
    # Filtros interativos
    st.subheader("🔍 Filtros de Análise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtro por tópico
        if 'topic' in topic_data.columns:
            topics = ['Todos'] + sorted(list(topic_data['topic'].unique()))
            selected_topic = st.selectbox("Tópico", topics)
        else:
            selected_topic = 'Todos'
    
    with col2:
        # Filtro por relevância do tópico
        if 'topic_relevance' in topic_data.columns:
            min_relevance = st.slider(
                "Relevância Mínima",
                min_value=float(topic_data['topic_relevance'].min()),
                max_value=float(topic_data['topic_relevance'].max()),
                value=float(topic_data['topic_relevance'].min()),
                step=0.01
            )
        else:
            min_relevance = 0.0
    
    with col3:
        # Filtro por cluster
        if clustering_data is not None and 'cluster' in clustering_data.columns:
            clusters = ['Todos'] + sorted(list(clustering_data['cluster'].unique()))
            selected_cluster = st.selectbox("Cluster", clusters)
        else:
            selected_cluster = 'Todos'
    
    # Aplicar filtros
    filtered_data = topic_data.copy()
    
    if selected_topic != 'Todos' and 'topic' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['topic'] == selected_topic]
    
    if 'topic_relevance' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['topic_relevance'] >= min_relevance]
    
    # Métricas principais
    st.subheader("📊 Métricas de Tópicos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_messages = len(filtered_data)
        st.metric("Total de Mensagens", f"{total_messages:,}")
    
    with col2:
        if 'topic' in filtered_data.columns:
            unique_topics = filtered_data['topic'].nunique()
            st.metric("Tópicos Identificados", unique_topics)
        else:
            st.metric("Tópicos", "N/A")
    
    with col3:
        if 'topic_relevance' in filtered_data.columns:
            avg_relevance = filtered_data['topic_relevance'].mean()
            st.metric("Relevância Média", f"{avg_relevance:.3f}")
        else:
            st.metric("Relevância", "N/A")
    
    with col4:
        if clustering_data is not None and 'cluster' in clustering_data.columns:
            unique_clusters = clustering_data['cluster'].nunique()
            st.metric("Clusters Semânticos", unique_clusters)
        else:
            st.metric("Clusters", "N/A")
    
    # Visualizações principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição de Tópicos")
        
        if 'topic' in filtered_data.columns:
            topic_counts = filtered_data['topic'].value_counts().head(10)
            
            fig_topics = go.Figure()
            fig_topics.add_trace(go.Bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                marker_color='lightblue',
                text=topic_counts.values,
                textposition='auto'
            ))
            
            fig_topics.update_layout(
                title="Top 10 Tópicos por Frequência",
                xaxis_title="Número de Mensagens",
                yaxis_title="Tópico",
                height=400
            )
            
            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("Dados de tópicos não disponíveis")
    
    with col2:
        st.subheader("🎯 Relevância dos Tópicos")
        
        if 'topic_relevance' in filtered_data.columns and 'topic' in filtered_data.columns:
            # Boxplot de relevância por tópico
            top_topics = filtered_data['topic'].value_counts().head(8).index
            plot_data = filtered_data[filtered_data['topic'].isin(top_topics)]
            
            fig_relevance = px.box(
                plot_data,
                x='topic',
                y='topic_relevance',
                title="Distribuição de Relevância por Tópico"
            )
            
            fig_relevance.update_layout(
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_relevance, use_container_width=True)
        else:
            st.info("Dados de relevância não disponíveis")
    
    # Visualização de clustering semântico
    if clustering_data is not None:
        st.subheader("🔬 Análise de Clustering Semântico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'cluster' in clustering_data.columns:
                cluster_counts = clustering_data['cluster'].value_counts()
                
                fig_clusters = go.Figure()
                fig_clusters.add_trace(go.Pie(
                    labels=[f"Cluster {c}" for c in cluster_counts.index],
                    values=cluster_counts.values,
                    hole=0.3,
                    textinfo='label+percent'
                ))
                
                fig_clusters.update_layout(
                    title="Distribuição de Clusters",
                    height=350
                )
                
                st.plotly_chart(fig_clusters, use_container_width=True)
            else:
                st.info("Dados de cluster não disponíveis")
        
        with col2:
            # Scatter plot de clustering (se tiver coordenadas 2D)
            if all(col in clustering_data.columns for col in ['embedding_2d_x', 'embedding_2d_y', 'cluster']):
                # Amostra para performance
                sample_data = clustering_data.sample(min(1000, len(clustering_data)))
                
                fig_scatter = px.scatter(
                    sample_data,
                    x='embedding_2d_x',
                    y='embedding_2d_y',
                    color='cluster',
                    title="Visualização 2D dos Clusters",
                    labels={'embedding_2d_x': 'Dimensão 1', 'embedding_2d_y': 'Dimensão 2'}
                )
                
                fig_scatter.update_layout(height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Coordenadas de embedding não disponíveis")
    
    # Análise de termos-chave por tópico
    st.subheader("🔑 Termos-Chave por Tópico")
    
    if 'topic' in filtered_data.columns:
        # Seletor de tópico específico
        available_topics = sorted(filtered_data['topic'].unique())
        selected_topic_detail = st.selectbox(
            "Selecionar tópico para análise detalhada:",
            available_topics,
            key="topic_detail"
        )
        
        if selected_topic_detail:
            topic_subset = filtered_data[filtered_data['topic'] == selected_topic_detail]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Mensagens no tópico**: {len(topic_subset):,}")
                
                if 'topic_relevance' in topic_subset.columns:
                    st.write(f"**Relevância média**: {topic_subset['topic_relevance'].mean():.3f}")
                
                # Palavras-chave do tópico (se disponível)
                if 'topic_keywords' in topic_subset.columns:
                    keywords = topic_subset['topic_keywords'].iloc[0] if not topic_subset.empty else "N/A"
                    st.write(f"**Palavras-chave**: {keywords}")
            
            with col2:
                # Exemplos de mensagens do tópico
                if 'text' in topic_subset.columns and len(topic_subset) > 0:
                    st.write("**Exemplos de mensagens:**")
                    
                    # Pegar top 3 mensagens com maior relevância
                    if 'topic_relevance' in topic_subset.columns:
                        examples = topic_subset.nlargest(3, 'topic_relevance')
                    else:
                        examples = topic_subset.head(3)
                    
                    for idx, row in examples.iterrows():
                        text_preview = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
                        relevance = f" (Relevância: {row['topic_relevance']:.3f})" if 'topic_relevance' in row else ""
                        st.write(f"• {text_preview}{relevance}")
    
    # Análise temporal dos tópicos
    if 'date' in filtered_data.columns or 'timestamp' in filtered_data.columns:
        st.subheader("📈 Evolução Temporal dos Tópicos")
        
        date_col = 'date' if 'date' in filtered_data.columns else 'timestamp'
        
        try:
            filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
            
            # Seletor de período e top tópicos
            col1, col2 = st.columns(2)
            
            with col1:
                period = st.selectbox("Período", ["Diário", "Semanal", "Mensal"], index=1, key="topic_period")
            
            with col2:
                top_n_topics = st.selectbox("Top N Tópicos", [3, 5, 8, 10], index=1)
            
            if period == "Diário":
                freq = 'D'
            elif period == "Semanal":
                freq = 'W'
            else:
                freq = 'M'
            
            # Pegar top tópicos
            top_topics = filtered_data['topic'].value_counts().head(top_n_topics).index
            temporal_data = filtered_data[filtered_data['topic'].isin(top_topics)]
            
            # Agrupar por período e tópico
            temporal_grouped = temporal_data.groupby([
                temporal_data[date_col].dt.to_period(freq),
                'topic'
            ]).size().reset_index(name='count')
            
            temporal_grouped[date_col] = temporal_grouped[date_col].dt.to_timestamp()
            
            fig_temporal = px.line(
                temporal_grouped,
                x=date_col,
                y='count',
                color='topic',
                title=f"Evolução {period} dos Top {top_n_topics} Tópicos",
                labels={'count': 'Número de Mensagens', date_col: 'Período'}
            )
            
            fig_temporal.update_layout(height=400)
            st.plotly_chart(fig_temporal, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Erro na análise temporal: {e}")
    
    # Matriz de similaridade entre tópicos
    if 'topic' in filtered_data.columns and len(filtered_data['topic'].unique()) > 1:
        st.subheader("🔗 Similaridade entre Tópicos")
        
        # Calcular co-ocorrência de palavras entre tópicos (aproximação)
        if 'text' in filtered_data.columns:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Agrupar textos por tópico
                topic_texts = filtered_data.groupby('topic')['text'].apply(lambda x: ' '.join(x)).to_dict()
                
                if len(topic_texts) > 1:
                    topics_list = list(topic_texts.keys())
                    texts_list = [topic_texts[topic] for topic in topics_list]
                    
                    # Calcular TF-IDF e similaridade
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(texts_list)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Criar heatmap
                    fig_similarity = go.Figure(data=go.Heatmap(
                        z=similarity_matrix,
                        x=topics_list,
                        y=topics_list,
                        colorscale='Blues',
                        texttemplate="%{z:.2f}",
                        textfont={"size": 10}
                    ))
                    
                    fig_similarity.update_layout(
                        title="Matriz de Similaridade entre Tópicos",
                        height=400
                    )
                    
                    st.plotly_chart(fig_similarity, use_container_width=True)
                else:
                    st.info("Insuficientes tópicos para análise de similaridade")
                    
            except ImportError:
                st.info("Biblioteca sklearn não disponível para análise de similaridade")
            except Exception as e:
                st.warning(f"Erro na análise de similaridade: {e}")
    
    # Estatísticas detalhadas
    st.subheader("📋 Estatísticas Detalhadas")
    
    if 'topic' in filtered_data.columns:
        # Tabela de estatísticas por tópico
        topic_stats = []
        
        for topic in filtered_data['topic'].unique():
            topic_subset = filtered_data[filtered_data['topic'] == topic]
            
            stats = {
                'Tópico': topic,
                'Mensagens': len(topic_subset),
                'Percentual': f"{(len(topic_subset) / len(filtered_data)) * 100:.1f}%"
            }
            
            if 'topic_relevance' in topic_subset.columns:
                stats['Relevância Média'] = f"{topic_subset['topic_relevance'].mean():.3f}"
                stats['Relevância Max'] = f"{topic_subset['topic_relevance'].max():.3f}"
            
            if 'text' in topic_subset.columns:
                avg_length = topic_subset['text'].str.len().mean()
                stats['Tamanho Médio'] = f"{avg_length:.0f} chars"
            
            topic_stats.append(stats)
        
        stats_df = pd.DataFrame(topic_stats)
        stats_df = stats_df.sort_values('Mensagens', ascending=False)
        
        st.dataframe(stats_df, use_container_width=True)
    
    # Exportação e controles
    st.subheader("🛠️ Ferramentas de Análise")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Exportar Tópicos", key="export_topics"):
            export_path = data_loader.export_data('topic_modeling', 'csv')
            if export_path:
                st.success(f"Dados exportados: {export_path.name}")
    
    with col2:
        if st.button("📊 Exportar Clusters", key="export_clusters"):
            if clustering_data is not None:
                export_path = data_loader.export_data('clustering_results', 'csv')
                if export_path:
                    st.success(f"Clusters exportados: {export_path.name}")
            else:
                st.warning("Dados de clustering não disponíveis")
    
    with col3:
        if st.button("🔄 Atualizar Cache", key="refresh_topics_cache"):
            data_loader.clear_cache()
            st.rerun()
    
    with col4:
        if st.button("🔍 Buscar Tópicos", key="search_topics"):
            search_term = st.text_input("Termo de busca:", key="topic_search_input")
            if search_term:
                results = data_loader.search_data(search_term, ['topic_modeling'])
                if results:
                    st.write(f"Encontrados {len(results['topic_modeling'])} resultados")
                else:
                    st.write("Nenhum resultado encontrado")
    
    # Insights automáticos
    st.subheader("💡 Insights da Modelagem de Tópicos")
    
    insights = []
    
    if 'topic' in filtered_data.columns:
        most_common_topic = filtered_data['topic'].value_counts().index[0]
        topic_percentage = (filtered_data['topic'].value_counts().iloc[0] / len(filtered_data)) * 100
        insights.append(f"• **Tópico Dominante**: {most_common_topic} ({topic_percentage:.1f}% das mensagens)")
        
        num_topics = filtered_data['topic'].nunique()
        coverage = (filtered_data['topic'].value_counts().head(5).sum() / len(filtered_data)) * 100
        insights.append(f"• **Concentração**: Top 5 tópicos cobrem {coverage:.1f}% do conteúdo ({num_topics} tópicos totais)")
    
    if 'topic_relevance' in filtered_data.columns:
        high_relevance_pct = (filtered_data['topic_relevance'] > 0.7).mean() * 100
        insights.append(f"• **Alta Relevância**: {high_relevance_pct:.1f}% das mensagens têm relevância > 0.7")
    
    if clustering_data is not None and 'cluster' in clustering_data.columns:
        num_clusters = clustering_data['cluster'].nunique()
        avg_cluster_size = len(clustering_data) / num_clusters
        insights.append(f"• **Clustering**: {num_clusters} clusters identificados (média: {avg_cluster_size:.0f} mensagens/cluster)")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Execute modelagem de tópicos completa para gerar insights automáticos")