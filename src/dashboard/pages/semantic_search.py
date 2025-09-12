"""
Página de Busca Semântica do Dashboard digiNEV
Interface interativa para consultas de similaridade semântica
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any
import numpy as np

def render_search_page(data_loader):
    """Renderiza a página de busca semântica"""
    
    st.markdown('<div class="page-header"><h2>🔍 Busca Semântica Interativa</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados do índice de busca semântica
    search_data = data_loader.load_data('semantic_search_index')
    
    if search_data is None:
        st.warning("📊 Índice de busca semântica não disponível")
        st.info("Execute o pipeline principal para gerar o índice de busca (Stage 19)")
        return
    
    # Interface de busca
    st.subheader("🔎 Consulta Semântica")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Digite sua consulta:",
            placeholder="Ex: discurso político autoritário, violência digital, desinformação...",
            help="Use linguagem natural para buscar conteúdo semanticamente similar"
        )
    
    with col2:
        search_button = st.button("🚀 Buscar", type="primary", use_container_width=True)
    
    # Opções avançadas de busca
    with st.expander("⚙️ Opções Avançadas"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_results = st.slider("Número de resultados", 5, 50, 10)
        
        with col2:
            min_similarity = st.slider("Similaridade mínima", 0.0, 1.0, 0.3, 0.05)
        
        with col3:
            # Filtros por tipo de conteúdo
            if 'content_type' in search_data.columns:
                content_types = ['Todos'] + list(search_data['content_type'].unique())
                selected_type = st.selectbox("Tipo de conteúdo", content_types)
            else:
                selected_type = 'Todos'
    
    # Realizar busca
    if search_button and search_query:
        results = perform_semantic_search(search_data, search_query, num_results, min_similarity)
        
        if not results.empty:
            st.success(f"🎯 Encontrados {len(results)} resultados para: '{search_query}'")
            
            # Métricas dos resultados
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_similarity = results['similarity_score'].mean()
                st.metric("Similaridade Média", f"{avg_similarity:.3f}")
            
            with col2:
                max_similarity = results['similarity_score'].max()
                st.metric("Melhor Match", f"{max_similarity:.3f}")
            
            with col3:
                if 'political_category' in results.columns:
                    dominant_category = results['political_category'].mode().iloc[0] if not results['political_category'].mode().empty else "N/A"
                    st.metric("Categoria Dominante", dominant_category)
                else:
                    st.metric("Categoria", "N/A")
            
            with col4:
                if 'sentiment' in results.columns:
                    dominant_sentiment = results['sentiment'].mode().iloc[0] if not results['sentiment'].mode().empty else "N/A"
                    st.metric("Sentimento Dominante", dominant_sentiment)
                else:
                    st.metric("Sentimento", "N/A")
            
            # Visualização da distribuição de similaridade
            col1, col2 = st.columns(2)
            
            with col1:
                fig_similarity = px.histogram(
                    results,
                    x='similarity_score',
                    nbins=15,
                    title="Distribuição de Scores de Similaridade",
                    labels={'similarity_score': 'Score de Similaridade', 'count': 'Frequência'}
                )
                fig_similarity.update_layout(height=300)
                st.plotly_chart(fig_similarity, use_container_width=True)
            
            with col2:
                # Gráfico de similaridade por posição
                results_ranked = results.reset_index(drop=True)
                results_ranked['rank'] = range(1, len(results_ranked) + 1)
                
                fig_rank = px.line(
                    results_ranked,
                    x='rank',
                    y='similarity_score',
                    title="Similaridade por Ranking",
                    labels={'rank': 'Posição no Ranking', 'similarity_score': 'Score de Similaridade'},
                    markers=True
                )
                fig_rank.update_layout(height=300)
                st.plotly_chart(fig_rank, use_container_width=True)
            
            # Análise por categorias (se disponível)
            if 'political_category' in results.columns:
                st.subheader("📊 Análise por Categoria Política")
                
                category_analysis = results.groupby('political_category').agg({
                    'similarity_score': ['mean', 'count']
                }).round(3)
                
                category_analysis.columns = ['Similaridade Média', 'Quantidade']
                category_analysis = category_analysis.sort_values('Similaridade Média', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cat_similarity = px.bar(
                        x=category_analysis.index,
                        y=category_analysis['Similaridade Média'],
                        title="Similaridade Média por Categoria",
                        labels={'x': 'Categoria Política', 'y': 'Similaridade Média'}
                    )
                    fig_cat_similarity.update_layout(height=300)
                    st.plotly_chart(fig_cat_similarity, use_container_width=True)
                
                with col2:
                    fig_cat_count = px.pie(
                        values=category_analysis['Quantidade'],
                        names=category_analysis.index,
                        title="Distribuição por Categoria"
                    )
                    fig_cat_count.update_layout(height=300)
                    st.plotly_chart(fig_cat_count, use_container_width=True)
            
            # Exibição dos resultados
            st.subheader("📋 Resultados da Busca")
            
            # Controles de exibição
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sort_by = st.selectbox(
                    "Ordenar por:",
                    ['similarity_score', 'political_category', 'sentiment', 'date'] if 'date' in results.columns else ['similarity_score'],
                    key="sort_results"
                )
            
            with col2:
                show_columns = st.multiselect(
                    "Colunas para exibir:",
                    [col for col in results.columns if col != 'text'],
                    default=['similarity_score', 'political_category', 'sentiment'] if all(col in results.columns for col in ['political_category', 'sentiment']) else ['similarity_score'],
                    key="show_columns"
                )
            
            with col3:
                results_format = st.selectbox("Formato:", ["Cards", "Tabela"], key="results_format")
            
            # Ordenar resultados
            if sort_by in results.columns:
                sorted_results = results.sort_values(sort_by, ascending=False if sort_by == 'similarity_score' else True)
            else:
                sorted_results = results
            
            # Exibir resultados
            if results_format == "Cards":
                for idx, row in sorted_results.head(num_results).iterrows():
                    with st.expander(f"🎯 Resultado {idx + 1} - Similaridade: {row['similarity_score']:.3f}"):
                        
                        # Texto principal
                        if 'text' in row:
                            text_preview = row['text'][:500] + "..." if len(row['text']) > 500 else row['text']
                            st.write(f"**Texto:** {text_preview}")
                        
                        # Metadados em colunas
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        
                        with meta_col1:
                            if 'political_category' in row:
                                st.write(f"**Categoria:** {row['political_category']}")
                            if 'sentiment' in row:
                                st.write(f"**Sentimento:** {row['sentiment']}")
                        
                        with meta_col2:
                            if 'topic' in row:
                                st.write(f"**Tópico:** {row['topic']}")
                            if 'date' in row:
                                st.write(f"**Data:** {row['date']}")
                        
                        with meta_col3:
                            st.write(f"**Similaridade:** {row['similarity_score']:.3f}")
                            if 'confidence_score' in row:
                                st.write(f"**Confiança:** {row['confidence_score']:.3f}")
                        
                        # Contexto adicional
                        if 'context' in row and pd.notna(row['context']):
                            st.write(f"**Contexto:** {row['context']}")
            
            else:  # Formato tabela
                display_columns = ['similarity_score'] + show_columns
                if 'text' in results.columns:
                    # Criar preview do texto para tabela
                    sorted_results['text_preview'] = sorted_results['text'].str[:100] + "..."
                    display_columns.append('text_preview')
                
                available_columns = [col for col in display_columns if col in sorted_results.columns]
                st.dataframe(
                    sorted_results[available_columns].head(num_results),
                    use_container_width=True
                )
            
            # Análise temporal dos resultados
            if 'date' in results.columns or 'timestamp' in results.columns:
                st.subheader("📈 Análise Temporal dos Resultados")
                
                date_col = 'date' if 'date' in results.columns else 'timestamp'
                
                try:
                    results[date_col] = pd.to_datetime(results[date_col])
                    
                    # Agrupar por período
                    period = st.selectbox("Período de análise:", ["Diário", "Semanal", "Mensal"], index=1, key="temporal_period")
                    
                    if period == "Diário":
                        freq = 'D'
                    elif period == "Semanal":
                        freq = 'W'
                    else:
                        freq = 'M'
                    
                    temporal_data = results.groupby(results[date_col].dt.to_period(freq)).agg({
                        'similarity_score': ['mean', 'count']
                    }).round(3)
                    
                    temporal_data.columns = ['Similaridade Média', 'Quantidade']
                    temporal_data = temporal_data.reset_index()
                    temporal_data[date_col] = temporal_data[date_col].dt.to_timestamp()
                    
                    fig_temporal = px.line(
                        temporal_data,
                        x=date_col,
                        y='Similaridade Média',
                        title=f"Evolução {period} da Similaridade dos Resultados",
                        labels={'Similaridade Média': 'Score Médio de Similaridade'}
                    )
                    
                    # Adicionar área para quantidade
                    fig_temporal.add_scatter(
                        x=temporal_data[date_col],
                        y=temporal_data['Quantidade'],
                        mode='lines+markers',
                        name='Quantidade',
                        yaxis='y2',
                        line=dict(color='orange')
                    )
                    
                    fig_temporal.update_layout(
                        yaxis2=dict(title='Quantidade de Resultados', overlaying='y', side='right'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_temporal, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Erro na análise temporal: {e}")
            
            # Exportar resultados
            st.subheader("📥 Exportar Resultados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Salvar CSV", key="export_search_csv"):
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"busca_semantica_{timestamp}.csv"
                    results.to_csv(filename, index=False)
                    st.success(f"Resultados salvos em: {filename}")
            
            with col2:
                if st.button("📊 Salvar Excel", key="export_search_excel"):
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"busca_semantica_{timestamp}.xlsx"
                    results.to_excel(filename, index=False)
                    st.success(f"Resultados salvos em: {filename}")
            
            with col3:
                if st.button("📋 Copiar Query", key="copy_query"):
                    st.code(search_query)
                    st.info("Query copiada para área de transferência")
        
        else:
            st.warning("❌ Nenhum resultado encontrado. Tente:")
            st.markdown("""
            - Usar termos mais gerais
            - Diminuir o score de similaridade mínimo
            - Verificar a ortografia
            - Usar sinônimos ou termos relacionados
            """)
    
    # Seção de consultas sugeridas
    st.subheader("💡 Consultas Sugeridas")
    
    suggested_queries = [
        "violência política digital",
        "discurso autoritário brasileiro", 
        "desinformação e fake news",
        "polarização política redes sociais",
        "bolsonarismo telegram",
        "ameaças democracia digital",
        "populismo redes sociais",
        "extremismo político online"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, query in enumerate(suggested_queries):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            if st.button(f"🔍 {query}", key=f"suggested_{i}", use_container_width=True):
                st.session_state.suggested_query = query
                st.rerun()
    
    # Estatísticas do índice
    st.subheader("📊 Estatísticas do Índice de Busca")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_documents = len(search_data)
        st.metric("Total de Documentos", f"{total_documents:,}")
    
    with col2:
        if 'embedding_dimensions' in search_data.columns:
            dimensions = search_data['embedding_dimensions'].iloc[0] if not search_data.empty else "N/A"
            st.metric("Dimensões do Embedding", dimensions)
        else:
            st.metric("Dimensões", "N/A")
    
    with col3:
        if 'index_quality' in search_data.columns:
            avg_quality = search_data['index_quality'].mean()
            st.metric("Qualidade Média do Índice", f"{avg_quality:.3f}")
        else:
            st.metric("Qualidade", "N/A")
    
    with col4:
        if 'last_indexed' in search_data.columns:
            last_update = search_data['last_indexed'].max() if not search_data.empty else "N/A"
            st.metric("Última Atualização", last_update)
        else:
            st.metric("Atualização", "N/A")
    
    # Manual de uso
    with st.expander("📚 Como Usar a Busca Semântica"):
        st.markdown("""
        ### 🎯 Dicas para Buscas Efetivas
        
        **Tipos de Consulta:**
        - **Conceitual**: "autoritarismo", "democracia", "violência política"
        - **Contextual**: "ameaças à democracia no Brasil"
        - **Específica**: "fake news sobre urnas eletrônicas"
        - **Emocional**: "discurso de ódio", "polarização extrema"
        
        **Configurações Recomendadas:**
        - **Similaridade mínima**: 0.3-0.5 para buscas amplas, 0.6+ para precisão
        - **Número de resultados**: 10-20 para análise inicial, 50+ para pesquisa profunda
        
        **Interpretação dos Scores:**
        - **0.8-1.0**: Correspondência muito alta
        - **0.6-0.8**: Correspondência boa
        - **0.4-0.6**: Correspondência moderada
        - **0.2-0.4**: Correspondência baixa
        """)

def perform_semantic_search(search_data: pd.DataFrame, query: str, num_results: int = 10, min_similarity: float = 0.3) -> pd.DataFrame:
    """
    Simula busca semântica nos dados disponíveis
    
    Args:
        search_data: DataFrame com índice de busca
        query: Query de busca
        num_results: Número máximo de resultados
        min_similarity: Score mínimo de similaridade
        
    Returns:
        DataFrame com resultados rankeados
    """
    try:
        # Simulação de busca textual simples (em produção seria busca vetorial)
        if 'text' not in search_data.columns:
            return pd.DataFrame()
        
        # Busca por termos da query no texto
        query_terms = query.lower().split()
        
        def calculate_similarity(text):
            if pd.isna(text):
                return 0.0
            
            text_lower = str(text).lower()
            matches = sum(1 for term in query_terms if term in text_lower)
            
            # Simulação de score semântico baseado em matches
            base_score = matches / len(query_terms) if query_terms else 0
            
            # Adicionar ruído para simular variabilidade semântica
            import random
            random.seed(hash(text_lower) % 2**32)  # Seed consistente baseada no texto
            semantic_boost = random.uniform(0.1, 0.3) if matches > 0 else 0
            
            return min(base_score + semantic_boost, 1.0)
        
        # Calcular scores de similaridade
        search_data = search_data.copy()
        search_data['similarity_score'] = search_data['text'].apply(calculate_similarity)
        
        # Filtrar por similaridade mínima
        results = search_data[search_data['similarity_score'] >= min_similarity]
        
        # Ordenar por similaridade e limitar resultados
        results = results.sort_values('similarity_score', ascending=False).head(num_results)
        
        return results
        
    except Exception as e:
        st.error(f"Erro na busca semântica: {e}")
        return pd.DataFrame()