"""
Página de Análise Política do Dashboard digiNEV
Análise de categorização e orientação política do discurso
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

def render_political_page(data_loader):
    """Renderiza a página de análise política"""
    
    st.markdown('<div class="page-header"><h2>🏛️ Análise Política</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados políticos
    political_data = data_loader.load_data('political_analysis')
    
    if political_data is None:
        st.warning("📊 Dados de análise política não disponíveis")
        st.info("Execute o pipeline principal para gerar a análise política (Stage 05)")
        return
    
    # Filtros interativos
    st.subheader("🔍 Filtros de Análise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtro por categoria política
        if 'political_category' in political_data.columns:
            categories = ['Todas'] + list(political_data['political_category'].unique())
            selected_category = st.selectbox("Categoria Política", categories)
        else:
            selected_category = 'Todas'
    
    with col2:
        # Filtro por orientação
        if 'political_orientation' in political_data.columns:
            orientations = ['Todas'] + list(political_data['political_orientation'].unique())
            selected_orientation = st.selectbox("Orientação Política", orientations)
        else:
            selected_orientation = 'Todas'
    
    with col3:
        # Filtro por período
        if 'date' in political_data.columns or 'timestamp' in political_data.columns:
            date_col = 'date' if 'date' in political_data.columns else 'timestamp'
            try:
                political_data[date_col] = pd.to_datetime(political_data[date_col])
                date_range = st.date_input(
                    "Período de Análise",
                    value=(political_data[date_col].min(), political_data[date_col].max()),
                    min_value=political_data[date_col].min(),
                    max_value=political_data[date_col].max()
                )
            except:
                date_range = None
        else:
            date_range = None
    
    # Aplicar filtros
    filtered_data = political_data.copy()
    
    if selected_category != 'Todas' and 'political_category' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['political_category'] == selected_category]
    
    if selected_orientation != 'Todas' and 'political_orientation' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['political_orientation'] == selected_orientation]
    
    if date_range and len(date_range) == 2:
        date_col = 'date' if 'date' in filtered_data.columns else 'timestamp'
        if date_col in filtered_data.columns:
            filtered_data = filtered_data[
                (filtered_data[date_col].dt.date >= date_range[0]) &
                (filtered_data[date_col].dt.date <= date_range[1])
            ]
    
    # Métricas principais
    st.subheader("📊 Métricas Políticas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_messages = len(filtered_data)
        st.metric("Total de Mensagens", f"{total_messages:,}")
    
    with col2:
        if 'political_category' in filtered_data.columns:
            categories_count = filtered_data['political_category'].nunique()
            st.metric("Categorias Identificadas", categories_count)
        else:
            st.metric("Categorias", "N/A")
    
    with col3:
        if 'political_intensity' in filtered_data.columns:
            avg_intensity = filtered_data['political_intensity'].mean()
            st.metric("Intensidade Média", f"{avg_intensity:.2f}")
        else:
            st.metric("Intensidade", "N/A")
    
    with col4:
        if 'confidence_score' in filtered_data.columns:
            avg_confidence = filtered_data['confidence_score'].mean()
            st.metric("Confiança Média", f"{avg_confidence:.2f}")
        else:
            st.metric("Confiança", "N/A")
    
    # Visualizações principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição por Categoria Política")
        
        if 'political_category' in filtered_data.columns:
            category_counts = filtered_data['political_category'].value_counts()
            
            fig_categories = go.Figure()
            fig_categories.add_trace(go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.3,
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Mensagens: %{value}<br>Porcentagem: %{percent}<extra></extra>'
            ))
            
            fig_categories.update_layout(
                title="Distribuição de Categorias Políticas",
                height=400
            )
            
            st.plotly_chart(fig_categories, use_container_width=True)
        else:
            st.info("Dados de categoria política não disponíveis")
    
    with col2:
        st.subheader("🎯 Orientação Política")
        
        if 'political_orientation' in filtered_data.columns:
            orientation_counts = filtered_data['political_orientation'].value_counts()
            
            fig_orientation = go.Figure()
            fig_orientation.add_trace(go.Bar(
                x=orientation_counts.index,
                y=orientation_counts.values,
                marker_color=['#ff4444', '#4444ff', '#44ff44', '#ffff44'][:len(orientation_counts)],
                text=orientation_counts.values,
                textposition='auto'
            ))
            
            fig_orientation.update_layout(
                title="Distribuição por Orientação Política",
                xaxis_title="Orientação",
                yaxis_title="Número de Mensagens",
                height=400
            )
            
            st.plotly_chart(fig_orientation, use_container_width=True)
        else:
            st.info("Dados de orientação política não disponíveis")
    
    # Análise temporal
    if date_range and 'date' in filtered_data.columns or 'timestamp' in filtered_data.columns:
        st.subheader("⏱️ Evolução Temporal do Discurso Político")
        
        date_col = 'date' if 'date' in filtered_data.columns else 'timestamp'
        
        # Agrupar por período
        period = st.selectbox("Período de Agrupamento", ["Diário", "Semanal", "Mensal"], index=1)
        
        if period == "Diário":
            freq = 'D'
        elif period == "Semanal":
            freq = 'W'
        else:
            freq = 'M'
        
        temporal_data = filtered_data.groupby([
            filtered_data[date_col].dt.to_period(freq),
            'political_category' if 'political_category' in filtered_data.columns else None
        ]).size().reset_index(name='count')
        
        if 'political_category' in filtered_data.columns:
            fig_temporal = px.line(
                temporal_data,
                x=date_col,
                y='count',
                color='political_category',
                title=f"Evolução {period} do Discurso por Categoria",
                labels={'count': 'Número de Mensagens', date_col: 'Período'}
            )
        else:
            fig_temporal = px.line(
                temporal_data,
                x=date_col,
                y='count',
                title=f"Evolução {period} do Discurso Político",
                labels={'count': 'Número de Mensagens', date_col: 'Período'}
            )
        
        fig_temporal.update_layout(height=400)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Análise de intensidade
    if 'political_intensity' in filtered_data.columns:
        st.subheader("🔥 Análise de Intensidade Política")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de intensidade
            fig_intensity = px.histogram(
                filtered_data,
                x='political_intensity',
                nbins=20,
                title="Distribuição de Intensidade Política",
                labels={'political_intensity': 'Intensidade', 'count': 'Frequência'}
            )
            
            fig_intensity.update_layout(height=300)
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        with col2:
            # Box plot por categoria
            if 'political_category' in filtered_data.columns:
                fig_intensity_cat = px.box(
                    filtered_data,
                    x='political_category',
                    y='political_intensity',
                    title="Intensidade por Categoria Política"
                )
                
                fig_intensity_cat.update_layout(height=300)
                st.plotly_chart(fig_intensity_cat, use_container_width=True)
            else:
                st.info("Análise por categoria não disponível")
    
    # Tabela detalhada
    st.subheader("📋 Dados Detalhados")
    
    # Configurar colunas para exibição
    display_columns = []
    for col in ['text', 'political_category', 'political_orientation', 'political_intensity', 'confidence_score', 'date', 'timestamp']:
        if col in filtered_data.columns:
            display_columns.append(col)
    
    if display_columns:
        # Opções de exibição
        col1, col2 = st.columns(2)
        
        with col1:
            show_rows = st.selectbox("Número de linhas", [10, 25, 50, 100], index=1)
        
        with col2:
            sort_column = st.selectbox("Ordenar por", display_columns)
        
        # Exibir tabela
        display_data = filtered_data[display_columns].head(show_rows)
        
        if sort_column in display_data.columns:
            display_data = display_data.sort_values(sort_column, ascending=False)
        
        st.dataframe(display_data, use_container_width=True)
        
        # Opções de exportação
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Exportar CSV"):
                export_path = data_loader.export_data('political_analysis', 'csv')
                if export_path:
                    st.success(f"Dados exportados para: {export_path.name}")
        
        with col2:
            if st.button("📊 Exportar Excel"):
                export_path = data_loader.export_data('political_analysis', 'excel')
                if export_path:
                    st.success(f"Dados exportados para: {export_path.name}")
        
        with col3:
            if st.button("🔄 Atualizar Dados"):
                data_loader.clear_cache()
                st.rerun()
    
    else:
        st.warning("Nenhuma coluna disponível para exibição")
    
    # Insights e estatísticas
    st.subheader("💡 Insights da Análise Política")
    
    insights = []
    
    if 'political_category' in filtered_data.columns:
        dominant_category = filtered_data['political_category'].mode().iloc[0]
        category_percentage = (filtered_data['political_category'].value_counts().iloc[0] / len(filtered_data)) * 100
        insights.append(f"• **Categoria Dominante**: {dominant_category} ({category_percentage:.1f}% das mensagens)")
    
    if 'political_intensity' in filtered_data.columns:
        high_intensity = (filtered_data['political_intensity'] > filtered_data['political_intensity'].quantile(0.75)).sum()
        intensity_percentage = (high_intensity / len(filtered_data)) * 100
        insights.append(f"• **Alta Intensidade**: {intensity_percentage:.1f}% das mensagens têm intensidade política elevada")
    
    if 'confidence_score' in filtered_data.columns:
        high_confidence = (filtered_data['confidence_score'] > 0.8).sum()
        confidence_percentage = (high_confidence / len(filtered_data)) * 100
        insights.append(f"• **Alta Confiança**: {confidence_percentage:.1f}% das classificações têm alta confiança (>0.8)")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Execute análise política completa para gerar insights automáticos")