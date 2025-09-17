"""
Página de Análise de Sentimento do Dashboard digiNEV
Análise de emoções e polarização no discurso político
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import numpy as np

def render_sentiment_page(data_loader):
    """Renderiza a página de análise de sentimento"""
    
    st.markdown('<div class="page-header"><h2>💭 Análise de Sentimento</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados de sentimento
    sentiment_data = data_loader.load_data('sentiment_analysis')
    
    if sentiment_data is None:
        st.warning("📊 Dados de análise de sentimento não disponíveis")
        st.info("Execute o pipeline principal para gerar a análise de sentimento (Stage 08)")
        return
    
    # Filtros interativos
    st.subheader("🔍 Filtros de Análise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtro por sentimento
        if 'sentiment' in sentiment_data.columns:
            sentiments = ['Todos'] + list(sentiment_data['sentiment'].unique())
            selected_sentiment = st.selectbox("Sentimento", sentiments)
        else:
            selected_sentiment = 'Todos'
    
    with col2:
        # Filtro por intensidade emocional
        if 'emotional_intensity' in sentiment_data.columns:
            intensity_range = st.slider(
                "Intensidade Emocional",
                min_value=float(sentiment_data['emotional_intensity'].min()),
                max_value=float(sentiment_data['emotional_intensity'].max()),
                value=(float(sentiment_data['emotional_intensity'].min()), 
                       float(sentiment_data['emotional_intensity'].max())),
                step=0.1
            )
        else:
            intensity_range = None
    
    with col3:
        # Filtro por confiança
        if 'confidence_score' in sentiment_data.columns:
            min_confidence = st.slider(
                "Confiança Mínima",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        else:
            min_confidence = 0.0
    
    # Aplicar filtros
    filtered_data = sentiment_data.copy()
    
    if selected_sentiment != 'Todos' and 'sentiment' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['sentiment'] == selected_sentiment]
    
    if intensity_range and 'emotional_intensity' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['emotional_intensity'] >= intensity_range[0]) &
            (filtered_data['emotional_intensity'] <= intensity_range[1])
        ]
    
    if 'confidence_score' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['confidence_score'] >= min_confidence]
    
    # Métricas principais
    st.subheader("📊 Métricas de Sentimento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_messages = len(filtered_data)
        st.metric("Total de Mensagens", f"{total_messages:,}")
    
    with col2:
        if 'sentiment' in filtered_data.columns:
            positive_pct = (filtered_data['sentiment'] == 'positive').mean() * 100
            st.metric("Sentimento Positivo", f"{positive_pct:.1f}%")
        else:
            st.metric("Positivo", "N/A")
    
    with col3:
        if 'emotional_intensity' in filtered_data.columns:
            avg_intensity = filtered_data['emotional_intensity'].mean()
            st.metric("Intensidade Média", f"{avg_intensity:.2f}")
        else:
            st.metric("Intensidade", "N/A")
    
    with col4:
        if 'polarization_score' in filtered_data.columns:
            avg_polarization = filtered_data['polarization_score'].mean()
            st.metric("Polarização Média", f"{avg_polarization:.2f}")
        else:
            st.metric("Polarização", "N/A")
    
    # Visualizações principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Distribuição de Sentimentos")
        
        if 'sentiment' in filtered_data.columns:
            sentiment_counts = filtered_data['sentiment'].value_counts()
            
            # Cores personalizadas para sentimentos
            color_map = {
                'positive': '#28a745',
                'negative': '#dc3545', 
                'neutral': '#6c757d',
                'mixed': '#ffc107'
            }
            
            colors = [color_map.get(sent, '#007bff') for sent in sentiment_counts.index]
            
            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                marker_colors=colors,
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Mensagens: %{value}<br>Porcentagem: %{percent}<extra></extra>'
            ))
            
            fig_sentiment.update_layout(
                title="Distribuição de Sentimentos",
                height=400
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("Dados de sentimento não disponíveis")
    
    with col2:
        st.subheader("🌡️ Intensidade Emocional")
        
        if 'emotional_intensity' in filtered_data.columns:
            fig_intensity = px.histogram(
                filtered_data,
                x='emotional_intensity',
                nbins=25,
                title="Distribuição de Intensidade Emocional",
                labels={'emotional_intensity': 'Intensidade Emocional', 'count': 'Frequência'},
                color_discrete_sequence=['#ff7f0e']
            )
            
            # Adicionar linha da média
            mean_intensity = filtered_data['emotional_intensity'].mean()
            fig_intensity.add_vline(
                x=mean_intensity,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {mean_intensity:.2f}"
            )
            
            fig_intensity.update_layout(height=400)
            st.plotly_chart(fig_intensity, use_container_width=True)
        else:
            st.info("Dados de intensidade emocional não disponíveis")
    
    # Análise de polarização
    if 'polarization_score' in filtered_data.columns:
        st.subheader("⚡ Análise de Polarização")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: intensidade vs polarização
            if 'emotional_intensity' in filtered_data.columns:
                fig_scatter = px.scatter(
                    filtered_data.sample(min(1000, len(filtered_data))),  # Amostra para performance
                    x='emotional_intensity',
                    y='polarization_score',
                    color='sentiment' if 'sentiment' in filtered_data.columns else None,
                    title="Intensidade vs Polarização",
                    labels={
                        'emotional_intensity': 'Intensidade Emocional',
                        'polarization_score': 'Score de Polarização'
                    },
                    opacity=0.6
                )
                
                fig_scatter.update_layout(height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Análise de correlação não disponível")
        
        with col2:
            # Boxplot de polarização por sentimento
            if 'sentiment' in filtered_data.columns:
                fig_polar_box = px.box(
                    filtered_data,
                    x='sentiment',
                    y='polarization_score',
                    title="Polarização por Sentimento",
                    labels={
                        'sentiment': 'Sentimento',
                        'polarization_score': 'Score de Polarização'
                    }
                )
                
                fig_polar_box.update_layout(height=350)
                st.plotly_chart(fig_polar_box, use_container_width=True)
            else:
                st.info("Análise por sentimento não disponível")
    
    # Análise temporal de sentimentos
    if 'date' in filtered_data.columns or 'timestamp' in filtered_data.columns:
        st.subheader("📈 Evolução Temporal dos Sentimentos")
        
        date_col = 'date' if 'date' in filtered_data.columns else 'timestamp'
        
        try:
            filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
            
            # Seletor de período
            period = st.selectbox("Período de Agrupamento", ["Diário", "Semanal", "Mensal"], index=1, key="sentiment_period")
            
            if period == "Diário":
                freq = 'D'
            elif period == "Semanal":
                freq = 'W'
            else:
                freq = 'M'
            
            # Agrupar dados temporais
            if 'sentiment' in filtered_data.columns:
                temporal_data = filtered_data.groupby([
                    filtered_data[date_col].dt.to_period(freq),
                    'sentiment'
                ]).size().reset_index(name='count')
                
                temporal_data[date_col] = temporal_data[date_col].dt.to_timestamp()
                
                fig_temporal = px.line(
                    temporal_data,
                    x=date_col,
                    y='count',
                    color='sentiment',
                    title=f"Evolução {period} dos Sentimentos",
                    labels={'count': 'Número de Mensagens', date_col: 'Período'}
                )
                
                fig_temporal.update_layout(height=400)
                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                st.info("Dados de sentimento temporal não disponíveis")
                
        except Exception as e:
            st.warning(f"Erro na análise temporal: {e}")
    
    # Análise de emoções específicas
    emotion_columns = [col for col in filtered_data.columns if 'emotion_' in col or col in ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust']]
    
    if emotion_columns:
        st.subheader("🎭 Análise de Emoções Específicas")
        
        # Média das emoções
        emotion_means = {}
        for col in emotion_columns:
            if pd.api.types.is_numeric_dtype(filtered_data[col]):
                emotion_means[col.replace('emotion_', '').replace('_', ' ').title()] = filtered_data[col].mean()
        
        if emotion_means:
            fig_emotions = go.Figure()
            fig_emotions.add_trace(go.Bar(
                x=list(emotion_means.keys()),
                y=list(emotion_means.values()),
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b'][:len(emotion_means)],
                text=[f"{v:.3f}" for v in emotion_means.values()],
                textposition='auto'
            ))
            
            fig_emotions.update_layout(
                title="Intensidade Média das Emoções",
                xaxis_title="Emoção",
                yaxis_title="Intensidade Média",
                height=400
            )
            
            st.plotly_chart(fig_emotions, use_container_width=True)
    
    # Análise de confiança
    if 'confidence_score' in filtered_data.columns:
        st.subheader("🎯 Análise de Confiança")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de confiança
            fig_confidence = px.histogram(
                filtered_data,
                x='confidence_score',
                nbins=20,
                title="Distribuição de Scores de Confiança",
                labels={'confidence_score': 'Score de Confiança', 'count': 'Frequência'}
            )
            
            fig_confidence.update_layout(height=300)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with col2:
            # Confiança por sentimento
            if 'sentiment' in filtered_data.columns:
                fig_conf_sent = px.violin(
                    filtered_data,
                    x='sentiment',
                    y='confidence_score',
                    title="Confiança por Sentimento",
                    labels={
                        'sentiment': 'Sentimento',
                        'confidence_score': 'Score de Confiança'
                    }
                )
                
                fig_conf_sent.update_layout(height=300)
                st.plotly_chart(fig_conf_sent, use_container_width=True)
            else:
                st.info("Análise por sentimento não disponível")
    
    # Tabela de exemplos
    st.subheader("📋 Exemplos de Mensagens por Sentimento")
    
    if 'sentiment' in filtered_data.columns and 'text' in filtered_data.columns:
        selected_sentiment_example = st.selectbox(
            "Selecionar sentimento para exemplos:",
            filtered_data['sentiment'].unique(),
            key="sentiment_examples"
        )
        
        sentiment_examples = filtered_data[
            filtered_data['sentiment'] == selected_sentiment_example
        ].head(5)
        
        if not sentiment_examples.empty:
            for idx, row in sentiment_examples.iterrows():
                with st.expander(f"Exemplo {idx + 1} - {row['sentiment'].title()}"):
                    st.write(f"**Texto:** {row['text'][:200]}{'...' if len(row['text']) > 200 else ''}")
                    
                    if 'emotional_intensity' in row:
                        st.write(f"**Intensidade:** {row['emotional_intensity']:.3f}")
                    
                    if 'confidence_score' in row:
                        st.write(f"**Confiança:** {row['confidence_score']:.3f}")
                    
                    if 'polarization_score' in row:
                        st.write(f"**Polarização:** {row['polarization_score']:.3f}")
    
    # Insights automáticos
    st.subheader("💡 Insights da Análise de Sentimento")
    
    insights = []
    
    if 'sentiment' in filtered_data.columns:
        sentiment_dist = filtered_data['sentiment'].value_counts(normalize=True) * 100
        dominant_sentiment = sentiment_dist.index[0]
        insights.append(f"• **Sentimento Dominante**: {dominant_sentiment.title()} ({sentiment_dist.iloc[0]:.1f}% das mensagens)")
    
    if 'emotional_intensity' in filtered_data.columns:
        high_intensity_pct = (filtered_data['emotional_intensity'] > filtered_data['emotional_intensity'].quantile(0.8)).mean() * 100
        insights.append(f"• **Alta Intensidade**: {high_intensity_pct:.1f}% das mensagens têm intensidade emocional elevada")
    
    if 'polarization_score' in filtered_data.columns:
        high_polar_pct = (filtered_data['polarization_score'] > 0.7).mean() * 100
        insights.append(f"• **Polarização**: {high_polar_pct:.1f}% das mensagens são altamente polarizadas")
    
    if 'confidence_score' in filtered_data.columns:
        reliable_pct = (filtered_data['confidence_score'] > 0.8).mean() * 100
        insights.append(f"• **Confiabilidade**: {reliable_pct:.1f}% das análises têm alta confiança")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Execute análise de sentimento completa para gerar insights automáticos")
    
    # Exportação de dados
    st.subheader("📥 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exportar CSV", key="export_sentiment_csv"):
            export_path = data_loader.export_data('sentiment_analysis', 'csv')
            if export_path:
                st.success(f"Dados exportados para: {export_path.name}")
    
    with col2:
        if st.button("📈 Exportar Excel", key="export_sentiment_excel"):
            export_path = data_loader.export_data('sentiment_analysis', 'excel')
            if export_path:
                st.success(f"Dados exportados para: {export_path.name}")
    
    with col3:
        if st.button("🔄 Atualizar Cache", key="refresh_sentiment_cache"):
            data_loader.clear_cache()
            st.rerun()