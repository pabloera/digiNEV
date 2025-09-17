"""
Página de Análise Temporal do Dashboard digiNEV
Análise de evolução temporal, trends e detecção de eventos
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional
from datetime import datetime, timedelta

def render_temporal_page(data_loader):
    """Renderiza a página de análise temporal"""
    
    st.markdown('<div class="page-header"><h2>⏱️ Análise Temporal</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados temporais
    temporal_data = data_loader.load_data('temporal_analysis')
    
    if temporal_data is None:
        st.warning("📊 Dados de análise temporal não disponíveis")
        st.info("Execute o pipeline principal para gerar a análise temporal (Stage 14)")
        return
    
    # Filtros interativos
    st.subheader("🔍 Filtros de Análise Temporal")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Período de análise
        if 'date' in temporal_data.columns or 'timestamp' in temporal_data.columns:
            date_col = 'date' if 'date' in temporal_data.columns else 'timestamp'
            temporal_data[date_col] = pd.to_datetime(temporal_data[date_col])
            
            min_date = temporal_data[date_col].min().date()
            max_date = temporal_data[date_col].max().date()
            
            date_range = st.date_input(
                "Período de Análise",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None
            date_col = None
    
    with col2:
        # Granularidade temporal
        time_granularity = st.selectbox(
            "Granularidade",
            ["Horário", "Diário", "Semanal", "Mensal"],
            index=1
        )
    
    with col3:
        # Tipo de evento
        if 'event_type' in temporal_data.columns:
            event_types = ['Todos'] + list(temporal_data['event_type'].unique())
            selected_event_type = st.selectbox("Tipo de Evento", event_types)
        else:
            selected_event_type = 'Todos'
    
    # Aplicar filtros
    filtered_data = temporal_data.copy()
    
    if date_range and len(date_range) == 2 and date_col:
        filtered_data = filtered_data[
            (filtered_data[date_col].dt.date >= date_range[0]) &
            (filtered_data[date_col].dt.date <= date_range[1])
        ]
    
    if selected_event_type != 'Todos' and 'event_type' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['event_type'] == selected_event_type]
    
    # Métricas principais
    st.subheader("📊 Métricas Temporais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_events = len(filtered_data)
        st.metric("Total de Eventos", f"{total_events:,}")
    
    with col2:
        if date_col and not filtered_data.empty:
            duration_days = (filtered_data[date_col].max() - filtered_data[date_col].min()).days
            st.metric("Período (dias)", duration_days)
        else:
            st.metric("Período", "N/A")
    
    with col3:
        if 'intensity' in filtered_data.columns:
            avg_intensity = filtered_data['intensity'].mean()
            st.metric("Intensidade Média", f"{avg_intensity:.2f}")
        else:
            st.metric("Intensidade", "N/A")
    
    with col4:
        if 'significance_score' in filtered_data.columns:
            significant_events = (filtered_data['significance_score'] > 0.7).sum()
            st.metric("Eventos Significativos", significant_events)
        else:
            st.metric("Eventos Sig.", "N/A")
    
    # Timeline principal
    st.subheader("📈 Timeline de Eventos")
    
    if date_col and not filtered_data.empty:
        # Configurar frequência baseada na granularidade
        freq_map = {
            "Horário": 'H',
            "Diário": 'D',
            "Semanal": 'W',
            "Mensal": 'M'
        }
        freq = freq_map[time_granularity]
        
        # Agrupar dados por período
        timeline_data = filtered_data.groupby(filtered_data[date_col].dt.to_period(freq)).agg({
            'intensity': 'mean' if 'intensity' in filtered_data.columns else 'count',
            'significance_score': 'mean' if 'significance_score' in filtered_data.columns else 'count'
        }).reset_index()
        
        timeline_data[date_col] = timeline_data[date_col].dt.to_timestamp()
        timeline_data.columns = [date_col, 'Intensidade Média', 'Significância Média']
        
        # Criar gráfico temporal
        fig_timeline = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Intensidade ao Longo do Tempo', 'Significância dos Eventos'),
            vertical_spacing=0.15
        )
        
        # Intensidade
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_data[date_col],
                y=timeline_data['Intensidade Média'],
                mode='lines+markers',
                name='Intensidade',
                line=dict(color='blue', width=2),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Significância
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_data[date_col],
                y=timeline_data['Significância Média'],
                mode='lines+markers',
                name='Significância',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig_timeline.update_layout(
            height=600,
            title=f"Evolução {time_granularity} dos Eventos",
            showlegend=False
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Dados temporais não disponíveis para visualização")
    
    # Detecção de picos e eventos significativos
    st.subheader("🎯 Detecção de Eventos Significativos")
    
    if 'significance_score' in filtered_data.columns and not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de significância
            fig_significance = px.histogram(
                filtered_data,
                x='significance_score',
                nbins=20,
                title="Distribuição de Scores de Significância",
                labels={'significance_score': 'Score de Significância', 'count': 'Frequência'}
            )
            
            # Adicionar linha de threshold
            threshold = st.slider("Threshold de Significância", 0.0, 1.0, 0.7, step=0.05)
            fig_significance.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold}"
            )
            
            fig_significance.update_layout(height=350)
            st.plotly_chart(fig_significance, use_container_width=True)
        
        with col2:
            # Top eventos significativos
            significant_events = filtered_data[filtered_data['significance_score'] > threshold].nlargest(5, 'significance_score')
            
            if not significant_events.empty:
                st.write("**Top 5 Eventos Mais Significativos:**")
                
                for idx, event in significant_events.iterrows():
                    score = event['significance_score']
                    date_str = event[date_col].strftime("%Y-%m-%d %H:%M") if date_col else "N/A"
                    
                    with st.expander(f"🎯 Score: {score:.3f} - {date_str}"):
                        if 'description' in event:
                            st.write(f"**Descrição:** {event['description']}")
                        if 'event_type' in event:
                            st.write(f"**Tipo:** {event['event_type']}")
                        if 'intensity' in event:
                            st.write(f"**Intensidade:** {event['intensity']:.2f}")
            else:
                st.info("Nenhum evento significativo encontrado com o threshold atual")
    
    # Análise de tendências
    st.subheader("📊 Análise de Tendências")
    
    if date_col and not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tendência de volume ao longo do tempo
            daily_volume = filtered_data.groupby(filtered_data[date_col].dt.date).size().reset_index(name='volume')
            daily_volume.columns = ['date', 'volume']
            
            # Calcular tendência (regressão linear simples)
            x_numeric = np.arange(len(daily_volume))
            coeffs = np.polyfit(x_numeric, daily_volume['volume'], 1)
            trend_line = np.poly1d(coeffs)(x_numeric)
            
            fig_trend = go.Figure()
            
            # Volume diário
            fig_trend.add_trace(go.Scatter(
                x=daily_volume['date'],
                y=daily_volume['volume'],
                mode='lines+markers',
                name='Volume Diário',
                line=dict(color='lightblue')
            ))
            
            # Linha de tendência
            fig_trend.add_trace(go.Scatter(
                x=daily_volume['date'],
                y=trend_line,
                mode='lines',
                name=f'Tendência (slope: {coeffs[0]:.2f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig_trend.update_layout(
                title="Tendência de Volume de Eventos",
                xaxis_title="Data",
                yaxis_title="Volume de Eventos",
                height=350
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Análise de sazonalidade (dia da semana)
            filtered_data['day_of_week'] = filtered_data[date_col].dt.day_name()
            weekly_pattern = filtered_data['day_of_week'].value_counts()
            
            # Reordenar dias da semana
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern = weekly_pattern.reindex([day for day in day_order if day in weekly_pattern.index])
            
            fig_weekly = px.bar(
                x=weekly_pattern.index,
                y=weekly_pattern.values,
                title="Padrão Semanal de Eventos",
                labels={'x': 'Dia da Semana', 'y': 'Número de Eventos'}
            )
            
            fig_weekly.update_layout(height=350)
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Análise de correlação temporal
    st.subheader("🔗 Correlações Temporais")
    
    if date_col and 'intensity' in filtered_data.columns and not filtered_data.empty:
        # Autocorrelação da intensidade
        col1, col2 = st.columns(2)
        
        with col1:
            # Preparar dados para autocorrelação
            daily_intensity = filtered_data.groupby(filtered_data[date_col].dt.date)['intensity'].mean()
            
            if len(daily_intensity) > 10:
                # Calcular autocorrelação simples
                lags = range(1, min(15, len(daily_intensity)))
                autocorr_values = []
                
                for lag in lags:
                    corr = daily_intensity.corr(daily_intensity.shift(lag))
                    autocorr_values.append(corr if not pd.isna(corr) else 0)
                
                fig_autocorr = go.Figure()
                fig_autocorr.add_trace(go.Bar(
                    x=lags,
                    y=autocorr_values,
                    name='Autocorrelação'
                ))
                
                fig_autocorr.update_layout(
                    title="Autocorrelação da Intensidade",
                    xaxis_title="Lag (dias)",
                    yaxis_title="Correlação",
                    height=350
                )
                
                st.plotly_chart(fig_autocorr, use_container_width=True)
            else:
                st.info("Dados insuficientes para análise de autocorrelação")
        
        with col2:
            # Correlação com outras métricas
            if 'significance_score' in filtered_data.columns:
                correlation_metrics = filtered_data[['intensity', 'significance_score']].corr()
                
                fig_corr_matrix = px.imshow(
                    correlation_metrics,
                    title="Matriz de Correlação",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                
                fig_corr_matrix.update_layout(height=350)
                st.plotly_chart(fig_corr_matrix, use_container_width=True)
            else:
                st.info("Métricas adicionais não disponíveis para correlação")
    
    # Análise de eventos por período
    st.subheader("🗓️ Análise por Período")
    
    if date_col and not filtered_data.empty:
        period_analysis = st.selectbox(
            "Tipo de análise:",
            ["Por Hora do Dia", "Por Dia do Mês", "Por Mês do Ano"],
            key="period_analysis"
        )
        
        if period_analysis == "Por Hora do Dia":
            filtered_data['hour'] = filtered_data[date_col].dt.hour
            hourly_pattern = filtered_data['hour'].value_counts().sort_index()
            
            fig_hourly = px.bar(
                x=hourly_pattern.index,
                y=hourly_pattern.values,
                title="Distribuição por Hora do Dia",
                labels={'x': 'Hora', 'y': 'Número de Eventos'}
            )
            
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
            
        elif period_analysis == "Por Dia do Mês":
            filtered_data['day'] = filtered_data[date_col].dt.day
            daily_pattern = filtered_data['day'].value_counts().sort_index()
            
            fig_daily = px.line(
                x=daily_pattern.index,
                y=daily_pattern.values,
                title="Distribuição por Dia do Mês",
                labels={'x': 'Dia do Mês', 'y': 'Número de Eventos'},
                markers=True
            )
            
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
            
        else:  # Por Mês do Ano
            filtered_data['month'] = filtered_data[date_col].dt.month
            monthly_pattern = filtered_data['month'].value_counts().sort_index()
            
            # Nomes dos meses
            month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            
            fig_monthly = px.bar(
                x=[month_names[i-1] for i in monthly_pattern.index],
                y=monthly_pattern.values,
                title="Distribuição por Mês do Ano",
                labels={'x': 'Mês', 'y': 'Número de Eventos'}
            )
            
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Tabela de estatísticas temporais
    st.subheader("📋 Estatísticas Temporais Detalhadas")
    
    if not filtered_data.empty:
        stats_data = []
        
        # Estatísticas gerais
        stats_data.append({
            'Métrica': 'Total de Eventos',
            'Valor': f"{len(filtered_data):,}",
            'Descrição': 'Número total de eventos no período'
        })
        
        if date_col:
            duration = filtered_data[date_col].max() - filtered_data[date_col].min()
            stats_data.append({
                'Métrica': 'Duração',
                'Valor': f"{duration.days} dias",
                'Descrição': 'Período total analisado'
            })
            
            events_per_day = len(filtered_data) / max(duration.days, 1)
            stats_data.append({
                'Métrica': 'Eventos por Dia',
                'Valor': f"{events_per_day:.1f}",
                'Descrição': 'Taxa média de eventos'
            })
        
        if 'intensity' in filtered_data.columns:
            stats_data.append({
                'Métrica': 'Intensidade Média',
                'Valor': f"{filtered_data['intensity'].mean():.3f}",
                'Descrição': 'Intensidade média dos eventos'
            })
            
            stats_data.append({
                'Métrica': 'Intensidade Máxima',
                'Valor': f"{filtered_data['intensity'].max():.3f}",
                'Descrição': 'Maior intensidade registrada'
            })
        
        if 'significance_score' in filtered_data.columns:
            significant_count = (filtered_data['significance_score'] > 0.7).sum()
            stats_data.append({
                'Métrica': 'Eventos Significativos',
                'Valor': f"{significant_count} ({significant_count/len(filtered_data)*100:.1f}%)",
                'Descrição': 'Eventos com alta significância (>0.7)'
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    # Ferramentas de análise
    st.subheader("🛠️ Ferramentas de Análise Temporal")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Exportar Dados", key="export_temporal"):
            export_path = data_loader.export_data('temporal_analysis', 'csv')
            if export_path:
                st.success(f"Dados exportados: {export_path.name}")
    
    with col2:
        if st.button("🔍 Detectar Anomalias", key="detect_anomalies"):
            if 'intensity' in filtered_data.columns:
                # Detecção simples de anomalias usando IQR
                Q1 = filtered_data['intensity'].quantile(0.25)
                Q3 = filtered_data['intensity'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = filtered_data[
                    (filtered_data['intensity'] < lower_bound) | 
                    (filtered_data['intensity'] > upper_bound)
                ]
                
                st.info(f"🚨 {len(anomalies)} anomalias detectadas na intensidade")
            else:
                st.warning("Dados de intensidade não disponíveis")
    
    with col3:
        if st.button("📈 Analisar Tendências", key="analyze_trends"):
            if date_col and not filtered_data.empty:
                daily_events = filtered_data.groupby(filtered_data[date_col].dt.date).size()
                
                if len(daily_events) > 5:
                    # Tendência simples
                    x = np.arange(len(daily_events))
                    coeffs = np.polyfit(x, daily_events.values, 1)
                    
                    if coeffs[0] > 0.1:
                        st.info("📈 Tendência crescente detectada")
                    elif coeffs[0] < -0.1:
                        st.info("📉 Tendência decrescente detectada")
                    else:
                        st.info("📊 Tendência estável")
                else:
                    st.warning("Dados insuficientes para análise de tendência")
            else:
                st.warning("Dados temporais não disponíveis")
    
    with col4:
        if st.button("🔄 Atualizar Cache", key="refresh_temporal_cache"):
            data_loader.clear_cache()
            st.rerun()
    
    # Insights automáticos
    st.subheader("💡 Insights da Análise Temporal")
    
    insights = []
    
    if date_col and not filtered_data.empty:
        # Período com mais atividade
        daily_events = filtered_data.groupby(filtered_data[date_col].dt.date).size()
        peak_date = daily_events.idxmax()
        peak_count = daily_events.max()
        insights.append(f"• **Pico de Atividade**: {peak_date} com {peak_count} eventos")
        
        # Dia da semana mais ativo
        weekly_events = filtered_data.groupby(filtered_data[date_col].dt.day_name()).size()
        most_active_day = weekly_events.idxmax()
        insights.append(f"• **Dia Mais Ativo**: {most_active_day} ({weekly_events.max()} eventos)")
    
    if 'intensity' in filtered_data.columns:
        high_intensity_events = (filtered_data['intensity'] > filtered_data['intensity'].quantile(0.9)).sum()
        insights.append(f"• **Alta Intensidade**: {high_intensity_events} eventos no top 10% de intensidade")
    
    if 'significance_score' in filtered_data.columns:
        avg_significance = filtered_data['significance_score'].mean()
        insights.append(f"• **Significância Média**: {avg_significance:.3f} (escala 0-1)")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Execute análise temporal completa para gerar insights automáticos")