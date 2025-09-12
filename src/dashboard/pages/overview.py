"""
Página de Visão Geral do Dashboard digiNEV
Apresenta métricas principais e resumo executivo dos dados
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

def render_overview_page(data_loader):
    """Renderiza a página de visão geral"""
    
    st.markdown('<div class="page-header"><h2>📋 Visão Geral do Dataset</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados essenciais
    data_types = ['dataset_stats', 'political_analysis', 'sentiment_analysis', 'topic_modeling']
    datasets = data_loader.load_multiple_data(data_types)
    
    # Status dos dados
    status = data_loader.get_data_status()
    
    # Métricas principais
    st.subheader("📊 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_messages = 0
        if datasets['dataset_stats'] is not None:
            total_messages = len(datasets['dataset_stats'])
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_messages:,}</div>
            <div class="metric-label">Total de Mensagens</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        available_analyses = status['available_files']
        total_analyses = status['total_files']
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{available_analyses}/{total_analyses}</div>
            <div class="metric-label">Análises Disponíveis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        completion_rate = round((available_analyses / total_analyses) * 100) if total_analyses > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{completion_rate}%</div>
            <div class="metric-label">Taxa de Completude</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        last_execution = status.get('last_execution', 'N/A')
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">📅</div>
            <div class="metric-label">{last_execution}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráficos de visão geral
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Transformações Realizadas")
        
        # Gráfico de progresso das análises
        analysis_progress = []
        analysis_names = []
        
        for data_type in data_types:
            if datasets[data_type] is not None:
                analysis_progress.append(100)
                analysis_names.append(data_type.replace('_', ' ').title())
            else:
                analysis_progress.append(0)
                analysis_names.append(data_type.replace('_', ' ').title())
        
        fig_progress = go.Figure()
        fig_progress.add_trace(go.Bar(
            x=analysis_names,
            y=analysis_progress,
            marker_color=['#1f77b4' if p == 100 else '#ff7f0e' for p in analysis_progress],
            text=[f"{p}%" for p in analysis_progress],
            textposition='auto'
        ))
        
        fig_progress.update_layout(
            title="Status das Análises",
            xaxis_title="Tipo de Análise",
            yaxis_title="Completude (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_progress, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribuição de Dados")
        
        # Gráfico de distribuição por tipo de análise
        if any(df is not None for df in datasets.values()):
            data_summary = data_loader.get_data_summary()
            
            if data_summary:
                types = list(data_summary.keys())
                sizes = [data_summary[t]['rows'] for t in types]
                
                fig_distribution = go.Figure()
                fig_distribution.add_trace(go.Pie(
                    labels=[t.replace('_', ' ').title() for t in types],
                    values=sizes,
                    hole=0.3,
                    textinfo='label+percent'
                ))
                
                fig_distribution.update_layout(
                    title="Distribuição de Registros por Análise",
                    height=400
                )
                
                st.plotly_chart(fig_distribution, use_container_width=True)
            else:
                st.info("Dados de distribuição não disponíveis")
        else:
            st.info("Execute o pipeline para gerar dados de análise")
    
    # Status detalhado dos arquivos
    st.subheader("📁 Status Detalhado dos Arquivos")
    
    if status['missing_files']:
        st.warning(f"⚠️ {len(status['missing_files'])} arquivo(s) em falta")
        
        # Tabela de status dos arquivos
        file_status = []
        for data_type, filename in data_loader.expected_files.items():
            is_available = data_type not in status['missing_files']
            size_mb = 0
            
            if is_available and data_type in status.get('file_sizes', {}):
                size_mb = round(status['file_sizes'][data_type] / 1024 / 1024, 2)
            
            file_status.append({
                'Análise': data_type.replace('_', ' ').title(),
                'Arquivo': filename,
                'Status': '✅ Disponível' if is_available else '❌ Ausente',
                'Tamanho (MB)': size_mb if is_available else '-'
            })
        
        status_df = pd.DataFrame(file_status)
        st.dataframe(status_df, use_container_width=True)
    else:
        st.success("✅ Todos os arquivos de análise estão disponíveis!")
    
    # Informações adicionais
    st.subheader("ℹ️ Informações do Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Diretório de Saída**  
        `{data_loader.output_dir}`
        
        **Status**: {'✅ Existe' if status['output_dir_exists'] else '❌ Não encontrado'}
        """)
    
    with col2:
        st.info(f"""
        **Última Execução**  
        {status['last_execution']}
        
        **Arquivos**: {status['available_files']}/{status['total_files']}
        """)
    
    with col3:
        st.info(f"""
        **Cache de Dados**  
        {len(data_loader._cache)} item(s) em cache
        
        **Performance**: Otimizada
        """)
    
    # Próximos passos
    if status['available_files'] < status['total_files']:
        st.subheader("🚀 Próximos Passos")
        
        st.markdown("""
        ### Para gerar dados completos:
        
        1. **Execute o pipeline principal**:
           ```bash
           poetry run python run_pipeline.py
           ```
        
        2. **Aguarde a conclusão** das 22 etapas de análise
        
        3. **Recarregue o dashboard** para ver os resultados
        
        ### Análises em falta:
        """)
        
        for missing in status['missing_files']:
            st.write(f"- {missing.replace('_', ' ').title()}")
    else:
        st.success("🎉 Todos os dados estão disponíveis! Explore as análises usando o menu lateral.")