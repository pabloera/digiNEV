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
import os
from pathlib import Path

def render_overview_page(data_loader):
    """Renderiza a página de visão geral"""
    
    st.markdown('<div class="page-header"><h2>📋 Visão Geral do Dataset</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Seção de Upload de Arquivo
    st.subheader("📁 Carregar Dados")
    
    # Configurar tamanho máximo para arquivos grandes
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <b>💡 Instruções:</b><br>
        • Selecione um arquivo CSV com os dados do Telegram para análise<br>
        • Suporte para arquivos grandes (até 200MB)<br>
        • O arquivo será processado automaticamente após o upload
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type=['csv'],
        help="Faça upload do arquivo CSV com dados do Telegram para análise",
        key="csv_uploader"
    )
    
    # Processar arquivo carregado
    uploaded_data = None
    if uploaded_file is not None:
        try:
            # Mostrar informações do arquivo
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.info(f"📄 **Arquivo**: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Carregar dados com progress bar
            with st.spinner("Carregando dados..."):
                # Ler CSV com encoding automático
                try:
                    uploaded_data = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        uploaded_data = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        uploaded_data = pd.read_csv(uploaded_file, encoding='cp1252')
            
            if uploaded_data is not None and not uploaded_data.empty:
                st.success(f"✅ **Dados carregados com sucesso!** {len(uploaded_data):,} registros, {len(uploaded_data.columns)} colunas")
                
                # Salvar arquivo na pasta de dados
                data_path = data_loader.data_dir / "uploaded_data"
                data_path.mkdir(exist_ok=True)
                
                # Salvar como CSV principal
                output_file = data_path / f"telegram_data_{uploaded_file.name}"
                uploaded_data.to_csv(output_file, index=False)
                st.info(f"💾 Dados salvos em: {output_file.name}")
                
                # Preview dos dados
                with st.expander("👀 Preview dos Dados"):
                    st.dataframe(uploaded_data.head(100), use_container_width=True)
                    
                    # Informações das colunas
                    st.write("**Colunas disponíveis:**")
                    col_info = []
                    for col in uploaded_data.columns:
                        col_type = str(uploaded_data[col].dtype)
                        non_null = uploaded_data[col].notna().sum()
                        col_info.append({
                            'Coluna': col,
                            'Tipo': col_type,
                            'Valores Válidos': f"{non_null:,}/{len(uploaded_data):,}"
                        })
                    
                    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                
                # Opções de ação
                st.markdown("---")
                
                # Seção do Pipeline de Análise Completa
                st.markdown("### 🚀 **Pipeline de Análise Completa**")
                
                # Importar componentes do pipeline
                try:
                    from dashboard.utils.pipeline_runner import get_pipeline_runner
                    from dashboard.components.pipeline_ui import create_pipeline_interface
                    
                    # Obter instância do pipeline runner
                    pipeline_runner = get_pipeline_runner()
                    
                    # Criar interface do pipeline
                    pipeline_interface = create_pipeline_interface(pipeline_runner)
                    
                    # Renderizar interface completa
                    pipeline_interface.render_complete_interface()
                    
                except Exception as e:
                    st.error(f"❌ Erro ao carregar interface do pipeline: {str(e)}")
                    # Fallback para o botão simples
                    if st.button("🚀 Iniciar Pipeline de Análise", type="primary"):
                        st.info("⚠️ Para executar o pipeline completo, use o script `run_pipeline.py` no terminal")
                        st.code("python run_pipeline.py", language="bash")
                
                st.markdown("---")
                
                # Opções adicionais
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Análise Rápida"):
                        st.session_state.quick_analysis_data = uploaded_data
                        st.rerun()
                
                with col2:
                    if st.button("💾 Salvar como Dataset Principal"):
                        main_data_file = data_loader.data_dir / "telegram_data.csv"
                        uploaded_data.to_csv(main_data_file, index=False)
                        st.success(f"✅ Dataset salvo como principal: {main_data_file.name}")
                        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
            st.info("Verifique se o arquivo é um CSV válido com encoding UTF-8, Latin-1 ou CP1252")
    
    # Análise rápida dos dados carregados
    if 'quick_analysis_data' in st.session_state:
        st.markdown("---")
        st.subheader("⚡ Análise Rápida")
        
        quick_data = st.session_state.quick_analysis_data
        
        # Estatísticas básicas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Registros", f"{len(quick_data):,}")
        
        with col2:
            st.metric("Colunas", len(quick_data.columns))
        
        with col3:
            # Tentar detectar coluna de texto
            text_cols = [col for col in quick_data.columns if 'text' in col.lower() or 'message' in col.lower() or 'content' in col.lower()]
            if text_cols:
                avg_length = quick_data[text_cols[0]].astype(str).str.len().mean()
                st.metric("Comprimento Médio do Texto", f"{avg_length:.0f}")
            else:
                st.metric("Colunas de Texto", "Não detectadas")
        
        with col4:
            # Tentar detectar coluna de data
            date_cols = [col for col in quick_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                try:
                    date_range = pd.to_datetime(quick_data[date_cols[0]], errors='coerce')
                    days_range = (date_range.max() - date_range.min()).days
                    st.metric("Período (dias)", f"{days_range:,}")
                except:
                    st.metric("Período", "Não calculado")
            else:
                st.metric("Colunas de Data", "Não detectadas")
        
        # Distribuição básica
        if text_cols:
            st.subheader("📈 Distribuição do Comprimento das Mensagens")
            text_lengths = quick_data[text_cols[0]].astype(str).str.len()
            
            import plotly.express as px
            fig = px.histogram(
                x=text_lengths,
                nbins=50,
                title="Distribuição do Comprimento das Mensagens",
                labels={'x': 'Comprimento do Texto', 'y': 'Frequência'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🗑️ Limpar Análise Rápida"):
            del st.session_state.quick_analysis_data
            st.rerun()
    
    st.markdown("---")
    
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