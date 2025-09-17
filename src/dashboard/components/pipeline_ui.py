#!/usr/bin/env python3
"""
Pipeline UI Components: Interface visual para execução de pipeline no dashboard
Function: Componentes Streamlit para mostrar progresso, logs e controles do pipeline
Usage: Interface integrada para execução completa do pipeline com feedback em tempo real
"""

import time
from datetime import datetime
from typing import List, Optional

import streamlit as st
import pandas as pd

from ..utils.pipeline_runner import DashboardPipelineRunner, PipelineStatus, StageProgress


class PipelineInterface:
    """Interface principal para execução do pipeline"""
    
    def __init__(self, runner: DashboardPipelineRunner):
        """Inicializa interface com pipeline runner"""
        self.runner = runner
    
    def render_pipeline_controls(self) -> bool:
        """Renderiza controles principais do pipeline"""
        st.markdown("### 🚀 **Execução do Pipeline de Análise**")
        
        progress = self.runner.get_progress()
        is_running = self.runner.is_running()
        
        # Botões de controle
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            if not is_running:
                if st.button("🚀 **Iniciar Pipeline Completo**", type="primary", use_container_width=True):
                    if self.runner.start_pipeline():
                        st.success("✅ Pipeline iniciado com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Erro ao iniciar pipeline")
            else:
                st.button("⚡ **Pipeline Executando...**", disabled=True, use_container_width=True)
        
        with col2:
            if is_running:
                if st.button("⏹️ **Parar**", type="secondary"):
                    if self.runner.stop_pipeline():
                        st.warning("⚠️ Cancelamento solicitado...")
                        time.sleep(1)
                        st.rerun()
            else:
                st.button("⏹️ **Parar**", disabled=True, type="secondary")
        
        with col3:
            if st.button("🔄 **Atualizar**"):
                st.rerun()
        
        with col4:
            if st.button("🧹 **Limpar Logs**"):
                # Limpar logs da sessão
                if 'pipeline_logs' in st.session_state:
                    del st.session_state['pipeline_logs']
                st.rerun()
        
        # Status atual
        status_colors = {
            PipelineStatus.IDLE: "🔵",
            PipelineStatus.RUNNING: "🟡",
            PipelineStatus.COMPLETED: "🟢", 
            PipelineStatus.ERROR: "🔴",
            PipelineStatus.CANCELLED: "🟠"
        }
        
        status_texts = {
            PipelineStatus.IDLE: "Aguardando",
            PipelineStatus.RUNNING: "Executando",
            PipelineStatus.COMPLETED: "Concluído",
            PipelineStatus.ERROR: "Erro",
            PipelineStatus.CANCELLED: "Cancelado"
        }
        
        status_icon = status_colors.get(progress.status, "⚪")
        status_text = status_texts.get(progress.status, "Desconhecido")
        
        st.markdown(f"**Status:** {status_icon} {status_text}")
        
        # Se houver erro, mostrar
        if progress.status == PipelineStatus.ERROR and progress.error_message:
            st.error(f"❌ **Erro:** {progress.error_message}")
        
        return is_running
    
    def render_progress_overview(self):
        """Renderiza visão geral do progresso"""
        progress = self.runner.get_progress()
        
        if progress.status == PipelineStatus.IDLE:
            st.info("📋 Pipeline pronto para execução. Clique em '🚀 Iniciar Pipeline Completo' para começar.")
            return
        
        # Métricas principais
        completion_pct = self.runner.get_completion_percentage()
        current_stage_name = self.runner.get_current_stage_name()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Progresso Geral",
                value=f"{completion_pct:.1f}%",
                delta=f"{progress.current_stage}/{progress.total_stages} etapas"
            )
        
        with col2:
            if progress.start_time:
                elapsed = datetime.now() - progress.start_time
                st.metric(
                    label="Tempo Decorrido",
                    value=f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"
                )
            else:
                st.metric(label="Tempo Decorrido", value="--")
        
        with col3:
            st.metric(
                label="Etapa Atual",
                value=f"{progress.current_stage}",
                delta=current_stage_name
            )
        
        with col4:
            completed_stages = sum(1 for stage in progress.stages if stage.status == "completed")
            st.metric(
                label="Etapas Concluídas",
                value=f"{completed_stages}",
                delta=f"{progress.total_stages - completed_stages} restantes"
            )
        
        # Barra de progresso principal
        st.markdown("#### 📊 **Progresso Geral**")
        progress_bar = st.progress(completion_pct / 100.0)
        st.markdown(f"**{completion_pct:.1f}% concluído** - {current_stage_name}")
    
    def render_stages_detail(self):
        """Renderiza detalhes das etapas"""
        progress = self.runner.get_progress()
        
        if not progress.stages:
            return
        
        st.markdown("#### 🔍 **Detalhes das Etapas**")
        
        # Criar DataFrame para exibição
        stages_data = []
        for i, stage in enumerate(progress.stages, 1):
            # Ícones de status
            status_icons = {
                "pending": "⏳",
                "running": "⚡",
                "completed": "✅",
                "error": "❌"
            }
            
            # Calcular tempo de execução
            execution_time = ""
            if stage.start_time and stage.end_time:
                duration = stage.end_time - stage.start_time
                execution_time = f"{duration.total_seconds():.1f}s"
            elif stage.start_time:
                duration = datetime.now() - stage.start_time
                execution_time = f"{duration.total_seconds():.1f}s (em execução)"
            
            stages_data.append({
                "Nº": i,
                "Status": status_icons.get(stage.status, "⚪"),
                "Etapa": stage.stage_name,
                "ID": stage.stage_id,
                "Tempo": execution_time,
                "Progresso": f"{stage.progress_percent:.0f}%" if stage.progress_percent > 0 else ""
            })
        
        # Exibir tabela
        df = pd.DataFrame(stages_data)
        
        # Configurar cores baseadas no status
        def style_status(val):
            if val == "✅":
                return "background-color: #d4edda; color: #155724;"
            elif val == "⚡":
                return "background-color: #fff3cd; color: #856404;"
            elif val == "❌":
                return "background-color: #f8d7da; color: #721c24;"
            else:
                return ""
        
        styled_df = df.style.applymap(style_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Mostrar etapa atual em destaque
        if progress.current_stage > 0 and progress.current_stage <= len(progress.stages):
            current_stage = progress.stages[progress.current_stage - 1]
            if current_stage.status == "running":
                st.info(f"🔄 **Executando agora:** {current_stage.stage_name}")
    
    def render_logs_panel(self):
        """Renderiza painel de logs"""
        st.markdown("#### 📝 **Logs de Execução**")
        
        # Controles de logs
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            auto_scroll = st.checkbox("🔄 Auto-scroll", value=True, help="Rolar automaticamente para logs mais recentes")
        
        with col2:
            max_lines = st.selectbox("📋 Mostrar", [50, 100, 200, 500], index=0, help="Número máximo de linhas")
        
        with col3:
            if st.button("📥 **Download Logs**"):
                logs = self.runner.get_logs_summary(max_lines=1000)
                if logs:
                    log_content = "\n".join(logs)
                    st.download_button(
                        label="💾 Baixar arquivo de log",
                        data=log_content,
                        file_name=f"pipeline_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        # Exibir logs
        logs = self.runner.get_logs_summary(max_lines=max_lines)
        
        if logs:
            # Container para logs
            log_container = st.container()
            
            with log_container:
                # Criar texto dos logs com formatação
                log_text = "\n".join(logs)
                
                # Se auto-scroll estiver ativo, mostrar apenas logs mais recentes primeiro
                if auto_scroll and len(logs) > 20:
                    recent_logs = logs[-20:]
                    older_logs = logs[:-20]
                    
                    # Mostrar logs mais recentes em destaque
                    st.markdown("**📍 Logs Mais Recentes:**")
                    st.code("\n".join(recent_logs), language="text")
                    
                    # Logs mais antigos em expander
                    if older_logs:
                        with st.expander(f"📚 Logs Anteriores ({len(older_logs)} linhas)"):
                            st.code("\n".join(older_logs), language="text")
                else:
                    # Mostrar todos os logs
                    st.code(log_text, language="text")
                
                # Auto-refresh se pipeline estiver executando
                if self.runner.is_running():
                    time.sleep(2)
                    st.rerun()
        else:
            st.info("📭 Nenhum log disponível ainda. Os logs aparecerão quando o pipeline for iniciado.")
    
    def render_complete_interface(self):
        """Renderiza interface completa do pipeline"""
        # Controles principais
        is_running = self.render_pipeline_controls()
        
        st.markdown("---")
        
        # Progresso geral
        self.render_progress_overview()
        
        st.markdown("---")
        
        # Abas para detalhes
        tab1, tab2 = st.tabs(["🔍 **Detalhes das Etapas**", "📝 **Logs de Execução**"])
        
        with tab1:
            self.render_stages_detail()
        
        with tab2:
            self.render_logs_panel()
        
        # Auto-refresh se executando
        if is_running:
            # Placeholder para forçar refresh automático
            if 'pipeline_refresh_count' not in st.session_state:
                st.session_state.pipeline_refresh_count = 0
            
            st.session_state.pipeline_refresh_count += 1
            
            # Refresh a cada 3 segundos quando executando
            time.sleep(3)
            st.rerun()


def create_pipeline_interface(runner: DashboardPipelineRunner) -> PipelineInterface:
    """Factory function para criar interface do pipeline"""
    return PipelineInterface(runner)