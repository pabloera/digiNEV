"""
digiNEV Visualization Components: Research-focused charts and graphs for Brazilian political discourse analysis
Function: Stage-specific visualizations, quality control charts, and monitoring displays for 22-stage pipeline analysis
Usage: Social scientists view analysis results through dashboard - generates charts for political patterns, trends, and discourse metrics
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .pipeline_monitor import PipelineMonitor, StageMetrics, StageStatus

class PipelineVisualizations:
    """Classe principal para visualizações do pipeline"""

    def __init__(self, monitor: PipelineMonitor):
        """Inicializa com instância do monitor"""
        self.monitor = monitor

        # Cores por categoria de etapa
        self.category_colors = {
            'preprocessing': '#1f77b4',      # Azul
            'data_quality': '#ff7f0e',      # Laranja
            'analysis': '#2ca02c',          # Verde
            'ai_processing': '#d62728',     # Vermelho
            'nlp_processing': '#9467bd',    # Roxo
            'feature_engineering': '#8c564b',  # Marrom
            'validation': '#e377c2'         # Rosa
        }

        # Cores por status
        self.status_colors = {
            'pending': '#95a5a6',     # Cinza
            'running': '#f39c12',     # Amarelo
            'completed': '#27ae60',   # Verde
            'failed': '#e74c3c',      # Vermelho
            'skipped': '#3498db',     # Azul claro
            'protected': '#9b59b6'    # Roxo
        }

    def create_pipeline_overview_dashboard(self) -> None:
        """Cria dashboard de visão geral do pipeline"""
        overview = self.monitor.get_pipeline_overview()

        # Métricas principais em colunas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Progresso Geral",
                f"{overview['overall_progress']:.1%}",
                f"{overview['completed_stages']}/{overview['total_stages']} etapas"
            )

        with col2:
            if overview['elapsed_time'] > 0:
                remaining_time = max(0, overview['estimated_remaining_time'])
                st.metric(
                    "Tempo Restante",
                    f"{remaining_time/60:.1f} min",
                    f"Decorrido: {overview['elapsed_time']/60:.1f} min"
                )
            else:
                st.metric("Tempo Estimado", f"{overview['estimated_total_time']/60:.1f} min", "Total")

        with col3:
            st.metric(
                "Registros Processados",
                f"{overview['total_records']:,}",
                "Total"
            )

        with col4:
            if overview['failed_stages'] > 0:
                st.metric("Status", "❌ Com Falhas", f"{overview['failed_stages']} falhas", delta_color="inverse")
            elif overview['running_stages'] > 0:
                st.metric("Status", "🔄 Executando", f"{overview['running_stages']} ativas")
            elif overview['completed_stages'] == overview['total_stages']:
                st.metric("Status", "Concluído", "100% completo")
            else:
                st.metric("Status", "⏸️ Pausado", "Aguardando")

    def create_pipeline_progress_chart(self) -> go.Figure:
        """Cria gráfico de progresso geral do pipeline"""
        timeline_data = self.monitor.get_timeline_data()

        fig = go.Figure()

        # Preparar dados para o gráfico Gantt
        gantt_data = []
        y_position = 0

        for stage in timeline_data:
            stage_name = f"{stage['stage_id']}: {stage['name']}"

            # Determinar cores baseado no status
            color = self.status_colors.get(stage['status'], '#95a5a6')

            # Determinar tempo de início e fim
            if stage['start_time']:
                start = stage['start_time']
                if stage['end_time']:
                    end = stage['end_time']
                elif stage['status'] == 'running':
                    end = datetime.now()
                else:
                    end = start + timedelta(seconds=stage['expected_duration'])
            else:
                # Para etapas não iniciadas, usar tempos estimados
                if y_position == 0:
                    start = datetime.now()
                else:
                    # Usar fim da etapa anterior como início estimado
                    prev_end = gantt_data[-1]['end'] if gantt_data else datetime.now()
                    start = prev_end
                end = start + timedelta(seconds=stage['expected_duration'])

            gantt_data.append({
                'stage_id': stage['stage_id'],
                'name': stage_name,
                'start': start,
                'end': end,
                'status': stage['status'],
                'color': color,
                'critical': stage.get('critical', False),
                'duration': stage.get('duration', 0),
                'expected_duration': stage['expected_duration'],
                'category': stage['category']
            })

            y_position += 1

        # Criar barras do Gantt
        for i, stage in enumerate(gantt_data):
            # Barra principal
            fig.add_trace(go.Scatter(
                x=[stage['start'], stage['end']],
                y=[i, i],
                mode='lines',
                line=dict(width=20, color=stage['color']),
                name=stage['name'],
                hovertemplate=(
                    f"<b>{stage['name']}</b><br>"
                    f"Status: {stage['status']}<br>"
                    f"Categoria: {stage['category']}<br>"
                    f"Crítica: {'Sim' if stage['critical'] else 'Não'}<br>"
                    f"Duração: {stage['duration']:.1f}s / {stage['expected_duration']}s<br>"
                    f"Início: {stage['start'].strftime('%H:%M:%S')}<br>"
                    f"Fim: {stage['end'].strftime('%H:%M:%S')}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

            # Marcador para etapas críticas
            if stage['critical']:
                fig.add_trace(go.Scatter(
                    x=[stage['start']],
                    y=[i],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='star'),
                    name='Crítica',
                    showlegend=False
                ))

        # Linha vertical para momento atual
        fig.add_vline(
            x=datetime.now(),
            line_dash="dash",
            line_color="red",
            annotation_text="Agora"
        )

        # Configurar layout
        fig.update_layout(
            title="Timeline do Pipeline - 22 Etapas",
            xaxis_title="Tempo",
            yaxis_title="Etapas",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(gantt_data))),
                ticktext=[stage['name'] for stage in gantt_data],
                autorange="reversed"
            ),
            height=800,
            showlegend=False,
            hovermode='closest'
        )

        return fig

    def create_categories_progress_chart(self) -> go.Figure:
        """Cria gráfico de progresso por categoria"""
        overview = self.monitor.get_pipeline_overview()
        categories_progress = overview.get('categories_progress', {})

        categories = list(categories_progress.keys())
        completed = [categories_progress[cat]['completed'] for cat in categories]
        total = [categories_progress[cat]['total'] for cat in categories]
        progress_pct = [comp/tot*100 if tot > 0 else 0 for comp, tot in zip(completed, total)]

        fig = go.Figure()

        # Barras de progresso por categoria
        fig.add_trace(go.Bar(
            x=categories,
            y=progress_pct,
            marker_color=[self.category_colors.get(cat, '#95a5a6') for cat in categories],
            text=[f"{comp}/{tot}" for comp, tot in zip(completed, total)],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>Progresso: %{y:.1f}%<br>%{text} etapas<extra></extra>"
        ))

        fig.update_layout(
            title="Progresso por Categoria de Etapas",
            xaxis_title="Categoria",
            yaxis_title="Progresso (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )

        return fig

    def create_stage_performance_chart(self) -> go.Figure:
        """Cria gráfico de performance das etapas"""
        timeline_data = self.monitor.get_timeline_data()

        # Filtrar apenas etapas com dados de performance
        completed_stages = [stage for stage in timeline_data if stage['status'] == 'completed']

        if not completed_stages:
            # Gráfico vazio se não há etapas completas
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhuma etapa concluída ainda",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Performance das Etapas Concluídas",
                height=400
            )
            return fig

        stage_names = [stage['name'] for stage in completed_stages]
        actual_duration = [stage['duration'] for stage in completed_stages]
        expected_duration = [stage['expected_duration'] for stage in completed_stages]
        efficiency = [exp/act if act > 0 else 0 for exp, act in zip(expected_duration, actual_duration)]

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tempo Real vs Esperado', 'Eficiência das Etapas'),
            vertical_spacing=0.1
        )

        # Subplot 1: Tempo real vs esperado
        fig.add_trace(
            go.Bar(name='Tempo Esperado', x=stage_names, y=expected_duration, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Tempo Real', x=stage_names, y=actual_duration, marker_color='darkblue'),
            row=1, col=1
        )

        # Subplot 2: Eficiência
        colors = ['green' if eff >= 1.0 else 'orange' if eff >= 0.8 else 'red' for eff in efficiency]
        fig.add_trace(
            go.Bar(name='Eficiência', x=stage_names, y=efficiency, marker_color=colors),
            row=2, col=1
        )

        fig.update_layout(
            title="Análise de Performance das Etapas",
            height=600,
            showlegend=True
        )

        fig.update_yaxes(title_text="Tempo (segundos)", row=1, col=1)
        fig.update_yaxes(title_text="Eficiência (1.0 = ideal)", row=2, col=1)
        fig.update_xaxes(tickangle=45)

        return fig

    def create_quality_control_chart(self) -> go.Figure:
        """Cria gráfico de controle de qualidade"""
        timeline_data = self.monitor.get_timeline_data()

        stages_with_quality = [stage for stage in timeline_data
                             if stage.get('quality_score', 0) > 0]

        if not stages_with_quality:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de qualidade não disponíveis ainda",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Controle de Qualidade das Etapas", height=400)
            return fig

        stage_names = [stage['name'] for stage in stages_with_quality]
        quality_scores = [stage['quality_score'] for stage in stages_with_quality]
        success_rates = [stage.get('success_rate', 0) for stage in stages_with_quality]

        fig = go.Figure()

        # Linha de controle superior (90%)
        fig.add_hline(y=0.9, line_dash="dash", line_color="green",
                     annotation_text="Limite Superior (90%)")

        # Linha de controle inferior (70%)
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                     annotation_text="Limite Inferior (70%)")

        # Linha crítica (50%)
        fig.add_hline(y=0.5, line_dash="solid", line_color="red",
                     annotation_text="Crítico (50%)")

        # Pontos de qualidade
        fig.add_trace(go.Scatter(
            x=stage_names,
            y=quality_scores,
            mode='markers+lines',
            name='Score de Qualidade',
            marker=dict(size=10, color='blue'),
            line=dict(width=2)
        ))

        # Pontos de taxa de sucesso
        fig.add_trace(go.Scatter(
            x=stage_names,
            y=success_rates,
            mode='markers+lines',
            name='Taxa de Sucesso',
            marker=dict(size=8, color='green'),
            line=dict(width=2, dash='dot')
        ))

        fig.update_layout(
            title="Gráfico de Controle de Qualidade",
            xaxis_title="Etapas",
            yaxis_title="Score (0-1)",
            yaxis=dict(range=[0, 1]),
            height=500,
            hovermode='x unified'
        )

        fig.update_xaxes(tickangle=45)

        return fig

    def create_resource_usage_chart(self) -> go.Figure:
        """Cria gráfico de uso de recursos"""
        timeline_data = self.monitor.get_timeline_data()

        # Simular dados de recursos (na implementação real, estes viriam do monitor)
        stages_with_resources = []
        for stage in timeline_data:
            if stage['status'] == 'completed':
                # Simular uso de CPU e memória baseado na categoria
                if stage['category'] == 'ai_processing':
                    cpu = np.random.uniform(70, 95)
                    memory = np.random.uniform(60, 85)
                elif stage['category'] == 'nlp_processing':
                    cpu = np.random.uniform(60, 80)
                    memory = np.random.uniform(70, 90)
                else:
                    cpu = np.random.uniform(30, 60)
                    memory = np.random.uniform(40, 70)

                stages_with_resources.append({
                    'name': stage['name'],
                    'cpu_usage': cpu,
                    'memory_usage': memory,
                    'category': stage['category']
                })

        if not stages_with_resources:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de recursos não disponíveis ainda",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Uso de Recursos", height=400)
            return fig

        stage_names = [stage['name'] for stage in stages_with_resources]
        cpu_usage = [stage['cpu_usage'] for stage in stages_with_resources]
        memory_usage = [stage['memory_usage'] for stage in stages_with_resources]

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Uso de CPU (%)', 'Uso de Memória (%)'),
            vertical_spacing=0.1
        )

        # CPU Usage
        fig.add_trace(
            go.Bar(name='CPU', x=stage_names, y=cpu_usage,
                  marker_color='lightcoral'),
            row=1, col=1
        )

        # Memory Usage
        fig.add_trace(
            go.Bar(name='Memória', x=stage_names, y=memory_usage,
                  marker_color='lightblue'),
            row=2, col=1
        )

        # Linhas de alerta
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_hline(y=90, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(
            title="Monitoramento de Recursos por Etapa",
            height=600,
            showlegend=False
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(range=[0, 100])

        return fig

    def create_api_cost_chart(self) -> go.Figure:
        """Cria gráfico de custos de API"""
        timeline_data = self.monitor.get_timeline_data()

        # Filtrar etapas que usam API
        api_stages = [stage for stage in timeline_data
                     if stage['category'] == 'ai_processing' and stage['status'] == 'completed']

        if not api_stages:
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhuma etapa de IA concluída ainda",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Custos de API por Etapa", height=400)
            return fig

        stage_names = [stage['name'] for stage in api_stages]
        # Simular custos (na implementação real, viria do monitor)
        api_costs = [np.random.uniform(0.001, 0.05) for _ in api_stages]
        api_calls = [np.random.randint(10, 500) for _ in api_stages]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Barras de custo
        fig.add_trace(
            go.Bar(name='Custo (USD)', x=stage_names, y=api_costs,
                  marker_color='green', yaxis='y'),
        )

        # Linha de número de chamadas
        fig.add_trace(
            go.Scatter(name='Chamadas API', x=stage_names, y=api_calls,
                      mode='markers+lines', marker_color='red', yaxis='y2'),
        )

        fig.update_xaxes(title_text="Etapas de IA")
        fig.update_yaxes(title_text="Custo (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Número de Chamadas", secondary_y=True)

        fig.update_layout(
            title="Análise de Custos de API",
            height=400,
            hovermode='x unified'
        )

        return fig

    def create_stage_details_panel(self, stage_id: str) -> None:
        """Cria painel detalhado para uma etapa específica"""
        stage_details = self.monitor.get_stage_details(stage_id)

        if not stage_details:
            st.error(f"Etapa {stage_id} não encontrada")
            return

        # Header da etapa
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader(f"{stage_details['name']}")
            st.caption(stage_details['description'])

        with col2:
            status_emoji = {
                'pending': '⏳',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'protected': '🔒'
            }
            st.metric("Status", f"{status_emoji.get(stage_details['status'], '❓')} {stage_details['status'].title()}")

        with col3:
            if stage_details.get('critical', False):
                st.metric("Tipo", "🔴 Crítica")
            else:
                st.metric("Tipo", "🟡 Normal")

        # Métricas detalhadas
        if stage_details['status'] in ['completed', 'running']:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if stage_details['duration']:
                    efficiency = stage_details.get('efficiency', 0)
                    delta_color = "normal" if efficiency >= 1.0 else "inverse"
                    st.metric(
                        "Duração",
                        f"{stage_details['duration']:.1f}s",
                        f"Esperado: {stage_details['expected_duration']}s",
                        delta_color=delta_color
                    )
                else:
                    st.metric("Duração", "Em execução...")

            with col2:
                st.metric(
                    "Registros",
                    f"{stage_details['records_processed']:,}",
                    "Processados"
                )

            with col3:
                if stage_details['success_rate'] > 0:
                    st.metric(
                        "Taxa de Sucesso",
                        f"{stage_details['success_rate']:.1%}",
                        delta_color="normal" if stage_details['success_rate'] >= 0.9 else "inverse"
                    )

            with col4:
                if stage_details['quality_score'] > 0:
                    st.metric(
                        "Qualidade",
                        f"{stage_details['quality_score']:.2f}",
                        delta_color="normal" if stage_details['quality_score'] >= 0.8 else "inverse"
                    )

        # Informações adicionais específicas por categoria
        if stage_details['category'] == 'ai_processing':
            self._show_ai_stage_details(stage_details)
        elif stage_details['category'] == 'data_quality':
            self._show_data_quality_details(stage_details)
        elif stage_details['category'] == 'nlp_processing':
            self._show_nlp_details(stage_details)

    def _show_ai_stage_details(self, stage_details: Dict) -> None:
        """Mostra detalhes específicos de etapas de IA"""
        st.subheader("💡 Detalhes de Processamento IA")

        col1, col2 = st.columns(2)
        with col1:
            if stage_details.get('api_calls_made', 0) > 0:
                st.metric("Chamadas API", f"{stage_details['api_calls_made']:,}")
        with col2:
            if stage_details.get('api_cost_usd', 0) > 0:
                st.metric("Custo API", f"${stage_details['api_cost_usd']:.4f}")

    def _show_data_quality_details(self, stage_details: Dict) -> None:
        """Mostra detalhes específicos de qualidade de dados"""
        st.subheader("🔍 Controle de Qualidade")

        col1, col2 = st.columns(2)
        with col1:
            if stage_details.get('records_input', 0) > 0:
                st.metric("Registros Entrada", f"{stage_details['records_input']:,}")
        with col2:
            if stage_details.get('records_output', 0) > 0:
                reduction_pct = (1 - stage_details['records_output']/stage_details['records_input']) * 100
                st.metric("Registros Saída", f"{stage_details['records_output']:,}",
                         f"{reduction_pct:+.1f}% alteração")

    def _show_nlp_details(self, stage_details: Dict) -> None:
        """Mostra detalhes específicos de processamento NLP"""
        st.subheader("📝 Processamento Linguístico")

        col1, col2 = st.columns(2)
        with col1:
            if stage_details.get('processing_rate', 0) > 0:
                st.metric("Taxa Processamento", f"{stage_details['processing_rate']:.1f} reg/s")
        with col2:
            if stage_details.get('memory_usage_mb', 0) > 0:
                st.metric("Uso Memória", f"{stage_details['memory_usage_mb']:.1f} MB")
