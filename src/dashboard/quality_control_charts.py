"""
Quality Control Charts v4.9.1 - Gráficos de Controle de Qualidade
=================================================================

Implementação de gráficos de controle estatístico para monitoramento
da qualidade do pipeline em tempo real com limites de controle,
alertas e análise de tendências.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import json
from pathlib import Path

@dataclass
class QualityMetric:
    """Métrica de qualidade para controle estatístico"""
    timestamp: datetime
    stage_id: str
    metric_name: str
    value: float
    upper_control_limit: float
    lower_control_limit: float
    target_value: float
    specification_upper: float
    specification_lower: float
    sample_size: int = 1

@dataclass
class ControlLimits:
    """Limites de controle estatístico"""
    center_line: float
    upper_control_limit: float
    lower_control_limit: float
    upper_spec_limit: float
    lower_spec_limit: float

class QualityControlCharts:
    """Classe principal para gráficos de controle de qualidade"""
    
    def __init__(self, project_root: Path):
        """Inicializa o sistema de controle de qualidade"""
        self.project_root = project_root
        self.quality_data_file = project_root / "logs" / "quality_metrics.json"
        self.quality_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurações de controle para cada tipo de métrica
        self.control_configs = {
            'success_rate': {
                'target': 0.95,
                'upper_spec': 1.0,
                'lower_spec': 0.8,
                'control_multiplier': 3  # 3-sigma
            },
            'quality_score': {
                'target': 0.85,
                'upper_spec': 1.0,
                'lower_spec': 0.7,
                'control_multiplier': 3
            },
            'processing_time': {
                'target': 1.0,  # Relativo ao tempo esperado
                'upper_spec': 2.0,
                'lower_spec': 0.2,
                'control_multiplier': 3
            },
            'memory_usage': {
                'target': 0.6,  # 60% da capacidade
                'upper_spec': 0.9,
                'lower_spec': 0.1,
                'control_multiplier': 3
            },
            'api_cost_efficiency': {
                'target': 1.0,
                'upper_spec': 2.0,
                'lower_spec': 0.5,
                'control_multiplier': 3
            }
        }
    
    def load_quality_data(self) -> List[QualityMetric]:
        """Carrega dados históricos de qualidade"""
        if not self.quality_data_file.exists():
            return []
        
        try:
            with open(self.quality_data_file, 'r') as f:
                data = json.load(f)
            
            metrics = []
            for item in data:
                metrics.append(QualityMetric(
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    stage_id=item['stage_id'],
                    metric_name=item['metric_name'],
                    value=item['value'],
                    upper_control_limit=item['upper_control_limit'],
                    lower_control_limit=item['lower_control_limit'],
                    target_value=item['target_value'],
                    specification_upper=item['specification_upper'],
                    specification_lower=item['specification_lower'],
                    sample_size=item.get('sample_size', 1)
                ))
            
            return metrics
            
        except Exception as e:
            st.error(f"Erro carregando dados de qualidade: {e}")
            return []
    
    def save_quality_metric(self, metric: QualityMetric) -> None:
        """Salva nova métrica de qualidade"""
        metrics = self.load_quality_data()
        metrics.append(metric)
        
        # Manter apenas últimos 1000 registros
        if len(metrics) > 1000:
            metrics = metrics[-1000:]
        
        try:
            data = []
            for metric in metrics:
                data.append({
                    'timestamp': metric.timestamp.isoformat(),
                    'stage_id': metric.stage_id,
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'upper_control_limit': metric.upper_control_limit,
                    'lower_control_limit': metric.lower_control_limit,
                    'target_value': metric.target_value,
                    'specification_upper': metric.specification_upper,
                    'specification_lower': metric.specification_lower,
                    'sample_size': metric.sample_size
                })
            
            with open(self.quality_data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            st.error(f"Erro salvando métrica de qualidade: {e}")
    
    def calculate_control_limits(self, values: List[float], metric_type: str) -> ControlLimits:
        """Calcula limites de controle estatístico"""
        if not values:
            config = self.control_configs.get(metric_type, self.control_configs['quality_score'])
            return ControlLimits(
                center_line=config['target'],
                upper_control_limit=config['upper_spec'],
                lower_control_limit=config['lower_spec'],
                upper_spec_limit=config['upper_spec'],
                lower_spec_limit=config['lower_spec']
            )
        
        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1) if len(values) > 1 else 0
        
        config = self.control_configs.get(metric_type, self.control_configs['quality_score'])
        multiplier = config['control_multiplier']
        
        return ControlLimits(
            center_line=mean_value,
            upper_control_limit=mean_value + multiplier * std_value,
            lower_control_limit=max(0, mean_value - multiplier * std_value),
            upper_spec_limit=config['upper_spec'],
            lower_spec_limit=config['lower_spec']
        )
    
    def create_control_chart(self, metric_name: str, stage_filter: Optional[str] = None) -> go.Figure:
        """Cria gráfico de controle para uma métrica específica"""
        metrics = self.load_quality_data()
        
        # Filtrar por métrica e etapa se especificado
        filtered_metrics = [m for m in metrics if m.metric_name == metric_name]
        if stage_filter:
            filtered_metrics = [m for m in filtered_metrics if m.stage_id == stage_filter]
        
        if not filtered_metrics:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Sem dados para {metric_name}" + (f" na etapa {stage_filter}" if stage_filter else ""),
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=f"Gráfico de Controle - {metric_name}",
                height=400
            )
            return fig
        
        # Preparar dados
        timestamps = [m.timestamp for m in filtered_metrics]
        values = [m.value for m in filtered_metrics]
        stages = [m.stage_id for m in filtered_metrics]
        
        # Calcular limites de controle
        limits = self.calculate_control_limits(values, metric_name)
        
        # Criar figura
        fig = go.Figure()
        
        # Linha central (média)
        fig.add_hline(
            y=limits.center_line,
            line_dash="solid",
            line_color="blue",
            annotation_text=f"Linha Central: {limits.center_line:.3f}"
        )
        
        # Limites de controle (3-sigma)
        fig.add_hline(
            y=limits.upper_control_limit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"LSC: {limits.upper_control_limit:.3f}"
        )
        
        fig.add_hline(
            y=limits.lower_control_limit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"LIC: {limits.lower_control_limit:.3f}"
        )
        
        # Limites de especificação
        if limits.upper_spec_limit != limits.upper_control_limit:
            fig.add_hline(
                y=limits.upper_spec_limit,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"LSE: {limits.upper_spec_limit:.3f}"
            )
        
        if limits.lower_spec_limit != limits.lower_control_limit:
            fig.add_hline(
                y=limits.lower_spec_limit,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"LIE: {limits.lower_spec_limit:.3f}"
            )
        
        # Pontos de dados
        colors = []
        for value in values:
            if value > limits.upper_control_limit or value < limits.lower_control_limit:
                colors.append('red')  # Fora de controle
            elif value > limits.upper_spec_limit or value < limits.lower_spec_limit:
                colors.append('orange')  # Fora de especificação
            else:
                colors.append('green')  # Dentro do controle
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='markers+lines',
            name='Valores Medidos',
            marker=dict(color=colors, size=8),
            line=dict(width=2),
            text=[f"Etapa: {stage}" for stage in stages],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Valor: %{y:.3f}<br>"
                "Data: %{x}<br>"
                "<extra></extra>"
            )
        ))
        
        # Identificar pontos fora de controle
        out_of_control = [(t, v, s) for t, v, s in zip(timestamps, values, stages)
                         if v > limits.upper_control_limit or v < limits.lower_control_limit]
        
        if out_of_control:
            oc_timestamps, oc_values, oc_stages = zip(*out_of_control)
            fig.add_trace(go.Scatter(
                x=oc_timestamps,
                y=oc_values,
                mode='markers',
                name='Fora de Controle',
                marker=dict(color='red', size=12, symbol='x'),
                text=[f"ALERTA: {stage}" for stage in oc_stages],
                hovertemplate=(
                    "<b>FORA DE CONTROLE</b><br>"
                    "%{text}<br>"
                    "Valor: %{y:.3f}<br>"
                    "Data: %{x}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title=f"Gráfico de Controle - {metric_name}" + (f" [{stage_filter}]" if stage_filter else ""),
            xaxis_title="Tempo",
            yaxis_title=metric_name.replace('_', ' ').title(),
            height=500,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def create_capability_analysis(self, metric_name: str) -> go.Figure:
        """Cria análise de capacidade do processo"""
        metrics = self.load_quality_data()
        filtered_metrics = [m for m in metrics if m.metric_name == metric_name]
        
        if not filtered_metrics:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Sem dados para análise de capacidade - {metric_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=f"Análise de Capacidade - {metric_name}", height=400)
            return fig
        
        values = [m.value for m in filtered_metrics]
        config = self.control_configs.get(metric_name, self.control_configs['quality_score'])
        
        # Calcular estatísticas
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        # Calcular índices de capacidade
        upper_spec = config['upper_spec']
        lower_spec = config['lower_spec']
        target = config['target']
        
        cp = (upper_spec - lower_spec) / (6 * std_val) if std_val > 0 else float('inf')
        cpk_upper = (upper_spec - mean_val) / (3 * std_val) if std_val > 0 else float('inf')
        cpk_lower = (mean_val - lower_spec) / (3 * std_val) if std_val > 0 else float('inf')
        cpk = min(cpk_upper, cpk_lower)
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribuição dos Valores',
                'Índices de Capacidade',
                'Histórico Temporal',
                'Resumo Estatístico'
            ],
            specs=[
                [{"type": "xy"}, {"type": "bar"}],
                [{"type": "xy"}, {"type": "table"}]
            ]
        )
        
        # 1. Histograma com curva normal
        fig.add_trace(
            go.Histogram(x=values, nbinsx=20, name='Distribuição', 
                        histnorm='probability density', opacity=0.7),
            row=1, col=1
        )
        
        # Curva normal teórica
        x_norm = np.linspace(min(values), max(values), 100)
        y_norm = stats.norm.pdf(x_norm, mean_val, std_val)
        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Teórica'),
            row=1, col=1
        )
        
        # Adicionar limites de especificação
        fig.add_vline(x=upper_spec, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_vline(x=lower_spec, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_vline(x=target, line_dash="solid", line_color="green", row=1, col=1)
        
        # 2. Índices de capacidade
        indices = ['Cp', 'Cpk', 'Cpu', 'Cpl']
        valores_indices = [cp, cpk, cpk_upper, cpk_lower]
        cores_indices = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in valores_indices]
        
        fig.add_trace(
            go.Bar(x=indices, y=valores_indices, marker_color=cores_indices, name='Índices'),
            row=1, col=2
        )
        
        # Linha de referência para Cp/Cpk
        fig.add_hline(y=1.33, line_dash="dash", line_color="green", row=1, col=2)
        fig.add_hline(y=1.0, line_dash="dash", line_color="orange", row=1, col=2)
        
        # 3. Série temporal
        timestamps = [m.timestamp for m in filtered_metrics]
        fig.add_trace(
            go.Scatter(x=timestamps, y=values, mode='lines+markers', name='Série Temporal'),
            row=2, col=1
        )
        
        fig.add_hline(y=upper_spec, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=lower_spec, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=target, line_dash="solid", line_color="green", row=2, col=1)
        
        # 4. Tabela de resumo
        resumo_data = [
            ['Média', f"{mean_val:.4f}"],
            ['Desvio Padrão', f"{std_val:.4f}"],
            ['Mínimo', f"{min(values):.4f}"],
            ['Máximo', f"{max(values):.4f}"],
            ['Cp', f"{cp:.3f}"],
            ['Cpk', f"{cpk:.3f}"],
            ['Amostras', f"{len(values)}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Estatística', 'Valor']),
                cells=dict(values=list(zip(*resumo_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Análise de Capacidade - {metric_name}",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_pareto_chart(self) -> go.Figure:
        """Cria gráfico de Pareto dos problemas de qualidade"""
        metrics = self.load_quality_data()
        
        if not metrics:
            fig = go.Figure()
            fig.add_annotation(
                text="Sem dados para análise de Pareto",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Análise de Pareto - Problemas de Qualidade", height=400)
            return fig
        
        # Identificar problemas por etapa
        problemas = {}
        for metric in metrics:
            limits = self.calculate_control_limits([], metric.metric_name)
            
            if (metric.value > limits.upper_spec_limit or 
                metric.value < limits.lower_spec_limit):
                
                stage_name = metric.stage_id
                if stage_name not in problemas:
                    problemas[stage_name] = 0
                problemas[stage_name] += 1
        
        if not problemas:
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhum problema de qualidade identificado",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Análise de Pareto - Sem Problemas", height=400)
            return fig
        
        # Ordenar por frequência
        problemas_ordenados = sorted(problemas.items(), key=lambda x: x[1], reverse=True)
        etapas, frequencias = zip(*problemas_ordenados)
        
        # Calcular percentual acumulado
        total = sum(frequencias)
        percentuais = [f/total*100 for f in frequencias]
        percentuais_acumulados = np.cumsum(percentuais)
        
        # Criar gráfico
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Barras de frequência
        fig.add_trace(
            go.Bar(x=etapas, y=frequencias, name='Frequência', marker_color='steelblue'),
            secondary_y=False
        )
        
        # Linha de percentual acumulado
        fig.add_trace(
            go.Scatter(x=etapas, y=percentuais_acumulados, mode='lines+markers',
                      name='% Acumulado', marker_color='red', line_width=3),
            secondary_y=True
        )
        
        # Linha 80% (Princípio de Pareto)
        fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                     annotation_text="80%", secondary_y=True)
        
        fig.update_xaxes(title_text="Etapas do Pipeline")
        fig.update_yaxes(title_text="Número de Problemas", secondary_y=False)
        fig.update_yaxes(title_text="Percentual Acumulado (%)", secondary_y=True, range=[0, 100])
        
        fig.update_layout(
            title="Análise de Pareto - Problemas de Qualidade por Etapa",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_quality_dashboard(self) -> None:
        """Cria dashboard completo de controle de qualidade"""
        st.header("📊 Dashboard de Controle de Qualidade")
        
        # Carregar dados
        metrics = self.load_quality_data()
        
        if not metrics:
            st.warning("⚠️ Nenhum dado de qualidade disponível ainda. Execute o pipeline para gerar métricas.")
            return
        
        # Métricas resumo
        col1, col2, col3, col4 = st.columns(4)
        
        # Calcular estatísticas gerais
        total_measurements = len(metrics)
        unique_stages = len(set(m.stage_id for m in metrics))
        
        # Contar problemas
        problems = 0
        for metric in metrics:
            limits = self.calculate_control_limits([], metric.metric_name)
            if (metric.value > limits.upper_spec_limit or 
                metric.value < limits.lower_spec_limit):
                problems += 1
        
        quality_rate = (total_measurements - problems) / total_measurements * 100 if total_measurements > 0 else 0
        
        with col1:
            st.metric("Taxa de Qualidade", f"{quality_rate:.1f}%", 
                     delta_color="normal" if quality_rate >= 90 else "inverse")
        
        with col2:
            st.metric("Total de Medições", f"{total_measurements:,}")
        
        with col3:
            st.metric("Etapas Monitoradas", f"{unique_stages}")
        
        with col4:
            st.metric("Problemas Identificados", f"{problems}", 
                     delta_color="inverse" if problems > 0 else "normal")
        
        # Seletores
        col1, col2 = st.columns(2)
        
        with col1:
            metric_types = list(set(m.metric_name for m in metrics))
            selected_metric = st.selectbox("Selecionar Métrica", metric_types)
        
        with col2:
            stages = ['Todas'] + list(set(m.stage_id for m in metrics))
            selected_stage = st.selectbox("Selecionar Etapa", stages)
            stage_filter = None if selected_stage == 'Todas' else selected_stage
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Gráfico de Controle")
            control_chart = self.create_control_chart(selected_metric, stage_filter)
            st.plotly_chart(control_chart, use_container_width=True)
        
        with col2:
            st.subheader("📊 Análise de Pareto")
            pareto_chart = self.create_pareto_chart()
            st.plotly_chart(pareto_chart, use_container_width=True)
        
        # Análise de capacidade
        st.subheader("🎯 Análise de Capacidade")
        capability_chart = self.create_capability_analysis(selected_metric)
        st.plotly_chart(capability_chart, use_container_width=True)
        
        # Alertas e recomendações
        self._show_quality_alerts(metrics)
    
    def _show_quality_alerts(self, metrics: List[QualityMetric]) -> None:
        """Mostra alertas e recomendações de qualidade"""
        st.subheader("🚨 Alertas e Recomendações")
        
        recent_metrics = [m for m in metrics if m.timestamp > datetime.now() - timedelta(hours=24)]
        
        if not recent_metrics:
            st.info("ℹ️ Nenhum alerta nas últimas 24 horas")
            return
        
        # Identificar problemas recentes
        alerts = []
        for metric in recent_metrics:
            limits = self.calculate_control_limits([], metric.metric_name)
            
            if metric.value > limits.upper_control_limit:
                alerts.append({
                    'type': 'error',
                    'stage': metric.stage_id,
                    'metric': metric.metric_name,
                    'message': f"Valor acima do limite superior de controle: {metric.value:.3f} > {limits.upper_control_limit:.3f}",
                    'timestamp': metric.timestamp
                })
            elif metric.value < limits.lower_control_limit:
                alerts.append({
                    'type': 'error',
                    'stage': metric.stage_id,
                    'metric': metric.metric_name,
                    'message': f"Valor abaixo do limite inferior de controle: {metric.value:.3f} < {limits.lower_control_limit:.3f}",
                    'timestamp': metric.timestamp
                })
            elif metric.value > limits.upper_spec_limit or metric.value < limits.lower_spec_limit:
                alerts.append({
                    'type': 'warning',
                    'stage': metric.stage_id,
                    'metric': metric.metric_name,
                    'message': f"Valor fora da especificação: {metric.value:.3f}",
                    'timestamp': metric.timestamp
                })
        
        if not alerts:
            st.success("✅ Todos os processos estão dentro dos limites de controle")
            return
        
        # Mostrar alertas
        for alert in sorted(alerts, key=lambda x: x['timestamp'], reverse=True):
            if alert['type'] == 'error':
                st.error(f"🔴 **{alert['stage']}** - {alert['metric']}: {alert['message']}")
            else:
                st.warning(f"🟡 **{alert['stage']}** - {alert['metric']}: {alert['message']}")
        
        # Recomendações
        st.subheader("💡 Recomendações")
        
        error_stages = set(alert['stage'] for alert in alerts if alert['type'] == 'error')
        if error_stages:
            st.markdown("**Ações Imediatas Recomendadas:**")
            for stage in error_stages:
                st.markdown(f"- Revisar configurações da etapa **{stage}**")
                st.markdown(f"- Verificar logs detalhados da execução")
                st.markdown(f"- Considerar ajuste de parâmetros")
        
        warning_stages = set(alert['stage'] for alert in alerts if alert['type'] == 'warning')
        if warning_stages:
            st.markdown("**Monitoramento Recomendado:**")
            for stage in warning_stages:
                st.markdown(f"- Acompanhar tendência da etapa **{stage}**")
                st.markdown(f"- Avaliar necessidade de otimização")