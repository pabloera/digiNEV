"""
Stage 13 Temporal Analysis Dashboard
Análise Temporal de Padrões de Coordenação no Discurso Político Brasileiro

Foco: Visualização de padrões temporais e coordenação de atividade
- Line chart: Volume de mensagens ao longo do tempo
- Event correlation: Picos de atividade vs eventos políticos
- Heatmap: Coordenação temporal entre usuários/canais
- Network graph: Clusters de atividade sincronizada
- Timeline: Períodos de alta coordenação identificados
- Sankey: Fluxo temporal → sentimento → affordances
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Analisador temporal com foco em coordenação e padrões de atividade."""

    def __init__(self):
        # Configurações de cores para consistência visual
        self.temporal_colors = {
            'high_activity': '#DC143C',     # Vermelho para alta atividade
            'medium_activity': '#FF8C00',   # Laranja para média atividade
            'low_activity': '#4682B4',      # Azul para baixa atividade
            'coordination': '#8B0000',      # Vermelho escuro para coordenação
            'background': '#F0F8FF'         # Azul claro para fundo
        }

        # Brazilian political events timeline
        self.political_events = {
            '2019-01-01': 'Início do Governo Bolsonaro',
            '2020-03-11': 'OMS declara pandemia COVID-19',
            '2020-04-24': 'Saída de Sergio Moro do governo',
            '2021-01-06': 'Início da vacinação no Brasil',
            '2021-10-07': 'Manifestações pró-Bolsonaro 7 de setembro',
            '2022-02-01': 'Início oficial da campanha eleitoral',
            '2022-10-02': 'Primeiro turno das eleições',
            '2022-10-30': 'Segundo turno das eleições',
            '2023-01-01': 'Início do terceiro governo Lula',
            '2023-01-08': 'Ataques aos Três Poderes'
        }

        # Configurações para gráficos
        self.chart_config = {
            'height': 500,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
            'showlegend': True
        }

    def validate_temporal_columns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Valida se as colunas temporais necessárias estão presentes."""
        required_columns = [
            'hour', 'day_of_week', 'month', 'year', 'day_of_year',
            'sender_frequency', 'is_frequent_sender', 'temporal_coordination',
            'is_weekend', 'is_business_hours'
        ]

        optional_columns = [
            'datetime', 'sender', 'sentiment_label', 'affordances_score'
        ]

        validation = {}

        # Verificar colunas obrigatórias
        for col in required_columns:
            validation[col] = col in df.columns

        # Verificar colunas opcionais
        for col in optional_columns:
            validation[f"{col}_optional"] = col in df.columns

        return validation

    def create_message_volume_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        1. Line chart: Volume de mensagens ao longo do tempo
        """
        try:
            # Verificar se temos dados temporais válidos
            if 'datetime' in df.columns:
                # Tentar converter datetime se disponível
                try:
                    df['parsed_datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    df_temp = df.dropna(subset=['parsed_datetime'])

                    if len(df_temp) > 0:
                        # Agrupar por data e contar mensagens
                        daily_volume = df_temp.groupby(df_temp['parsed_datetime'].dt.date).size().reset_index()
                        daily_volume.columns = ['date', 'message_count']
                        daily_volume['date'] = pd.to_datetime(daily_volume['date'])

                        fig = go.Figure()

                        # Linha principal de volume
                        fig.add_trace(go.Scatter(
                            x=daily_volume['date'],
                            y=daily_volume['message_count'],
                            mode='lines+markers',
                            name='Volume de Mensagens',
                            line=dict(color=self.temporal_colors['high_activity'], width=2),
                            marker=dict(size=4)
                        ))

                        # Adicionar eventos políticos como annotations
                        for date_str, event in self.political_events.items():
                            event_date = pd.to_datetime(date_str)
                            if daily_volume['date'].min() <= event_date <= daily_volume['date'].max():
                                fig.add_vline(
                                    x=event_date,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=event,
                                    annotation_position="top"
                                )

                        fig.update_layout(
                            title="Volume de Mensagens ao Longo do Tempo",
                            xaxis_title="Data",
                            yaxis_title="Número de Mensagens",
                            **self.chart_config
                        )

                        return fig

                except Exception as e:
                    logger.warning(f"Erro ao processar datetime: {e}")

            # Fallback: usar ano/mês/dia disponíveis
            if all(col in df.columns for col in ['year', 'month']):
                df_grouped = df.groupby(['year', 'month']).size().reset_index(name='message_count')
                df_grouped['date'] = pd.to_datetime(df_grouped[['year', 'month']].assign(day=1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_grouped['date'],
                    y=df_grouped['message_count'],
                    mode='lines+markers',
                    name='Volume Mensal',
                    line=dict(color=self.temporal_colors['medium_activity'], width=2)
                ))

                fig.update_layout(
                    title="Volume de Mensagens por Mês",
                    xaxis_title="Data",
                    yaxis_title="Número de Mensagens",
                    **self.chart_config
                )

                return fig

            # Último fallback: distribuição por hora
            if 'hour' in df.columns:
                hourly_volume = df.groupby('hour').size().reset_index(name='message_count')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hourly_volume['hour'],
                    y=hourly_volume['message_count'],
                    name='Volume por Hora',
                    marker_color=self.temporal_colors['low_activity']
                ))

                fig.update_layout(
                    title="Distribuição de Mensagens por Hora do Dia",
                    xaxis_title="Hora",
                    yaxis_title="Número de Mensagens",
                    **self.chart_config
                )

                return fig

        except Exception as e:
            logger.error(f"Erro ao criar timeline de volume: {e}")

        # Gráfico de erro
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Dados temporais não disponíveis",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Volume de Mensagens - Dados Indisponíveis")
        return fig

    def create_event_correlation_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        2. Event correlation: Picos de atividade vs eventos políticos
        """
        try:
            if 'datetime' in df.columns and 'temporal_coordination' in df.columns:
                # Tentar processar dados temporais
                try:
                    df['parsed_datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    df_temp = df.dropna(subset=['parsed_datetime'])

                    if len(df_temp) > 0:
                        # Agrupar por data com coordenação temporal
                        daily_data = df_temp.groupby(df_temp['parsed_datetime'].dt.date).agg({
                            'temporal_coordination': 'mean',
                            'sender_frequency': 'mean'
                        }).reset_index()
                        daily_data.columns = ['date', 'avg_coordination', 'avg_sender_freq']
                        daily_data['date'] = pd.to_datetime(daily_data['date'])
                        daily_data['message_count'] = df_temp.groupby(df_temp['parsed_datetime'].dt.date).size().values

                        # Criar subplot com dois eixos Y
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Volume de Mensagens vs Eventos Políticos', 'Coordenação Temporal'),
                            vertical_spacing=0.1
                        )

                        # Volume de mensagens
                        fig.add_trace(
                            go.Scatter(
                                x=daily_data['date'],
                                y=daily_data['message_count'],
                                mode='lines',
                                name='Volume de Mensagens',
                                line=dict(color=self.temporal_colors['high_activity'])
                            ),
                            row=1, col=1
                        )

                        # Coordenação temporal
                        fig.add_trace(
                            go.Scatter(
                                x=daily_data['date'],
                                y=daily_data['avg_coordination'],
                                mode='lines',
                                name='Coordenação Temporal',
                                line=dict(color=self.temporal_colors['coordination'])
                            ),
                            row=2, col=1
                        )

                        # Adicionar eventos políticos
                        for date_str, event in self.political_events.items():
                            event_date = pd.to_datetime(date_str)
                            if daily_data['date'].min() <= event_date <= daily_data['date'].max():
                                # Linha vertical no primeiro subplot
                                fig.add_vline(
                                    x=event_date,
                                    line_dash="dash",
                                    line_color="red",
                                    row=1, col=1
                                )
                                # Linha vertical no segundo subplot
                                fig.add_vline(
                                    x=event_date,
                                    line_dash="dash",
                                    line_color="red",
                                    row=2, col=1
                                )

                        fig.update_layout(
                            title="Correlação de Atividade com Eventos Políticos Brasileiros",
                            height=600,
                            showlegend=True
                        )

                        return fig

                except Exception as e:
                    logger.warning(f"Erro ao processar correlação temporal: {e}")

            # Fallback: análise por coordenação temporal disponível
            if 'temporal_coordination' in df.columns and 'hour' in df.columns:
                hourly_coord = df.groupby('hour')['temporal_coordination'].mean().reset_index()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hourly_coord['hour'],
                    y=hourly_coord['temporal_coordination'],
                    name='Coordenação por Hora',
                    marker_color=self.temporal_colors['coordination']
                ))

                fig.update_layout(
                    title="Coordenação Temporal por Hora do Dia",
                    xaxis_title="Hora",
                    yaxis_title="Coordenação Temporal Média",
                    **self.chart_config
                )

                return fig

        except Exception as e:
            logger.error(f"Erro ao criar análise de correlação: {e}")

        # Gráfico de erro
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Dados de correlação não disponíveis",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Correlação de Eventos - Dados Indisponíveis")
        return fig

    def create_coordination_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        3. Heatmap: Coordenação temporal entre usuários/canais
        """
        try:
            if all(col in df.columns for col in ['hour', 'day_of_week', 'temporal_coordination']):
                # Criar matriz de coordenação hora x dia da semana
                coord_matrix = df.groupby(['day_of_week', 'hour'])['temporal_coordination'].mean().reset_index()
                coord_pivot = coord_matrix.pivot(index='day_of_week', columns='hour', values='temporal_coordination')

                # Nomes dos dias da semana
                day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                coord_pivot.index = [day_names[i] if i < len(day_names) else f'Dia {i}' for i in coord_pivot.index]

                fig = go.Figure(data=go.Heatmap(
                    z=coord_pivot.values,
                    x=[f'{h:02d}h' for h in coord_pivot.columns],
                    y=coord_pivot.index,
                    colorscale='Reds',
                    colorbar=dict(title="Coordenação Temporal"),
                    hoverongaps=False
                ))

                fig.update_layout(
                    title="Heatmap de Coordenação Temporal (Dia da Semana x Hora)",
                    xaxis_title="Hora do Dia",
                    yaxis_title="Dia da Semana",
                    **self.chart_config
                )

                return fig

            elif all(col in df.columns for col in ['sender', 'temporal_coordination']) and 'sender' in df.columns:
                # Fallback: coordenação por sender se disponível
                if df['sender'].nunique() > 1:
                    top_senders = df['sender'].value_counts().head(20).index
                    df_top = df[df['sender'].isin(top_senders)]

                    sender_coord = df_top.groupby(['sender', 'hour'])['temporal_coordination'].mean().reset_index()
                    coord_pivot = sender_coord.pivot(index='sender', columns='hour', values='temporal_coordination')
                    coord_pivot = coord_pivot.fillna(0)

                    fig = go.Figure(data=go.Heatmap(
                        z=coord_pivot.values,
                        x=[f'{h:02d}h' for h in coord_pivot.columns],
                        y=[f'User_{i+1}' for i in range(len(coord_pivot.index))],
                        colorscale='Reds',
                        colorbar=dict(title="Coordenação"),
                        hoverongaps=False
                    ))

                    fig.update_layout(
                        title="Coordenação Temporal entre Top 20 Usuários",
                        xaxis_title="Hora do Dia",
                        yaxis_title="Usuários",
                        **self.chart_config
                    )

                    return fig

        except Exception as e:
            logger.error(f"Erro ao criar heatmap de coordenação: {e}")

        # Gráfico de erro ou dados simulados para demonstração
        fig = go.Figure()

        # Criar dados simulados para demonstração
        hours = list(range(24))
        days = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']

        # Simular padrão realista: maior coordenação em horários de pico
        z_data = []
        for day_idx in range(7):
            row = []
            for hour in hours:
                # Picos manhã (8-10h) e noite (19-22h), mais baixo fim de semana
                base_coord = 0.1
                if hour in range(8, 11) or hour in range(19, 23):
                    base_coord = 0.8 if day_idx < 5 else 0.4  # Menor coordenação nos fins de semana
                elif hour in range(12, 14):  # Pico do almoço
                    base_coord = 0.6 if day_idx < 5 else 0.3
                else:
                    base_coord = 0.2 if day_idx < 5 else 0.1

                row.append(base_coord + np.random.normal(0, 0.05))  # Adicionar um pouco de ruído
            z_data.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f'{h:02d}h' for h in hours],
            y=days,
            colorscale='Reds',
            colorbar=dict(title="Coordenação Temporal"),
            hoverongaps=False
        ))

        fig.update_layout(
            title="Heatmap de Coordenação Temporal (Padrão Demonstrativo)",
            xaxis_title="Hora do Dia",
            yaxis_title="Dia da Semana",
            **self.chart_config
        )

        return fig

    def create_network_graph(self, df: pd.DataFrame) -> go.Figure:
        """
        4. Network graph: Clusters de atividade sincronizada
        """
        try:
            if all(col in df.columns for col in ['sender', 'temporal_coordination', 'hour']):
                # Filtrar usuários com alta coordenação
                high_coord_threshold = df['temporal_coordination'].quantile(0.8)
                df_high_coord = df[df['temporal_coordination'] >= high_coord_threshold]

                if len(df_high_coord) > 10:  # Precisamos de dados suficientes
                    # Criar rede baseada em coordenação temporal
                    G = nx.Graph()

                    # Agrupar por hora e sender
                    sender_hour_groups = df_high_coord.groupby(['hour', 'sender']).size().reset_index(name='activity')

                    # Adicionar nós (senders)
                    senders = sender_hour_groups['sender'].unique()
                    for sender in senders[:50]:  # Limitar a 50 para performance
                        G.add_node(sender)

                    # Adicionar arestas baseadas em atividade sincronizada
                    hour_groups = sender_hour_groups.groupby('hour')
                    for hour, group in hour_groups:
                        senders_in_hour = group['sender'].tolist()
                        # Conectar senders que estavam ativos na mesma hora
                        for i, sender1 in enumerate(senders_in_hour):
                            for sender2 in senders_in_hour[i+1:]:
                                if G.has_node(sender1) and G.has_node(sender2):
                                    if G.has_edge(sender1, sender2):
                                        G[sender1][sender2]['weight'] += 1
                                    else:
                                        G.add_edge(sender1, sender2, weight=1)

                    # Calcular layout
                    pos = nx.spring_layout(G, k=1, iterations=50)

                    # Extrair coordenadas dos nós
                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []

                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(f'User: {str(node)[:10]}...<br>Conexões: {G.degree[node]}')
                        node_size.append(G.degree[node] * 5 + 10)

                    # Extrair coordenadas das arestas
                    edge_x = []
                    edge_y = []

                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    # Criar gráfico
                    fig = go.Figure()

                    # Adicionar arestas
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='lightgray'),
                        hoverinfo='none',
                        mode='lines',
                        name='Conexões'
                    ))

                    # Adicionar nós
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        text=node_text,
                        marker=dict(
                            size=node_size,
                            color=self.temporal_colors['coordination'],
                            line=dict(width=1, color='white')
                        ),
                        name='Usuários'
                    ))

                    fig.update_layout(
                        title="Rede de Atividade Sincronizada (Clusters de Coordenação)",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[
                            dict(
                                text="Tamanho do nó = número de conexões<br>Conexões = atividade sincronizada",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(size=12)
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=self.chart_config['height']
                    )

                    return fig

        except Exception as e:
            logger.error(f"Erro ao criar network graph: {e}")

        # Gráfico de erro ou dados simulados
        fig = go.Figure()

        # Criar rede simulada para demonstração
        G = nx.karate_club_graph()
        pos = nx.spring_layout(G)

        # Extrair coordenadas
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [f'Usuário {node}<br>Conexões: {G.degree[node]}' for node in G.nodes()]
        node_size = [G.degree[node] * 3 + 8 for node in G.nodes()]

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Adicionar arestas
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            name='Conexões'
        ))

        # Adicionar nós
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=self.temporal_colors['coordination'],
                line=dict(width=1, color='white')
            ),
            name='Usuários'
        ))

        fig.update_layout(
            title="Rede de Coordenação Temporal (Dados Demonstrativos)",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=self.chart_config['height']
        )

        return fig

    def create_coordination_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        5. Timeline: Períodos de alta coordenação identificados
        """
        try:
            if 'temporal_coordination' in df.columns:
                # Calcular threshold para alta coordenação
                high_coord_threshold = df['temporal_coordination'].quantile(0.9)

                if 'datetime' in df.columns:
                    try:
                        df['parsed_datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                        df_temp = df.dropna(subset=['parsed_datetime'])

                        if len(df_temp) > 0:
                            # Identificar períodos de alta coordenação
                            high_coord_periods = df_temp[df_temp['temporal_coordination'] >= high_coord_threshold]

                            if len(high_coord_periods) > 0:
                                # Agrupar por data
                                daily_coord = df_temp.groupby(df_temp['parsed_datetime'].dt.date).agg({
                                    'temporal_coordination': ['mean', 'max', 'count']
                                }).reset_index()
                                daily_coord.columns = ['date', 'avg_coord', 'max_coord', 'count']
                                daily_coord['date'] = pd.to_datetime(daily_coord['date'])

                                # Identificar dias com alta coordenação
                                high_coord_days = daily_coord[daily_coord['avg_coord'] >= high_coord_threshold]

                                fig = go.Figure()

                                # Linha de coordenação média
                                fig.add_trace(go.Scatter(
                                    x=daily_coord['date'],
                                    y=daily_coord['avg_coord'],
                                    mode='lines',
                                    name='Coordenação Média',
                                    line=dict(color=self.temporal_colors['medium_activity'])
                                ))

                                # Destacar períodos de alta coordenação
                                if len(high_coord_days) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=high_coord_days['date'],
                                        y=high_coord_days['avg_coord'],
                                        mode='markers',
                                        name='Alta Coordenação',
                                        marker=dict(
                                            size=10,
                                            color=self.temporal_colors['coordination'],
                                            symbol='diamond'
                                        )
                                    ))

                                # Linha de threshold
                                fig.add_hline(
                                    y=high_coord_threshold,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Threshold (90º percentil: {high_coord_threshold:.3f})"
                                )

                                # Adicionar eventos políticos
                                for date_str, event in self.political_events.items():
                                    event_date = pd.to_datetime(date_str)
                                    if daily_coord['date'].min() <= event_date <= daily_coord['date'].max():
                                        fig.add_vline(
                                            x=event_date,
                                            line_dash="dot",
                                            line_color="blue",
                                            annotation_text=event,
                                            annotation_position="top"
                                        )

                                fig.update_layout(
                                    title="Timeline de Períodos de Alta Coordenação",
                                    xaxis_title="Data",
                                    yaxis_title="Coordenação Temporal",
                                    **self.chart_config
                                )

                                return fig

                    except Exception as e:
                        logger.warning(f"Erro ao processar timeline temporal: {e}")

                # Fallback: análise por hora
                if 'hour' in df.columns:
                    hourly_coord = df.groupby('hour')['temporal_coordination'].agg(['mean', 'max', 'count']).reset_index()
                    high_coord_hours = hourly_coord[hourly_coord['mean'] >= high_coord_threshold]

                    fig = go.Figure()

                    # Barras de coordenação por hora
                    fig.add_trace(go.Bar(
                        x=hourly_coord['hour'],
                        y=hourly_coord['mean'],
                        name='Coordenação por Hora',
                        marker_color=[
                            self.temporal_colors['coordination'] if coord >= high_coord_threshold
                            else self.temporal_colors['medium_activity']
                            for coord in hourly_coord['mean']
                        ]
                    ))

                    # Linha de threshold
                    fig.add_hline(
                        y=high_coord_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {high_coord_threshold:.3f}"
                    )

                    fig.update_layout(
                        title="Períodos de Alta Coordenação por Hora do Dia",
                        xaxis_title="Hora",
                        yaxis_title="Coordenação Temporal Média",
                        **self.chart_config
                    )

                    return fig

        except Exception as e:
            logger.error(f"Erro ao criar timeline de coordenação: {e}")

        # Gráfico de erro
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Dados de coordenação temporal não disponíveis",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Timeline de Coordenação - Dados Indisponíveis")
        return fig

    def create_temporal_sentiment_sankey(self, df: pd.DataFrame) -> go.Figure:
        """
        6. Sankey: Fluxo temporal → sentimento → affordances
        """
        try:
            # Verificar colunas necessárias
            required_cols = ['hour', 'sentiment_label', 'affordances_score']
            optional_cols = ['is_weekend', 'is_business_hours']

            # Usar colunas disponíveis
            available_cols = [col for col in required_cols if col in df.columns]

            if len(available_cols) >= 2:  # Pelo menos 2 dimensões para Sankey
                # Simplificar categorias temporais
                if 'hour' in df.columns:
                    df = df.copy()
                    df['time_period'] = df['hour'].apply(lambda x:
                        'Madrugada (0-6h)' if x <= 6
                        else 'Manhã (7-12h)' if x <= 12
                        else 'Tarde (13-18h)' if x <= 18
                        else 'Noite (19-23h)'
                    )
                elif 'is_business_hours' in df.columns:
                    df = df.copy()
                    df['time_period'] = df['is_business_hours'].apply(lambda x:
                        'Horário Comercial' if x else 'Fora do Horário'
                    )
                else:
                    df = df.copy()
                    df['time_period'] = 'Período Geral'

                # Simplificar sentimento
                if 'sentiment_label' in df.columns:
                    sentiment_map = {
                        'positive': 'Positivo',
                        'negative': 'Negativo',
                        'neutral': 'Neutro'
                    }
                    df['sentiment_clean'] = df['sentiment_label'].map(sentiment_map).fillna('Neutro')
                else:
                    df['sentiment_clean'] = 'Neutro'

                # Simplificar affordances
                if 'affordances_score' in df.columns:
                    df['affordances_level'] = pd.cut(
                        df['affordances_score'].fillna(0),
                        bins=3,
                        labels=['Baixo', 'Médio', 'Alto']
                    ).astype(str)
                else:
                    df['affordances_level'] = 'Médio'

                # Contar fluxos
                flow_counts = df.groupby(['time_period', 'sentiment_clean', 'affordances_level']).size().reset_index(name='count')

                if len(flow_counts) > 0:
                    # Criar listas únicas de labels
                    time_labels = df['time_period'].unique().tolist()
                    sentiment_labels = df['sentiment_clean'].unique().tolist()
                    affordances_labels = df['affordances_level'].unique().tolist()

                    # Criar mapeamento de índices
                    all_labels = time_labels + sentiment_labels + affordances_labels
                    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

                    # Preparar dados do Sankey
                    source = []
                    target = []
                    value = []

                    # Fluxo 1: Tempo → Sentimento
                    temp_sentiment = df.groupby(['time_period', 'sentiment_clean']).size().reset_index(name='count')
                    for _, row in temp_sentiment.iterrows():
                        source.append(label_to_idx[row['time_period']])
                        target.append(label_to_idx[row['sentiment_clean']])
                        value.append(row['count'])

                    # Fluxo 2: Sentimento → Affordances
                    sentiment_affordances = df.groupby(['sentiment_clean', 'affordances_level']).size().reset_index(name='count')
                    for _, row in sentiment_affordances.iterrows():
                        source.append(label_to_idx[row['sentiment_clean']])
                        target.append(label_to_idx[row['affordances_level']])
                        value.append(row['count'])

                    # Cores para os nós
                    node_colors = (
                        ['#4682B4'] * len(time_labels) +      # Azul para tempo
                        ['#FF8C00', '#DC143C', '#708090'] +   # Laranja, vermelho, cinza para sentimentos
                        ['#228B22'] * len(affordances_labels)  # Verde para affordances
                    )

                    fig = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=all_labels,
                            color=node_colors[:len(all_labels)]
                        ),
                        link=dict(
                            source=source,
                            target=target,
                            value=value,
                            color='rgba(70, 130, 180, 0.4)'  # Azul transparente
                        )
                    )])

                    fig.update_layout(
                        title="Fluxo Temporal → Sentimento → Affordances",
                        font_size=12,
                        **self.chart_config
                    )

                    return fig

        except Exception as e:
            logger.error(f"Erro ao criar Sankey temporal: {e}")

        # Sankey simplificado de demonstração
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Manhã", "Tarde", "Noite", "Positivo", "Negativo", "Neutro", "Baixo", "Médio", "Alto"],
                color=["#4682B4", "#4682B4", "#4682B4", "#FF8C00", "#DC143C", "#708090", "#228B22", "#228B22", "#228B22"]
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2, 3, 4, 5, 3, 4, 5],
                target=[3, 4, 4, 5, 3, 5, 6, 7, 8, 7, 8, 6],
                value=[30, 20, 40, 35, 25, 30, 15, 25, 35, 20, 30, 25]
            )
        )])

        fig.update_layout(
            title="Fluxo Temporal → Sentimento → Affordances (Demonstrativo)",
            font_size=12,
            **self.chart_config
        )

        return fig


def main_dashboard(df: pd.DataFrame) -> None:
    """
    Função principal do dashboard de análise temporal.
    """
    analyzer = TemporalAnalyzer()

    st.header("📊 Stage 13 - Análise Temporal de Coordenação")
    st.markdown("""
    **Análise temporal de padrões de coordenação no discurso político brasileiro**

    Esta análise examina dimensões temporais da atividade através de:
    - Volume e distribuição temporal de mensagens
    - Correlação com eventos políticos brasileiros
    - Padrões de coordenação entre usuários/canais
    - Redes de atividade sincronizada
    - Fluxos temporais integrados com sentimento e affordances
    """)

    # Validação de dados
    validation = analyzer.validate_temporal_columns(df)

    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_messages = len(df)
        st.metric("Total de Mensagens", f"{total_messages:,}")

    with col2:
        if 'temporal_coordination' in df.columns:
            avg_coordination = df['temporal_coordination'].mean()
            st.metric("Coordenação Média", f"{avg_coordination:.3f}")
        else:
            st.metric("Coordenação Média", "N/A")

    with col3:
        if 'is_frequent_sender' in df.columns:
            frequent_senders = df['is_frequent_sender'].sum()
            st.metric("Usuários Frequentes", f"{frequent_senders:,}")
        else:
            st.metric("Usuários Frequentes", "N/A")

    with col4:
        if 'hour' in df.columns:
            peak_hour = df['hour'].mode().iloc[0] if len(df['hour'].mode()) > 0 else 'N/A'
            st.metric("Hora de Pico", f"{peak_hour}h" if peak_hour != 'N/A' else "N/A")
        else:
            st.metric("Hora de Pico", "N/A")

    # Seção 1: Volume temporal
    st.subheader("1. 📈 Volume de Mensagens ao Longo do Tempo")
    timeline_fig = analyzer.create_message_volume_timeline(df)
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Seção 2: Correlação com eventos
    st.subheader("2. 🎯 Correlação com Eventos Políticos")
    correlation_fig = analyzer.create_event_correlation_analysis(df)
    st.plotly_chart(correlation_fig, use_container_width=True)

    # Seção 3: Heatmap de coordenação
    st.subheader("3. 🔥 Heatmap de Coordenação Temporal")
    heatmap_fig = analyzer.create_coordination_heatmap(df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Seção 4: Network graph
    st.subheader("4. 🕸️ Rede de Atividade Sincronizada")
    network_fig = analyzer.create_network_graph(df)
    st.plotly_chart(network_fig, use_container_width=True)

    # Seção 5: Timeline de coordenação
    st.subheader("5. ⏱️ Timeline de Alta Coordenação")
    coordination_timeline_fig = analyzer.create_coordination_timeline(df)
    st.plotly_chart(coordination_timeline_fig, use_container_width=True)

    # Seção 6: Sankey temporal
    st.subheader("6. 🌊 Fluxo Temporal → Sentimento → Affordances")
    sankey_fig = analyzer.create_temporal_sentiment_sankey(df)
    st.plotly_chart(sankey_fig, use_container_width=True)

    # Informações de validação
    st.subheader("📋 Validação de Dados")

    # Separar colunas obrigatórias e opcionais
    required_cols = [k for k, v in validation.items() if not k.endswith('_optional')]
    optional_cols = [k for k, v in validation.items() if k.endswith('_optional')]

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Colunas Obrigatórias:**")
        for col, present in [(k, validation[k]) for k in required_cols]:
            status = "✅" if present else "❌"
            st.write(f"{status} {col}")

    with col2:
        st.write("**Colunas Opcionais:**")
        for col, present in [(k.replace('_optional', ''), validation[k]) for k in optional_cols]:
            status = "✅" if present else "❌"
            st.write(f"{status} {col}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Temporal Analysis Dashboard",
        page_icon="⏰",
        layout="wide"
    )

    # Para teste local
    st.title("⏰ Stage 13 - Análise Temporal")
    st.info("Esta é uma pré-visualização do dashboard. Use através do sistema principal para dados reais.")