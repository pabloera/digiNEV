"""
Stage 06 Affordances Classification Dashboard
===========================================

Dashboard acadêmico para análise de classificação de affordances em discurso político brasileiro.
Implementa visualizações científicas com foco em fluxos, conexões e evolução temporal.

Autor: digiNEV v.final
Data: Outubro 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta


class Stage06AffordancesDashboard:
    """Dashboard para análise de classificação de affordances em mensagens políticas."""

    def __init__(self):
        """Inicializar dashboard de affordances."""
        self.logger = logging.getLogger(__name__)

        # Configurações das categorias de affordances
        self.affordance_categories = [
            'noticia', 'midia_social', 'video_audio_gif', 'opiniao',
            'mobilizacao', 'ataque', 'interacao', 'is_forwarded'
        ]

        # Cores acadêmicas para visualizações
        self.color_palette = {
            'noticia': '#1f77b4',      # Azul - informação
            'midia_social': '#ff7f0e',  # Laranja - social
            'video_audio_gif': '#2ca02c', # Verde - multimídia
            'opiniao': '#d62728',       # Vermelho - opinião
            'mobilizacao': '#9467bd',   # Roxo - ação
            'ataque': '#8c564b',        # Marrom - agressão
            'interacao': '#e377c2',     # Rosa - interação
            'is_forwarded': '#7f7f7f'   # Cinza - encaminhamento
        }

        # Labels em português para apresentação
        self.affordance_labels = {
            'noticia': 'Notícia',
            'midia_social': 'Mídia Social',
            'video_audio_gif': 'Multimídia',
            'opiniao': 'Opinião',
            'mobilizacao': 'Mobilização',
            'ataque': 'Ataque',
            'interacao': 'Interação',
            'is_forwarded': 'Encaminhado'
        }

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Carregar dados processados do Stage 06."""
        try:
            # Buscar arquivos processados mais recentes
            dashboard_results_path = Path("src/dashboard/data/dashboard_results")

            if not dashboard_results_path.exists():
                st.error("Diretório de resultados do dashboard não encontrado.")
                return None

            # Procurar arquivo mais recente com Stage 06 ou posterior
            pattern_files = list(dashboard_results_path.glob("*_06_*affordances*.csv"))
            if not pattern_files:
                # Buscar qualquer arquivo com dados de affordances
                all_files = list(dashboard_results_path.glob("*.csv"))
                pattern_files = []
                for file in all_files:
                    # Verificar se contém dados de affordances lendo uma amostra
                    try:
                        sample_df = pd.read_csv(file, nrows=5)
                        if any(col.startswith('aff_') for col in sample_df.columns):
                            pattern_files.append(file)
                    except:
                        continue

            if not pattern_files:
                st.warning("Nenhum arquivo com dados de affordances encontrado.")
                return None

            # Selecionar arquivo mais recente
            latest_file = max(pattern_files, key=lambda x: x.stat().st_mtime)

            self.logger.info(f"Carregando dados de: {latest_file}")
            df = pd.read_csv(latest_file)

            # Validar se possui colunas de affordances
            affordance_columns = [f'aff_{cat}' for cat in self.affordance_categories]
            missing_columns = [col for col in affordance_columns if col not in df.columns]

            if missing_columns:
                st.warning(f"Colunas de affordances não encontradas: {missing_columns}")
                return None

            # Adicionar informações de processamento
            if 'affordance_categories' in df.columns:
                # Converter string de lista para lista real se necessário
                if df['affordance_categories'].dtype == 'object':
                    try:
                        df['affordance_categories'] = df['affordance_categories'].apply(
                            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
                        )
                    except:
                        df['affordance_categories'] = [[] for _ in range(len(df))]
            else:
                # Criar lista de categorias baseada nas colunas binárias
                df['affordance_categories'] = df.apply(
                    lambda row: [cat for cat in self.affordance_categories
                               if row.get(f'aff_{cat}', 0) == 1], axis=1
                )

            self.logger.info(f"Dados carregados: {len(df)} registros com {len([c for c in df.columns if c.startswith('aff_')])} categorias de affordances")
            return df

        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            st.error(f"Erro ao carregar dados processados: {e}")
            return None

    def create_sankey_diagram(self, df: pd.DataFrame) -> go.Figure:
        """Criar diagrama Sankey mostrando fluxo entre categorias de affordances."""
        try:
            # Preparar dados para o Sankey
            # Contar affordances individuais vs combinadas

            # Preparar nós
            nodes = []
            node_colors = []

            # Adicionar categoria "Individual" e "Múltiplas"
            nodes.extend(["Individual", "Múltiplas"])
            node_colors.extend(["#e6e6e6", "#cccccc"])

            # Adicionar categorias de affordances
            for cat in self.affordance_categories:
                nodes.append(self.affordance_labels[cat])
                node_colors.append(self.color_palette[cat])

            # Calcular fluxos
            source_indices = []
            target_indices = []
            values = []

            # Fluxo de Individual/Múltiplas para categorias específicas
            for i, cat in enumerate(self.affordance_categories):
                # Mensagens com apenas esta categoria
                single_cat = df[
                    (df[f'aff_{cat}'] == 1) &
                    (df[[f'aff_{c}' for c in self.affordance_categories]].sum(axis=1) == 1)
                ]

                if len(single_cat) > 0:
                    source_indices.append(0)  # Individual
                    target_indices.append(i + 2)  # Categoria específica
                    values.append(len(single_cat))

                # Mensagens com esta categoria + outras
                multi_cat = df[
                    (df[f'aff_{cat}'] == 1) &
                    (df[[f'aff_{c}' for c in self.affordance_categories]].sum(axis=1) > 1)
                ]

                if len(multi_cat) > 0:
                    source_indices.append(1)  # Múltiplas
                    target_indices.append(i + 2)  # Categoria específica
                    values.append(len(multi_cat))

            # Criar figura Sankey
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=node_colors
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color=[f"rgba({int(node_colors[target][1:3], 16)}, {int(node_colors[target][3:5], 16)}, {int(node_colors[target][5:7], 16)}, 0.4)"
                          for target in target_indices]  # Convert to rgba with transparency
                )
            )])

            fig.update_layout(
                title=dict(
                    text="Fluxo de Categorias de Affordances<br><sub>Distribuição entre classificações individuais e múltiplas</sub>",
                    x=0.5,
                    font=dict(size=16, color="#2E4057")
                ),
                font=dict(size=11, family="Arial"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500
            )

            return fig

        except Exception as e:
            self.logger.error(f"Erro ao criar diagrama Sankey: {e}")
            # Retornar figura vazia em caso de erro
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erro ao gerar Sankey: {str(e)[:100]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

    def create_network_graph(self, df: pd.DataFrame) -> go.Figure:
        """Criar grafo de rede mostrando conexões entre affordances combinadas."""
        try:
            # Criar grafo de co-ocorrência de affordances
            G = nx.Graph()

            # Adicionar nós para cada categoria
            for cat in self.affordance_categories:
                count = df[f'aff_{cat}'].sum()
                G.add_node(cat, count=count, label=self.affordance_labels[cat])

            # Adicionar arestas baseadas em co-ocorrência
            for i, cat1 in enumerate(self.affordance_categories):
                for cat2 in self.affordance_categories[i+1:]:
                    # Contar mensagens que têm ambas as categorias
                    co_occurrence = len(df[
                        (df[f'aff_{cat1}'] == 1) & (df[f'aff_{cat2}'] == 1)
                    ])

                    if co_occurrence > 0:
                        G.add_edge(cat1, cat2, weight=co_occurrence)

            # Posicionamento usando spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)

            # Preparar dados para Plotly
            edge_x = []
            edge_y = []
            edge_info = []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                weight = G[edge[0]][edge[1]]['weight']
                edge_info.append(f"{G.nodes[edge[0]]['label']} ↔ {G.nodes[edge[1]]['label']}: {weight} co-ocorrências")

            # Criar trace para arestas
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            # Preparar dados para nós
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            node_hover = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                # Informações do nó
                count = G.nodes[node]['count']
                label = G.nodes[node]['label']

                node_text.append(label)
                node_size.append(max(20, min(60, count / 5)))  # Tamanho proporcional
                node_color.append(self.color_palette[node])

                # Conexões do nó
                connections = list(G.neighbors(node))
                connection_info = [f"{G.nodes[conn]['label']}: {G[node][conn]['weight']}"
                                 for conn in connections]

                hover_text = f"<b>{label}</b><br>"
                hover_text += f"Mensagens: {count}<br>"
                hover_text += f"Conexões: {len(connections)}<br>"
                if connection_info:
                    hover_text += "<br>Co-ocorrências:<br>" + "<br>".join(connection_info[:5])
                    if len(connection_info) > 5:
                        hover_text += f"<br>... e mais {len(connection_info) - 5}"

                node_hover.append(hover_text)

            # Criar trace para nós
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_hover,
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color="white")
                )
            )

            # Criar figura
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(
                                  text="Rede de Co-ocorrência de Affordances<br><sub>Conexões entre categorias presentes nas mesmas mensagens</sub>",
                                  x=0.5,
                                  font=dict(size=16, color="#2E4057")
                              ),
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=80),
                              annotations=[ dict(
                                  text="Tamanho dos nós = número de mensagens | Espessura das linhas = co-ocorrências",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color="#888", size=10)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='white',
                              paper_bgcolor='white',
                              height=600
                          ))

            return fig

        except Exception as e:
            self.logger.error(f"Erro ao criar grafo de rede: {e}")
            # Retornar figura vazia em caso de erro
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erro ao gerar grafo: {str(e)[:100]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

    def create_timeline_evolution(self, df: pd.DataFrame) -> go.Figure:
        """Criar timeline mostrando evolução das affordances ao longo do tempo."""
        try:
            # Verificar se existe coluna de data/tempo
            date_columns = ['timestamp', 'datetime', 'date', 'created_at', 'time']
            date_col = None

            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break

            if date_col is None:
                # Criar timeline artificial baseada no índice
                self.logger.warning("Coluna de data não encontrada, criando timeline artificial")
                df['timeline_date'] = pd.date_range(
                    start='2022-01-01',
                    periods=len(df),
                    freq='h'
                )
                date_col = 'timeline_date'

            # Converter para datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            # Remover linhas com datas inválidas
            df_valid = df.dropna(subset=[date_col])

            if len(df_valid) == 0:
                raise ValueError("Nenhuma data válida encontrada para timeline")

            # Agrupar por período (diário)
            df_valid['date_period'] = df_valid[date_col].dt.date

            # Calcular proporções de cada affordance por período
            timeline_data = []

            for date in sorted(df_valid['date_period'].unique()):
                day_data = df_valid[df_valid['date_period'] == date]
                total_messages = len(day_data)

                if total_messages == 0:
                    continue

                period_record = {
                    'date': date,
                    'total_messages': total_messages
                }

                for cat in self.affordance_categories:
                    count = day_data[f'aff_{cat}'].sum()
                    proportion = count / total_messages if total_messages > 0 else 0
                    period_record[f'{cat}_count'] = count
                    period_record[f'{cat}_prop'] = proportion

                timeline_data.append(period_record)

            timeline_df = pd.DataFrame(timeline_data)

            if len(timeline_df) == 0:
                raise ValueError("Nenhum dado temporal válido para timeline")

            # Criar subplot com duas visualizações
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Evolução das Proporções de Affordances",
                    "Volume Absoluto por Categoria"
                ),
                vertical_spacing=0.12,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )

            # Gráfico 1: Proporções (área empilhada)
            for cat in self.affordance_categories:
                fig.add_trace(
                    go.Scatter(
                        x=timeline_df['date'],
                        y=timeline_df[f'{cat}_prop'],
                        name=self.affordance_labels[cat],
                        fill='tonexty' if cat != self.affordance_categories[0] else 'tozeroy',
                        stackgroup='one',
                        line=dict(width=1, color=self.color_palette[cat]),
                        hovertemplate=f"<b>{self.affordance_labels[cat]}</b><br>" +
                                    "Data: %{x}<br>" +
                                    "Proporção: %{y:.1%}<br>" +
                                    "<extra></extra>"
                    ),
                    row=1, col=1
                )

            # Gráfico 2: Contagens absolutas (linhas)
            for cat in self.affordance_categories:
                fig.add_trace(
                    go.Scatter(
                        x=timeline_df['date'],
                        y=timeline_df[f'{cat}_count'],
                        name=self.affordance_labels[cat],
                        line=dict(width=2, color=self.color_palette[cat]),
                        mode='lines+markers',
                        marker=dict(size=4),
                        showlegend=False,
                        hovertemplate=f"<b>{self.affordance_labels[cat]}</b><br>" +
                                    "Data: %{x}<br>" +
                                    "Mensagens: %{y}<br>" +
                                    "<extra></extra>"
                    ),
                    row=2, col=1
                )

            # Configurar layout
            fig.update_layout(
                title=dict(
                    text="Evolução Temporal das Affordances<br><sub>Análise de distribuição e volume ao longo do tempo</sub>",
                    x=0.5,
                    font=dict(size=16, color="#2E4057")
                ),
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial", size=11)
            )

            # Configurar eixos
            fig.update_xaxes(title_text="Período", row=2, col=1)
            fig.update_yaxes(title_text="Proporção", tickformat=".0%", row=1, col=1)
            fig.update_yaxes(title_text="Número de Mensagens", row=2, col=1)

            return fig

        except Exception as e:
            self.logger.error(f"Erro ao criar timeline: {e}")
            # Retornar figura vazia em caso de erro
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erro ao gerar timeline: {str(e)[:100]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

    def render_affordance_metrics(self, df: pd.DataFrame):
        """Renderizar métricas resumidas de affordances."""
        try:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_messages = len(df)
                st.metric(
                    label="Total de Mensagens",
                    value=f"{total_messages:,}",
                    help="Número total de mensagens processadas"
                )

            with col2:
                if 'affordance_confidence' in df.columns:
                    avg_confidence = df['affordance_confidence'].mean()
                    st.metric(
                        label="Confiança Média",
                        value=f"{avg_confidence:.2%}",
                        help="Confiança média da classificação de affordances"
                    )
                else:
                    st.metric(
                        label="Método",
                        value="Heurístico",
                        help="Classificação por método heurístico"
                    )

            with col3:
                # Calcular mensagens com múltiplas affordances
                multi_affordance = df[
                    df[[f'aff_{cat}' for cat in self.affordance_categories]].sum(axis=1) > 1
                ]
                multi_pct = len(multi_affordance) / len(df) if len(df) > 0 else 0
                st.metric(
                    label="Múltiplas Affordances",
                    value=f"{multi_pct:.1%}",
                    help="Percentual de mensagens com múltiplas categorias"
                )

            with col4:
                # Categoria mais comum
                category_counts = {}
                for cat in self.affordance_categories:
                    category_counts[cat] = df[f'aff_{cat}'].sum()

                top_category = max(category_counts, key=category_counts.get)
                top_count = category_counts[top_category]

                st.metric(
                    label="Categoria Principal",
                    value=self.affordance_labels[top_category],
                    delta=f"{top_count} mensagens",
                    help=f"Categoria mais frequente: {self.affordance_labels[top_category]}"
                )

        except Exception as e:
            st.error(f"Erro ao calcular métricas: {e}")

    def render_affordance_distribution(self, df: pd.DataFrame):
        """Renderizar gráfico de distribuição das affordances."""
        try:
            # Calcular contagens para cada categoria
            category_data = []
            for cat in self.affordance_categories:
                count = df[f'aff_{cat}'].sum()
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                category_data.append({
                    'categoria': self.affordance_labels[cat],
                    'categoria_id': cat,
                    'count': count,
                    'percentage': percentage
                })

            category_df = pd.DataFrame(category_data)
            category_df = category_df.sort_values('count', ascending=True)

            # Criar gráfico horizontal
            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=category_df['categoria'],
                x=category_df['count'],
                orientation='h',
                marker=dict(
                    color=[self.color_palette[cat_id] for cat_id in category_df['categoria_id']],
                    line=dict(color='white', width=1)
                ),
                text=[f"{count} ({pct:.1f}%)" for count, pct in
                      zip(category_df['count'], category_df['percentage'])],
                textposition='inside',
                textfont=dict(color='white', size=11),
                hovertemplate="<b>%{y}</b><br>" +
                            "Mensagens: %{x}<br>" +
                            "Percentual: %{customdata:.1f}%<br>" +
                            "<extra></extra>",
                customdata=category_df['percentage']
            ))

            fig.update_layout(
                title=dict(
                    text="Distribuição de Categorias de Affordances",
                    x=0.5,
                    font=dict(size=14, color="#2E4057")
                ),
                xaxis_title="Número de Mensagens",
                yaxis_title="Categoria de Affordance",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial", size=11),
                height=400,
                margin=dict(l=120, r=20, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao criar distribuição: {e}")

    def render_confidence_analysis(self, df: pd.DataFrame):
        """Renderizar análise de confiança da classificação."""
        try:
            if 'affordance_confidence' not in df.columns:
                st.info("Análise de confiança disponível apenas para classificação via API")
                return

            col1, col2 = st.columns(2)

            with col1:
                # Histograma de confiança
                fig_hist = go.Figure()

                fig_hist.add_trace(go.Histogram(
                    x=df['affordance_confidence'],
                    nbinsx=20,
                    marker=dict(color='#1f77b4', opacity=0.7, line=dict(color='white', width=1)),
                    name='Distribuição'
                ))

                fig_hist.update_layout(
                    title="Distribuição de Confiança",
                    xaxis_title="Confiança",
                    yaxis_title="Número de Mensagens",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=300
                )

                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Confiança por categoria
                confidence_by_category = []
                for cat in self.affordance_categories:
                    cat_messages = df[df[f'aff_{cat}'] == 1]
                    if len(cat_messages) > 0:
                        avg_conf = cat_messages['affordance_confidence'].mean()
                        confidence_by_category.append({
                            'categoria': self.affordance_labels[cat],
                            'confianca': avg_conf
                        })

                if confidence_by_category:
                    conf_df = pd.DataFrame(confidence_by_category)
                    conf_df = conf_df.sort_values('confianca', ascending=True)

                    fig_conf = go.Figure()

                    fig_conf.add_trace(go.Bar(
                        y=conf_df['categoria'],
                        x=conf_df['confianca'],
                        orientation='h',
                        marker=dict(color='#ff7f0e', opacity=0.8),
                        text=[f"{conf:.2%}" for conf in conf_df['confianca']],
                        textposition='inside'
                    ))

                    fig_conf.update_layout(
                        title="Confiança Média por Categoria",
                        xaxis_title="Confiança Média",
                        yaxis_title="Categoria",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300,
                        margin=dict(l=100)
                    )

                    st.plotly_chart(fig_conf, use_container_width=True)

        except Exception as e:
            st.error(f"Erro na análise de confiança: {e}")

    def render_dashboard(self):
        """Renderizar dashboard completo de affordances."""
        try:
            # Header do dashboard
            st.markdown("""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
                <h1 style="margin: 0; text-align: center;">🤖 Stage 06: Análise de Affordances</h1>
                <p style="margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;">
                    Classificação inteligente de mensagens em categorias de affordances comunicativas
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Carregar dados
            df = self.load_processed_data()

            if df is None:
                st.error("Não foi possível carregar dados processados do Stage 06.")
                st.info("Execute o pipeline completo para gerar dados de affordances.")
                return

            # Controles de filtro
            st.markdown("### 🎛️ Controles de Análise")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Filtro de confiança (se disponível)
                if 'affordance_confidence' in df.columns:
                    min_confidence = st.slider(
                        "Confiança Mínima",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1,
                        help="Filtrar por confiança mínima da classificação"
                    )
                    df_filtered = df[df['affordance_confidence'] >= min_confidence]
                else:
                    df_filtered = df
                    st.info("Filtro de confiança: N/A (método heurístico)")

            with col2:
                # Filtro de categorias múltiplas
                multi_only = st.checkbox(
                    "Apenas Múltiplas Affordances",
                    help="Mostrar apenas mensagens com múltiplas categorias"
                )

                if multi_only:
                    multi_mask = df_filtered[[f'aff_{cat}' for cat in self.affordance_categories]].sum(axis=1) > 1
                    df_filtered = df_filtered[multi_mask]

            with col3:
                # Seleção de período (se disponível)
                date_columns = ['timestamp', 'datetime', 'date', 'created_at', 'time']
                date_col = None
                for col in date_columns:
                    if col in df_filtered.columns:
                        date_col = col
                        break

                if date_col:
                    st.selectbox(
                        "Período de Análise",
                        ["Todos os períodos", "Último mês", "Últimos 3 meses"],
                        help="Filtrar por período temporal"
                    )
                else:
                    st.info("Filtro temporal: N/A")

            # Métricas principais
            st.markdown("### 📊 Métricas Principais")
            self.render_affordance_metrics(df_filtered)

            # Distribuição básica
            st.markdown("### 📈 Distribuição de Affordances")
            self.render_affordance_distribution(df_filtered)

            # Três visualizações principais
            st.markdown("### 🔬 Análises Avançadas")

            # Tabs para as três visualizações
            tab1, tab2, tab3 = st.tabs([
                "🌊 Fluxo Sankey",
                "🕸️ Rede de Conexões",
                "📅 Evolução Temporal"
            ])

            with tab1:
                st.markdown("#### Diagrama Sankey: Fluxo entre Categorias")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                <strong>Interpretação:</strong> O diagrama Sankey mostra como as mensagens fluem entre
                classificações individuais (uma única affordance) e múltiplas (combinações de affordances).
                </div>
                """, unsafe_allow_html=True)

                sankey_fig = self.create_sankey_diagram(df_filtered)
                st.plotly_chart(sankey_fig, use_container_width=True)

            with tab2:
                st.markdown("#### Grafo de Rede: Conexões entre Affordances")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                <strong>Interpretação:</strong> A rede mostra quais categorias de affordances tendem a
                aparecer juntas nas mesmas mensagens, revelando padrões de comunicação política.
                </div>
                """, unsafe_allow_html=True)

                network_fig = self.create_network_graph(df_filtered)
                st.plotly_chart(network_fig, use_container_width=True)

            with tab3:
                st.markdown("#### Timeline: Evolução das Affordances")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                <strong>Interpretação:</strong> A timeline revela como diferentes tipos de affordances
                evoluem ao longo do tempo, mostrando tendências no discurso político.
                </div>
                """, unsafe_allow_html=True)

                timeline_fig = self.create_timeline_evolution(df_filtered)
                st.plotly_chart(timeline_fig, use_container_width=True)

            # Análise de confiança
            st.markdown("### 🎯 Análise de Confiança")
            self.render_confidence_analysis(df_filtered)

            # Dados técnicos
            with st.expander("🔧 Dados Técnicos"):
                st.markdown("#### Informações do Dataset")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Registros Originais", len(df))
                    st.metric("Registros Filtrados", len(df_filtered))
                    st.metric("Colunas Totais", len(df_filtered.columns))

                with col2:
                    affordance_cols = [col for col in df_filtered.columns if col.startswith('aff_')]
                    st.metric("Colunas de Affordances", len(affordance_cols))

                    if 'affordance_confidence' in df_filtered.columns:
                        st.metric("Método de Classificação", "API + Heurístico")
                    else:
                        st.metric("Método de Classificação", "Heurístico")

                st.markdown("#### Amostra dos Dados")
                st.dataframe(
                    df_filtered[[col for col in df_filtered.columns if col.startswith('aff_') or
                                col in ['affordance_categories', 'affordance_confidence']]].head(),
                    use_container_width=True
                )

        except Exception as e:
            self.logger.error(f"Erro ao renderizar dashboard: {e}")
            st.error(f"Erro ao carregar dashboard de affordances: {e}")