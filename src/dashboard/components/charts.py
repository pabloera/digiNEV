"""
Componentes de gráficos reutilizáveis para o dashboard digiNEV
Gráficos padronizados e configuráveis para visualizações
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      delta_color: str = "normal") -> str:
    """
    Cria um card de métrica estilizado
    
    Args:
        title: Título da métrica
        value: Valor principal
        delta: Valor de mudança (opcional)
        delta_color: Cor do delta (normal, good, bad)
    
    Returns:
        HTML string do card
    """
    delta_html = ""
    if delta:
        color_map = {
            "good": "#28a745",
            "bad": "#dc3545", 
            "normal": "#6c757d"
        }
        color = color_map.get(delta_color, "#6c757d")
        delta_html = f'<div style="color: {color}; font-size: 0.8rem; margin-top: 0.25rem;">{delta}</div>'
    
    return f"""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 2rem; font-weight: bold; color: #1f77b4; margin: 0.5rem 0;">
            {value}
        </div>
        <div style="font-size: 0.9rem; color: #666; margin: 0;">
            {title}
        </div>
        {delta_html}
    </div>
    """

def create_distribution_pie(data: pd.Series, title: str, hole: float = 0.3,
                          colors: Optional[List[str]] = None) -> go.Figure:
    """
    Cria gráfico de pizza para distribuições
    
    Args:
        data: Série com dados para o gráfico
        title: Título do gráfico
        hole: Tamanho do buraco central (0-1)
        colors: Lista de cores personalizadas
    
    Returns:
        Figura do Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=data.index,
        values=data.values,
        hole=hole,
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Valor: %{value}<br>Porcentagem: %{percent}<extra></extra>',
        marker_colors=colors
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    
    return fig

def create_time_series(data: pd.DataFrame, x_col: str, y_col: str, 
                      color_col: Optional[str] = None, title: str = "",
                      line_type: str = "lines+markers") -> go.Figure:
    """
    Cria gráfico de série temporal
    
    Args:
        data: DataFrame com os dados
        x_col: Nome da coluna do eixo X (temporal)
        y_col: Nome da coluna do eixo Y
        color_col: Coluna para colorir as linhas (opcional)
        title: Título do gráfico
        line_type: Tipo de linha (lines, markers, lines+markers)
    
    Returns:
        Figura do Plotly
    """
    if color_col and color_col in data.columns:
        fig = px.line(
            data, x=x_col, y=y_col, color=color_col,
            title=title, markers=True if "markers" in line_type else False
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode=line_type,
            name=y_col,
            line=dict(width=2)
        ))
        fig.update_layout(title=title)
    
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )
    
    return fig

def create_correlation_heatmap(data: pd.DataFrame, title: str = "Matriz de Correlação",
                             columns: Optional[List[str]] = None) -> go.Figure:
    """
    Cria heatmap de correlação
    
    Args:
        data: DataFrame com dados numéricos
        title: Título do gráfico
        columns: Colunas específicas para correlação (opcional)
    
    Returns:
        Figura do Plotly
    """
    if columns:
        correlation_data = data[columns]
    else:
        correlation_data = data.select_dtypes(include=[np.number])
    
    correlation_matrix = correlation_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        texttemplate="%{z:.2f}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

def create_bar_chart(data: pd.Series, title: str, orientation: str = "v",
                    color_scale: Optional[str] = None, top_n: Optional[int] = None) -> go.Figure:
    """
    Cria gráfico de barras
    
    Args:
        data: Série com dados para o gráfico
        title: Título do gráfico
        orientation: Orientação (v=vertical, h=horizontal)
        color_scale: Escala de cores
        top_n: Mostrar apenas top N valores
    
    Returns:
        Figura do Plotly
    """
    if top_n:
        plot_data = data.head(top_n)
    else:
        plot_data = data
    
    if orientation == "h":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_data.values,
            y=plot_data.index,
            orientation='h',
            text=plot_data.values,
            textposition='auto',
            marker_color=plot_data.values if color_scale else None,
            marker_colorscale=color_scale
        ))
        fig.update_layout(
            xaxis_title="Valor",
            yaxis_title="Categoria"
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_data.index,
            y=plot_data.values,
            text=plot_data.values,
            textposition='auto',
            marker_color=plot_data.values if color_scale else None,
            marker_colorscale=color_scale
        ))
        fig.update_layout(
            xaxis_title="Categoria",
            yaxis_title="Valor"
        )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    return fig

def create_histogram(data: pd.Series, title: str, bins: int = 20,
                    add_mean_line: bool = True, add_median_line: bool = False) -> go.Figure:
    """
    Cria histograma
    
    Args:
        data: Série com dados numéricos
        title: Título do gráfico
        bins: Número de bins
        add_mean_line: Adicionar linha da média
        add_median_line: Adicionar linha da mediana
    
    Returns:
        Figura do Plotly
    """
    fig = px.histogram(
        x=data,
        nbins=bins,
        title=title,
        labels={'x': data.name or 'Valor', 'count': 'Frequência'}
    )
    
    if add_mean_line:
        mean_val = data.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Média: {mean_val:.2f}"
        )
    
    if add_median_line:
        median_val = data.median()
        fig.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mediana: {median_val:.2f}"
        )
    
    fig.update_layout(height=400)
    
    return fig

def create_box_plot(data: pd.DataFrame, x_col: Optional[str], y_col: str,
                   title: str) -> go.Figure:
    """
    Cria box plot
    
    Args:
        data: DataFrame com os dados
        x_col: Coluna categórica (opcional)
        y_col: Coluna numérica
        title: Título do gráfico
    
    Returns:
        Figura do Plotly
    """
    if x_col and x_col in data.columns:
        fig = px.box(data, x=x_col, y=y_col, title=title)
    else:
        fig = go.Figure()
        fig.add_trace(go.Box(y=data[y_col], name=y_col))
        fig.update_layout(title=title)
    
    fig.update_layout(height=400)
    
    return fig

def create_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str,
                       color_col: Optional[str] = None, size_col: Optional[str] = None,
                       title: str = "", add_trendline: bool = False) -> go.Figure:
    """
    Cria gráfico de dispersão
    
    Args:
        data: DataFrame com os dados
        x_col: Coluna do eixo X
        y_col: Coluna do eixo Y
        color_col: Coluna para colorir pontos (opcional)
        size_col: Coluna para tamanho dos pontos (opcional)
        title: Título do gráfico
        add_trendline: Adicionar linha de tendência
    
    Returns:
        Figura do Plotly
    """
    fig = px.scatter(
        data, x=x_col, y=y_col, color=color_col, size=size_col,
        title=title, opacity=0.6,
        trendline="ols" if add_trendline else None
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 1,
                      thresholds: Optional[Dict[str, float]] = None) -> go.Figure:
    """
    Cria gráfico de gauge (velocímetro)
    
    Args:
        value: Valor atual
        title: Título do gráfico
        min_val: Valor mínimo
        max_val: Valor máximo
        thresholds: Dicionário com thresholds {'baixo': 0.3, 'médio': 0.7, 'alto': 1.0}
    
    Returns:
        Figura do Plotly
    """
    # Configurar steps baseado nos thresholds
    steps = []
    if thresholds:
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
        colors = ['lightgray', 'yellow', 'lightgreen', 'green']
        
        prev_val = min_val
        for i, (name, thresh_val) in enumerate(sorted_thresholds):
            steps.append({
                'range': [prev_val, thresh_val],
                'color': colors[i % len(colors)]
            })
            prev_val = thresh_val
    else:
        steps = [
            {'range': [min_val, max_val * 0.5], 'color': "lightgray"},
            {'range': [max_val * 0.5, max_val * 0.8], 'color': "yellow"},
            {'range': [max_val * 0.8, max_val], 'color': "lightgreen"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': max_val * 0.8},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.8
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig

def create_multi_line_chart(data: pd.DataFrame, x_col: str, y_cols: List[str],
                           title: str, colors: Optional[List[str]] = None) -> go.Figure:
    """
    Cria gráfico com múltiplas linhas
    
    Args:
        data: DataFrame com os dados
        x_col: Coluna do eixo X
        y_cols: Lista de colunas para as linhas
        title: Título do gráfico
        colors: Lista de cores para as linhas
    
    Returns:
        Figura do Plotly
    """
    fig = go.Figure()
    
    if not colors:
        colors = px.colors.qualitative.Set1
    
    for i, y_col in enumerate(y_cols):
        if y_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name=y_col.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title="Valor",
        hovermode='x unified'
    )
    
    return fig

def create_funnel_chart(data: pd.Series, title: str) -> go.Figure:
    """
    Cria gráfico de funil
    
    Args:
        data: Série com dados ordenados para o funil
        title: Título do gráfico
    
    Returns:
        Figura do Plotly
    """
    fig = go.Figure(go.Funnel(
        y=data.index,
        x=data.values,
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig

def create_treemap(data: pd.DataFrame, values_col: str, labels_col: str,
                  parent_col: Optional[str] = None, title: str = "") -> go.Figure:
    """
    Cria treemap
    
    Args:
        data: DataFrame com os dados
        values_col: Coluna com valores
        labels_col: Coluna com rótulos
        parent_col: Coluna com categorias pai (opcional)
        title: Título do gráfico
    
    Returns:
        Figura do Plotly
    """
    if parent_col and parent_col in data.columns:
        fig = go.Figure(go.Treemap(
            labels=data[labels_col],
            values=data[values_col],
            parents=data[parent_col],
            textinfo="label+value"
        ))
    else:
        fig = go.Figure(go.Treemap(
            labels=data[labels_col],
            values=data[values_col],
            textinfo="label+value"
        ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig