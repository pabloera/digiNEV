"""
Componentes de filtros interativos para o dashboard digiNEV
Filtros reutilizáveis e configuráveis para análise de dados
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date

def create_date_range_filter(data: pd.DataFrame, date_column: str, 
                           key: str = "date_filter") -> Optional[Tuple[date, date]]:
    """
    Cria filtro de intervalo de datas
    
    Args:
        data: DataFrame com os dados
        date_column: Nome da coluna de data
        key: Chave única para o widget
    
    Returns:
        Tupla com (data_inicio, data_fim) ou None
    """
    if date_column not in data.columns:
        st.warning(f"Coluna {date_column} não encontrada")
        return None
    
    try:
        # Converter para datetime se necessário
        data[date_column] = pd.to_datetime(data[date_column])
        
        min_date = data[date_column].min().date()
        max_date = data[date_column].max().date()
        
        return st.date_input(
            "Período de Análise",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=key
        )
    
    except Exception as e:
        st.error(f"Erro ao processar datas: {e}")
        return None

def create_categorical_filter(data: pd.DataFrame, column: str, 
                            label: str = None, include_all: bool = True,
                            key: str = None) -> Any:
    """
    Cria filtro para variáveis categóricas
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna categórica
        label: Rótulo do filtro
        include_all: Incluir opção "Todos"
        key: Chave única para o widget
    
    Returns:
        Valor selecionado
    """
    if column not in data.columns:
        st.warning(f"Coluna {column} não encontrada")
        return None
    
    if not label:
        label = column.replace('_', ' ').title()
    
    if not key:
        key = f"{column}_filter"
    
    unique_values = sorted(data[column].dropna().unique())
    
    if include_all:
        options = ['Todos'] + list(unique_values)
    else:
        options = list(unique_values)
    
    return st.selectbox(label, options, key=key)

def create_multiselect_filter(data: pd.DataFrame, column: str,
                            label: str = None, key: str = None) -> List[Any]:
    """
    Cria filtro de múltipla seleção
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna
        label: Rótulo do filtro
        key: Chave única para o widget
    
    Returns:
        Lista de valores selecionados
    """
    if column not in data.columns:
        st.warning(f"Coluna {column} não encontrada")
        return []
    
    if not label:
        label = column.replace('_', ' ').title()
    
    if not key:
        key = f"{column}_multiselect"
    
    unique_values = sorted(data[column].dropna().unique())
    
    return st.multiselect(
        label,
        options=unique_values,
        default=[],
        key=key
    )

def create_numeric_range_filter(data: pd.DataFrame, column: str,
                              label: str = None, step: float = 0.01,
                              key: str = None) -> Tuple[float, float]:
    """
    Cria filtro de intervalo numérico
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna numérica
        label: Rótulo do filtro
        step: Passo do slider
        key: Chave única para o widget
    
    Returns:
        Tupla com (valor_min, valor_max)
    """
    if column not in data.columns:
        st.warning(f"Coluna {column} não encontrada")
        return (0.0, 1.0)
    
    if not label:
        label = column.replace('_', ' ').title()
    
    if not key:
        key = f"{column}_range"
    
    try:
        min_val = float(data[column].min())
        max_val = float(data[column].max())
        
        return st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=step,
            key=key
        )
    
    except Exception as e:
        st.error(f"Erro ao processar valores numéricos: {e}")
        return (0.0, 1.0)

def create_text_search_filter(label: str = "Buscar texto", 
                            placeholder: str = "Digite para buscar...",
                            key: str = "text_search") -> str:
    """
    Cria filtro de busca textual
    
    Args:
        label: Rótulo do filtro
        placeholder: Texto placeholder
        key: Chave única para o widget
    
    Returns:
        Texto de busca
    """
    return st.text_input(
        label,
        placeholder=placeholder,
        key=key
    )

def create_top_n_filter(max_value: int = 100, default_value: int = 10,
                       label: str = "Número de resultados",
                       key: str = "top_n") -> int:
    """
    Cria filtro para top N resultados
    
    Args:
        max_value: Valor máximo
        default_value: Valor padrão
        label: Rótulo do filtro
        key: Chave única para o widget
    
    Returns:
        Número selecionado
    """
    return st.selectbox(
        label,
        options=[5, 10, 20, 25, 50, 100],
        index=[5, 10, 20, 25, 50, 100].index(default_value) if default_value in [5, 10, 20, 25, 50, 100] else 1,
        key=key
    )

def create_sort_filter(columns: List[str], label: str = "Ordenar por",
                      key: str = "sort_by") -> str:
    """
    Cria filtro de ordenação
    
    Args:
        columns: Lista de colunas disponíveis
        label: Rótulo do filtro
        key: Chave única para o widget
    
    Returns:
        Coluna selecionada
    """
    display_columns = [col.replace('_', ' ').title() for col in columns]
    column_map = dict(zip(display_columns, columns))
    
    selected_display = st.selectbox(label, display_columns, key=key)
    return column_map[selected_display]

def create_confidence_filter(data: pd.DataFrame, confidence_column: str = 'confidence_score',
                           label: str = "Confiança mínima", key: str = "confidence_filter") -> float:
    """
    Cria filtro de confiança/score
    
    Args:
        data: DataFrame com os dados
        confidence_column: Nome da coluna de confiança
        label: Rótulo do filtro
        key: Chave única para o widget
    
    Returns:
        Valor de confiança mínima
    """
    if confidence_column not in data.columns:
        return 0.0
    
    try:
        min_conf = float(data[confidence_column].min())
        max_conf = float(data[confidence_column].max())
        
        return st.slider(
            label,
            min_value=min_conf,
            max_value=max_conf,
            value=min_conf,
            step=0.05,
            key=key
        )
    except:
        return 0.0

def create_period_aggregation_filter(label: str = "Período de agrupamento",
                                   key: str = "period_agg") -> str:
    """
    Cria filtro para agregação temporal
    
    Args:
        label: Rótulo do filtro
        key: Chave única para o widget
    
    Returns:
        Período selecionado
    """
    periods = ["Horário", "Diário", "Semanal", "Mensal"]
    return st.selectbox(label, periods, index=1, key=key)

def apply_filters(data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Aplica múltiplos filtros aos dados
    
    Args:
        data: DataFrame original
        filters: Dicionário com filtros aplicados
    
    Returns:
        DataFrame filtrado
    """
    filtered_data = data.copy()
    
    for filter_name, filter_value in filters.items():
        try:
            if filter_name.endswith('_date_range') and filter_value:
                # Filtro de data
                column = filter_name.replace('_date_range', '')
                if column in filtered_data.columns and len(filter_value) == 2:
                    filtered_data[column] = pd.to_datetime(filtered_data[column])
                    filtered_data = filtered_data[
                        (filtered_data[column].dt.date >= filter_value[0]) &
                        (filtered_data[column].dt.date <= filter_value[1])
                    ]
            
            elif filter_name.endswith('_categorical') and filter_value != 'Todos':
                # Filtro categórico
                column = filter_name.replace('_categorical', '')
                if column in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[column] == filter_value]
            
            elif filter_name.endswith('_multiselect') and filter_value:
                # Filtro múltipla seleção
                column = filter_name.replace('_multiselect', '')
                if column in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
            
            elif filter_name.endswith('_range') and filter_value:
                # Filtro numérico
                column = filter_name.replace('_range', '')
                if column in filtered_data.columns and len(filter_value) == 2:
                    filtered_data = filtered_data[
                        (filtered_data[column] >= filter_value[0]) &
                        (filtered_data[column] <= filter_value[1])
                    ]
            
            elif filter_name.endswith('_text_search') and filter_value:
                # Busca textual (em todas as colunas de texto)
                text_columns = filtered_data.select_dtypes(include=['object']).columns
                mask = pd.Series([False] * len(filtered_data))
                
                for col in text_columns:
                    mask |= filtered_data[col].astype(str).str.contains(
                        filter_value, case=False, na=False
                    )
                
                filtered_data = filtered_data[mask]
            
            elif filter_name.endswith('_confidence') and filter_value:
                # Filtro de confiança
                column = filter_name.replace('_confidence', '')
                if column in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[column] >= filter_value]
        
        except Exception as e:
            st.warning(f"Erro ao aplicar filtro {filter_name}: {e}")
            continue
    
    return filtered_data

def create_filter_summary(filters: Dict[str, Any]) -> str:
    """
    Cria resumo dos filtros aplicados
    
    Args:
        filters: Dicionário com filtros
    
    Returns:
        String com resumo
    """
    active_filters = []
    
    for filter_name, filter_value in filters.items():
        if filter_value is None:
            continue
        
        # Extrair nome limpo do filtro
        clean_name = filter_name.replace('_filter', '').replace('_', ' ').title()
        
        if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
            if filter_name.endswith('_date_range'):
                active_filters.append(f"{clean_name}: {filter_value[0]} a {filter_value[1]}")
            elif filter_name.endswith('_range'):
                active_filters.append(f"{clean_name}: {filter_value[0]:.2f} - {filter_value[1]:.2f}")
        
        elif isinstance(filter_value, list) and filter_value:
            active_filters.append(f"{clean_name}: {', '.join(map(str, filter_value))}")
        
        elif isinstance(filter_value, str) and filter_value and filter_value != 'Todos':
            active_filters.append(f"{clean_name}: {filter_value}")
        
        elif isinstance(filter_value, (int, float)) and filter_value > 0:
            active_filters.append(f"{clean_name}: {filter_value}")
    
    if active_filters:
        return "🔍 Filtros ativos: " + " | ".join(active_filters)
    else:
        return "🔍 Nenhum filtro aplicado"

def create_filter_sidebar(data: pd.DataFrame, available_filters: List[str]) -> Dict[str, Any]:
    """
    Cria barra lateral com filtros
    
    Args:
        data: DataFrame com os dados
        available_filters: Lista de filtros disponíveis
    
    Returns:
        Dicionário com valores dos filtros
    """
    filters = {}
    
    with st.sidebar:
        st.subheader("🔍 Filtros")
        
        for filter_type in available_filters:
            if filter_type == 'date_range' and ('date' in data.columns or 'timestamp' in data.columns):
                date_col = 'date' if 'date' in data.columns else 'timestamp'
                filters['date_range'] = create_date_range_filter(data, date_col)
            
            elif filter_type == 'political_category' and 'political_category' in data.columns:
                filters['political_category'] = create_categorical_filter(
                    data, 'political_category', 'Categoria Política'
                )
            
            elif filter_type == 'sentiment' and 'sentiment' in data.columns:
                filters['sentiment'] = create_categorical_filter(
                    data, 'sentiment', 'Sentimento'
                )
            
            elif filter_type == 'topic' and 'topic' in data.columns:
                filters['topic'] = create_categorical_filter(
                    data, 'topic', 'Tópico'
                )
            
            elif filter_type == 'confidence' and 'confidence_score' in data.columns:
                filters['confidence'] = create_confidence_filter(data)
            
            elif filter_type == 'intensity' and 'intensity' in data.columns:
                filters['intensity'] = create_numeric_range_filter(
                    data, 'intensity', 'Intensidade'
                )
            
            elif filter_type == 'text_search':
                filters['text_search'] = create_text_search_filter()
            
            elif filter_type == 'top_n':
                filters['top_n'] = create_top_n_filter()
    
    return filters

def create_quick_filters(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Cria filtros rápidos na área principal
    
    Args:
        data: DataFrame com os dados
    
    Returns:
        Dicionário com filtros
    """
    filters = {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'political_category' in data.columns:
            filters['political_category'] = create_categorical_filter(
                data, 'political_category', key='quick_political'
            )
    
    with col2:
        if 'sentiment' in data.columns:
            filters['sentiment'] = create_categorical_filter(
                data, 'sentiment', key='quick_sentiment'
            )
    
    with col3:
        if 'confidence_score' in data.columns:
            filters['confidence'] = create_confidence_filter(
                data, key='quick_confidence'
            )
    
    with col4:
        filters['top_n'] = create_top_n_filter(key='quick_top_n')
    
    return filters