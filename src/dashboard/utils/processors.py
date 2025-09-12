"""
Processadores de dados para o dashboard digiNEV
Funções utilitárias para processamento e transformação de dados
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re

def clean_text_data(data: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Limpa dados de texto
    
    Args:
        data: DataFrame com os dados
        text_columns: Colunas de texto para limpar
    
    Returns:
        DataFrame com texto limpo
    """
    cleaned_data = data.copy()
    
    for column in text_columns:
        if column in cleaned_data.columns:
            # Remover caracteres especiais e normalizar
            cleaned_data[column] = cleaned_data[column].astype(str)
            cleaned_data[column] = cleaned_data[column].str.strip()
            cleaned_data[column] = cleaned_data[column].str.replace(r'\s+', ' ', regex=True)
            
            # Remover valores vazios
            cleaned_data[column] = cleaned_data[column].replace(['', 'nan', 'None'], pd.NA)
    
    return cleaned_data

def normalize_scores(data: pd.DataFrame, score_columns: List[str], 
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normaliza colunas de scores
    
    Args:
        data: DataFrame com os dados
        score_columns: Colunas de scores para normalizar
        method: Método de normalização ('minmax', 'zscore', 'robust')
    
    Returns:
        DataFrame com scores normalizados
    """
    normalized_data = data.copy()
    
    for column in score_columns:
        if column in normalized_data.columns:
            values = normalized_data[column].dropna()
            
            if len(values) == 0:
                continue
            
            if method == 'minmax':
                min_val = values.min()
                max_val = values.max()
                if max_val != min_val:
                    normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = values.mean()
                std_val = values.std()
                if std_val != 0:
                    normalized_data[column] = (normalized_data[column] - mean_val) / std_val
            
            elif method == 'robust':
                median_val = values.median()
                mad_val = (values - median_val).abs().median()
                if mad_val != 0:
                    normalized_data[column] = (normalized_data[column] - median_val) / mad_val
    
    return normalized_data

def aggregate_temporal_data(data: pd.DataFrame, date_column: str, 
                          value_columns: List[str], frequency: str = 'D') -> pd.DataFrame:
    """
    Agrega dados temporais
    
    Args:
        data: DataFrame com os dados
        date_column: Coluna de data
        value_columns: Colunas de valores para agregar
        frequency: Frequência de agregação ('H', 'D', 'W', 'M')
    
    Returns:
        DataFrame agregado
    """
    if date_column not in data.columns:
        return pd.DataFrame()
    
    # Converter para datetime
    temp_data = data.copy()
    temp_data[date_column] = pd.to_datetime(temp_data[date_column])
    
    # Definir função de agregação baseada no tipo de dados
    agg_funcs = {}
    for col in value_columns:
        if col in temp_data.columns:
            if temp_data[col].dtype in ['int64', 'float64']:
                agg_funcs[col] = ['mean', 'sum', 'count']
            else:
                agg_funcs[col] = 'count'
    
    if not agg_funcs:
        return pd.DataFrame()
    
    # Agregar por período
    aggregated = temp_data.groupby(pd.Grouper(key=date_column, freq=frequency)).agg(agg_funcs)
    
    # Flattened column names
    aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                         for col in aggregated.columns]
    
    return aggregated.reset_index()

def calculate_trend_metrics(data: pd.DataFrame, value_column: str, 
                          periods: int = 7) -> Dict[str, float]:
    """
    Calcula métricas de tendência
    
    Args:
        data: DataFrame com dados ordenados temporalmente
        value_column: Coluna de valores
        periods: Número de períodos para análise
    
    Returns:
        Dicionário com métricas de tendência
    """
    if value_column not in data.columns or len(data) < periods:
        return {}
    
    values = data[value_column].dropna()
    
    if len(values) < periods:
        return {}
    
    # Tendência linear
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    trend_slope = coeffs[0]
    
    # Variação percentual
    recent_values = values.tail(periods)
    older_values = values.head(periods)
    
    recent_mean = recent_values.mean()
    older_mean = older_values.mean()
    
    pct_change = ((recent_mean - older_mean) / older_mean * 100) if older_mean != 0 else 0
    
    # Volatilidade
    volatility = values.std()
    
    # Momentum (aceleração da tendência)
    if len(values) >= periods * 2:
        mid_point = len(values) // 2
        first_half_trend = np.polyfit(x[:mid_point], values[:mid_point], 1)[0]
        second_half_trend = np.polyfit(x[mid_point:], values[mid_point:], 1)[0]
        momentum = second_half_trend - first_half_trend
    else:
        momentum = 0
    
    return {
        'trend_slope': trend_slope,
        'percent_change': pct_change,
        'volatility': volatility,
        'momentum': momentum,
        'direction': 'crescente' if trend_slope > 0 else 'decrescente' if trend_slope < 0 else 'estável'
    }

def detect_outliers(data: pd.DataFrame, columns: List[str], 
                   method: str = 'iqr') -> pd.DataFrame:
    """
    Detecta outliers nos dados
    
    Args:
        data: DataFrame com os dados
        columns: Colunas para análise de outliers
        method: Método de detecção ('iqr', 'zscore', 'isolation')
    
    Returns:
        DataFrame com coluna is_outlier adicionada
    """
    result_data = data.copy()
    outlier_flags = pd.Series([False] * len(result_data))
    
    for column in columns:
        if column not in result_data.columns:
            continue
        
        values = result_data[column].dropna()
        
        if len(values) == 0:
            continue
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = (result_data[column] < lower_bound) | (result_data[column] > upper_bound)
        
        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val != 0:
                z_scores = np.abs((result_data[column] - mean_val) / std_val)
                column_outliers = z_scores > 3
            else:
                column_outliers = pd.Series([False] * len(result_data))
        
        else:  # isolation forest would require sklearn
            # Fallback to IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = (result_data[column] < lower_bound) | (result_data[column] > upper_bound)
        
        outlier_flags |= column_outliers.fillna(False)
    
    result_data['is_outlier'] = outlier_flags
    return result_data

def create_categorical_summary(data: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict]:
    """
    Cria resumo de variáveis categóricas
    
    Args:
        data: DataFrame com os dados
        categorical_columns: Colunas categóricas para analisar
    
    Returns:
        Dicionário com resumos das variáveis categóricas
    """
    summary = {}
    
    for column in categorical_columns:
        if column not in data.columns:
            continue
        
        values = data[column].dropna()
        value_counts = values.value_counts()
        
        summary[column] = {
            'total_values': len(values),
            'unique_values': len(value_counts),
            'most_frequent': value_counts.index[0] if not value_counts.empty else None,
            'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'frequency_distribution': value_counts.to_dict(),
            'entropy': calculate_entropy(value_counts),
            'concentration_ratio': value_counts.iloc[0] / len(values) if not value_counts.empty and len(values) > 0 else 0
        }
    
    return summary

def calculate_entropy(value_counts: pd.Series) -> float:
    """
    Calcula entropia de uma distribuição
    
    Args:
        value_counts: Série com contagens de valores
    
    Returns:
        Valor de entropia
    """
    if len(value_counts) == 0:
        return 0
    
    probabilities = value_counts / value_counts.sum()
    probabilities = probabilities[probabilities > 0]  # Remover zeros
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def filter_by_confidence(data: pd.DataFrame, confidence_column: str, 
                        threshold: float = 0.7) -> pd.DataFrame:
    """
    Filtra dados por nível de confiança
    
    Args:
        data: DataFrame com os dados
        confidence_column: Coluna de confiança
        threshold: Threshold mínimo de confiança
    
    Returns:
        DataFrame filtrado
    """
    if confidence_column not in data.columns:
        return data
    
    return data[data[confidence_column] >= threshold].copy()

def create_time_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Cria features temporais
    
    Args:
        data: DataFrame com os dados
        date_column: Coluna de data
    
    Returns:
        DataFrame com features temporais adicionadas
    """
    if date_column not in data.columns:
        return data
    
    result_data = data.copy()
    result_data[date_column] = pd.to_datetime(result_data[date_column])
    
    # Features básicas
    result_data[f'{date_column}_year'] = result_data[date_column].dt.year
    result_data[f'{date_column}_month'] = result_data[date_column].dt.month
    result_data[f'{date_column}_day'] = result_data[date_column].dt.day
    result_data[f'{date_column}_weekday'] = result_data[date_column].dt.dayofweek
    result_data[f'{date_column}_hour'] = result_data[date_column].dt.hour
    
    # Features derivadas
    result_data[f'{date_column}_is_weekend'] = result_data[f'{date_column}_weekday'].isin([5, 6])
    result_data[f'{date_column}_quarter'] = result_data[date_column].dt.quarter
    result_data[f'{date_column}_week_of_year'] = result_data[date_column].dt.isocalendar().week
    
    # Período do dia
    def get_period_of_day(hour):
        if pd.isna(hour):
            return 'unknown'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    
    result_data[f'{date_column}_period'] = result_data[f'{date_column}_hour'].apply(get_period_of_day)
    
    return result_data

def calculate_correlation_matrix(data: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Calcula matriz de correlação com tratamento de valores ausentes
    
    Args:
        data: DataFrame com os dados
        numeric_columns: Colunas numéricas para correlação
    
    Returns:
        Matriz de correlação
    """
    available_columns = [col for col in numeric_columns if col in data.columns]
    
    if len(available_columns) < 2:
        return pd.DataFrame()
    
    correlation_data = data[available_columns].select_dtypes(include=[np.number])
    
    if correlation_data.empty:
        return pd.DataFrame()
    
    # Remover colunas com variância zero
    non_constant_columns = []
    for col in correlation_data.columns:
        if correlation_data[col].nunique() > 1:
            non_constant_columns.append(col)
    
    if len(non_constant_columns) < 2:
        return pd.DataFrame()
    
    correlation_matrix = correlation_data[non_constant_columns].corr()
    return correlation_matrix

def prepare_data_for_visualization(data: pd.DataFrame, max_categories: int = 20) -> pd.DataFrame:
    """
    Prepara dados para visualização agrupando categorias raras
    
    Args:
        data: DataFrame com os dados
        max_categories: Número máximo de categorias por variável
    
    Returns:
        DataFrame preparado para visualização
    """
    viz_data = data.copy()
    
    # Processar colunas categóricas
    categorical_columns = viz_data.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        value_counts = viz_data[column].value_counts()
        
        if len(value_counts) > max_categories:
            # Manter top N-1 categorias e agrupar o resto em "Outros"
            top_categories = value_counts.head(max_categories - 1).index
            viz_data[column] = viz_data[column].apply(
                lambda x: x if x in top_categories else 'Outros'
            )
    
    return viz_data

def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida qualidade dos dados
    
    Args:
        data: DataFrame para validar
    
    Returns:
        Dicionário com métricas de qualidade
    """
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_data_pct': (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100).round(2),
        'duplicate_rows': data.duplicated().sum(),
        'duplicate_rows_pct': (data.duplicated().sum() / len(data) * 100).round(2),
        'column_quality': {}
    }
    
    for column in data.columns:
        col_quality = {
            'dtype': str(data[column].dtype),
            'missing_pct': (data[column].isnull().sum() / len(data) * 100).round(2),
            'unique_values': data[column].nunique(),
            'uniqueness_pct': (data[column].nunique() / data[column].notnull().sum() * 100).round(2) if data[column].notnull().sum() > 0 else 0
        }
        
        # Score de qualidade da coluna
        completeness = 1 - (col_quality['missing_pct'] / 100)
        uniqueness = min(col_quality['uniqueness_pct'] / 100, 1.0)
        
        col_quality['quality_score'] = (completeness * 0.7 + uniqueness * 0.3).round(3)
        
        quality_report['column_quality'][column] = col_quality
    
    # Score geral de qualidade
    avg_quality = np.mean([col['quality_score'] for col in quality_report['column_quality'].values()])
    quality_report['overall_quality_score'] = round(avg_quality, 3)
    
    return quality_report