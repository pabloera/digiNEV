#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_01.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_01_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 01: Extração e identificação automática de features (Python puro).

    SEMPRE O PRIMEIRO STAGE - identifica colunas disponíveis e extrai features se necessário.
    """
    ctx.logger.info("🔍 STAGE 01: Feature Extraction")

    # Identificar coluna de texto principal
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Verificar se contém texto substancial
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > 20:  # Textos com mais de 20 caracteres em média
                    text_columns.append(col)

    # Selecionar melhor coluna de texto
    if not text_columns:
        raise ValueError("❌ Nenhuma coluna de texto encontrada")

    # Priorizar colunas comuns
    priority_columns = ['text', 'body', 'message', 'content', 'texto', 'mensagem']
    main_text_column = None

    for priority in priority_columns:
        if priority in text_columns:
            main_text_column = priority
            break

    if not main_text_column:
        main_text_column = text_columns[0]

    # Identificar coluna de timestamp (se disponível)
    timestamp_column = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
            timestamp_column = col
            break

    # === PADRONIZAÇÃO DE DATETIME ===
    if timestamp_column:
        df = _standardize_datetime_column(df, timestamp_column)
        # Após padronização, a coluna se chama 'datetime'
        timestamp_column = 'datetime'

    # DETECÇÃO AUTOMÁTICA DE FEATURES EXISTENTES
    features_detected = _detect_existing_features(df)

    # EXTRAÇÃO AUTOMÁTICA DE FEATURES (se não existem)
    df = _extract_missing_features(df, main_text_column, features_detected)

    # === CONTAR COLUNAS DE METADADOS ===
    # Metadados = todas as colunas exceto texto principal e datetime padronizado
    metadata_columns = []
    for col in df.columns:
        if col not in [main_text_column, 'datetime'] and not col.startswith(('emojis_', 'hashtags_', 'urls_', 'mentions_')):
            metadata_columns.append(col)

    # Adicionar features identificadas
    df['main_text_column'] = main_text_column
    df['timestamp_column'] = timestamp_column if timestamp_column else 'none'
    df['metadata_columns_count'] = len(metadata_columns)
    df['has_timestamp'] = timestamp_column is not None

    ctx.stats['stages_completed'] += 1
    ctx.stats['features_extracted'] += 4 + len(features_detected['extracted'])

    ctx.logger.info(f"✅ Features: text={main_text_column}, timestamp={timestamp_column}")
    ctx.logger.info(f"✅ Features detectadas: {list(features_detected['existing'].keys())}")
    ctx.logger.info(f"✅ Features extraídas: {features_detected['extracted']}")
    if timestamp_column:
        ctx.logger.info(f"📅 Datetime otimizado: coluna única 'datetime'")
    return df
def _standardize_datetime_column(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """
    Padronizar coluna de datetime para formato único DD/MM/AAAA HH:MM:SS.
    Remove coluna original e substitui por versão padronizada.
    
    Args:
        df: DataFrame com dados
        timestamp_column: Nome da coluna de timestamp identificada
        
    Returns:
        DataFrame com coluna datetime padronizada (substitui a original)
    """
    ctx.logger.info(f"📅 Padronizando datetime da coluna: {timestamp_column}")
    
    def parse_datetime(datetime_str):
        """Tentar múltiplos formatos de datetime."""
        if pd.isna(datetime_str):
            return None
            
        datetime_str = str(datetime_str).strip()
        
        # Formatos comuns para tentar
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S',      # 2019-07-02 01:10:00
            '%d/%m/%Y %H:%M:%S',      # 02/07/2019 01:10:00
            '%Y-%m-%d',               # 2019-07-02
            '%d/%m/%Y',               # 02/07/2019
            '%Y-%m-%d %H:%M',         # 2019-07-02 01:10
            '%d/%m/%Y %H:%M',         # 02/07/2019 01:10
            '%Y/%m/%d %H:%M:%S',      # 2019/07/02 01:10:00
            '%m/%d/%Y %H:%M:%S',      # 07/02/2019 01:10:00 (formato americano)
        ]
        
        for fmt in formats_to_try:
            try:
                parsed_dt = pd.to_datetime(datetime_str, format=fmt)
                # Converter para formato padrão brasileiro DD/MM/AAAA HH:MM:SS
                return parsed_dt.strftime('%d/%m/%Y %H:%M:%S')
            except (ValueError, TypeError):
                continue
                
        # Se nenhum formato funcionou, tentar parse genérico do pandas
        try:
            parsed_dt = pd.to_datetime(datetime_str, infer_datetime_format=True)
            return parsed_dt.strftime('%d/%m/%Y %H:%M:%S')
        except:
            return None
    
    # Aplicar padronização
    datetime_standardized = df[timestamp_column].apply(parse_datetime)
    
    # === SUBSTITUIR COLUNA ORIGINAL ===
    # Remover coluna original e usar nome 'datetime' para a versão padronizada
    df = df.drop(columns=[timestamp_column])
    df['datetime'] = datetime_standardized
    
    # Estatísticas de conversão
    valid_datetimes = df['datetime'].notna().sum()
    total_records = len(df)
    success_rate = (valid_datetimes / total_records) * 100
    
    ctx.logger.info(f"✅ Datetime padronizado e substituído: {valid_datetimes}/{total_records} ({success_rate:.1f}%) válidos")
    
    # Amostras do resultado
    sample_standardized = df['datetime'].dropna().head(3).tolist()
    
    ctx.logger.info(f"📋 Formato final:")
    for i, std in enumerate(sample_standardized):
        ctx.logger.info(f"   {i+1}. {std}")
    
    return df
def _detect_existing_features(df: pd.DataFrame) -> Dict:
    """
    Detecta features que já existem como colunas no DataFrame.
    """
    existing_features = {}

    # Features de interesse para detectar
    feature_patterns = {
        'hashtags': ['hashtag', 'hashtags', 'tags'],
        'urls': ['url', 'urls', 'links', 'link'],
        'mentions': ['mention', 'mentions', 'user_mentions', 'usuarios'],
        'emojis': ['emoji', 'emojis', 'emoticon'],
        'reply_count': ['reply', 'replies', 'respostas'],
        'retweet_count': ['retweet', 'retweets', 'rt_count'],
        'like_count': ['like', 'likes', 'curtidas', 'fav'],
        'user_info': ['user', 'username', 'author', 'usuario']
    }

    for feature_name, patterns in feature_patterns.items():
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern in col.lower()]
            if matching_cols:
                existing_features[feature_name] = matching_cols[0]
                break

    return {
        'existing': existing_features,
        'extracted': []
    }
def _extract_missing_features(df: pd.DataFrame, text_column: str, features_info: Dict) -> pd.DataFrame:
    """
    Extrai apenas features essenciais do texto principal.
    """
    extracted_features = []

    # Verificar se a coluna de texto existe
    if text_column not in df.columns:
        ctx.logger.error(f"❌ Coluna de texto '{text_column}' não encontrada no DataFrame")
        ctx.logger.error(f"Colunas disponíveis: {list(df.columns)}")
        # Usar primeira coluna disponível como fallback
        text_column = df.columns[0] if len(df.columns) > 0 else 'body'
        ctx.logger.warning(f"⚠️ Usando coluna '{text_column}' como fallback")

    # Só extrair se não existir coluna correspondente
    if 'hashtags' not in features_info['existing']:
        df['hashtags_extracted'] = df[text_column].astype(str).str.findall(r'#\w+')
        extracted_features.append('hashtags')

    if 'urls' not in features_info['existing']:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        df['urls_extracted'] = df[text_column].astype(str).str.findall(url_pattern)
        extracted_features.append('urls')

    if 'mentions' not in features_info['existing']:
        # Padrão para @mentions
        df['mentions_extracted'] = df[text_column].astype(str).str.findall(r'@\w+')
        extracted_features.append('mentions')

    if 'emojis' not in features_info['existing']:
        # Padrão básico para emojis (Unicode)
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]'
        df['emojis_extracted'] = df[text_column].astype(str).str.findall(emoji_pattern)
        extracted_features.append('emojis')

    # REMOVIDAS: has_interrogation, has_exclamation, has_caps_words, has_portuguese_words
    # Estas colunas não são necessárias para a análise

    features_info['extracted'] = extracted_features
    return df

