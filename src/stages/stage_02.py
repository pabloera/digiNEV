#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_02.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
import unicodedata


def _stage_02_text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 02: Validação de Features + Limpeza de Texto.

    Para datasets com estrutura correta (datetime, body, url, hashtag, channel, etc):
    1. Validar features existentes contra coluna 'body' 
    2. Corrigir features incorretas/vazias
    3. Limpar body_cleaned (texto sem features)
    4. Aplicar normalização de texto
    
    USA: Detecção automática da estrutura do dataset
    """
    ctx.logger.info("🧹 STAGE 02: Feature Validation + Text Preprocessing")

    # === DETECTAR ESTRUTURA DO DATASET ===
    expected_columns = ['datetime', 'body', 'url', 'hashtag', 'channel', 'is_fwrd', 'mentions', 'sender', 'media_type', 'domain', 'body_cleaned']
    
    if all(col in df.columns for col in expected_columns[:5]):  # Verificar colunas essenciais
        ctx.logger.info("✅ Dataset estruturado detectado - validando features existentes")
        
        # === FASE 1: VALIDAÇÃO DE FEATURES EXISTENTES ===
        df = _extract_and_validate_features(df, 'body')
        
        # === FASE 2: LIMPEZA DE TEXTO (usar body como principal) ===
        main_text_col = 'body'
        
    else:
        ctx.logger.info("⚠️ Dataset não estruturado - usando coluna principal")
        
        # Obter nome da coluna principal de texto (armazenado no Stage 01)
        if 'main_text_column' in df.columns and len(df) > 0:
            main_text_col = df['main_text_column'].iloc[0]
            ctx.logger.info(f"🔍 Coluna principal identificada: {main_text_col}")
            
            # Verificar se a coluna existe realmente
            if main_text_col not in df.columns:
                ctx.logger.warning(f"⚠️ Coluna '{main_text_col}' não encontrada, buscando alternativa")
                # Buscar coluna de texto válida
                text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in ['main_text_column', 'timestamp_column']]
                if text_columns:
                    main_text_col = text_columns[0]
                    ctx.logger.info(f"✅ Usando coluna alternativa: {main_text_col}")
                else:
                    raise ValueError("❌ Nenhuma coluna de texto válida encontrada")
        else:
            # Fallback: buscar coluna de texto
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if text_columns:
                main_text_col = text_columns[0]
                ctx.logger.warning(f"⚠️ Usando primeira coluna de texto disponível: {main_text_col}")
            else:
                raise ValueError("❌ Nenhuma coluna de texto encontrada")
        
        # === FASE 1: EXTRAÇÃO DE FEATURES ===
        df = _extract_and_validate_features(df, main_text_col)
    
    # === FASE 2: NORMALIZAÇÃO DE TEXTO ===
    def clean_text(text):
        """Limpar texto usando Python puro."""
        if pd.isna(text):
            return ""

        text = str(text)

        # Normalizar unicode
        text = unicodedata.normalize('NFKD', text)

        # Remover caracteres especiais mas preservar acentos
        text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)

        # Normalizar espaços
        text = re.sub(r'\s+', ' ', text).strip()

        # Converter para lowercase
        text = text.lower()

        return text

    # Aplicar limpeza ao texto principal
    df['normalized_text'] = df[main_text_col].apply(clean_text)

    ctx.stats['stages_completed'] += 1
    ctx.stats['features_extracted'] += 2

    ctx.logger.info(f"✅ Stage 02 concluído: {df['normalized_text'].str.len().mean():.1f} chars média")
    return df


def _extract_and_validate_features(ctx, df: pd.DataFrame, main_text_col: str) -> pd.DataFrame:
    """
    Validar features existentes contra coluna 'body' e corrigir se necessário.
    
    Dataset já tem: datetime, body, url, hashtag, channel, is_fwrd, mentions, sender, media_type, domain, body_cleaned
    """
    ctx.logger.info("🔍 Validando features existentes contra coluna 'body'...")
    
    # === VERIFICAR SE DATASET TEM ESTRUTURA CORRETA ===
    expected_columns = ['datetime', 'body', 'url', 'hashtag', 'channel', 'is_fwrd', 'mentions', 'sender', 'media_type', 'domain', 'body_cleaned']
    
    # Se o dataset tem as colunas corretas, usar body como texto principal
    if all(col in df.columns for col in expected_columns[:5]):  # Verificar colunas essenciais
        ctx.logger.info("✅ Dataset com estrutura correta detectado")
        
        # === REMOVER BODY_CLEANED (duplicação desnecessária) ===
        if 'body_cleaned' in df.columns:
            df = df.drop(columns=['body_cleaned'])
            ctx.logger.info("🗑️ body_cleaned removido (duplicação desnecessária)")
        
        # === VALIDAR FEATURES CONTRA BODY ===
        corrections_made = 0
        
        # Validar URL
        if 'url' in df.columns and 'body' in df.columns:
            corrections_made += _validate_feature_against_body(df, 'url', 'body', [r'https?://\S+', r'www\.\S+'])
        
        # Validar Hashtags
        if 'hashtag' in df.columns and 'body' in df.columns:
            corrections_made += _validate_feature_against_body(df, 'hashtag', 'body', [r'#\w+'])
        
        # Validar Mentions
        if 'mentions' in df.columns and 'body' in df.columns:
            corrections_made += _validate_feature_against_body(df, 'mentions', 'body', [r'@\w+'])
        
        ctx.logger.info(f"✅ Validação concluída: {corrections_made} correções aplicadas")
        
    else:
        # === DATASET SEM ESTRUTURA PADRÃO - EXTRAIR TUDO ===
        ctx.logger.info("⚠️ Dataset sem estrutura padrão - extraindo features do texto principal")
        df = _extract_features_from_text(df, main_text_col)
    
    return df


def _validate_feature_against_body(ctx, df: pd.DataFrame, feature_col: str, body_col: str, patterns: list) -> int:
    """Validar feature específica contra body."""
    corrections = 0
    
    for idx, row in df.iterrows():
        body_text = str(row[body_col]) if pd.notna(row[body_col]) else ""
        existing_feature = row[feature_col] if pd.notna(row[feature_col]) else ""
        
        # Extrair feature do body
        extracted_features = []
        for pattern in patterns:
            matches = re.findall(pattern, body_text, re.IGNORECASE)
            extracted_features.extend(matches)
        
        # Se encontrou features no body mas coluna está vazia, corrigir
        if extracted_features and not existing_feature:
            if len(extracted_features) == 1:
                df.at[idx, feature_col] = extracted_features[0]
            else:
                df.at[idx, feature_col] = ';'.join(extracted_features)  # Múltiplas features
            corrections += 1
    
    if corrections > 0:
        ctx.logger.info(f"🔧 {feature_col}: {corrections} correções aplicadas")
    
    return corrections


def _clean_body_text(ctx, df: pd.DataFrame):
    """
    REMOVIDO: body_cleaned não é mais necessário.
    O texto limpo é gerado como 'normalized_text' no Stage 02.
    """
    # Não fazer nada - body_cleaned removido para evitar duplicação
    ctx.logger.info("🗑️ body_cleaned removido (duplicação desnecessária)")
    pass


def _extract_features_from_text(ctx, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Extrair features de dataset sem estrutura padrão (fallback)."""
    
    # Features essenciais para extrair
    feature_patterns = {
        'urls': [r'https?://\S+', r'www\.\S+'],
        'hashtags': [r'#\w+'],
        'mentions': [r'@\w+'],
        'channel_name': []  # Usar valor padrão
    }
    
    for feature_name, patterns in feature_patterns.items():
        if patterns:
            def extract_feature(text):
                if pd.isna(text):
                    return []
                
                extracted = []
                for pattern in patterns:
                    matches = re.findall(pattern, str(text), re.IGNORECASE)
                    extracted.extend(matches)
                return list(set(extracted)) if extracted else []
            
            df[feature_name] = df[text_col].apply(extract_feature)
        else:
            df[feature_name] = "unknown_channel"
    
    ctx.logger.info("🆕 Features extraídas de dataset sem estrutura padrão")
    return df


def _find_existing_feature_column(df: pd.DataFrame, possible_names: list) -> str:
    """Encontrar coluna existente para uma feature."""
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    return None


def _validate_and_correct_feature(ctx, df: pd.DataFrame, existing_col: str, feature_name: str, patterns: list, text_col: str) -> pd.DataFrame:
    """Validar e corrigir feature existente."""
    if not patterns:  # Channel name não tem padrão regex
        ctx.logger.info(f"✅ Feature {existing_col} mantida (sem validação regex)")
        return df
    
    # Extrair valores corretos do texto
    def extract_correct_values(text):
        if pd.isna(text):
            return []
        
        text = str(text)
        extracted = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted.extend(matches)
        return list(set(extracted))  # Remover duplicatas
    
    # Validar contra texto original
    df['_temp_extracted'] = df[text_col].apply(extract_correct_values)
    
    # Comparar e corrigir se necessário
    corrections_made = 0
    
    def validate_and_fix(row):
        nonlocal corrections_made
        existing_value = row[existing_col]
        correct_value = row['_temp_extracted']
        
        # Se valor existente está vazio ou incorreto, corrigir
        if pd.isna(existing_value) or existing_value == [] or existing_value == '':
            if correct_value:
                corrections_made += 1
                return correct_value
        
        return existing_value
    
    df[existing_col] = df.apply(validate_and_fix, axis=1)
    df = df.drop('_temp_extracted', axis=1)
    
    if corrections_made > 0:
        ctx.logger.info(f"🔧 Feature {existing_col}: {corrections_made} correções aplicadas")
    else:
        ctx.logger.info(f"✅ Feature {existing_col}: validação OK, sem correções necessárias")
    
    return df


def _extract_new_feature(ctx, df: pd.DataFrame, feature_name: str, patterns: list, text_col: str) -> pd.DataFrame:
    """Extrair nova feature do texto."""
    def extract_feature(text):
        if pd.isna(text):
            return []
        
        text = str(text)
        extracted = []
        
        if patterns:  # Features com padrão regex
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted.extend(matches)
        else:  # Channel name - tentar extrair de metadados ou usar valor padrão
            return "unknown_channel"
        
        return list(set(extracted)) if extracted else []
    
    # Extrair feature
    df[feature_name] = df[text_col].apply(extract_feature)
    
    # Estatísticas
    non_empty = df[feature_name].apply(lambda x: len(x) > 0 if isinstance(x, list) else bool(x)).sum()
    total = len(df)
    
    ctx.logger.info(f"📊 Feature {feature_name}: {non_empty}/{total} registros ({non_empty/total*100:.1f}%)")
    
    return df

