#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_04.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_04_statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 04: Statistical Analysis
    
    Comparar início do dataset com o dataset reduzido.
    Gerar estatísticas para classificação e gráficos.
    
    Processamentos:
    - Contagem de dados antes e depois
    - Proporção de duplicadas
    - Proporção de hashtags
    - Detecção de repetições excessivas para tabela com 10 principais casos
    """
    try:
        ctx.logger.info("📊 STAGE 04: Statistical Analysis")
        
        text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
        
        # === ANÁLISE DE DUPLICAÇÃO ===
        total_registros = len(df)
        registros_unicos = len(df[df['dupli_freq'] == 1])
        registros_duplicados = total_registros - registros_unicos
        
        duplicacao_pct = (registros_duplicados / total_registros * 100) if total_registros > 0 else 0
        
        # === ANÁLISE DE HASHTAGS ===
        # FIX: usar coluna 'hashtags_extracted' (Stage 01) ou 'body' (# removido de normalized_text)
        has_hashtags = 0
        if 'hashtags_extracted' in df.columns:
            has_hashtags = df['hashtags_extracted'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else bool(x)
            ).sum()
        elif 'body' in df.columns:
            has_hashtags = df['body'].str.contains('#', na=False).sum()
        elif text_column in df.columns:
            has_hashtags = df[text_column].str.contains('#', na=False).sum()
        
        hashtag_pct = (has_hashtags / total_registros * 100) if total_registros > 0 else 0
        
        # === TOP 10 REPETIÇÕES EXCESSIVAS ===
        top_duplicates = df[df['dupli_freq'] > 1].nlargest(10, 'dupli_freq')[
            [text_column, 'dupli_freq', 'channels_found', 'date_span_days']
        ].to_dict('records')
        
        # === ESTATÍSTICAS BÁSICAS DE TEXTO ===
        if text_column in df.columns:
            char_counts = df[text_column].str.len().fillna(0)
            word_counts = df[text_column].str.split().str.len().fillna(0)
            
            df['char_count'] = char_counts
            df['word_count'] = word_counts
            
            avg_chars = char_counts.mean()
            avg_words = word_counts.mean()
        else:
            avg_chars = 0
            avg_words = 0
            df['char_count'] = 0
            df['word_count'] = 0
        
        # === PROPORÇÕES DE QUALIDADE ===
        # FIX: emoji_ratio e caps_ratio devem usar 'body' (texto cru) — normalized_text
        # é lowercase e sem emojis, o que faz essas métricas retornarem sempre 0.0
        raw_col = 'body' if 'body' in df.columns else text_column
        if raw_col in df.columns:
            df['emoji_ratio'] = df[raw_col].apply(_calculate_emoji_ratio)
            df['caps_ratio'] = df[raw_col].apply(_calculate_caps_ratio)
            df['repetition_ratio'] = df[raw_col].apply(_calculate_repetition_ratio)

            # Detecção de idioma básica (pode usar normalized_text — lowercase ok)
            df['likely_portuguese'] = df[text_column].apply(_detect_portuguese) if text_column in df.columns else True
        else:
            df['emoji_ratio'] = 0.0
            df['caps_ratio'] = 0.0
            df['repetition_ratio'] = 0.0
            df['likely_portuguese'] = True
        
        # === CONSOLIDAÇÃO DE ESTATÍSTICAS ===
        # Consolidar estatísticas globais em objeto summary
        summary_stats = {
            'total_dataset_size': total_registros,
            'unique_texts_count': registros_unicos,
            'duplication_percentage': round(duplicacao_pct, 2),
            'hashtag_percentage': round(hashtag_pct, 2),
            'avg_chars_per_text': round(avg_chars, 1),
            'avg_words_per_text': round(avg_words, 1)
        }

        # Salvar no contexto para acesso posterior
        ctx.global_stats = summary_stats
        
        # Log das estatísticas
        ctx.logger.info(f"✅ Análise estatística concluída:")
        ctx.logger.info(f"   📊 Total de registros: {total_registros:,}")
        ctx.logger.info(f"   🔄 Duplicação: {duplicacao_pct:.1f}%")
        ctx.logger.info(f"   # Hashtags: {hashtag_pct:.1f}%")
        ctx.logger.info(f"   📝 Média: {avg_words:.1f} palavras, {avg_chars:.0f} chars")
        
        if top_duplicates:
            ctx.logger.info(f"   🔝 Maior repetição: {top_duplicates[0]['dupli_freq']} ocorrências")
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 11
        
        return df
        
    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 04: {e}")
        ctx.stats['processing_errors'] += 1
        return df

