#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_12.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_12_semantic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 12: Análise semântica.
    
    Análise semântica e de sentimento dos textos.
    """
    try:
        ctx.logger.info("🔄 Stage 12: Análise semântica")
        
        text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
        
        # Análise de sentimento básica
        df['sentiment_polarity'] = df[text_column].apply(_calculate_sentiment_polarity)
        df['sentiment_label'] = df['sentiment_polarity'].apply(
            lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
        )
        
        # Análise de emoções básicas (usar body original para detectar !, ?, CAPS)
        raw_col = 'body' if 'body' in df.columns else text_column
        df['emotion_intensity'] = df.apply(
            lambda row: _calculate_emotion_intensity(
                str(row.get(text_column, '')),
                raw_text=str(row.get(raw_col, ''))
            ), axis=1
        )
        df['has_aggressive_language'] = df[text_column].apply(_detect_aggressive_language)
        
        # Complexidade semântica
        # FIX: usar 'spacy_tokens' (output real do Stage 07) em vez de 'tokens' (inexistente)
        if 'spacy_tokens' in df.columns:
            df['semantic_diversity'] = df['spacy_tokens'].apply(
                lambda x: len(set(x)) / len(x) if isinstance(x, list) and len(x) > 0 else 0
            )
        else:
            df['semantic_diversity'] = df[text_column].apply(
                lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
            )
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 5
        
        ctx.logger.info(f"✅ Stage 12 concluído: {len(df)} registros processados")
        return df
        
    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 12: {e}")
        ctx.stats['processing_errors'] += 1
        return df

