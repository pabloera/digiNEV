#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_16.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_16_event_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 16: Análise de contexto de eventos.

    Detecta contextos políticos e eventos relevantes.
    """
    try:
        ctx.logger.info("🔄 Stage 16: Análise de contexto de eventos")

        # FIX: preferir lemmatized_text (output do spaCy) para melhor matching de contextos
        if 'lemmatized_text' in df.columns:
            text_column = 'lemmatized_text'
        elif 'normalized_text' in df.columns:
            text_column = 'normalized_text'
        else:
            text_column = 'body'
        
        # Contextos políticos brasileiros
        df['political_context'] = df[text_column].apply(_detect_political_context)
        df['mentions_government'] = df[text_column].apply(_mentions_government)
        df['mentions_opposition'] = df[text_column].apply(_mentions_opposition)
        
        # Eventos específicos (eleições, manifestações, etc.)
        df['election_context'] = df[text_column].apply(_detect_election_context)
        df['protest_context'] = df[text_column].apply(_detect_protest_context)
        
        # Frame Analysis - Entman (1993), J Communication 43(4): 51-58
        frame_results = df[text_column].apply(_analyze_political_frames)
        df['frame_conflito'] = frame_results.apply(lambda x: x.get('conflito', 0.0))
        df['frame_responsabilizacao'] = frame_results.apply(lambda x: x.get('responsabilizacao', 0.0))
        df['frame_moralista'] = frame_results.apply(lambda x: x.get('moralista', 0.0))
        df['frame_economico'] = frame_results.apply(lambda x: x.get('economico', 0.0))

        # Análise temporal de eventos
        if 'datetime' in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]) if 'day_of_week' in df.columns else False
            df['is_business_hours'] = df['hour'].between(9, 17) if 'hour' in df.columns else False
        else:
            df['is_weekend'] = False
            df['is_business_hours'] = True

        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 11

        ctx.logger.info(f"✅ Stage 16 concluído: {len(df)} registros, 4 frames Entman extraídos")
        return df
        
    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 16: {e}")
        ctx.stats['processing_errors'] += 1
        return df

