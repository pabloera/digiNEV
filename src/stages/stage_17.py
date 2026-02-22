#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_17.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_17_channel_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 17: Análise de canais/fontes.

    Classifica canais e fontes de informação.
    """
    try:
        ctx.logger.info("🔄 Stage 17: Análise de canais")
        
        # Análise de canais
        if 'channel' in df.columns:
            df['channel_type'] = df['channel'].apply(_classify_channel_type)
            
            channel_counts = df['channel'].value_counts()
            df['channel_activity'] = df['channel'].map(channel_counts)
            df['is_active_channel'] = df['channel_activity'] > df['channel_activity'].median()
        else:
            df['channel_type'] = 'unknown'
            df['channel_activity'] = 1
            df['is_active_channel'] = False
        
        # Análise de mídia
        if 'media_type' in df.columns:
            df['content_type'] = df['media_type'].fillna('text')
            df['has_media'] = df['media_type'].notna()
        else:
            df['content_type'] = 'text'
            df['has_media'] = False
        
        # Padrões de forwarding
        if 'is_fwrd' in df.columns:
            df['is_forwarded'] = df['is_fwrd'].fillna(False)
            forwarded_ratio = df['is_forwarded'].mean()
            df['forwarding_context'] = forwarded_ratio
        else:
            df['is_forwarded'] = False
            df['forwarding_context'] = 0.0
        
        # Influência do canal
        if 'sender' in df.columns and 'channel' in df.columns:
            sender_channel_counts = df.groupby(['sender', 'channel']).size()
            df['sender_channel_influence'] = df.apply(
                lambda row: sender_channel_counts.get((row['sender'], row['channel']), 0), axis=1
            )
        else:
            df['sender_channel_influence'] = 1
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 7
        
        ctx.logger.info(f"✅ Stage 17 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 17: {e}")
        ctx.stats['processing_errors'] += 1
        return df

# ==========================================
# HELPER METHODS FOR ANALYSIS STAGES
# (Integrado com lexico_unified_system.json: 956 termos, 9 macrotemas)
# ==========================================

