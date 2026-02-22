#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_14.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_14_network_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 14: Análise de rede (coordenação e padrões).

    Detecta padrões de coordenação e comportamento de rede.
    """
    try:
        ctx.logger.info("🔄 Stage 14: Análise de rede")
        
        # Análise de coordenação básica
        if 'sender' in df.columns:
            sender_counts = df['sender'].value_counts()
            df['sender_frequency'] = df['sender'].map(sender_counts)
            df['is_frequent_sender'] = df['sender_frequency'] > df['sender_frequency'].median()
        else:
            df['sender_frequency'] = 1
            df['is_frequent_sender'] = False
        
        # Análise de URLs compartilhadas
        if 'urls_extracted' in df.columns:
            # URLs mais compartilhadas
            all_urls = []
            for urls in df['urls_extracted'].fillna('[]'):
                if isinstance(urls, str):
                    try:
                        url_list = eval(urls) if urls.startswith('[') else [urls]
                        all_urls.extend(url_list)
                    except:
                        pass
            
            url_counts = pd.Series(all_urls).value_counts()
            df['shared_url_frequency'] = df['urls_extracted'].apply(
                lambda x: max([url_counts.get(url, 0) for url in (eval(x) if isinstance(x, str) and x.startswith('[') else [])], default=0)
            )
        else:
            df['shared_url_frequency'] = 0
        
        # Coordenação temporal (mensagens em horários similares)
        if 'hour' in df.columns:
            hour_counts = df['hour'].value_counts()
            df['temporal_coordination'] = df['hour'].map(hour_counts) / len(df)
        else:
            df['temporal_coordination'] = 0.0
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 4
        
        ctx.logger.info(f"✅ Stage 14 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 14: {e}")
        ctx.stats['processing_errors'] += 1
        return df

