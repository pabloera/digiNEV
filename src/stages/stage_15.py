#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_15.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse


def _stage_15_domain_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 15: Análise de domínios.

    Analisa domínios e URLs para identificar padrões de mídia.
    """
    try:
        ctx.logger.info("🔄 Stage 15: Análise de domínios")
        
        # Análise de domínios com trust score (Page et al. 1999, adaptado)
        if 'domain' in df.columns:
            df['domain_type'] = df['domain'].apply(_classify_domain_type)
            df['domain_trust_score'] = df['domain'].apply(_calculate_domain_trust_score)

            domain_counts = df['domain'].value_counts()
            df['domain_frequency'] = df['domain'].map(domain_counts)

            # Mídia mainstream vs alternativa (baseado em domain_type classificado)
            mainstream_types = ['mainstream_news', 'government']
            df['is_mainstream_media'] = df['domain_type'].isin(mainstream_types)
        else:
            df['domain_type'] = 'unknown'
            df['domain_trust_score'] = 0.0
            df['domain_frequency'] = 0
            df['is_mainstream_media'] = False
        
        # Análise de URLs
        if 'urls_extracted' in df.columns:
            df['url_count'] = df['urls_extracted'].apply(
                lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else (1 if x else 0)
            )
            df['has_external_links'] = df['url_count'] > 0
        else:
            df['url_count'] = 0
            df['has_external_links'] = False
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 5
        
        ctx.logger.info(f"✅ Stage 15 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 15: {e}")
        ctx.stats['processing_errors'] += 1
        return df


