#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_13.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_13_temporal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 13: Análise temporal.

    Extrai padrões temporais dos timestamps.
    """
    try:
        ctx.logger.info("🔄 Stage 13: Análise temporal")
        
        if 'datetime' not in df.columns:
            ctx.logger.warning("⚠️ datetime não encontrado")
            df['hour'] = 12
            df['day_of_week'] = 1
            df['month'] = 1
        else:
            # Converter datetime para análise temporal
            try:
                datetime_series = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                
                df['hour'] = datetime_series.dt.hour
                df['day_of_week'] = datetime_series.dt.dayofweek
                df['month'] = datetime_series.dt.month
                df['year'] = datetime_series.dt.year
                df['day_of_year'] = datetime_series.dt.dayofyear
                
            except Exception as e:
                ctx.logger.warning(f"⚠️ Erro conversão datetime: {e}")
                df['hour'] = 12
                df['day_of_week'] = 1
                df['month'] = 1
                df['year'] = 2020
                df['day_of_year'] = 1
        
        # Burst Detection - Kleinberg (2003), KDD
        # Detecta dias com volume anormal de mensagens
        df['is_burst_day'] = False
        if 'datetime' in df.columns:
            try:
                dt_series = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                dates = dt_series.dt.date
                daily_counts = dates.value_counts()
                if len(daily_counts) >= 3:
                    mean_count = daily_counts.mean()
                    std_count = daily_counts.std()
                    burst_threshold = mean_count + 2 * std_count  # 2 desvios padrão
                    burst_dates = daily_counts[daily_counts > burst_threshold].index.tolist()
                    df['is_burst_day'] = dates.isin(burst_dates)
                    if burst_dates:
                        ctx.logger.info(f"📈 Burst detection: {len(burst_dates)} dias com volume anormal")
            except Exception as e:
                ctx.logger.warning(f"⚠️ Burst detection: {e}")

        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 6

        ctx.logger.info(f"✅ Stage 13 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 13: {e}")
        ctx.stats['processing_errors'] += 1
        return df

