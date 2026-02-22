#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_03.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_03_cross_dataset_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 03: Cross-Dataset Deduplication
    
    Eliminação de duplicatas entre TODOS os datasets com contador de frequência.
    Algoritmo: Agrupar por texto idêntico, manter registro mais antigo, 
    contar duplicatas com dupli_freq.
    
    Redução esperada: 40-50% (300k → 180k)
    """
    try:
        ctx.logger.info("🔄 STAGE 03: Cross-Dataset Deduplication")
        
        text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
        datetime_column = 'datetime' if 'datetime' in df.columns else df.columns[df.columns.str.contains('date|time', case=False)].tolist()[0] if any(df.columns.str.contains('date|time', case=False)) else None
        
        initial_count = len(df)
        ctx.logger.info(f"📊 Registros iniciais: {initial_count:,}")
        
        # Agrupar por texto idêntico
        grouping_columns = [text_column]
        
        # Preparar dados para agrupamento
        dedup_data = []
        
        for text, group in df.groupby(text_column):
            if pd.isna(text) or text.strip() == '':
                continue
                
            # Manter registro mais antigo (primeiro datetime)
            if datetime_column and datetime_column in group.columns:
                # Converter datetime para ordenação
                group_sorted = group.copy()
                if group_sorted[datetime_column].dtype == 'object':
                    try:
                        group_sorted['datetime_parsed'] = pd.to_datetime(group_sorted[datetime_column], 
                                                                       format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    except:
                        group_sorted['datetime_parsed'] = pd.to_datetime(group_sorted[datetime_column], errors='coerce')
                    
                    # Ordenar por datetime e pegar o mais antigo
                    oldest_record = group_sorted.sort_values('datetime_parsed').iloc[0]
                else:
                    oldest_record = group.iloc[0]
            else:
                oldest_record = group.iloc[0]
            
            # Contador de duplicatas
            dupli_freq = len(group)
            
            # Metadados de dispersão
            channels_found = []
            if 'channel' in group.columns:
                channels_found = group['channel'].dropna().unique().tolist()
            elif 'sender_id' in group.columns:
                channels_found = group['sender_id'].dropna().unique().tolist()
            
            # Período de ocorrência
            date_span_days = 0
            if datetime_column and datetime_column in group.columns:
                try:
                    dates = pd.to_datetime(group[datetime_column], errors='coerce').dropna()
                    if len(dates) > 1:
                        date_span_days = (dates.max() - dates.min()).days
                except:
                    pass
            
            # Criar registro deduplificado
            dedup_record = oldest_record.copy()
            dedup_record['dupli_freq'] = dupli_freq
            dedup_record['channels_found'] = len(channels_found)
            dedup_record['date_span_days'] = date_span_days
            
            dedup_data.append(dedup_record)
        
        # Criar DataFrame deduplificado
        if dedup_data:
            df_deduplicated = pd.DataFrame(dedup_data)
            df_deduplicated = df_deduplicated.reset_index(drop=True)
        else:
            df_deduplicated = df.copy()
            df_deduplicated['dupli_freq'] = 1
            df_deduplicated['channels_found'] = 0
            df_deduplicated['date_span_days'] = 0
        
        final_count = len(df_deduplicated)
        reduction_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
        
        # Estatísticas de deduplicação
        unique_texts = df_deduplicated['dupli_freq'].value_counts().sort_index()
        total_duplicates = df_deduplicated[df_deduplicated['dupli_freq'] > 1]['dupli_freq'].sum()
        
        ctx.logger.info(f"✅ Deduplicação concluída:")
        ctx.logger.info(f"   📉 {initial_count:,} → {final_count:,} registros")
        ctx.logger.info(f"   📊 Redução: {reduction_pct:.1f}%")
        ctx.logger.info(f"   🔄 Duplicatas processadas: {total_duplicates:,}")
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 3
        
        return df_deduplicated
        
    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 03: {e}")
        ctx.stats['processing_errors'] += 1
        # Em caso de erro, adicionar colunas padrão
        df['dupli_freq'] = 1
        df['channels_found'] = 0
        df['date_span_days'] = 0
        return df

