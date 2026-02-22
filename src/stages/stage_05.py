#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_05.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _stage_05_content_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 05: Content Quality Filter
    
    Filtrar conteúdo por qualidade e completude.
    Input: Dados deduplificados
    Output: Apenas conteúdo de qualidade
    
    Filtros:
    - Comprimento: < 10 chars ou > 2000 chars
    - Qualidade: emoji_ratio > 70%, caps_ratio > 80%, repetition_ratio > 50%
    - Idioma: Manter apenas likely_portuguese = True
    
    Redução esperada: 15-25% (180k → 135k)
    """
    try:
        ctx.logger.info("🎯 STAGE 05: Content Quality Filter")
        
        text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
        initial_count = len(df)
        
        # === FILTROS DE COMPRIMENTO ===
        # Muito curto: < 10 chars (só emoji/URL)
        length_filter = (df['char_count'] >= 10) & (df['char_count'] <= 2000)
        
        # === FILTROS DE QUALIDADE ===
        # emoji_ratio > 70% = ruído
        emoji_filter = df['emoji_ratio'] <= 0.70
        
        # caps_ratio > 80% = spam  
        caps_filter = df['caps_ratio'] <= 0.80
        
        # repetition_ratio > 50% = baixa qualidade
        repetition_filter = df['repetition_ratio'] <= 0.50
        
        # === FILTROS DE IDIOMA ===
        # Manter apenas likely_portuguese = True
        language_filter = df['likely_portuguese'] == True
        
        # === APLICAR TODOS OS FILTROS ===
        quality_mask = length_filter & emoji_filter & caps_filter & repetition_filter & language_filter
        
        # === GERAR COLUNAS DE QUALIDADE ===
        # Contar problemas por tipo para logging
        problems = {
            'length_issue': (~length_filter).sum(),
            'excessive_emojis': (~emoji_filter).sum(),
            'excessive_caps': (~caps_filter).sum(),
            'excessive_repetition': (~repetition_filter).sum(),
            'non_portuguese': (~language_filter).sum()
        }
        
        # Content quality score (0-100)
        quality_components = [
            length_filter.astype(int) * 20,  # 20 pontos para comprimento adequado
            emoji_filter.astype(int) * 20,   # 20 pontos para emojis adequados
            caps_filter.astype(int) * 20,    # 20 pontos para caps adequados
            repetition_filter.astype(int) * 20, # 20 pontos para repetição adequada
            language_filter.astype(int) * 20    # 20 pontos para português
        ]
        
        df['content_quality_score'] = sum(quality_components)
        
        # === APLICAR FILTRO ===
        df_filtered = df[quality_mask].copy().reset_index(drop=True)
        
        final_count = len(df_filtered)
        reduction_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
        
        # === ESTATÍSTICAS DOS FILTROS ===
        avg_quality_score = df_filtered['content_quality_score'].mean()

        ctx.logger.info(f"✅ Filtro de qualidade aplicado:")
        ctx.logger.info(f"   📉 {initial_count:,} → {final_count:,} registros")
        ctx.logger.info(f"   📊 Redução: {reduction_pct:.1f}%")
        ctx.logger.info(f"   🎯 Score qualidade médio: {avg_quality_score:.1f}/100")
        ctx.logger.info(f"   ❌ Rejeitados: comprimento={problems['length_issue']}, emojis={problems['excessive_emojis']}")
        ctx.logger.info(f"      caps={problems['excessive_caps']}, repetição={problems['excessive_repetition']}, idioma={problems['non_portuguese']}")

        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 1
        
        return df_filtered
        
    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 05: {e}")
        ctx.stats['processing_errors'] += 1
        # Em caso de erro, retornar dados originais com colunas padrão
        df['content_quality_score'] = 80
        return df

