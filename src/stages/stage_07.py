#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_07.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any

# spaCy (optional)
try:
    import spacy
    try:
        nlp = spacy.load("pt_core_news_lg")
        SPACY_AVAILABLE = True
    except OSError:
        try:
            nlp = spacy.load("pt_core_news_sm")
            SPACY_AVAILABLE = True
        except OSError:
            nlp = None
            SPACY_AVAILABLE = False
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False


def _stage_07_linguistic_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 07: Processamento linguístico com spaCy.

    USA: normalized_text do Stage 02
    GERA: tokens, lemmas, POS tags, entidades nomeadas
    """
    ctx.logger.info("🔤 Stage 07: Linguistic Processing (spaCy)")

    if not SPACY_AVAILABLE:
        ctx.logger.warning("⚠️ spaCy não disponível - usando processamento básico")
        return _linguistic_fallback(df)

    def process_text_with_spacy(text):
        """Processar texto com spaCy otimizado."""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'entities': [],
                'tokens_count': 0,
                'entities_count': 0
            }

        try:
            # Processar texto (limitando para performance)
            doc = nlp(str(text)[:1000])  # Limitar a 1000 chars para performance

            tokens = [token.text for token in doc if not token.is_space]
            lemmas = [token.lemma_ for token in doc if not token.is_space and token.lemma_ != '-PRON-']
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            return {
                'tokens': tokens,
                'lemmas': lemmas,
                'tokens_count': len(tokens),
                'entities_count': len(entities)
            }
        except Exception as e:
            ctx.logger.warning(f"Erro spaCy: {e}")
            return {
                'tokens': [],
                'lemmas': [],
                'tokens_count': 0,
                'entities_count': 0
            }

    # FIX: spaCy deve receber 'body' (texto cru) — normalized_text é lowercase
    # e sem pontuação, o que degrada NER, POS tagging e sentence splitting
    spacy_input_col = 'body' if 'body' in df.columns else 'normalized_text'
    spacy_results = df[spacy_input_col].apply(process_text_with_spacy)

    # Extrair dados do spaCy (removidos pos_tags e entities - não utilizados)
    df['spacy_tokens'] = spacy_results.apply(lambda x: x['tokens'])
    df['spacy_lemmas'] = spacy_results.apply(lambda x: x['lemmas'])
    df['spacy_tokens_count'] = spacy_results.apply(lambda x: x['tokens_count'])
    df['spacy_entities_count'] = spacy_results.apply(lambda x: x['entities_count'])

    # Criar texto processado com lemmas para stages posteriores
    df['lemmatized_text'] = df['spacy_lemmas'].apply(lambda x: ' '.join(x) if x else '')

    ctx.stats['stages_completed'] += 1
    ctx.stats['features_extracted'] += 4

    avg_tokens = df['spacy_tokens_count'].mean()
    avg_entities = df['spacy_entities_count'].mean()
    ctx.logger.info(f"✅ spaCy processado: {avg_tokens:.1f} tokens, {avg_entities:.1f} entidades média")
    return df
def _linguistic_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback básico quando spaCy não está disponível."""
    # FIX: usar body como input (consistente com spaCy path)
    fallback_col = 'body' if 'body' in df.columns else 'normalized_text'
    df['spacy_tokens'] = df[fallback_col].str.split()
    df['spacy_tokens_count'] = df['spacy_tokens'].str.len()
    df['spacy_lemmas'] = df['spacy_tokens']  # sem spaCy, lemmas = tokens
    df['spacy_entities_count'] = 0
    df['lemmatized_text'] = df[fallback_col].str.lower()  # fallback: lowercase do body

    ctx.stats['stages_completed'] += 1
    ctx.stats['features_extracted'] += 3
    return df

