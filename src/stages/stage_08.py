#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_08.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


def _stage_08_political_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 08: Classificação política brasileira.

    REFORMULADO: usa spacy_lemmas (Stage 07) com token matching via set lookup.
    Aplica léxico unificado (914+ termos, 11 macrotemas) para classificar textos.
    Escopo: discurso bolsonarista/direita brasileira (2019-2023).
    Orientações retornadas: extrema-direita, direita, centro-direita, neutral.
    Nota: léxico não inclui termos de esquerda (fora do escopo do projeto).
    """
    try:
        ctx.logger.info("🔄 Stage 08: Classificação política brasileira (token matching via spaCy lemmas)")

        # Determinar coluna de input: preferir spacy_lemmas > lemmatized_text > normalized_text
        if 'spacy_lemmas' in df.columns:
            input_col = 'spacy_lemmas'
            ctx.logger.info("   📥 Input: spacy_lemmas (token-level matching)")
        elif 'lemmatized_text' in df.columns:
            input_col = 'lemmatized_text'
            ctx.logger.info("   📥 Input: lemmatized_text (string fallback)")
        else:
            input_col = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            ctx.logger.warning(f"   ⚠️ spaCy output não disponível, fallback: {input_col}")

        # Classificação política usando léxico unificado
        df['political_orientation'] = df[input_col].apply(_classify_political_orientation)
        df['political_keywords'] = df[input_col].apply(_extract_political_keywords)
        df['political_intensity'] = df[input_col].apply(_calculate_political_intensity)

        # Classificação temática - 12 categorias (political_keywords_dict.py)
        try:
            from src.core.political_keywords_dict import POLITICAL_KEYWORDS
            import re as _re

            for cat_name, cat_terms in POLITICAL_KEYWORDS.items():
                col_name = 'cat_' + _re.sub(r'^cat\d+_', '', cat_name)
                # Token matching: set intersection para single-word, substring para multi-word
                cat_single = set(t for t in cat_terms if ' ' not in t)
                cat_multi = [t for t in cat_terms if ' ' in t]

                def _count_cat_matches(lemmas_or_text, s=cat_single, m=cat_multi):
                    if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
                        return 0
                    if isinstance(lemmas_or_text, list):
                        tset = set(t.lower() for t in lemmas_or_text if t)
                    else:
                        tset = set(str(lemmas_or_text).lower().split())
                    count = len(tset & s)
                    if m:
                        joined = ' '.join(sorted(tset))
                        count += sum(1 for t in m if t in joined)
                    return count

                df[col_name] = df[input_col].apply(_count_cat_matches)

            ctx.logger.info(f"📊 Classificação temática: {len(POLITICAL_KEYWORDS)} categorias aplicadas (token matching)")
        except ImportError:
            ctx.logger.warning("⚠️ political_keywords_dict.py não encontrado, pulando categorias temáticas")

        # === CODIFICAÇÃO TCW (Tabela-Categoria-Palavra) ===
        # Integrado do classificador TCW: adiciona códigos numéricos 3-dígitos
        # e grau de concordância entre as 3 tabelas LLM
        try:
            import json as _json
            from pathlib import Path as _Path

            tcw_path = _Path(__file__).parent / 'core' / 'tcw_codes.json'
            if tcw_path.exists():
                with open(tcw_path, 'r', encoding='utf-8') as f:
                    tcw_codes = _json.load(f)

                # Construir lookup: word → list of codes
                word_to_codes = {}
                for code, info in tcw_codes.items():
                    word = info['word'].lower()
                    if word not in word_to_codes:
                        word_to_codes[word] = []
                    word_to_codes[word].append({
                        'code': code,
                        'table': info['table'],
                        'category': info['category'],
                        'category_name': info['category_name']
                    })

                # Single-word e multi-word lookup sets
                tcw_single = set(w for w in word_to_codes if ' ' not in w)
                tcw_multi = [w for w in word_to_codes if ' ' in w]

                def _tcw_classify(lemmas_or_text):
                    """Classificar texto usando codificação TCW."""
                    if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
                        return [], [], 0.0
                    if isinstance(lemmas_or_text, list):
                        tset = set(t.lower() for t in lemmas_or_text if t)
                    else:
                        tset = set(str(lemmas_or_text).lower().split())

                    codes_found = []
                    categories_found = set()

                    # Single-word matches
                    for word in tset & tcw_single:
                        for entry in word_to_codes[word]:
                            codes_found.append(entry['code'])
                            categories_found.add(entry['category_name'])

                    # Multi-word matches
                    if tcw_multi:
                        joined = ' '.join(sorted(tset))
                        for mw in tcw_multi:
                            if mw in joined:
                                for entry in word_to_codes[mw]:
                                    codes_found.append(entry['code'])
                                    categories_found.add(entry['category_name'])

                    # Concordância: quantas tabelas (1-3) concordam nos códigos encontrados
                    if codes_found:
                        tables_seen = set()
                        for code in codes_found:
                            if code in tcw_codes:
                                tables_seen.add(tcw_codes[code]['table'])
                        agreement = len(tables_seen) / 3.0  # 0.33, 0.67, 1.0
                    else:
                        agreement = 0.0

                    return codes_found, list(categories_found), agreement

                tcw_results = df[input_col].apply(_tcw_classify)
                df['tcw_codes'] = tcw_results.apply(lambda x: x[0])
                df['tcw_categories'] = tcw_results.apply(lambda x: x[1])
                df['tcw_agreement'] = tcw_results.apply(lambda x: x[2])
                df['tcw_code_count'] = df['tcw_codes'].apply(len)

                tcw_classified = (df['tcw_code_count'] > 0).sum()
                ctx.logger.info(f"🔢 TCW: {tcw_classified}/{len(df)} textos classificados ({tcw_classified/len(df)*100:.1f}%)")
            else:
                ctx.logger.warning("⚠️ tcw_codes.json não encontrado em src/core/, pulando TCW")
        except Exception as tcw_err:
            ctx.logger.warning(f"⚠️ Erro TCW: {tcw_err}")

        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 19  # 15 base + 4 TCW

        ctx.logger.info(f"✅ Stage 08 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 08: {e}")
        ctx.stats['processing_errors'] += 1
        return df

