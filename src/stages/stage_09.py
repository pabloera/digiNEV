#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_09.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
# sklearn imported lazily inside function


def _stage_09_tfidf_vectorization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 09: Vetorização TF-IDF com tokens spaCy.
    
    Calcula TF-IDF usando tokens processados pelo spaCy.
    Trata casos de vocabulário vazio em chunks pequenos.
    """
    try:
        ctx.logger.info("🔄 Stage 09: Vetorização TF-IDF")
        
        # Verificar se há dados suficientes
        if len(df) < 2:
            ctx.logger.warning(f"⚠️ Dados insuficientes para TF-IDF ({len(df)} documentos), preenchendo com padrões")
            df['tfidf_score_mean'] = 0.0
            df['tfidf_score_max'] = 0.0
            df['tfidf_top_terms'] = [[] for _ in range(len(df))]
            ctx.stats['features_extracted'] += 3
            return df
        
        # FIX: usar 'lemmatized_text' (output do Stage 07 spaCy) em vez de 'tokens' (inexistente)
        if 'lemmatized_text' in df.columns:
            text_data = df['lemmatized_text'].fillna('').tolist()
        elif 'spacy_tokens' in df.columns:
            text_data = df['spacy_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('').tolist()
        else:
            ctx.logger.warning("⚠️ lemmatized_text/spacy_tokens não encontrados, usando normalized_text")
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            text_data = df[text_column].fillna('').tolist()
        
        # Verificar se há texto não-vazio
        non_empty_texts = [text for text in text_data if text.strip()]
        if len(non_empty_texts) < 2:
            ctx.logger.warning(f"⚠️ Textos vazios demais para TF-IDF ({len(non_empty_texts)} válidos), usando fallback")
            df['tfidf_score_mean'] = 0.1
            df['tfidf_score_max'] = 0.2
            df['tfidf_top_terms'] = [['texto', 'palavra'] for _ in range(len(df))]
            ctx.stats['features_extracted'] += 3
            return df
        
        # TF-IDF com configuração adaptativa para chunks pequenos
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        # Ajustar max_features baseado no tamanho do chunk
        chunk_size = len(df)
        if chunk_size < 50:
            max_features = min(20, chunk_size * 2)  # Muito conservador
        elif chunk_size < 200:
            max_features = min(50, chunk_size)  # Conservador
        else:
            max_features = 100  # Padrão
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,  # Aceitar termos que aparecem pelo menos 1 vez
            stop_words=None,  # Já removemos stopwords no spaCy
            lowercase=False,   # Já normalizado
            token_pattern=r'\S+',  # Aceitar qualquer token não-espaço
            ngram_range=(1, 1)  # Apenas unigramas para chunks pequenos
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            
            # Verificar se conseguiu gerar features
            if tfidf_matrix.shape[1] == 0:
                raise ValueError("Vocabulário vazio após vectorização")
            
            # Converter para array denso para cálculos
            tfidf_dense = tfidf_matrix.toarray()
            
            # Scores médios por documento
            df['tfidf_score_mean'] = np.mean(tfidf_dense, axis=1)
            df['tfidf_score_max'] = np.max(tfidf_dense, axis=1)
            
            # Top terms por documento 
            top_terms_count = min(5, len(feature_names))  # Adaptar ao vocabulário disponível
            df['tfidf_top_terms'] = [
                [feature_names[i] for i in row.argsort()[::-1][:top_terms_count] if row[i] > 0]
                for row in tfidf_dense
            ]
            
            ctx.logger.info(f"✅ TF-IDF: {len(feature_names)} features, max_features={max_features}")
            
        except (ValueError, Exception) as ve:
            ctx.logger.warning(f"⚠️ Erro na vectorização TF-IDF: {ve}, usando fallback simples")
            
            # Fallback: análise simples baseada em frequência de palavras
            import re
            from collections import Counter
            
            all_words = []
            for text in text_data:
                if text and text.strip():
                    # Extrair palavras simples
                    words = re.findall(r'\w+', text.lower())
                    words = [w for w in words if len(w) > 2]  # Filtrar palavras muito curtas
                    all_words.extend(words)
            
            if all_words:
                word_freq = Counter(all_words)
                common_words = [word for word, _ in word_freq.most_common(10)]
                
                # Scores baseados em presença de palavras comuns
                df['tfidf_score_mean'] = [
                    len([w for w in re.findall(r'\w+', str(text).lower()) if w in common_words]) / max(1, len(common_words)) * 0.5
                    for text in text_data
                ]
                df['tfidf_score_max'] = df['tfidf_score_mean'] * 1.5
                df['tfidf_top_terms'] = [
                    [w for w in re.findall(r'\w+', str(text).lower()) if w in common_words][:5]
                    for text in text_data
                ]
            else:
                # Última opção: valores padrão
                df['tfidf_score_mean'] = 0.1
                df['tfidf_score_max'] = 0.2
                df['tfidf_top_terms'] = [[] for _ in range(len(df))]
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 3
        
        ctx.logger.info(f"✅ Stage 09 concluído: {len(df)} registros processados")
        return df
        
    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 09: {e}")
        ctx.stats['processing_errors'] += 1
        
        # Valores padrão em caso de erro
        df['tfidf_score_mean'] = 0.0
        df['tfidf_score_max'] = 0.0
        df['tfidf_top_terms'] = [[] for _ in range(len(df))]
        ctx.stats['features_extracted'] += 3
        return df
