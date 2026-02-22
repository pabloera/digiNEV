#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_11.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
# sklearn imported lazily inside function


def _stage_11_topic_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 11: Topic modeling com LDA.

    Descoberta automática de tópicos nos textos.
    """
    try:
        ctx.logger.info("🔄 Stage 11: Topic modeling")
        
        # FIX: usar 'lemmatized_text' (output do Stage 07 spaCy) em vez de 'tokens' (inexistente)
        if 'lemmatized_text' in df.columns:
            text_data = df['lemmatized_text'].fillna('').tolist()
        elif 'spacy_tokens' in df.columns:
            text_data = df['spacy_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('').tolist()
        else:
            ctx.logger.warning("⚠️ lemmatized_text/spacy_tokens não encontrados, usando normalized_text")
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            text_data = df[text_column].fillna('').tolist()
        
        # Topic modeling básico com LDA
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Stopwords PT (termos funcionais que poluem LDA)
        pt_stopwords = [
            'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas',
            'um', 'uma', 'uns', 'umas', 'por', 'para', 'com', 'sem', 'sob',
            'que', 'se', 'não', 'mais', 'muito', 'como', 'mas', 'ou', 'já',
            'também', 'só', 'seu', 'sua', 'seus', 'suas', 'ele', 'ela', 'eles',
            'elas', 'isso', 'isto', 'esse', 'essa', 'este', 'esta', 'aqui',
            'ali', 'lá', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas',
            'entre', 'sobre', 'após', 'até', 'quando', 'onde', 'quem', 'qual',
            'foi', 'ser', 'ter', 'está', 'são', 'tem', 'era', 'vai', 'pode',
            'nos', 'me', 'te', 'lhe', 'o', 'a', 'os', 'as', 'e', 'é',
            'eu', 'tu', 'nós', 'vós', 'meu', 'minha', 'teu', 'tua',
            'nosso', 'nossa', 'nossos', 'nossas', 'todo', 'toda', 'todos', 'todas',
            'outro', 'outra', 'outros', 'outras', 'mesmo', 'mesma', 'cada',
            'ainda', 'então', 'depois', 'antes', 'bem', 'agora', 'sempre',
            'nunca', 'nada', 'tudo', 'algo', 'assim', 'aquele', 'aquela',
            'http', 'https', 'www', 'com', 'org', 'br', 'the', 'and', 'for'
        ]

        # Preparar dados (com remoção de stopwords PT)
        vectorizer = CountVectorizer(max_features=50, stop_words=pt_stopwords)
        doc_term_matrix = vectorizer.fit_transform(text_data)
        
        # LDA simples
        n_topics = min(5, len(df) // 20 + 1)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        doc_topic_matrix = lda.fit_transform(doc_term_matrix)
        
        # Tópico dominante para cada documento
        df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
        df['topic_probability'] = doc_topic_matrix.max(axis=1)
        
        # Palavras-chave dos tópicos
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[::-1][:3]]
            topic_keywords.append(top_words)
        
        df['topic_keywords'] = df['dominant_topic'].apply(lambda x: topic_keywords[x] if x < len(topic_keywords) else [])
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 3
        
        ctx.logger.info(f"✅ Stage 11 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 11: {e}")
        ctx.stats['processing_errors'] += 1
        return df
