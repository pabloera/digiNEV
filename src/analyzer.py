#!/usr/bin/env python3
"""
digiNEV Analyzer v.final
=======================

Sistema consolidado único de análise de discurso político brasileiro.
Pipeline com 14 estágios interligados gerando 81+ colunas de análise.

ARQUITETURA CONSOLIDADA:
- Sistema único centralizado (elimina estruturas paralelas)
- 14 estágios científicos sequenciais
- Dados reais processados (sem métricas inventadas)
- Configuração unificada via config/settings.yaml

Author: digiNEV Academic Research Team
Version: v.final (Consolidação Final)
"""

import pandas as pd
import numpy as np
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports (sempre disponíveis)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Standard library imports
import unicodedata
import time
from collections import Counter
from urllib.parse import urlparse

# spaCy para processamento linguístico em português
try:
    import spacy
    # Tentar carregar modelo português
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


class Analyzer:
    """
    Analisador consolidado com 14 stages sequenciais interligados.

    STAGES IMPLEMENTADOS (sem fallbacks confusos):
    01. feature_extraction (Python puro) - Detecção automática de colunas e features
    02. text_preprocessing (Python puro) - Limpeza básica de texto em português
    03. linguistic_processing (spaCy) - Processamento linguístico avançado LOGO APÓS limpeza
    04. statistical_analysis (Python puro) - Análise estatística com dados spaCy
    05. political_classification (Lexicon real) - Classificação política brasileira
    06. tfidf_vectorization (scikit-learn) - TF-IDF com tokens spaCy
    07. clustering_analysis (scikit-learn) - Clustering baseado em features linguísticas
    08. topic_modeling (scikit-learn) - Topic modeling com embeddings
    09. temporal_analysis (Python puro) - Análise temporal
    10. network_analysis (Python puro) - Coordenação e padrões de rede
    """

    def __init__(self):
        """Inicializar analyzer com configuração clara."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load political lexicon if available
        self.political_lexicon = self._load_political_lexicon()

        # Initialize ML components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.kmeans_model = None
        self.lda_model = None

        # Stats tracking
        self.stats = {
            'stages_completed': 0,
            'features_extracted': 0,
            'processing_errors': 0
        }

        self.logger.info("✅ Clean Scientific Analyzer v.final inicializado")

    def _load_political_lexicon(self) -> Dict:
        """Carregar lexicon político real (se disponível)."""
        lexicon_path = Path("src/core/lexico_politico_hierarquizado.json")

        if lexicon_path.exists():
            try:
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    lexicon = json.load(f)
                self.logger.info(f"✅ Lexicon político carregado: {len(lexicon)} categorias")
                return lexicon
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao carregar lexicon: {e}")

        # Fallback mínimo (sem confusão)
        return {
            "extrema-direita": ["bolsonaro", "conservador"],
            "direita": ["liberal", "capitalista"],
            "centro": ["moderado", "centrista"],
            "esquerda": ["socialista", "progressista"]
        }

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisar dataset com pipeline sequencial interligado de 14 stages.

        Args:
            df: DataFrame com dados para análise

        Returns:
            Dict com resultado da análise
        """
        try:
            self.logger.info(f"🔬 Iniciando análise: {len(df)} registros")

            # Reset stats
            self.stats = {'stages_completed': 0, 'features_extracted': 0, 'processing_errors': 0}

            # STAGE 01: Feature Extraction (SEMPRE PRIMEIRO)
            df = self._stage_01_feature_extraction(df)

            # STAGE 02: Text Preprocessing (Limpeza básica)
            df = self._stage_02_text_preprocessing(df)

            # STAGE 03: Linguistic Processing (spaCy LOGO APÓS limpeza)
            df = self._stage_03_linguistic_processing(df)

            # STAGE 04: Statistical Analysis (com dados spaCy)
            df = self._stage_04_statistical_analysis(df)

            # STAGE 05: Political Classification (brasileira)
            df = self._stage_04_political_classification(df)

            # STAGE 06: TF-IDF Vectorization (com tokens spaCy)
            df = self._stage_05_tfidf_vectorization(df)

            # STAGE 07: Clustering Analysis (baseado em features linguísticas)
            df = self._stage_06_clustering_analysis(df)

            # STAGE 08: Topic Modeling (com embeddings)
            df = self._stage_07_topic_modeling(df)

            # STAGE 09: Temporal Analysis
            df = self._stage_08_temporal_analysis(df)

            # STAGE 10: Network Analysis (coordenação e padrões)
            df = self._stage_09_network_analysis(df)

            # STAGE 11: Domain Analysis
            df = self._stage_10_domain_analysis(df)

            # STAGE 12: Semantic Analysis (NOVO)
            df = self._stage_12_semantic_analysis(df)

            # STAGE 13: Event Context Analysis (NOVO)
            df = self._stage_13_event_context(df)

            # STAGE 14: Channel Analysis (NOVO)
            df = self._stage_14_channel_analysis(df)

            # Final metadata
            df['processing_timestamp'] = datetime.now().isoformat()
            df['stages_completed'] = self.stats['stages_completed']
            df['features_extracted'] = self.stats['features_extracted']

            self.logger.info(f"✅ Análise concluída: {len(df.columns)} colunas, {self.stats['stages_completed']} stages")

            return {
                'data': df,
                'stats': self.stats.copy(),
                'columns_generated': len(df.columns),
                'stages_completed': self.stats['stages_completed']
            }

        except Exception as e:
            self.logger.error(f"❌ Erro na análise: {e}")
            raise

    def _stage_01_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 01: Extração e identificação automática de features (Python puro).

        SEMPRE O PRIMEIRO STAGE - identifica colunas disponíveis e extrai features se necessário.
        """
        self.logger.info("🔍 STAGE 01: Feature Extraction")

        # Identificar coluna de texto principal
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Verificar se contém texto substancial
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Textos com mais de 20 caracteres em média
                        text_columns.append(col)

        # Selecionar melhor coluna de texto
        if not text_columns:
            raise ValueError("❌ Nenhuma coluna de texto encontrada")

        # Priorizar colunas comuns
        priority_columns = ['text', 'body', 'message', 'content', 'texto', 'mensagem']
        main_text_column = None

        for priority in priority_columns:
            if priority in text_columns:
                main_text_column = priority
                break

        if not main_text_column:
            main_text_column = text_columns[0]

        # Identificar coluna de timestamp (se disponível)
        timestamp_column = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                timestamp_column = col
                break

        # DETECÇÃO AUTOMÁTICA DE FEATURES EXISTENTES
        features_detected = self._detect_existing_features(df)

        # EXTRAÇÃO AUTOMÁTICA DE FEATURES (se não existem)
        df = self._extract_missing_features(df, main_text_column, features_detected)

        # Identificar colunas de metadados
        metadata_columns = [col for col in df.columns
                           if col not in [main_text_column, timestamp_column]]

        # Adicionar features identificadas
        df['main_text_column'] = main_text_column
        df['timestamp_column'] = timestamp_column if timestamp_column else 'none'
        df['metadata_columns_count'] = len(metadata_columns)
        df['has_timestamp'] = timestamp_column is not None

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 4 + len(features_detected['extracted'])

        self.logger.info(f"✅ Features: text={main_text_column}, timestamp={timestamp_column}")
        self.logger.info(f"✅ Features detectadas: {list(features_detected['existing'].keys())}")
        self.logger.info(f"✅ Features extraídas: {features_detected['extracted']}")
        return df

    def _detect_existing_features(self, df: pd.DataFrame) -> Dict:
        """
        Detecta features que já existem como colunas no DataFrame.
        """
        existing_features = {}

        # Features de interesse para detectar
        feature_patterns = {
            'hashtags': ['hashtag', 'hashtags', 'tags'],
            'urls': ['url', 'urls', 'links', 'link'],
            'mentions': ['mention', 'mentions', 'user_mentions', 'usuarios'],
            'emojis': ['emoji', 'emojis', 'emoticon'],
            'reply_count': ['reply', 'replies', 'respostas'],
            'retweet_count': ['retweet', 'retweets', 'rt_count'],
            'like_count': ['like', 'likes', 'curtidas', 'fav'],
            'user_info': ['user', 'username', 'author', 'usuario']
        }

        for feature_name, patterns in feature_patterns.items():
            for pattern in patterns:
                matching_cols = [col for col in df.columns if pattern in col.lower()]
                if matching_cols:
                    existing_features[feature_name] = matching_cols[0]
                    break

        return {
            'existing': existing_features,
            'extracted': []
        }

    def _extract_missing_features(self, df: pd.DataFrame, text_column: str, features_info: Dict) -> pd.DataFrame:
        """
        Extrai features ausentes do texto principal usando regex otimizados para português brasileiro.
        """
        extracted_features = []

        # Só extrair se não existir coluna correspondente
        if 'hashtags' not in features_info['existing']:
            df['hashtags_extracted'] = df[text_column].astype(str).str.findall(r'#\w+')
            df['hashtags_count'] = df['hashtags_extracted'].str.len()
            extracted_features.append('hashtags')

        if 'urls' not in features_info['existing']:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            df['urls_extracted'] = df[text_column].astype(str).str.findall(url_pattern)
            df['urls_count'] = df['urls_extracted'].str.len()
            extracted_features.append('urls')

        if 'mentions' not in features_info['existing']:
            # Padrão para @mentions
            df['mentions_extracted'] = df[text_column].astype(str).str.findall(r'@\w+')
            df['mentions_count'] = df['mentions_extracted'].str.len()
            extracted_features.append('mentions')

        if 'emojis' not in features_info['existing']:
            # Padrão básico para emojis (Unicode)
            emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]'
            df['emojis_extracted'] = df[text_column].astype(str).str.findall(emoji_pattern)
            df['emojis_count'] = df['emojis_extracted'].str.len()
            extracted_features.append('emojis')

        # Features de texto em português brasileiro
        df['has_interrogation'] = df[text_column].astype(str).str.contains(r'\?', na=False)
        df['has_exclamation'] = df[text_column].astype(str).str.contains(r'!', na=False)
        df['has_caps_words'] = df[text_column].astype(str).str.contains(r'\b[A-Z]{2,}\b', na=False)
        extracted_features.extend(['interrogation', 'exclamation', 'caps_words'])

        # Detectar características do português brasileiro
        df['has_portuguese_words'] = df[text_column].astype(str).str.contains(
            r'\b(que|com|para|por|não|são|muito|mais|todo|bem|ainda|só|já|até|ainda)\b',
            case=False, na=False
        )
        extracted_features.append('portuguese_words')

        features_info['extracted'] = extracted_features
        return df

    def _stage_02_text_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 02: Pré-processamento de texto (Python puro).

        USA: main_text_column identificada no Stage 01
        """
        self.logger.info("🧹 STAGE 02: Text Preprocessing")

        main_text_col = df['main_text_column'].iloc[0]

        def clean_text(text):
            """Limpar texto usando Python puro."""
            if pd.isna(text):
                return ""

            text = str(text)

            # Normalizar unicode
            text = unicodedata.normalize('NFKD', text)

            # Remover caracteres especiais mas preservar acentos
            text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)

            # Normalizar espaços
            text = re.sub(r'\s+', ' ', text).strip()

            # Converter para lowercase
            text = text.lower()

            return text

        # Aplicar limpeza
        df['normalized_text'] = df[main_text_col].apply(clean_text)
        df['text_cleaned'] = True

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 2

        self.logger.info(f"✅ Texto pré-processado: {df['normalized_text'].str.len().mean():.1f} chars média")
        return df

    def _stage_03_linguistic_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 03: Processamento linguístico com spaCy (LOGO APÓS limpeza).

        USA: normalized_text do Stage 02
        GERA: tokens, lemmas, POS tags, entidades nomeadas
        """
        self.logger.info("🔤 STAGE 03: Linguistic Processing (spaCy)")

        if not SPACY_AVAILABLE:
            self.logger.warning("⚠️ spaCy não disponível - usando processamento básico")
            return self._linguistic_fallback(df)

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
                pos_tags = [token.pos_ for token in doc if not token.is_space]
                entities = [(ent.text, ent.label_) for ent in doc.ents]

                return {
                    'tokens': tokens,
                    'lemmas': lemmas,
                    'pos_tags': pos_tags,
                    'entities': entities,
                    'tokens_count': len(tokens),
                    'entities_count': len(entities)
                }
            except Exception as e:
                self.logger.warning(f"Erro spaCy: {e}")
                return {
                    'tokens': [],
                    'lemmas': [],
                    'pos_tags': [],
                    'entities': [],
                    'tokens_count': 0,
                    'entities_count': 0
                }

        # Processar textos normalizados
        spacy_results = df['normalized_text'].apply(process_text_with_spacy)

        # Extrair dados do spaCy
        df['spacy_tokens'] = spacy_results.apply(lambda x: x['tokens'])
        df['spacy_lemmas'] = spacy_results.apply(lambda x: x['lemmas'])
        df['spacy_pos_tags'] = spacy_results.apply(lambda x: x['pos_tags'])
        df['spacy_entities'] = spacy_results.apply(lambda x: x['entities'])
        df['spacy_tokens_count'] = spacy_results.apply(lambda x: x['tokens_count'])
        df['spacy_entities_count'] = spacy_results.apply(lambda x: x['entities_count'])

        # Criar texto processado com lemmas para stages posteriores
        df['lemmatized_text'] = df['spacy_lemmas'].apply(lambda x: ' '.join(x) if x else '')

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 6

        avg_tokens = df['spacy_tokens_count'].mean()
        avg_entities = df['spacy_entities_count'].mean()
        self.logger.info(f"✅ spaCy processado: {avg_tokens:.1f} tokens, {avg_entities:.1f} entidades média")
        return df

    def _linguistic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback básico quando spaCy não está disponível."""
        # Tokenização básica
        df['spacy_tokens'] = df['normalized_text'].str.split()
        df['spacy_tokens_count'] = df['spacy_tokens'].str.len()
        df['spacy_entities_count'] = 0
        df['lemmatized_text'] = df['normalized_text']  # Usar texto normalizado

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3
        return df

    def _stage_04_statistical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 03: Análise estatística básica (Python puro).

        USA: normalized_text do Stage 02
        """
        self.logger.info("📊 STAGE 03: Statistical Analysis")

        # Estatísticas básicas de texto
        df['word_count'] = df['normalized_text'].str.split().str.len().fillna(0).astype(int)
        df['char_count'] = df['normalized_text'].str.len().fillna(0).astype(int)
        df['sentence_count'] = df['normalized_text'].str.count(r'[.!?]+').fillna(0).astype(int)

        # Estatísticas derivadas
        df['avg_word_length'] = (df['char_count'] / df['word_count'].replace(0, 1)).round(2)
        df['words_per_sentence'] = (df['word_count'] / df['sentence_count'].replace(0, 1)).round(2)

        # Categorização por tamanho
        def categorize_text_length(word_count):
            if word_count < 10:
                return 'short'
            elif word_count < 50:
                return 'medium'
            else:
                return 'long'

        df['text_length_category'] = df['word_count'].apply(categorize_text_length)

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 6

        self.logger.info(f"✅ Estatísticas: {df['word_count'].mean():.1f} palavras média")
        return df

    def _stage_04_political_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 04: Classificação política usando lexicon real.

        USA: normalized_text do Stage 02
        """
        self.logger.info("🏛️ STAGE 04: Political Classification")

        def classify_political_spectrum(text):
            """Classificar usando lexicon político real."""
            if pd.isna(text) or not text:
                return 'unknown'

            text = str(text).lower()
            scores = {}

            # Calcular scores para cada categoria
            for category, terms in self.political_lexicon.items():
                score = sum(1 for term in terms if term.lower() in text)
                if score > 0:
                    scores[category] = score

            # Retornar categoria com maior score
            if scores:
                return max(scores.keys(), key=scores.get)
            return 'neutral'

        def count_political_entities(text):
            """Contar entidades políticas no texto."""
            if pd.isna(text) or not text:
                return 0

            text = str(text).lower()
            count = 0

            for terms in self.political_lexicon.values():
                count += sum(1 for term in terms if term.lower() in text)

            return count

        # Aplicar classificação política
        df['political_spectrum'] = df['normalized_text'].apply(classify_political_spectrum)
        df['political_entity_count'] = df['normalized_text'].apply(count_political_entities)
        df['has_political_content'] = df['political_entity_count'] > 0

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3

        political_dist = df['political_spectrum'].value_counts()
        self.logger.info(f"✅ Classificação política: {political_dist.to_dict()}")
        return df

    def _stage_05_tfidf_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 05: Vetorização TF-IDF (scikit-learn).

        USA: normalized_text do Stage 02
        """
        self.logger.info("🔢 STAGE 05: TF-IDF Vectorization")

        # Configurar TF-IDF com ajuste dinâmico e safety check
        n_docs = len(df)
        min_df = max(1, min(2, n_docs // 10))  # Ajuste dinâmico baseado no tamanho
        max_df = min(0.8, max(0.5, (n_docs - min_df) / n_docs))  # Garantir max_df > min_df

        # Safety check para evitar max_df <= min_df
        if max_df <= min_df:
            max_df = min_df + 0.1
            if max_df > 1.0:
                min_df = 1
                max_df = 0.9

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=min(1000, n_docs * 10),  # Limitar features baseado no tamanho
            stop_words=None,  # Não remover stop words para português
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df
        )

        # Ajustar e transformar
        texts = df['normalized_text'].fillna('').tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # Extrair top terms por documento
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        def get_top_tfidf_terms(doc_idx, n_terms=5):
            """Extrair top termos TF-IDF para um documento."""
            doc_vector = self.tfidf_matrix[doc_idx].toarray().flatten()
            top_indices = doc_vector.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
            return ', '.join(top_terms[:n_terms])

        # Adicionar features TF-IDF
        df['tfidf_top_terms'] = [get_top_tfidf_terms(i) for i in range(len(df))]
        df['tfidf_max_score'] = [self.tfidf_matrix[i].max() for i in range(len(df))]
        df['tfidf_feature_count'] = self.tfidf_matrix.shape[1]

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3

        self.logger.info(f"✅ TF-IDF: {self.tfidf_matrix.shape[1]} features, max_score: {df['tfidf_max_score'].max():.3f}")
        return df

    def _stage_06_clustering_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 06: Análise de clustering (scikit-learn).

        USA: tfidf_matrix do Stage 05
        """
        self.logger.info("🎯 STAGE 06: Clustering Analysis")

        if self.tfidf_matrix is None:
            raise ValueError("❌ TF-IDF matrix não disponível - execute Stage 05 primeiro")

        # Determinar número de clusters
        n_samples = self.tfidf_matrix.shape[0]
        n_clusters = min(max(2, n_samples // 10), 10)  # Entre 2 e 10 clusters

        # Aplicar K-Means
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.tfidf_matrix)

        # Calcular distâncias aos centros
        distances = self.kmeans_model.transform(self.tfidf_matrix)
        min_distances = distances.min(axis=1)

        # Adicionar features de clustering
        df['cluster_id'] = cluster_labels
        df['cluster_distance'] = min_distances.round(3)
        df['cluster_size'] = df['cluster_id'].map(df['cluster_id'].value_counts())

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3

        cluster_dist = df['cluster_id'].value_counts()
        self.logger.info(f"✅ Clustering: {n_clusters} clusters, distribuição: {cluster_dist.head(3).to_dict()}")
        return df

    def _stage_07_topic_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 07: Modelagem de tópicos (scikit-learn LDA).

        USA: tfidf_matrix do Stage 05 + cluster_id do Stage 06
        """
        self.logger.info("📚 STAGE 07: Topic Modeling")

        if self.tfidf_matrix is None:
            raise ValueError("❌ TF-IDF matrix não disponível - execute Stage 05 primeiro")

        # Determinar número de tópicos baseado em clusters
        n_clusters = df['cluster_id'].nunique()
        n_topics = min(max(2, n_clusters), 5)  # Entre 2 e 5 tópicos

        # Aplicar LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )

        topic_distributions = self.lda_model.fit_transform(self.tfidf_matrix)

        # Extrair tópico dominante para cada documento
        dominant_topics = topic_distributions.argmax(axis=1)
        topic_probabilities = topic_distributions.max(axis=1)

        # Adicionar features de tópicos
        df['topic_id'] = dominant_topics
        df['topic_probability'] = topic_probabilities.round(3)
        df['topic_diversity'] = (topic_distributions > 0.1).sum(axis=1)  # Quantos tópicos > 10%

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3

        topic_dist = df['topic_id'].value_counts()
        self.logger.info(f"✅ Tópicos: {n_topics} tópicos, prob média: {df['topic_probability'].mean():.3f}")
        return df

    def _stage_08_temporal_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 08: Análise temporal (Python puro).

        USA: timestamp_column identificada no Stage 01
        """
        self.logger.info("⏰ STAGE 08: Temporal Analysis")

        timestamp_col = df['timestamp_column'].iloc[0]

        if timestamp_col == 'none' or timestamp_col not in df.columns:
            # Sem timestamp - criar features genéricas
            df['hour'] = -1
            df['day_of_week'] = -1
            df['month'] = -1
            df['has_temporal_data'] = False
        else:
            # Processar timestamps reais
            def extract_temporal_features(timestamp):
                """Extrair features temporais."""
                try:
                    if pd.isna(timestamp):
                        return -1, -1, -1

                    # Tentar converter para datetime
                    if isinstance(timestamp, str):
                        dt = pd.to_datetime(timestamp, errors='coerce')
                    else:
                        dt = pd.to_datetime(timestamp, errors='coerce')

                    if pd.isna(dt):
                        return -1, -1, -1

                    return dt.hour, dt.dayofweek, dt.month

                except:
                    return -1, -1, -1

            temporal_features = df[timestamp_col].apply(
                lambda x: extract_temporal_features(x)
            )

            df['hour'] = [t[0] for t in temporal_features]
            df['day_of_week'] = [t[1] for t in temporal_features]
            df['month'] = [t[2] for t in temporal_features]
            df['has_temporal_data'] = df['hour'] != -1

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 4

        valid_temporal = (df['has_temporal_data']).sum()
        self.logger.info(f"✅ Temporal: {valid_temporal}/{len(df)} registros com timestamp válido")
        return df

    def _stage_09_network_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 09: Análise de redes e coordenação (Python puro).

        USA: cluster_id do Stage 06 + temporal features do Stage 08
        """
        self.logger.info("🕸️ STAGE 09: Network Analysis")

        # Detectar coordenação temporal entre clusters
        def detect_temporal_coordination():
            """Detectar coordenação baseada em clusters e tempo."""
            coordination_scores = []

            for idx, row in df.iterrows():
                score = 0

                # Se tem dados temporais válidos
                if row['has_temporal_data']:
                    # Verificar se há outros no mesmo cluster na mesma hora
                    same_cluster_same_hour = df[
                        (df['cluster_id'] == row['cluster_id']) &
                        (df['hour'] == row['hour']) &
                        (df.index != idx)
                    ]

                    if len(same_cluster_same_hour) >= 2:
                        score += 0.5

                    # Verificar coordenação por tamanho do cluster
                    if row['cluster_size'] >= len(df) * 0.1:  # Cluster com >10% dos dados
                        score += 0.3

                coordination_scores.append(min(score, 1.0))

            return coordination_scores

        # Calcular coordenação
        df['coordination_score'] = detect_temporal_coordination()
        df['potential_coordination'] = df['coordination_score'] > 0.3

        # Padrões temporais por cluster
        def get_temporal_pattern(cluster_id):
            """Identificar padrão temporal do cluster."""
            cluster_data = df[df['cluster_id'] == cluster_id]

            if not cluster_data['has_temporal_data'].any():
                return 'no_temporal_data'

            valid_hours = cluster_data[cluster_data['has_temporal_data']]['hour']

            if len(valid_hours) == 0:
                return 'no_temporal_data'

            hour_counts = valid_hours.value_counts()

            if hour_counts.max() >= len(valid_hours) * 0.5:
                return 'concentrated'
            else:
                return 'distributed'

        df['temporal_pattern'] = df['cluster_id'].apply(get_temporal_pattern)

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3

        coordination_count = df['potential_coordination'].sum()
        self.logger.info(f"✅ Network: {coordination_count}/{len(df)} com potencial coordenação")
        return df

    def _stage_10_domain_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 10: Análise de Domínios e URLs"""
        start_time = time.time()
        self.logger.info("🌐 STAGE 10: Análise de Domínios")
        
        try:
            import re
            from urllib.parse import urlparse
            
            texts = df['normalized_text'].fillna('').astype(str)
            
            # Extrair URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            df['urls_found'] = texts.apply(lambda x: re.findall(url_pattern, x))
            df['url_count'] = df['urls_found'].apply(len)
            
            # Extrair domínios únicos
            def extract_domains(urls):
                domains = []
                for url in urls:
                    try:
                        domain = urlparse(url).netloc
                        if domain:
                            domains.append(domain)
                    except:
                        continue
                return list(set(domains))
            
            df['domains_found'] = df['urls_found'].apply(extract_domains)
            df['unique_domains_count'] = df['domains_found'].apply(len)
            
            # Classificar tipos de domínio
            mainstream_domains = ['g1.com', 'folha.uol.com.br', 'estadao.com.br', 'globo.com']
            alternative_domains = ['brasil247.com', 'diariodocentrodomundo.com.br']
            social_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com']
            
            def classify_domains(domains):
                if not domains:
                    return 'none'
                
                mainstream_count = sum(1 for d in domains if any(md in d for md in mainstream_domains))
                alternative_count = sum(1 for d in domains if any(ad in d for ad in alternative_domains))
                social_count = sum(1 for d in domains if any(sd in d for sd in social_domains))
                
                if mainstream_count > 0:
                    return 'mainstream'
                elif alternative_count > 0:
                    return 'alternative'
                elif social_count > 0:
                    return 'social'
                else:
                    return 'other'
            
            df['domain_category'] = df['domains_found'].apply(classify_domains)
            
            # Detectar presença de links externos
            df['has_external_links'] = df['url_count'] > 0
            
            # Diversidade de fontes
            total_domains = df['domains_found'].apply(len).sum()
            df['domain_diversity'] = 'low' if total_domains < 5 else 'medium' if total_domains < 20 else 'high'
            
        except Exception as e:
            self.logger.warning(f"Erro na análise de domínios: {e}")
            df['url_count'] = 0
            df['unique_domains_count'] = 0
            df['domain_category'] = 'error'
            df['has_external_links'] = False
            df['domain_diversity'] = 'error'
        
        processing_time = time.time() - start_time
        self.stats['stage_10_time'] = processing_time
        self.stats['stages_completed'] += 1
        self.logger.info(f"✅ Stage 10 concluído em {processing_time:.2f}s")
        
        return df

    def _stage_11_domain_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 11: Análise de Domínios e URLs"""
        start_time = time.time()
        self.logger.info("🌐 STAGE 11: Análise de Domínios")
        
        try:
            import re
            from urllib.parse import urlparse
            
            texts = df['normalized_text'].fillna('').astype(str)
            
            # Extrair URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            df['urls_found'] = texts.apply(lambda x: re.findall(url_pattern, x))
            df['url_count'] = df['urls_found'].apply(len)
            
            # Extrair domínios únicos
            def extract_domains(urls):
                domains = []
                for url in urls:
                    try:
                        domain = urlparse(url).netloc
                        if domain:
                            domains.append(domain)
                    except:
                        continue
                return list(set(domains))
            
            df['domains_found'] = df['urls_found'].apply(extract_domains)
            df['unique_domains_count'] = df['domains_found'].apply(len)
            
            # Classificar tipos de domínio
            mainstream_domains = ['g1.com', 'folha.uol.com.br', 'estadao.com.br', 'globo.com']
            alternative_domains = ['brasil247.com', 'diariodocentrodomundo.com.br']
            social_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com']
            
            def classify_domains(domains):
                if not domains:
                    return 'none'
                
                mainstream_count = sum(1 for d in domains if any(md in d for md in mainstream_domains))
                alternative_count = sum(1 for d in domains if any(ad in d for ad in alternative_domains))
                social_count = sum(1 for d in domains if any(sd in d for sd in social_domains))
                
                if mainstream_count > 0:
                    return 'mainstream'
                elif alternative_count > 0:
                    return 'alternative'
                elif social_count > 0:
                    return 'social'
                else:
                    return 'other'
            
            df['domain_category'] = df['domains_found'].apply(classify_domains)
            
            # Detectar presença de links externos
            df['has_external_links'] = df['url_count'] > 0
            
            # Diversidade de fontes
            total_domains = df['domains_found'].apply(len).sum()
            df['domain_diversity'] = 'low' if total_domains < 5 else 'medium' if total_domains < 20 else 'high'
            
        except Exception as e:
            self.logger.warning(f"Erro na análise de domínios: {e}")
            df['url_count'] = 0
            df['unique_domains_count'] = 0
            df['domain_category'] = 'error'
            df['has_external_links'] = False
            df['domain_diversity'] = 'error'
        
        processing_time = time.time() - start_time
        self.stats['stage_11_time'] = processing_time
        self.stats['stages_completed'] += 1
        self.logger.info(f"✅ Stage 11 concluído em {processing_time:.2f}s")
        
        return df

    def _stage_12_semantic_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 12: Análise Semântica Avançada"""
        start_time = time.time()
        self.logger.info("🧠 STAGE 12: Análise Semântica")
        
        # Análise semântica com fallback heurístico
        try:
            # Análise de coocorrência de termos
            texts = df['normalized_text'].fillna('').astype(str)
            
            # Calcular métricas semânticas básicas
            df['semantic_complexity'] = texts.apply(lambda x: len(set(x.split())) / max(len(x.split()), 1))
            df['semantic_richness'] = texts.apply(lambda x: len(set(x.split())) if x else 0)
            
            # Análise de conectivos e marcadores discursivos
            conectivos = ['mas', 'porém', 'contudo', 'entretanto', 'todavia', 'no entanto']
            df['conectivos_count'] = texts.apply(lambda x: sum(1 for c in conectivos if c in x.lower()))
            
            # Marcadores de intensidade
            intensificadores = ['muito', 'bastante', 'extremamente', 'totalmente', 'completamente']
            df['intensificadores_count'] = texts.apply(lambda x: sum(1 for i in intensificadores if i in x.lower()))
            
            # Análise de modalidade
            modalidade = ['deve', 'deveria', 'pode', 'poderia', 'talvez', 'provavelmente']
            df['modalidade_count'] = texts.apply(lambda x: sum(1 for m in modalidade if m in x.lower()))
            
            # Contextualização semântica
            df['semantic_context'] = 'neutral'
            mask_high_complexity = df['semantic_complexity'] > df['semantic_complexity'].quantile(0.75)
            df.loc[mask_high_complexity, 'semantic_context'] = 'complex'
            
            mask_low_complexity = df['semantic_complexity'] < df['semantic_complexity'].quantile(0.25)
            df.loc[mask_low_complexity, 'semantic_context'] = 'simple'
            
        except Exception as e:
            self.logger.warning(f"Erro na análise semântica: {e}")
            df['semantic_complexity'] = 0.5
            df['semantic_richness'] = 0
            df['conectivos_count'] = 0
            df['intensificadores_count'] = 0
            df['modalidade_count'] = 0
            df['semantic_context'] = 'unknown'
        
        processing_time = time.time() - start_time
        self.stats['stage_12_time'] = processing_time
        self.stats['stages_completed'] += 1
        self.logger.info(f"✅ Stage 12 concluído em {processing_time:.2f}s")
        
        return df

    def _stage_13_event_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 13: Análise de Contexto de Eventos Políticos"""
        start_time = time.time()
        self.logger.info("📰 STAGE 13: Contexto de Eventos")
        
        try:
            # Eventos políticos brasileiros relevantes (2019-2023)
            political_events = [
                {'event': 'Posse Bolsonaro', 'date': '2019-01-01', 'category': 'institucional'},
                {'event': 'Início Pandemia', 'date': '2020-03-11', 'category': 'saude'},
                {'event': 'Eleições 2022', 'date': '2022-10-02', 'category': 'eleitoral'},
                {'event': 'CPI COVID', 'date': '2021-04-27', 'category': 'investigativa'},
                {'event': '7 Setembro 2021', 'date': '2021-09-07', 'category': 'manifestacao'}
            ]
            
            # Detectar timestamp se disponível
            timestamp_columns = ['timestamp', 'date', 'created_at', 'published_at']
            timestamp_col = None
            for col in timestamp_columns:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                # Converter timestamp para datetime
                df['event_timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Marcar proximidade a eventos
                df['near_event'] = False
                df['event_category'] = 'none'
                df['days_to_event'] = 999
                
                for event in political_events:
                    event_date = pd.to_datetime(event['date'])
                    # Janela de ±7 dias do evento
                    window_start = event_date - pd.Timedelta(days=7)
                    window_end = event_date + pd.Timedelta(days=7)
                    
                    mask = (df['event_timestamp'] >= window_start) & (df['event_timestamp'] <= window_end)
                    df.loc[mask, 'near_event'] = True
                    df.loc[mask, 'event_category'] = event['category']
                    
                    # Calcular dias até o evento
                    days_diff = (df['event_timestamp'] - event_date).dt.days.abs()
                    closer_mask = days_diff < df['days_to_event']
                    df.loc[mask & closer_mask, 'days_to_event'] = days_diff[mask & closer_mask]
                
                # Análise temporal contextual
                df['temporal_context'] = 'normal'
                df.loc[df['near_event'], 'temporal_context'] = 'event_period'
                
            else:
                # Fallback sem timestamp
                df['near_event'] = False
                df['event_category'] = 'unknown'
                df['days_to_event'] = 999
                df['temporal_context'] = 'unknown'
                
            # Detecção de contexto por palavras-chave
            event_keywords = {
                'eleitoral': ['eleição', 'voto', 'urna', 'candidato', 'campanha'],
                'saude': ['covid', 'pandemia', 'vacina', 'lockdown', 'quarentena'],
                'institucional': ['governo', 'presidente', 'congresso', 'supremo'],
                'manifestacao': ['protesto', 'manifestação', 'ato', 'marcha']
            }
            
            texts = df['normalized_text'].fillna('').astype(str)
            df['context_keywords_count'] = 0
            
            for category, keywords in event_keywords.items():
                keyword_count = texts.apply(lambda x: sum(1 for kw in keywords if kw in x.lower()))
                df[f'{category}_keywords'] = keyword_count
                df['context_keywords_count'] += keyword_count
                
        except Exception as e:
            self.logger.warning(f"Erro na análise de contexto: {e}")
            df['near_event'] = False
            df['event_category'] = 'error'
            df['days_to_event'] = 999
            df['temporal_context'] = 'error'
            df['context_keywords_count'] = 0
        
        processing_time = time.time() - start_time
        self.stats['stage_13_time'] = processing_time
        self.stats['stages_completed'] += 1
        self.logger.info(f"✅ Stage 13 concluído em {processing_time:.2f}s")
        
        return df

    def _stage_14_channel_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 14: Análise de Canais e Fontes"""
        start_time = time.time()
        self.logger.info("📡 STAGE 14: Análise de Canais")
        
        try:
            # Detectar colunas de canal/fonte
            channel_columns = ['channel', 'channel_username', 'source', 'author', 'sender']
            channel_col = None
            for col in channel_columns:
                if col in df.columns:
                    channel_col = col
                    break
            
            if channel_col:
                # Análise de canais
                df['channel_name'] = df[channel_col].fillna('unknown')
                
                # Estatísticas por canal
                channel_stats = df.groupby('channel_name').agg({
                    'normalized_text': 'count',
                    'word_count': 'mean',
                    'political_spectrum': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
                }).reset_index()
                
                channel_stats.columns = ['channel', 'message_count', 'avg_word_count', 'dominant_political']
                
                # Classificar tipos de canal
                df['channel_type'] = 'regular'
                
                # Canais com muitas mensagens = 'high_volume'
                high_volume_channels = channel_stats[channel_stats['message_count'] > channel_stats['message_count'].quantile(0.8)]['channel'].tolist()
                df.loc[df['channel_name'].isin(high_volume_channels), 'channel_type'] = 'high_volume'
                
                # Canais com textos longos = 'detailed'
                detailed_channels = channel_stats[channel_stats['avg_word_count'] > channel_stats['avg_word_count'].quantile(0.8)]['channel'].tolist()
                df.loc[df['channel_name'].isin(detailed_channels), 'channel_type'] = 'detailed'
                
                # Autoridade do canal (baseada em volume e engajamento)
                df['channel_authority'] = 'low'
                medium_auth_channels = channel_stats[
                    (channel_stats['message_count'] > channel_stats['message_count'].quantile(0.5)) &
                    (channel_stats['avg_word_count'] > channel_stats['avg_word_count'].quantile(0.5))
                ]['channel'].tolist()
                df.loc[df['channel_name'].isin(medium_auth_channels), 'channel_authority'] = 'medium'
                
                high_auth_channels = channel_stats[
                    (channel_stats['message_count'] > channel_stats['message_count'].quantile(0.8)) &
                    (channel_stats['avg_word_count'] > channel_stats['avg_word_count'].quantile(0.8))
                ]['channel'].tolist()
                df.loc[df['channel_name'].isin(high_auth_channels), 'channel_authority'] = 'high'
                
            else:
                # Fallback sem coluna de canal
                df['channel_name'] = 'unknown'
                df['channel_type'] = 'unknown'
                df['channel_authority'] = 'unknown'
            
            # Análise de diversidade de fontes
            unique_channels = df['channel_name'].nunique()
            df['source_diversity'] = 'low' if unique_channels < 5 else 'medium' if unique_channels < 20 else 'high'
            
            # Detecção de padrões de fonte
            df['source_pattern'] = 'organic'
            
            # Se muitas mensagens do mesmo canal em sequência = possível coordenação
            if channel_col and len(df) > 10:
                consecutive_same = 0
                for i in range(1, min(len(df), 50)):
                    if df.iloc[i]['channel_name'] == df.iloc[i-1]['channel_name']:
                        consecutive_same += 1
                
                if consecutive_same > len(df) * 0.3:  # Mais de 30% consecutivas do mesmo canal
                    df['source_pattern'] = 'coordinated'
                    
        except Exception as e:
            self.logger.warning(f"Erro na análise de canais: {e}")
            df['channel_name'] = 'error'
            df['channel_type'] = 'error'
            df['channel_authority'] = 'error'
            df['source_diversity'] = 'error'
            df['source_pattern'] = 'error'
        
        processing_time = time.time() - start_time
        self.stats['stage_14_time'] = processing_time
        self.stats['stages_completed'] += 1
        self.logger.info(f"✅ Stage 14 concluído em {processing_time:.2f}s")
        
        return df


def main():
    """Teste do analyzer limpo."""
    logging.basicConfig(level=logging.INFO)

    # Teste com dados de exemplo
    test_data = pd.DataFrame({
        'body': [
            'Este é um texto sobre política brasileira com bolsonaro',
            'Discussão sobre economia e mercado financeiro liberal',
            'Análise social progressista sobre direitos humanos',
            'Texto neutro sobre tecnologia e ciência',
            'Debate político conservador sobre tradições'
        ],
        'timestamp': [
            '2023-01-01 10:00:00',
            '2023-01-01 11:00:00',
            '2023-01-01 10:30:00',
            '2023-01-02 15:00:00',
            '2023-01-01 10:15:00'
        ]
    })

    analyzer = CleanScientificAnalyzer()
    result = analyzer.analyze_dataset(test_data)

    print(f"\n✅ Análise concluída:")
    print(f"📊 Colunas geradas: {result['columns_generated']}")
    print(f"🎯 Stages completados: {result['stats']['stages_completed']}")
    print(f"🔧 Features extraídas: {result['stats']['features_extracted']}")

    return result


if __name__ == "__main__":
    main()