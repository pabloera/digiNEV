#!/usr/bin/env python3
"""
================================================================================
BATCH ANALYSIS INTEGRATED - digiNEV v5.0
================================================================================
Sistema integrado de análise em lote para discurso político brasileiro no Telegram

DESCRIÇÃO:
----------
Este script realiza análise completa de datasets de mensagens do Telegram com
foco em política brasileira (2019-2023), integrando APIs de IA (Anthropic Claude
e Voyage.ai) para análise avançada de discurso político, sentimento e semântica.

FUNCIONALIDADES:
---------------
- 13 estágios de análise organizados em 3 fases
- Integração com APIs de IA para análise avançada
- Processamento otimizado para grandes volumes de dados
- Suporte completo para português brasileiro
- Detecção de coordenação e padrões de rede
- Análise temporal e contextual de eventos políticos

REQUISITOS:
----------
- Python 3.12+
- APIs: Anthropic Claude, Voyage.ai (opcional)
- Memória: 4GB RAM mínimo
- Dependências: pandas, numpy, spacy, sklearn

AUTOR: Sistema digiNEV - Digital Network Violence Monitor
DATA: Setembro 2025
VERSÃO: 5.0.0-integrated
================================================================================
"""

import os
import sys
import json
import logging
import hashlib
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from functools import lru_cache
import re
import time

# Core dependencies
import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BatchAnalysis')

# Suppress warnings
warnings.filterwarnings('ignore')

# ================================================================================
# API IMPORTS AND CONFIGURATION
# ================================================================================

# Batch analyzer operates independently without external modules
API_MODULES_AVAILABLE = False

# Optional advanced dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy available")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("⚠️ spaCy not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
    logger.info("✅ scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️ scikit-learn not available")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    logger.info("✅ NetworkX available")
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("⚠️ NetworkX not available")

# ================================================================================
# CONFIGURATION
# ================================================================================

class BatchConfig:
    """Configuration class for batch analysis"""

    # API Configuration
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')

    # Model Settings
    ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"  # Cost-optimized
    MAX_TOKENS = 1000
    TEMPERATURE = 0.3

    # Batch Processing
    BATCH_SIZE = 100
    SAMPLE_SIZE = 1000
    MAX_WORKERS = 4

    # Cache Settings
    ENABLE_CACHE = True
    CACHE_TTL = 72  # hours

    # Lexicon Configuration
    LEXICON_FILE = "batch_analyzer/lexico_politico_hierarquizado.json"

    @classmethod
    def load_political_lexicon(cls):
        """Load political lexicon from JSON file"""
        try:
            lexicon_path = Path(cls.LEXICON_FILE)
            if not lexicon_path.exists():
                # Try relative to current file
                lexicon_path = Path(__file__).parent / "lexico_politico_hierarquizado.json"

            if lexicon_path.exists():
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return cls._transform_lexicon(data)
            else:
                logger.warning(f"Lexicon file not found: {lexicon_path}")
                return cls._get_default_keywords()
        except Exception as e:
            logger.error(f"Error loading lexicon: {e}")
            return cls._get_default_keywords()

    @classmethod
    def _transform_lexicon(cls, data):
        """Transform hierarchical lexicon into flat keyword structure"""
        political_keywords = {}
        transversal_keywords = {}

        if "lexico" in data:
            # Process each macrotema
            for macrotema_key, macrotema_data in data["lexico"].items():
                if isinstance(macrotema_data, dict):
                    # Collect all words from this macrotema for transversal keywords
                    macrotema_words = []

                    # Process subtemas
                    if "subtemas" in macrotema_data:
                        for subtema_key, subtema_data in macrotema_data["subtemas"].items():
                            if isinstance(subtema_data, dict) and "palavras" in subtema_data:
                                # Add to political keywords (using subtema as category)
                                political_keywords[subtema_key] = subtema_data["palavras"]
                                # Collect for macrotema
                                macrotema_words.extend(subtema_data["palavras"])

                    # Add to transversal keywords (using macrotema as category)
                    if macrotema_words:
                        transversal_keywords[macrotema_key] = macrotema_words

        return political_keywords, transversal_keywords
            'ustra', 'brilhante ustra', 'carlos alberto brilhante ustra',

# MAIN ANALYZER CLASS
# ================================================================================

class IntegratedBatchAnalyzer:
    """
    Sistema integrado de análise em lote com APIs de IA

    Este analisador processa datasets de mensagens do Telegram através de
    13 estágios de análise, integrando APIs de IA quando disponíveis para
    maximizar a qualidade da análise.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Inicializa o analisador com configuração e módulos de API

        Args:
            config: Configuração customizada ou usa padrão
        """
        self.config = config or BatchConfig()
        self.results = {}
        self.df = None
        self.api_stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'heuristic_count': 0,
            'estimated_cost': 0.0
        }

        # Load political lexicon dynamically
        self._load_lexicon()

        # Initialize API modules if available
        self._initialize_api_modules()

        # Initialize NLP models
        self._initialize_nlp_models()

        # Cache for API responses
        self.cache = {}

        logger.info("=" * 80)
        logger.info("🚀 INTEGRATED BATCH ANALYZER INITIALIZED")
        logger.info(f"📊 API Status: {'✅ Connected' if self.apis_available else '⚠️ Método Heurístico'}")
        logger.info(f"🧠 NLP Status: {'✅ spaCy Ready' if self.nlp else '⚠️ Limited'}")
        logger.info(f"📚 Lexicon: {len(self.political_keywords)} categories, {len(self.transversal_keywords)} macrotemas")
        logger.info("=" * 80)

    def _load_lexicon(self):
        """Load political lexicon from JSON file or use defaults"""
        try:
            # Try to load from JSON file
            self.political_keywords, self.transversal_keywords = self.config.load_political_lexicon()
            logger.info(f"✅ Loaded lexicon from {self.config.LEXICON_FILE}")
        except Exception as e:
            logger.warning(f"⚠️ Using default lexicon: {e}")
            # Use the hardcoded keywords as fallback
            self.political_keywords = self.config.POLITICAL_KEYWORDS
            self.transversal_keywords = self.config.TRANSVERSAL_KEYWORDS

    def _initialize_api_modules(self):
        """Initialize API modules with proper configuration"""
        self.apis_available = False
        self.api_modules = {}

        if not API_MODULES_AVAILABLE:
            logger.warning("API modules not imported - using heuristic methods")
            return

        if not self.config.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set - API features disabled")
            return

        try:
            # Initialize Anthropic-based analyzers
            anthropic_config = {
                'api_key': self.config.ANTHROPIC_API_KEY,
                'model': self.config.ANTHROPIC_MODEL,
                'max_tokens': self.config.MAX_TOKENS,
                'temperature': self.config.TEMPERATURE
            }

            # Political Analyzer
            self.api_modules['political'] = PoliticalAnalyzer(
                anthropic_config,
                Path.cwd()
            )
            logger.info("✅ Political Analyzer API initialized")

            # Sentiment Analyzer
            self.api_modules['sentiment'] = SentimentAnalyzer(
                anthropic_config,
                Path.cwd()
            )
            logger.info("✅ Sentiment Analyzer API initialized")

            # Initialize Voyage.ai modules if key available
            if self.config.VOYAGE_API_KEY:
                voyage_config = {
                    'voyage_api_key': self.config.VOYAGE_API_KEY,
                    **anthropic_config
                }

                # Topic Modeler
                self.api_modules['topic'] = VoyageTopicModeler(
                    voyage_config,
                    Path.cwd()
                )
                logger.info("✅ Topic Modeler API initialized")

                # TF-IDF Analyzer
                self.api_modules['tfidf'] = SemanticTfidfAnalyzer(
                    voyage_config,
                    Path.cwd()
                )
                logger.info("✅ TF-IDF Analyzer API initialized")

                # Clustering Analyzer
                self.api_modules['clustering'] = VoyageClusteringAnalyzer(
                    voyage_config,
                    Path.cwd()
                )
                logger.info("✅ Clustering Analyzer API initialized")

            # Network Analyzer
            self.api_modules['network'] = IntelligentNetworkAnalyzer(
                anthropic_config,
                Path.cwd()
            )
            logger.info("✅ Network Analyzer API initialized")

            # Domain Analyzer
            self.api_modules['domain'] = IntelligentDomainAnalyzer(
                anthropic_config,
                Path.cwd()
            )
            logger.info("✅ Domain Analyzer API initialized")

            self.apis_available = True
            logger.info("🎯 All API modules successfully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize API modules: {e}")
            self.apis_available = False

    def _initialize_nlp_models(self):
        """Initialize NLP models (spaCy)"""
        self.nlp = None

        if SPACY_AVAILABLE:
            try:
                # Try to load Portuguese model
                self.nlp = spacy.load('pt_core_news_lg')
                logger.info("✅ spaCy Portuguese model loaded")
            except:
                try:
                    # Use simpler model
                    self.nlp = spacy.load('pt_core_news_sm')
                    logger.info("✅ spaCy Portuguese model (small) loaded")
                except:
                    logger.warning("spaCy models not installed - run: python -m spacy download pt_core_news_lg")

    # ================================================================================
    # STAGE 01: PREPROCESSING
    # ================================================================================

    def stage_01_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 01: Pré-processamento e Limpeza de Dados

        Realiza:
        - Validação de dados
        - Normalização de texto
        - Deduplicação
        - Filtragem de qualidade

        Args:
            df: DataFrame com mensagens brutas

        Returns:
            DataFrame limpo e preparado
        """
        logger.info("=" * 80)
        logger.info("🔧 STAGE 01: PREPROCESSING")
        logger.info("=" * 80)

        stage_start = datetime.now()
        initial_count = len(df)

        # Data validation
        logger.info(f"📥 Initial records: {initial_count:,}")

        # Auto-detect text column
        text_column = None
        for col in ['text', 'body', 'message', 'content', 'texto', 'mensagem']:
            if col in df.columns:
                text_column = col
                logger.info(f"📝 Using text column: '{text_column}'")
                break

        if text_column is None:
            logger.error("❌ No text column found in dataset")
            return df

        # Standardize column names for compatibility
        df['body'] = df[text_column].fillna('')
        df['body_cleaned'] = df.get('body_cleaned', df['body']).fillna('')

        # Text normalization
        df['text_normalized'] = df['body_cleaned'].apply(self._normalize_text)

        # Remove empty messages
        df = df[df['text_normalized'].str.len() > 10]  # Minimum 10 characters
        logger.info(f"📝 After removing empty: {len(df):,}")

        # Deduplicate exact matches
        df = df.drop_duplicates(subset=['text_normalized'], keep='first')
        logger.info(f"🔄 After deduplication: {len(df):,}")

        # Add preprocessing metadata
        df['preprocessing_timestamp'] = datetime.now().isoformat()
        df['text_length'] = df['text_normalized'].str.len()
        df['word_count'] = df['text_normalized'].str.split().str.len()

        # Quality filtering
        if 'quality_score' in df.columns:
            quality_threshold = 0.3
            df = df[df['quality_score'] > quality_threshold]
            logger.info(f"✅ After quality filter (>{quality_threshold}): {len(df):,}")

        # Calculate statistics
        final_count = len(df)
        processing_time = (datetime.now() - stage_start).total_seconds()

        self.results['preprocessing'] = {
            'initial_records': initial_count,
            'final_records': final_count,
            'removed_records': initial_count - final_count,
            'removal_rate': (initial_count - final_count) / initial_count if initial_count > 0 else 0,
            'processing_time': processing_time,
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'quality_threshold': 0.3 if 'quality_score' in df.columns else None
        }

        logger.info(f"✅ Preprocessing complete: {final_count:,}/{initial_count:,} records retained")
        logger.info(f"⏱️ Processing time: {processing_time:.2f}s")

        return df

    # ================================================================================
    # STAGE 02: TEXT MINING WITH APIs
    # ================================================================================

    def stage_02_text_mining(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 02: Text Mining com APIs de IA

        Realiza:
        - Classificação política (Anthropic API)
        - Análise de sentimento (Anthropic API)
        - Extração de tópicos básicos
        - Extração de entidades (hashtags, menções, URLs)

        Args:
            df: DataFrame preprocessado

        Returns:
            DataFrame com análises de text mining
        """
        logger.info("=" * 80)
        logger.info("🔍 STAGE 02: TEXT MINING (with APIs)")
        logger.info("=" * 80)

        stage_start = datetime.now()

        # Political Classification with API
        if self.apis_available and 'political' in self.api_modules:
            logger.info("🤖 Using Political Analyzer API...")
            try:
                # Use the API module with proper method
                result = self.api_modules['political'].analyze_political_content(
                    df,
                    text_column='text_normalized'
                )
                # Check if result is tuple (df, metadata) or just df
                if isinstance(result, tuple):
                    df, metadata = result
                    self.api_stats['api_calls'] += metadata.get('api_calls', 10)
                else:
                    df = result
                    self.api_stats['api_calls'] += 10
                logger.info(f"✅ Political classification via API complete")
            except Exception as e:
                logger.error(f"API failed, using heuristic method: {e}")
                df = self._heuristic_political_classification(df)
                self.api_stats['heuristic_count'] += 1
        else:
            df = self._heuristic_political_classification(df)
            self.api_stats['heuristic_count'] += 1

        # Sentiment Analysis with API
        if self.apis_available and 'sentiment' in self.api_modules:
            logger.info("🤖 Using Sentiment Analyzer API...")
            try:
                # Use the API module with proper method
                df = self.api_modules['sentiment'].analyze_sentiment_ultra_optimized(
                    df,
                    text_column='text_normalized'
                )
                self.api_stats['api_calls'] += len(df) // 10  # Estimate API calls
                logger.info("✅ Sentiment analysis via API complete")
            except Exception as e:
                logger.error(f"API failed, using heuristic method: {e}")
                df = self._heuristic_sentiment_analysis(df)
                self.api_stats['heuristic_count'] += 1
        else:
            df = self._heuristic_sentiment_analysis(df)
            self.api_stats['heuristic_count'] += 1

        # Basic topic extraction
        df = self._extract_basic_topics(df)

        # Entity extraction
        df['hashtags'] = df['text_normalized'].apply(self._extract_hashtags)
        df['mentions'] = df['text_normalized'].apply(self._extract_mentions)
        df['urls'] = df['text_normalized'].apply(self._extract_urls)
        df['has_media'] = df['urls'].apply(lambda x: len(x) > 0)

        processing_time = (datetime.now() - stage_start).total_seconds()

        # Store results
        self.results['text_mining'] = {
            'political_distribution': df['political_category'].value_counts().to_dict() if 'political_category' in df else {},
            'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df else {},
            'avg_sentiment_score': df['sentiment_score'].mean() if 'sentiment_score' in df else 0,
            'topics_extracted': df['topic'].nunique() if 'topic' in df else 0,
            'api_calls': self.api_stats['api_calls'],
            'heuristic_used': self.api_stats['heuristic_count'],
            'processing_time': processing_time
        }

        logger.info(f"✅ Text mining complete in {processing_time:.2f}s")
        logger.info(f"📊 API calls: {self.api_stats['api_calls']}, Heuristics: {self.api_stats['heuristic_count']}")

        return df

    # ================================================================================
    # STAGE 03-13: ADVANCED ANALYSIS STAGES
    # ================================================================================

    def stage_03_statistical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 03: Análise Estatística Avançada"""
        logger.info("=" * 80)
        logger.info("📊 STAGE 03: STATISTICAL ANALYSIS")

        stage_start = datetime.now()

        stats = {
            'message_length': {
                'mean': df['text_length'].mean(),
                'median': df['text_length'].median(),
                'std': df['text_length'].std(),
                'min': df['text_length'].min(),
                'max': df['text_length'].max(),
                'q25': df['text_length'].quantile(0.25),
                'q75': df['text_length'].quantile(0.75)
            },
            'word_count': {
                'mean': df['word_count'].mean(),
                'median': df['word_count'].median(),
                'std': df['word_count'].std()
            }
        }

        # Correlation analysis
        if 'sentiment_score' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()
                stats['correlations'] = correlations.to_dict()

        # Distribution analysis
        if 'political_category' in df.columns:
            stats['political_entropy'] = self._calculate_entropy(
                df['political_category'].value_counts()
            )

        self.results['statistical_analysis'] = stats
        logger.info(f"✅ Complete in {(datetime.now() - stage_start).total_seconds():.2f}s")

        return df

    def stage_04_semantic_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 04: Análise Semântica com APIs"""
        logger.info("=" * 80)
        logger.info("🧠 STAGE 04: SEMANTIC ANALYSIS")

        if self.apis_available and 'tfidf' in self.api_modules:
            logger.info("🤖 Using Semantic TF-IDF API...")
            try:
                # Use the correct API method
                df = self.api_modules['tfidf'].extract_semantic_tfidf(
                    df,
                    text_column='text_normalized'
                )
                self.api_stats['api_calls'] += len(df) // 100  # Estimate
            except Exception as e:
                logger.error(f"API failed: {e}")
                df = self._heuristic_semantic_analysis(df)
                self.api_stats['heuristic_count'] += 1
        else:
            df = self._heuristic_semantic_analysis(df)
            self.api_stats['heuristic_count'] += 1

        return df

    def stage_05_tfidf_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 05: TF-IDF Analysis"""
        logger.info("=" * 80)
        logger.info("📈 STAGE 05: TF-IDF ANALYSIS")

        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available")
            return df

        stage_start = datetime.now()

        # TF-IDF by political category
        tfidf_results = {}

        if 'political_category' in df.columns:
            for category in df['political_category'].unique():
                if category and len(df[df['political_category'] == category]) >= 5:
                    category_df = df[df['political_category'] == category]

                    vectorizer = TfidfVectorizer(
                        max_features=30,
                        min_df=2,
                        max_df=0.8,
                        ngram_range=(1, 2)
                    )

                    try:
                        tfidf_matrix = vectorizer.fit_transform(category_df['text_normalized'])
                        feature_names = vectorizer.get_feature_names_out()
                        scores = tfidf_matrix.sum(axis=0).A1
                        top_indices = scores.argsort()[-10:][::-1]
                        top_keywords = [(feature_names[i], float(scores[i])) for i in top_indices]

                        tfidf_results[category] = {
                            'top_keywords': top_keywords,
                            'vocabulary_size': len(feature_names)
                        }
                    except Exception as e:
                        logger.warning(f"TF-IDF failed for {category}: {e}")

        self.results['tfidf_analysis'] = tfidf_results
        logger.info(f"✅ Complete in {(datetime.now() - stage_start).total_seconds():.2f}s")

        return df

    def stage_06_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 06: Clustering com APIs"""
        logger.info("=" * 80)
        logger.info("🔮 STAGE 06: CLUSTERING")

        if self.apis_available and 'clustering' in self.api_modules:
            logger.info("🤖 Using Voyage Clustering API...")
            try:
                # Use the correct API method with text column
                df = self.api_modules['clustering'].perform_semantic_clustering(
                    df,
                    text_column='text_normalized'
                )
                self.api_stats['api_calls'] += 10  # Estimate API calls
            except Exception as e:
                logger.error(f"API failed: {e}")
                df = self._heuristic_clustering(df)
        else:
            df = self._fallback_clustering(df)

        return df

    def stage_07_topic_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 07: Topic Modeling com APIs"""
        logger.info("=" * 80)
        logger.info("🎯 STAGE 07: TOPIC MODELING")

        if self.apis_available and 'topic' in self.api_modules:
            logger.info("🤖 Using Voyage Topic Modeler API...")
            try:
                # Use the correct API method
                df = self.api_modules['topic'].extract_semantic_topics(
                    df,
                    text_column='text_normalized'
                )
                self.api_stats['api_calls'] += 10  # Estimate API calls
            except Exception as e:
                logger.error(f"API failed: {e}")
                df = self._heuristic_topic_modeling(df)
        else:
            df = self._fallback_topic_modeling(df)

        return df

    def stage_08_evolution_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 08: Análise de Evolução Temporal"""
        logger.info("=" * 80)
        logger.info("📅 STAGE 08: EVOLUTION ANALYSIS")

        stage_start = datetime.now()

        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column")
            return df

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        temporal_df = df.dropna(subset=['timestamp']).copy()

        if len(temporal_df) == 0:
            return df

        # Extract temporal features
        temporal_df['date'] = temporal_df['timestamp'].dt.date
        temporal_df['year'] = temporal_df['timestamp'].dt.year
        temporal_df['month'] = temporal_df['timestamp'].dt.month
        temporal_df['day_of_week'] = temporal_df['timestamp'].dt.dayofweek
        temporal_df['hour'] = temporal_df['timestamp'].dt.hour
        temporal_df['year_month'] = temporal_df['timestamp'].dt.to_period('M')

        # Political evolution over time
        if 'political_category' in temporal_df.columns:
            political_evolution = temporal_df.groupby(
                ['year_month', 'political_category']
            ).size().unstack(fill_value=0)

            self.results['evolution_analysis'] = {
                'political_trends': political_evolution.to_dict(),
                'total_months': len(political_evolution),
                'processing_time': (datetime.now() - stage_start).total_seconds()
            }

        # Merge temporal features back
        for col in ['year', 'month', 'day_of_week', 'hour']:
            if col in temporal_df.columns:
                df[col] = temporal_df[col]

        logger.info(f"✅ Complete in {(datetime.now() - stage_start).total_seconds():.2f}s")

        return df

    def stage_09_network_coordination(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 09: Network & Coordination Analysis com APIs"""
        logger.info("=" * 80)
        logger.info("🌐 STAGE 09: NETWORK & COORDINATION")

        if self.apis_available and 'network' in self.api_modules:
            logger.info("🤖 Using Network Analyzer API...")
            try:
                # Use the correct API method - returns Dict
                results = self.api_modules['network'].analyze_networks(df)
                # Add network analysis results to dataframe
                if 'coordination_score' in results:
                    df['coordination_score'] = results.get('coordination_score', 0)
                if 'network_density' in results:
                    df['network_density'] = results.get('network_density', 0)
                self.api_stats['api_calls'] += 10  # Estimate API calls
            except Exception as e:
                logger.error(f"API failed: {e}")
                df = self._heuristic_network_analysis(df)
        else:
            df = self._fallback_network_analysis(df)

        return df

    def stage_10_domain_url_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 10: Domain & URL Analysis com APIs"""
        logger.info("=" * 80)
        logger.info("🔗 STAGE 10: DOMAIN & URL ANALYSIS")

        if self.apis_available and 'domain' in self.api_modules:
            logger.info("🤖 Using Domain Analyzer API...")
            try:
                # Use the correct API method - returns DataFrame
                df = self.api_modules['domain'].analyze_domains(
                    df,
                    text_column='text_normalized'
                )
                self.api_stats['api_calls'] += 5  # Estimate API calls
            except Exception as e:
                logger.error(f"API failed: {e}")
                df = self._heuristic_domain_analysis(df)
        else:
            df = self._fallback_domain_analysis(df)

        return df

    def stage_11_event_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 11: Brazilian Political Event Context Analysis"""
        logger.info("=" * 80)
        logger.info("📰 STAGE 11: EVENT CONTEXT ANALYSIS")

        stage_start = datetime.now()

        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column")
            return df

        # Convert events to datetime
        events = self.config.POLITICAL_EVENTS.copy()
        for event in events:
            event['date'] = pd.to_datetime(event['date'])

        # Analyze message proximity to events
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        event_analysis = []

        for event in events:
            # Messages within 7 days of event
            window_start = event['date'] - timedelta(days=3)
            window_end = event['date'] + timedelta(days=4)

            event_messages = df[
                (df['timestamp'] >= window_start) &
                (df['timestamp'] <= window_end)
            ]

            if len(event_messages) > 0:
                event_stats = {
                    'event': event['event'],
                    'category': event['category'],
                    'date': event['date'].strftime('%Y-%m-%d'),
                    'message_count': len(event_messages)
                }

                if 'sentiment_score' in event_messages.columns:
                    event_stats['sentiment_avg'] = event_messages['sentiment_score'].mean()

                if 'political_category' in event_messages.columns:
                    event_stats['political_distribution'] = event_messages['political_category'].value_counts().to_dict()

                event_analysis.append(event_stats)

        # Mark messages near events
        df['near_event'] = False
        df['event_name'] = None

        for event in events:
            window_start = event['date'] - timedelta(days=3)
            window_end = event['date'] + timedelta(days=4)

            mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
            df.loc[mask, 'near_event'] = True
            df.loc[mask, 'event_name'] = event['event']

        self.results['event_context'] = {
            'events_analyzed': len(events),
            'event_details': event_analysis,
            'messages_near_events': df['near_event'].sum(),
            'processing_time': (datetime.now() - stage_start).total_seconds()
        }

        logger.info(f"✅ Complete in {(datetime.now() - stage_start).total_seconds():.2f}s")

        return df

    def stage_12_channel_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 12: Telegram Channel Analysis"""
        logger.info("=" * 80)
        logger.info("📡 STAGE 12: CHANNEL ANALYSIS")

        stage_start = datetime.now()

        if 'channel_username' not in df.columns:
            logger.warning("No channel information")
            return df

        channel_stats = []

        for channel in df['channel_username'].dropna().unique():
            channel_df = df[df['channel_username'] == channel]

            stats = {
                'channel': channel,
                'message_count': len(channel_df),
                'unique_users': channel_df['sender_id'].nunique() if 'sender_id' in channel_df else None,
                'avg_text_length': channel_df['text_length'].mean() if 'text_length' in channel_df else None
            }

            if 'sentiment_score' in channel_df.columns:
                stats['avg_sentiment'] = channel_df['sentiment_score'].mean()

            if 'political_category' in channel_df.columns:
                mode = channel_df['political_category'].mode()
                if len(mode) > 0:
                    stats['dominant_political'] = mode.iloc[0]

            channel_stats.append(stats)

        # Sort by message count
        channel_stats = sorted(channel_stats, key=lambda x: x['message_count'], reverse=True)

        self.results['channel_analysis'] = {
            'total_channels': len(channel_stats),
            'top_channels': channel_stats[:10],
            'processing_time': (datetime.now() - stage_start).total_seconds()
        }

        logger.info(f"✅ Complete in {(datetime.now() - stage_start).total_seconds():.2f}s")

        return df

    def stage_13_linguistic_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """STAGE 13: Linguistic Analysis with spaCy"""
        logger.info("=" * 80)
        logger.info("🔤 STAGE 13: LINGUISTIC ANALYSIS (spaCy)")

        stage_start = datetime.now()

        if not self.nlp:
            logger.warning("spaCy not available")
            return df

        # Sample for efficiency
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)

        linguistic_features = []
        entities_found = defaultdict(list)

        for idx, text in sample_df['text_normalized'].items():
            try:
                doc = self.nlp(text[:1000])  # Limit text length

                # Extract features
                features = {
                    'n_tokens': len(doc),
                    'n_sentences': len(list(doc.sents)),
                    'n_entities': len(doc.ents),
                }

                # Named entities
                for ent in doc.ents:
                    entities_found[ent.label_].append(ent.text)

                # POS tags
                pos_counts = Counter([token.pos_ for token in doc])
                features.update({
                    'n_verbs': pos_counts.get('VERB', 0),
                    'n_nouns': pos_counts.get('NOUN', 0),
                    'n_adjectives': pos_counts.get('ADJ', 0)
                })

                linguistic_features.append(features)

            except Exception as e:
                logger.warning(f"spaCy error: {e}")

        if linguistic_features:
            avg_features = {
                'avg_tokens': np.mean([f['n_tokens'] for f in linguistic_features]),
                'avg_sentences': np.mean([f['n_sentences'] for f in linguistic_features]),
                'avg_entities': np.mean([f['n_entities'] for f in linguistic_features]),
                'top_entities': {
                    label: Counter(ents).most_common(5)
                    for label, ents in entities_found.items()
                }
            }

            self.results['linguistic_analysis'] = avg_features

        logger.info(f"✅ Complete in {(datetime.now() - stage_start).total_seconds():.2f}s")

        return df

    # ================================================================================
    # HELPER METHODS
    # ================================================================================

    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis"""
        if not isinstance(text, str):
            return ""

        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http\S+', ' URL ', text)
        text = re.sub(r'@\w+', ' MENTION ', text)
        text = re.sub(r'#(\w+)', r' HASHTAG_\1 ', text)

        return text

    def _heuristic_political_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thematic classification of political discourse during Bolsonaro government"""
        logger.info("⚠️ Using thematic political classification")

        def classify(text):
            text_lower = text.lower()
            category_scores = {}
            detected_themes = []

            # Score each thematic category
            for category, keywords in self.political_keywords.items():
                score = 0
                matched_keywords = []

                for keyword in keywords:
                    if keyword in text_lower:
                        # Give higher weight to more specific terms
                        weight = 2.0 if len(keyword.split()) > 2 else 1.5 if len(keyword.split()) > 1 else 1.0
                        score += weight
                        matched_keywords.append(keyword)

                if score > 0:
                    category_scores[category] = score
                    # Extract category number and name for reporting
                    cat_num = category.split('_')[0].replace('cat', '')
                    cat_name = '_'.join(category.split('_')[1:])
                    detected_themes.append(f"{cat_num}:{cat_name}")

            # Apply bonus scores from transversal categories
            if hasattr(self, 'transversal_keywords'):
                # Pandemic context - typically aligns with cat1_pandemia and cat0_autoritarismo
                if 'pandemia' in self.transversal_keywords:
                    pandemic_terms = ['cloroquina', 'ivermectina', 'tratamento precoce', 'gripezinha']
                    if any(term in text_lower for term in pandemic_terms):
                        category_scores['cat1_pandemia'] = category_scores.get('cat1_pandemia', 0) + 3
                        # Pandemic denialism often linked to authoritarianism
                        if 'negacionismo' in text_lower or 'gripezinha' in text_lower:
                            category_scores['cat0_autoritarismo'] = category_scores.get('cat0_autoritarismo', 0) + 1

                # Corruption context - cat7_corrupcao
                if 'corrupcao' in self.transversal_keywords:
                    corruption_terms = ['lava jato', 'petrolão', 'mensalão', 'rachadinha', 'orçamento secreto']
                    if any(term in text_lower for term in corruption_terms):
                        category_scores['cat7_corrupcao'] = category_scores.get('cat7_corrupcao', 0) + 2

                # Political violence - often related to cat0_autoritarismo and cat5_militarismo
                if 'violencia_politica' in self.transversal_keywords:
                    violence_terms = ['8 de janeiro', 'oito de janeiro', 'golpista', 'golpe']
                    if any(term in text_lower for term in violence_terms):
                        category_scores['cat0_autoritarismo'] = category_scores.get('cat0_autoritarismo', 0) + 2
                        category_scores['cat5_militarismo'] = category_scores.get('cat5_militarismo', 0) + 1

                # Religious context - cat9_religiao
                if 'religiao' in self.transversal_keywords:
                    religious_terms = ['valores cristãos', 'família tradicional', 'ideologia de gênero']
                    if any(term in text_lower for term in religious_terms):
                        category_scores['cat9_religiao'] = category_scores.get('cat9_religiao', 0) + 2

            # Identify cross-category patterns (Bolsonarismo markers)
            bolsonarismo_score = 0
            if 'bolsonaro' in text_lower or 'mito' in text_lower:
                bolsonarismo_score += 3
            if any(cat in category_scores for cat in ['cat0_autoritarismo', 'cat5_militarismo']):
                bolsonarismo_score += 1
            if any(cat in category_scores for cat in ['cat1_pandemia', 'cat6_ideologia']):
                bolsonarismo_score += 1

            # No clear theme detected
            if not category_scores:
                return 'sem_categoria', [], 0.1, 0

            # Get primary category (highest score)
            primary_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[primary_category]

            # Get secondary categories (score > threshold)
            threshold = max_score * 0.4  # Categories with at least 40% of max score
            secondary_categories = [cat for cat, score in category_scores.items()
                                   if cat != primary_category and score >= threshold]

            # Calculate confidence based on score strength and clarity
            total_score = sum(category_scores.values())
            primary_ratio = max_score / total_score if total_score > 0 else 0

            # Higher confidence if one category dominates
            confidence = min(primary_ratio + (max_score / 20), 1.0)

            # Format result
            category_label = primary_category
            if secondary_categories:
                category_label = f"{primary_category}+{len(secondary_categories)}"

            return category_label, detected_themes, confidence, bolsonarismo_score

        # Apply classification
        results = df['text_normalized'].apply(classify)

        # Extract results into separate columns
        df['thematic_category'] = results.apply(lambda x: x[0])
        df['detected_themes'] = results.apply(lambda x: x[1])
        df['theme_confidence'] = results.apply(lambda x: x[2])
        df['bolsonarismo_score'] = results.apply(lambda x: x[3])

        # Add binary flag for high bolsonarismo alignment
        df['bolsonarismo_aligned'] = df['bolsonarismo_score'] >= 3

        return df

    def _heuristic_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic sentiment analysis without API"""
        logger.info("⚠️ Using heuristic sentiment analysis")

        positive_words = ['bom', 'ótimo', 'excelente', 'parabéns', 'sucesso', 'vitória', 'feliz', 'amor', 'conquista']
        negative_words = ['ruim', 'péssimo', 'horrível', 'fracasso', 'derrota', 'triste', 'corrupto', 'crime', 'violência']

        def calculate_sentiment(text):
            text_lower = text.lower()
            pos_score = sum(1 for word in positive_words if word in text_lower)
            neg_score = sum(1 for word in negative_words if word in text_lower)

            if pos_score > neg_score:
                return 'positivo', (pos_score - neg_score) / (pos_score + neg_score + 1)
            elif neg_score > pos_score:
                return 'negativo', -(neg_score - pos_score) / (pos_score + neg_score + 1)
            else:
                return 'neutro', 0.0

        sentiment_results = df['text_normalized'].apply(calculate_sentiment)
        df['sentiment'] = sentiment_results.apply(lambda x: x[0])
        df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])

        return df

    def _extract_basic_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic topics from text"""
        topics_keywords = {
            'economia': ['economia', 'inflação', 'emprego', 'salário', 'mercado', 'dólar', 'pib'],
            'saúde': ['saúde', 'vacina', 'covid', 'hospital', 'médico', 'sus', 'pandemia'],
            'educação': ['educação', 'escola', 'universidade', 'professor', 'aluno', 'enem', 'ensino'],
            'segurança': ['segurança', 'polícia', 'crime', 'violência', 'prisão', 'bandido', 'tráfico'],
            'corrupção': ['corrupção', 'propina', 'desvio', 'lava jato', 'investigação', 'petrolão'],
            'eleições': ['eleição', 'voto', 'candidato', 'urna', 'campanha', 'tse', 'pesquisa'],
            'meio_ambiente': ['ambiente', 'amazônia', 'desmatamento', 'clima', 'sustentável'],
            'direitos_sociais': ['direitos', 'igualdade', 'racismo', 'feminismo', 'lgbt', 'inclusão']
        }

        def identify_topic(text):
            text_lower = text.lower()
            for topic, keywords in topics_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    return topic
            return 'outros'

        df['topic'] = df['text_normalized'].apply(identify_topic)
        return df

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags"""
        return re.findall(r'#\w+', text)

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions"""
        return re.findall(r'@\w+', text)

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs"""
        return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    def _calculate_entropy(self, counts):
        """Calculate Shannon entropy"""
        if len(counts) == 0:
            return 0
        probs = counts / counts.sum()
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def _heuristic_semantic_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic semantic analysis without API"""
        logger.info("⚠️ Using heuristic semantic analysis")

        if not SKLEARN_AVAILABLE:
            df['semantic_diversity'] = df['text_normalized'].apply(lambda x: len(set(x.split())) / (len(x.split()) + 1))
            return df

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Sample for efficiency
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
            tfidf_matrix = vectorizer.fit_transform(sample_df['text_normalized'])

            # Calculate average similarity
            avg_similarity = self._calculate_sample_similarity(tfidf_matrix, n_samples=100)

            # Find similar groups
            similar_groups = self._find_similar_groups(tfidf_matrix, threshold=0.7)

            # Add semantic uniqueness
            df['semantic_uniqueness'] = 0.5  # Default
            for idx in sample_df.index:
                if idx in df.index:
                    df.loc[idx, 'semantic_uniqueness'] = self._calculate_uniqueness(idx, tfidf_matrix, sample_df.index)

            self.results['semantic_analysis'] = {
                'avg_semantic_similarity': float(avg_similarity),
                'similar_groups_found': len(similar_groups),
                'unique_messages_ratio': (df['semantic_uniqueness'] > 0.7).sum() / len(df)
            }
        except Exception as e:
            logger.error(f"Semantic analysis error: {e}")
            df['semantic_diversity'] = df['text_normalized'].apply(lambda x: len(set(x.split())) / (len(x.split()) + 1))

        return df

    def _heuristic_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic clustering without API"""
        logger.info("⚠️ Using heuristic clustering")

        if not SKLEARN_AVAILABLE:
            df['cluster_id'] = df['text_length'].apply(lambda x: int(x / 100))  # Simple length-based clustering
            return df

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans, DBSCAN

            # Sample for efficiency
            sample_size = min(500, len(df))
            sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

            # TF-IDF features for clustering
            vectorizer = TfidfVectorizer(max_features=50, min_df=2, max_df=0.8)
            features = vectorizer.fit_transform(sample_df['text_normalized'])

            # K-Means clustering
            n_clusters = min(8, len(sample_df) // 10) if len(sample_df) >= 20 else 2
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(features)
                sample_df['kmeans_cluster'] = kmeans_labels

                # DBSCAN for anomaly detection
                dbscan = DBSCAN(eps=0.3, min_samples=3)
                dbscan_labels = dbscan.fit_predict(features.toarray())
                sample_df['dbscan_cluster'] = dbscan_labels

                # Merge results back
                if 'kmeans_cluster' not in df.columns:
                    df['kmeans_cluster'] = -1
                if 'dbscan_cluster' not in df.columns:
                    df['dbscan_cluster'] = -1

                for idx in sample_df.index:
                    if idx in df.index:
                        df.loc[idx, 'kmeans_cluster'] = sample_df.loc[idx, 'kmeans_cluster']
                        df.loc[idx, 'dbscan_cluster'] = sample_df.loc[idx, 'dbscan_cluster']

                self.results['clustering'] = {
                    'n_kmeans_clusters': n_clusters,
                    'n_outliers': (sample_df['dbscan_cluster'] == -1).sum(),
                    'outlier_ratio': (sample_df['dbscan_cluster'] == -1).sum() / len(sample_df)
                }
            else:
                df['cluster_id'] = 0
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            df['cluster_id'] = df['text_length'].apply(lambda x: int(x / 100))

        return df

    def _heuristic_topic_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic topic modeling without API"""
        logger.info("⚠️ Using heuristic topic modeling")

        if not SKLEARN_AVAILABLE:
            df['lda_topic'] = df['topic']  # Use basic topics as heuristic
            return df

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            # Sample for LDA
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

            # Prepare features
            vectorizer = TfidfVectorizer(max_features=100, min_df=3, max_df=0.7)
            doc_term_matrix = vectorizer.fit_transform(sample_df['text_normalized'])
            feature_names = vectorizer.get_feature_names_out()

            # LDA model
            n_topics = 5
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda_features = lda.fit_transform(doc_term_matrix)

            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                top_weights = [topic[i] for i in top_indices]

                # Interpret topic
                topic_label = self._interpret_topic(top_words)

                topics.append({
                    'topic_id': topic_idx,
                    'label': topic_label,
                    'top_words': list(zip(top_words, top_weights))[:5],
                    'coherence': self._calculate_topic_coherence(top_words)
                })

            # Assign dominant topic
            sample_df['lda_topic_id'] = lda_features.argmax(axis=1)
            sample_df['lda_topic_prob'] = lda_features.max(axis=1)

            # Merge back
            if 'lda_topic' not in df.columns:
                df['lda_topic'] = -1
            if 'lda_topic_prob' not in df.columns:
                df['lda_topic_prob'] = 0

            for idx in sample_df.index:
                if idx in df.index:
                    df.loc[idx, 'lda_topic'] = sample_df.loc[idx, 'lda_topic_id']
                    df.loc[idx, 'lda_topic_prob'] = sample_df.loc[idx, 'lda_topic_prob']

            self.results['topic_modeling'] = {
                'n_topics': n_topics,
                'topics': topics,
                'topic_distribution': df['lda_topic'].value_counts().to_dict()
            }

        except Exception as e:
            logger.error(f"Topic modeling error: {e}")
            df['lda_topic'] = df['topic']  # Use basic topics as heuristic

        return df

    def _heuristic_network_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic network analysis without API"""
        logger.info("⚠️ Using heuristic network analysis")

        # Simple duplicate detection
        df['content_hash'] = df['text_normalized'].apply(lambda x: hash(x))
        duplicate_counts = df.groupby('content_hash').size()
        df['is_duplicate'] = df['content_hash'].isin(duplicate_counts[duplicate_counts > 1].index)

        return df

    def _heuristic_domain_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic domain analysis without API"""
        logger.info("⚠️ Using heuristic domain analysis")

        def extract_domain(url):
            try:
                from urllib.parse import urlparse
                return urlparse(url).netloc.lower().replace('www.', '')
            except:
                return None

        # Extract domains
        all_domains = []
        for urls in df['urls']:
            for url in urls:
                domain = extract_domain(url)
                if domain:
                    all_domains.append(domain)

        # Store top domains
        if all_domains:
            domain_counts = Counter(all_domains)
            self.results['domain_analysis'] = {
                'top_domains': dict(domain_counts.most_common(10)),
                'unique_domains': len(set(all_domains))
            }

        return df

    # Additional helper methods from unified script
    def _calculate_sample_similarity(self, tfidf_matrix, n_samples=100):
        """Calculate average similarity on sample pairs"""
        n_docs = tfidf_matrix.shape[0]
        if n_docs < 2:
            return 0

        similarities = []
        for _ in range(min(n_samples, n_docs * (n_docs - 1) // 2)):
            i, j = np.random.choice(n_docs, 2, replace=False)
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0]
            similarities.append(sim)

        return np.mean(similarities) if similarities else 0

    def _find_similar_groups(self, tfidf_matrix, threshold=0.7):
        """Find groups of similar messages"""
        n_docs = tfidf_matrix.shape[0]
        if n_docs < 2:
            return []

        groups = []
        visited = set()

        for i in range(min(n_docs, 100)):  # Limit for efficiency
            if i not in visited:
                group = [i]
                visited.add(i)

                for j in range(i + 1, min(n_docs, 100)):
                    if j not in visited:
                        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0]
                        if sim >= threshold:
                            group.append(j)
                            visited.add(j)

                if len(group) > 1:
                    groups.append(group)

        return groups

    def _calculate_uniqueness(self, idx, tfidf_matrix, indices):
        """Calculate semantic uniqueness score"""
        try:
            idx_pos = list(indices).index(idx)
            similarities = cosine_similarity(tfidf_matrix[idx_pos], tfidf_matrix).flatten()
            similarities[idx_pos] = 0  # Exclude self
            return 1 - similarities.max()
        except:
            return 0.5

    def _interpret_topic(self, words: List[str]) -> str:
        """Interpret topic based on top words"""
        word_set = set(words)

        # Political topics
        if any(w in word_set for w in ['bolsonaro', 'presidente', 'governo']):
            return 'política_executiva'
        if any(w in word_set for w in ['lula', 'pt', 'esquerda']):
            return 'política_esquerda'
        if any(w in word_set for w in ['eleição', 'voto', 'candidato']):
            return 'processo_eleitoral'

        # Social topics
        if any(w in word_set for w in ['covid', 'vacina', 'pandemia']):
            return 'saúde_pandemia'
        if any(w in word_set for w in ['economia', 'emprego', 'inflação']):
            return 'economia'

        return 'tema_geral'

    def _calculate_topic_coherence(self, words: List[str]) -> float:
        """Calculate topic coherence score"""
        # Simplified coherence based on word co-occurrence
        return min(1.0, len(set(words)) / len(words)) if words else 0

    def _calculate_trend(self, time_series):
        """Calculate trend direction"""
        if len(time_series) < 2:
            return 'insufficient_data'

        # Simple trend based on first and last periods
        if isinstance(time_series, pd.DataFrame):
            first = time_series.iloc[0].sum()
            last = time_series.iloc[-1].sum()
        else:
            first = time_series.iloc[0]
            last = time_series.iloc[-1]

        if last > first * 1.1:
            return 'increasing'
        elif last < first * 0.9:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_sentiment_trend(self, sentiment_series):
        """Calculate sentiment trend"""
        if len(sentiment_series) < 2:
            return 'insufficient_data'

        first_half = sentiment_series.iloc[:len(sentiment_series)//2].mean()
        second_half = sentiment_series.iloc[len(sentiment_series)//2:].mean()

        if second_half > first_half + 0.1:
            return 'improving'
        elif second_half < first_half - 0.1:
            return 'worsening'
        else:
            return 'stable'

    def _identify_emerging_topics(self, topic_evolution):
        """Identify emerging topics"""
        if len(topic_evolution) < 2:
            return []

        emerging = []
        for col in topic_evolution.columns:
            if topic_evolution[col].iloc[-1] > topic_evolution[col].iloc[0] * 2:
                emerging.append(col)

        return emerging

    def _identify_declining_topics(self, topic_evolution):
        """Identify declining topics"""
        if len(topic_evolution) < 2:
            return []

        declining = []
        for col in topic_evolution.columns:
            if topic_evolution[col].iloc[-1] < topic_evolution[col].iloc[0] * 0.5:
                declining.append(col)

        return declining

    # ================================================================================
    # MAIN EXECUTION
    # ================================================================================

    def run_analysis(self, dataset_path: str, sample_size: Optional[int] = None) -> Dict:
        """
        Execute complete batch analysis

        Args:
            dataset_path: Path to CSV dataset
            sample_size: Optional sample size (default from config)

        Returns:
            Dictionary with all analysis results
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING INTEGRATED BATCH ANALYSIS")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info("=" * 80)

        start_time = datetime.now()
        sample_size = sample_size or self.config.SAMPLE_SIZE

        # Load dataset
        try:
            logger.info(f"📥 Loading dataset (sample: {sample_size})...")
            df = pd.read_csv(dataset_path, nrows=sample_size)
            logger.info(f"✅ Loaded {len(df):,} records")
            self.df = df
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return {'error': str(e)}

        # Execute all stages
        stages = [
            ('01_preprocessing', self.stage_01_preprocessing),
            ('02_text_mining', self.stage_02_text_mining),
            ('03_statistical_analysis', self.stage_03_statistical_analysis),
            ('04_semantic_analysis', self.stage_04_semantic_analysis),
            ('05_tfidf_analysis', self.stage_05_tfidf_analysis),
            ('06_clustering', self.stage_06_clustering),
            ('07_topic_modeling', self.stage_07_topic_modeling),
            ('08_evolution_analysis', self.stage_08_evolution_analysis),
            ('09_network_coordination', self.stage_09_network_coordination),
            ('10_domain_url_analysis', self.stage_10_domain_url_analysis),
            ('11_event_context', self.stage_11_event_context),
            ('12_channel_analysis', self.stage_12_channel_analysis),
            ('13_linguistic_analysis', self.stage_13_linguistic_analysis)
        ]

        for stage_name, stage_func in stages:
            try:
                df = stage_func(df)
                logger.info(f"✅ {stage_name} completed")
            except Exception as e:
                logger.error(f"❌ {stage_name} failed: {e}")
                self.results[stage_name] = {'error': str(e)}

        # Calculate final statistics
        total_time = (datetime.now() - start_time).total_seconds()

        # Estimate API costs
        if self.api_stats['api_calls'] > 0:
            # Rough cost estimation (adjust based on actual pricing)
            cost_per_call = 0.001  # Estimated API cost
            self.api_stats['estimated_cost'] = self.api_stats['api_calls'] * cost_per_call

        # Final summary
        self.results['summary'] = {
            'total_records_analyzed': len(df),
            'total_processing_time': total_time,
            'api_statistics': self.api_stats,
            'stages_completed': len([s for s in stages if s[0] not in [k for k in self.results if 'error' in self.results.get(k, {})]]),
            'stages_failed': len([k for k in self.results if 'error' in self.results.get(k, {})]),
            'analysis_date': datetime.now().isoformat()
        }

        # Save results
        self._save_results(df)

        # Print summary
        self._print_summary()

        return self.results

    def _save_results(self, df: pd.DataFrame):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('batch_analysis_results')
        output_dir.mkdir(exist_ok=True)

        # Save processed dataframe sample
        output_csv = output_dir / f"processed_data_{timestamp}.csv"
        df.head(100).to_csv(output_csv, index=False)
        logger.info(f"💾 Saved processed data to {output_csv}")

        # Save analysis results as JSON
        output_json = output_dir / f"integrated_analysis_{timestamp}.json"

        # Convert to serializable format
        serializable_results = self._convert_to_serializable(self.results)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved analysis results to {output_json}")

    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable"""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 80)
        print("📊 INTEGRATED BATCH ANALYSIS SUMMARY")
        print("=" * 80)

        summary = self.results.get('summary', {})

        print(f"\n🔍 Records Analyzed: {summary.get('total_records_analyzed', 0):,}")
        print(f"⏱️ Total Time: {summary.get('total_processing_time', 0):.2f}s")
        print(f"✅ Stages Completed: {summary.get('stages_completed', 0)}/13")

        print(f"\n🤖 API Statistics:")
        print(f"   • API Calls: {self.api_stats['api_calls']}")
        print(f"   • Cache Hits: {self.api_stats['cache_hits']}")
        print(f"   • Heuristic Methods Used: {self.api_stats['heuristic_count']}")
        print(f"   • Estimated Cost: ${self.api_stats['estimated_cost']:.4f}")

        # Political distribution
        if 'text_mining' in self.results:
            political = self.results['text_mining'].get('political_distribution', {})
            if political:
                print(f"\n🗳️ Political Distribution:")
                for category, count in sorted(political.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"   • {category}: {count}")

        # Sentiment distribution
        if 'text_mining' in self.results:
            sentiment = self.results['text_mining'].get('sentiment_distribution', {})
            if sentiment:
                print(f"\n😊 Sentiment Distribution:")
                for category, count in sentiment.items():
                    print(f"   • {category}: {count}")

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated Batch Analysis for Brazilian Political Discourse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/telegram.csv
  %(prog)s --dataset data/telegram.csv --sample 5000
  %(prog)s --dataset data/telegram.csv --no-api
  %(prog)s --dataset data/telegram.csv --api-key sk-ant-...
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to CSV dataset'
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=1000,
        help='Number of records to analyze (default: 1000)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='batch_analysis_results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Disable API calls, use only heuristic methods'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Anthropic API key (overrides environment variable)'
    )

    parser.add_argument(
        '--voyage-key',
        type=str,
        help='Voyage.ai API key (overrides environment variable)'
    )

    args = parser.parse_args()

    # Configure based on arguments
    config = BatchConfig()

    if args.api_key:
        config.ANTHROPIC_API_KEY = args.api_key

    if args.voyage_key:
        config.VOYAGE_API_KEY = args.voyage_key

    if args.no_api:
        config.ANTHROPIC_API_KEY = None
        config.VOYAGE_API_KEY = None
        logger.info("🚫 API calls disabled - using heuristic methods only")

    # Check dataset exists
    if not Path(args.dataset).exists():
        logger.error(f"❌ Dataset not found: {args.dataset}")
        return 1

    # Run analysis
    analyzer = IntegratedBatchAnalyzer(config)
    results = analyzer.run_analysis(args.dataset, args.sample)

    if 'error' in results:
        logger.error(f"❌ Analysis failed: {results['error']}")
        return 1

    logger.info(f"\n✅ Analysis completed successfully!")
    logger.info(f"📁 Results saved to: batch_analysis_results/")

    return 0


if __name__ == "__main__":
    sys.exit(main())