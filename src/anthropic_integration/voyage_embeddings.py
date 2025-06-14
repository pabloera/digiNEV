"""
Voyage.ai Embeddings Integration for Political Discourse Dataset Analysis
Provides advanced text embedding capabilities for semantic analysis
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try to import voyageai, handle missing dependency gracefully
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False
    voyageai = None
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ✅ EMERGENCY CACHE INTEGRATION - Critical Performance Fix
try:
    from ..optimized.emergency_embeddings import get_global_embeddings_cache
    EMERGENCY_CACHE_AVAILABLE = True
except ImportError:
    EMERGENCY_CACHE_AVAILABLE = False
    get_global_embeddings_cache = None

from .base import AnthropicBase

# ✅ WEEK 2 ADVANCED OPTIMIZATIONS - Unified Embeddings Engine + Smart Claude Cache + Performance Monitor
try:
    from ..optimized.unified_embeddings_engine import get_global_unified_engine, EmbeddingRequest
    from ..optimized.smart_claude_cache import get_global_claude_cache, ClaudeRequest
    from ..optimized.performance_monitor import get_global_performance_monitor
    WEEK2_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    WEEK2_OPTIMIZATIONS_AVAILABLE = False
    get_global_unified_engine = None
    get_global_claude_cache = None
    get_global_performance_monitor = None
    EmbeddingRequest = None
    ClaudeRequest = None

logger = logging.getLogger(__name__)


class VoyageEmbeddingAnalyzer(AnthropicBase):
    """
    Advanced embedding analyzer using Voyage.ai API

    Capabilities:
    - High-quality text embeddings for semantic analysis
    - Batch processing for large datasets
    - Caching for performance optimization
    - Integration with existing semantic components
    - Clustering and similarity analysis
    - Multilingual support for Portuguese text
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Initialize Voyage client
        self.voyage_client = None
        self._initialize_voyage_client()

        # Configuration - OTIMIZADO PARA VOYAGE-3.5-LITE
        embedding_config = config.get('embeddings', {})
        self.model_name = embedding_config.get('model', 'voyage-3.5-lite')  # Modelo mais econômico
        self.batch_size = embedding_config.get('batch_size', 128)  # Aumentado para melhor throughput
        self.max_tokens = embedding_config.get('max_tokens', 32000)
        self.cache_embeddings = embedding_config.get('cache_embeddings', True)
        self.similarity_threshold = embedding_config.get('similarity_threshold', 0.75)  # Reduzido para performance

        # Cost optimization settings
        self.cost_optimization = embedding_config.get('cost_optimization', {})
        self.enable_sampling = self.cost_optimization.get('enable_sampling', False)
        self.max_messages_per_dataset = self.cost_optimization.get('max_messages_per_dataset', 50000)
        self.sampling_strategy = self.cost_optimization.get('sampling_strategy', 'strategic')
        self.min_text_length = self.cost_optimization.get('min_text_length', 50)
        self.require_political_keywords = self.cost_optimization.get('require_political_keywords', False)
        self.temporal_sampling = self.cost_optimization.get('temporal_sampling', False)

        # Political keywords for filtering
        self.political_keywords = [
            'bolsonaro', 'lula', 'eleição', 'eleições', 'voto', 'urna', 'stf', 'supremo',
            'presidente', 'governo', 'congresso', 'senado', 'câmara', 'política', 'político',
            'direita', 'esquerda', 'conservador', 'liberal', 'pts', 'pl', 'psol',
            'vacina', 'pandemia', 'covid', 'lockdown', 'isolamento', 'cloroquina',
            'fake news', 'mídia', 'golpe', 'democracia', 'ditadura', 'comunismo',
            'moro', 'dallagnol', 'glenn', 'intercept', 'operação', 'lava jato'
        ]

        # Cache directory
        self.cache_dir = Path(config.get('data', {}).get('interim_path', 'data/interim')) / 'embeddings_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Available Voyage models with their capabilities - ATUALIZADO 2025
        self.voyage_models = {
            'voyage-3.5-lite': {
                'description': 'Most economical model - RECOMMENDED',
                'max_tokens': 32000,
                'embedding_size': 1024,
                'languages': ['en', 'pt', 'es', 'fr', 'de', 'it'],
                'use_case': 'Cost-optimized analysis',
                'price_per_1k_tokens': 0.00002,
                'free_quota': 200000000
            },
            'voyage-3.5': {
                'description': 'Balanced quality and cost',
                'max_tokens': 32000,
                'embedding_size': 1536,
                'languages': ['en', 'pt', 'es', 'fr', 'de', 'it'],
                'use_case': 'High-quality analysis',
                'price_per_1k_tokens': 0.00006,
                'free_quota': 200000000
            },
            'voyage-large-2': {
                'description': 'Legacy high-performance model',
                'max_tokens': 32000,
                'embedding_size': 1536,
                'languages': ['en', 'pt', 'es', 'fr', 'de', 'it'],
                'use_case': 'Legacy support',
                'price_per_1k_tokens': 0.00012,
                'free_quota': 0
            },
            'voyage-code-2': {
                'description': 'Optimized for code and technical content',
                'max_tokens': 16000,
                'embedding_size': 1536,
                'languages': ['en'],
                'use_case': 'Code analysis, technical documents',
                'price_per_1k_tokens': 0.00012,
                'free_quota': 50000000
            }
        }

        # Validar configuração do modelo
        self._validate_model_configuration()

        logger.info(f"VoyageEmbeddingAnalyzer initialized with model: {self.model_name}")
        if self.enable_sampling:
            logger.info(f"Cost optimization ENABLED - max {self.max_messages_per_dataset} messages per dataset")
        else:
            logger.info("Cost optimization DISABLED - processing all messages")

    def _initialize_voyage_client(self):
        """Initialize Voyage.ai client with error handling"""
        try:
            if not VOYAGEAI_AVAILABLE:
                logger.warning("voyageai package not available - semantic search will use fallback methods")
                self.voyage_available = False
                return

            import os
            api_key = os.getenv('VOYAGE_API_KEY')
            if not api_key:
                logger.warning("VOYAGE_API_KEY not found in environment variables")
                self.voyage_available = False
                return

            self.voyage_client = voyageai.Client(api_key=api_key)
            self.voyage_available = True
            logger.info("Voyage.ai client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Voyage.ai client: {e}")
            self.voyage_available = False

    def _validate_model_configuration(self):
        """Valida configuração do modelo e emite alertas de custo"""

        if self.model_name not in self.voyage_models:
            logger.warning(f"Model {self.model_name} not in known models list - using default configuration")
            return

        model_info = self.voyage_models[self.model_name]

        # Alertas sobre custos e cotas
        if model_info.get('free_quota', 0) == 0:
            logger.warning(f"⚠️  Model {self.model_name} has NO FREE QUOTA - all usage will be charged")
        else:
            free_quota_millions = model_info['free_quota'] / 1000000
            logger.info(f"✅ Model {self.model_name} has {free_quota_millions:.0f}M free tokens")

        # Recomendações baseadas na configuração
        if not self.enable_sampling and self.model_name != 'voyage-3.5-lite':
            logger.warning("💰 COST OPTIMIZATION DISABLED with non-lite model - consider enabling sampling")

        if self.enable_sampling and self.model_name == 'voyage-3.5-lite':
            estimated_tokens = self.max_messages_per_dataset * 60  # ~60 tokens per message
            estimated_cost = (estimated_tokens / 1000) * model_info['price_per_1k_tokens']
            logger.info(f"💡 Estimated cost per dataset: ${estimated_cost:.4f} (likely FREE within quota)")

        # Log configuração otimizada
        logger.info(f"📊 Model: {self.model_name} | Sampling: {self.enable_sampling} | Batch: {self.batch_size}")

    def apply_cost_optimized_sampling(self, df: pd.DataFrame, text_column: str = 'body_cleaned') -> pd.DataFrame:
        """
        Aplica amostragem inteligente para otimização de custos

        Args:
            df: DataFrame original
            text_column: Coluna com texto para análise

        Returns:
            DataFrame amostrado otimizado para Voyage
        """
        if not self.enable_sampling:
            return df

        logger.info(f"Aplicando amostragem inteligente - Original: {len(df)} mensagens")

        # 1. Filtros básicos de qualidade
        filtered_df = df.copy()

        # Filtrar por comprimento mínimo
        if self.min_text_length > 0:
            length_mask = filtered_df[text_column].str.len() >= self.min_text_length
            filtered_df = filtered_df[length_mask]
            logger.info(f"Após filtro de comprimento (>={self.min_text_length}): {len(filtered_df)} mensagens")

        # Filtrar por palavras-chave políticas
        if self.require_political_keywords:
            keyword_pattern = '|'.join(self.political_keywords)
            political_mask = filtered_df[text_column].str.contains(
                keyword_pattern, case=False, na=False, regex=True
            )
            filtered_df = filtered_df[political_mask]
            logger.info(f"Após filtro político: {len(filtered_df)} mensagens")

        # 2. Amostragem estratégica
        if len(filtered_df) > self.max_messages_per_dataset:
            if self.sampling_strategy == 'strategic':
                sampled_df = self._strategic_sampling(filtered_df, text_column)
            elif self.sampling_strategy == 'temporal':
                sampled_df = self._temporal_sampling(filtered_df)
            else:  # random
                sampled_df = filtered_df.sample(n=self.max_messages_per_dataset)

            logger.info(f"Após amostragem {self.sampling_strategy}: {len(sampled_df)} mensagens")
            return sampled_df

        logger.info(f"Dataset final para Voyage: {len(filtered_df)} mensagens")
        return filtered_df

    def _strategic_sampling(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Amostragem estratégica baseada em importância"""

        # Calcular scores de importância
        df_scored = df.copy()

        # Score baseado em comprimento (textos mais longos = mais informativos)
        df_scored['length_score'] = df_scored[text_column].str.len() / df_scored[text_column].str.len().max()

        # Score baseado em hashtags (mais hashtags = mais engajamento)
        if 'hashtag' in df.columns:
            df_scored['hashtag_score'] = df_scored['hashtag'].fillna('').str.count(',').fillna(0) / 10
        else:
            df_scored['hashtag_score'] = 0

        # Score baseado em menções (mais menções = mais rede social)
        if 'mentions' in df.columns:
            df_scored['mention_score'] = df_scored['mentions'].fillna('').str.count(',').fillna(0) / 5
        else:
            df_scored['mention_score'] = 0

        # Score baseado em palavras-chave importantes
        important_keywords = ['bolsonaro', 'lula', 'eleição', 'stf', 'vacina', 'golpe']
        keyword_pattern = '|'.join(important_keywords)
        df_scored['keyword_score'] = df_scored[text_column].str.count(
            keyword_pattern, flags=re.IGNORECASE
        ).fillna(0) / 3

        # Score composto
        df_scored['importance_score'] = (
            df_scored['length_score'] * 0.3 +
            df_scored['hashtag_score'] * 0.2 +
            df_scored['mention_score'] * 0.2 +
            df_scored['keyword_score'] * 0.3
        )

        # Amostragem estratificada por importância
        # 70% das mensagens de alta importância, 30% aleatória
        high_importance_count = int(self.max_messages_per_dataset * 0.7)
        random_count = self.max_messages_per_dataset - high_importance_count

        # Top mensagens por importância
        high_importance = df_scored.nlargest(high_importance_count, 'importance_score')

        # Amostra aleatória do restante
        remaining = df_scored.drop(high_importance.index)
        if len(remaining) > 0:
            random_sample = remaining.sample(n=min(random_count, len(remaining)))
            result = pd.concat([high_importance, random_sample])
        else:
            result = high_importance

        return result.drop(columns=['length_score', 'hashtag_score', 'mention_score', 'keyword_score', 'importance_score'])

    def _temporal_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Amostragem baseada em períodos temporais chave"""

        if 'datetime' not in df.columns:
            logger.warning("Coluna datetime não encontrada, usando amostragem aleatória")
            return df.sample(n=self.max_messages_per_dataset)

        key_periods = self.cost_optimization.get('key_periods', [])

        if not key_periods:
            logger.warning("Períodos chave não configurados, usando amostragem aleatória")
            return df.sample(n=self.max_messages_per_dataset)

        sampled_dfs = []
        messages_per_period = self.max_messages_per_dataset // len(key_periods)

        df['datetime'] = pd.to_datetime(df['datetime'])

        for period in key_periods:
            start_date = pd.to_datetime(period['start'])
            end_date = pd.to_datetime(period['end'])
            sample_rate = period.get('sample_rate', 0.1)

            period_mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
            period_df = df[period_mask]

            if len(period_df) > 0:
                # Calcular número de amostras baseado na taxa e disponibilidade
                target_samples = min(
                    int(len(period_df) * sample_rate),
                    messages_per_period
                )

                if target_samples > 0:
                    period_sample = period_df.sample(n=target_samples)
                    sampled_dfs.append(period_sample)
                    logger.info(f"Período {period['description']}: {len(period_sample)} amostras de {len(period_df)} disponíveis")

        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            # Se ainda não chegamos ao limite, completar com amostra aleatória
            if len(result) < self.max_messages_per_dataset:
                remaining_needed = self.max_messages_per_dataset - len(result)
                unused_df = df.drop(result.index)
                if len(unused_df) > 0:
                    additional_sample = unused_df.sample(n=min(remaining_needed, len(unused_df)))
                    result = pd.concat([result, additional_sample], ignore_index=True)

            return result
        else:
            logger.warning("Nenhuma mensagem encontrada nos períodos chave, usando amostragem aleatória")
            return df.sample(n=self.max_messages_per_dataset)

    def generate_embeddings(
        self,
        texts: List[str],
        input_type: str = "document",
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a list of texts with Emergency Cache Integration

        Args:
            texts: List of texts to embed
            input_type: Type of input ("document", "query", "classification")
            cache_key: Optional cache key for storing results

        Returns:
            Dictionary with embeddings and metadata
        """
        if not self.voyage_available:
            logger.error("Voyage.ai client not available")
            return self._fallback_embeddings(texts)

        logger.info(f"🚀 Generating embeddings for {len(texts)} texts with model {self.model_name}")

        # ✅ WEEK 2 UNIFIED EMBEDDINGS ENGINE - Advanced hierarchical cache with L1/L2 levels
        if WEEK2_OPTIMIZATIONS_AVAILABLE:
            try:
                unified_engine = get_global_unified_engine()
                performance_monitor = get_global_performance_monitor()
                
                # Create embedding request for unified engine
                request = EmbeddingRequest(
                    texts=texts,
                    model=self.model_name,
                    stage_name=cache_key or 'voyage_general',
                    input_type=input_type
                )
                
                # Process through unified engine with advanced cache
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                engine_result = loop.run_until_complete(
                    unified_engine.get_embeddings(
                        request, 
                        lambda txt_batch, model: self._compute_embeddings_direct(txt_batch, model, input_type=input_type)
                    )
                )
                
                if engine_result and len(engine_result.embeddings) > 0:
                    # Record performance metrics
                    performance_monitor.record_stage_completion(
                        stage_name=cache_key or 'voyage_embeddings',
                        records_processed=len(texts),
                        processing_time=engine_result.total_time,
                        success_rate=1.0,
                        api_calls=0 if engine_result.cache_hit else 1,
                        cost_usd=0.0  # Voyage.ai cost calculation would go here
                    )
                    
                    # Convert to expected format
                    result = {
                        'embeddings': engine_result.embeddings.tolist() if hasattr(engine_result.embeddings, 'tolist') else engine_result.embeddings,
                        'model': engine_result.model,
                        'embedding_size': len(engine_result.embeddings[0]) if len(engine_result.embeddings) > 0 else 0,
                        'processing_stats': {
                            'total_texts': len(texts),
                            'successful_embeddings': len(engine_result.embeddings),
                            'failed_embeddings': 0,
                            'total_tokens': 0,  # Would need to calculate from texts
                            'batches_processed': 1,
                            'cache_hit': engine_result.cache_hit,
                            'total_time': engine_result.total_time,
                            'cache_level': engine_result.cache_level,
                            'compression_ratio': engine_result.compression_ratio
                        },
                        'timestamp': datetime.now().isoformat(),
                        'input_type': input_type,
                        'unified_engine_used': True
                    }
                    
                    cache_status = f"{engine_result.cache_level.upper()}" if engine_result.cache_hit else "COMPUTED"
                    logger.info(f"🚀 Unified Engine {cache_status}: {len(texts)} texts in {engine_result.total_time:.2f}s")
                    return result
                    
            except Exception as e:
                logger.warning(f"⚠️ Unified engine failed, falling back to emergency cache: {e}")
        
        # ✅ EMERGENCY CACHE FALLBACK - Original emergency cache as backup
        if EMERGENCY_CACHE_AVAILABLE:
            try:
                emergency_cache = get_global_embeddings_cache()
                embeddings, cache_stats = emergency_cache.get_stage_embeddings(
                    stage_name=cache_key or "unknown",
                    texts=texts,
                    compute_func=self._compute_embeddings_direct,
                    model=self.model_name,
                    input_type=input_type
                )
                
                if embeddings is not None and len(embeddings) > 0:
                    # Converter para formato esperado
                    result = {
                        'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                        'model': self.model_name,
                        'embedding_size': embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]) if len(embeddings) > 0 else 0,
                        'processing_stats': {
                            'total_texts': cache_stats.get('text_count', len(texts)),
                            'successful_embeddings': cache_stats.get('text_count', len(texts)),
                            'failed_embeddings': 0,
                            'cache_hit': cache_stats.get('cache_hit', False),
                            'compute_time': cache_stats.get('compute_time', 0),
                            'total_time': cache_stats.get('total_time', 0)
                        },
                        'timestamp': datetime.now().isoformat(),
                        'input_type': input_type,
                        'emergency_cache_used': True
                    }
                    
                    cache_status = "HIT" if cache_stats.get('cache_hit') else "COMPUTED"
                    logger.info(f"✅ Emergency Cache {cache_status}: {len(texts)} texts in {cache_stats.get('total_time', 0):.2f}s")
                    return result
                    
            except Exception as e:
                logger.warning(f"⚠️ Emergency cache failed, falling back to standard process: {e}")

        # Original cache check (fallback)
        if cache_key and self.cache_embeddings:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info(f"📦 Loaded embeddings from standard cache: {cache_key}")
                return cached_result

        # Process texts in batches with token counting
        all_embeddings = []
        processing_stats = {
            'total_texts': len(texts),
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'total_tokens': 0,
            'batches_processed': 0
        }

        try:
            i = 0
            while i < len(texts):
                # Create batch with very conservative token limit
                batch_texts, batch_size = self._create_token_limited_batch(texts[i:], max_batch_tokens=50000)

                if not batch_texts:
                    i += 1
                    continue

                # Filter and preprocess batch
                processed_batch = self._preprocess_texts(batch_texts)

                if not processed_batch:
                    i += batch_size
                    continue

                # Generate embeddings for batch
                batch_result = self._generate_batch_embeddings(
                    processed_batch,
                    input_type
                )

                if batch_result['success']:
                    all_embeddings.extend(batch_result['embeddings'])
                    processing_stats['successful_embeddings'] += len(batch_result['embeddings'])
                    processing_stats['total_tokens'] += batch_result.get('token_count', 0)
                else:
                    processing_stats['failed_embeddings'] += len(processed_batch)
                    logger.warning(f"Batch {i//self.batch_size + 1} failed: {batch_result.get('error', 'Unknown error')}")

                processing_stats['batches_processed'] += 1
                i += batch_size

                # Rate limiting - small delay between batches
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return self._fallback_embeddings(texts)

        # Prepare result
        result = {
            'embeddings': all_embeddings,
            'model': self.model_name,
            'embedding_size': len(all_embeddings[0]) if all_embeddings else 0,
            'processing_stats': processing_stats,
            'timestamp': datetime.now().isoformat(),
            'input_type': input_type
        }

        # Cache result if requested
        if cache_key and self.cache_embeddings:
            self._save_to_cache(cache_key, result)

        logger.info(f"Embedding generation completed: {processing_stats['successful_embeddings']}/{processing_stats['total_texts']} successful")

        return result

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for embedding generation with aggressive truncation"""
        processed = []
        # Much more conservative: limit each text to 1000 tokens max (3000 chars)
        max_chars_per_text = 3000

        for text in texts:
            if not text or not isinstance(text, str):
                continue

            # Clean and truncate text aggressively
            cleaned_text = str(text).strip()

            # Skip empty texts
            if not cleaned_text:
                continue

            # Aggressive truncation to ensure no single text is too large
            if len(cleaned_text) > max_chars_per_text:
                cleaned_text = cleaned_text[:max_chars_per_text] + "..."
                logger.debug(f"Text aggressively truncated to {max_chars_per_text} characters")

            processed.append(cleaned_text)

        return processed

    def _create_token_limited_batch(
        self,
        texts: List[str],
        max_batch_tokens: int = 100000
    ) -> Tuple[List[str], int]:
        """
        Create a batch of texts that doesn't exceed token limit

        Args:
            texts: List of texts to batch
            max_batch_tokens: Maximum tokens per batch

        Returns:
            Tuple of (batch_texts, number_of_texts_processed)
        """
        if not texts:
            return [], 0

        batch_texts = []
        current_tokens = 0
        texts_processed = 0

        for text in texts:
            # More conservative token estimation: 1 token ≈ 3 characters for Portuguese
            # This accounts for longer words and special characters
            estimated_tokens = len(str(text)) // 3

            # Check if adding this text would exceed limit
            if current_tokens + estimated_tokens > max_batch_tokens and batch_texts:
                break

            batch_texts.append(text)
            current_tokens += estimated_tokens
            texts_processed += 1

            # Also respect the original batch size limit
            if len(batch_texts) >= self.batch_size:
                break

        if not batch_texts and texts:
            # If even a single text is too large, include it anyway but truncate
            text = str(texts[0])
            max_chars = max_batch_tokens * 3
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                logger.warning(f"Text truncated to fit token limit: {len(text)} chars")
            batch_texts = [text]
            texts_processed = 1

        logger.debug(f"Created batch with {len(batch_texts)} texts, estimated {current_tokens} tokens")
        return batch_texts, texts_processed

    def _generate_batch_embeddings(
        self,
        texts: List[str],
        input_type: str
    ) -> Dict[str, Any]:
        """Generate embeddings for a single batch with safety checks"""
        try:
            # Final safety check: estimate total tokens and split if needed
            total_estimated_tokens = sum(len(str(text)) // 3 for text in texts)

            if total_estimated_tokens > 100000:  # Still too large, split further
                logger.warning(f"Batch still too large ({total_estimated_tokens} tokens), splitting...")
                mid = len(texts) // 2
                if mid == 0:  # Single text is too large, truncate it
                    texts = [str(texts[0])[:30000] + "..." if len(str(texts[0])) > 30000 else str(texts[0])]
                else:
                    # Process only first half
                    texts = texts[:mid]

            response = self.voyage_client.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type
            )

            return {
                'success': True,
                'embeddings': response.embeddings,
                'token_count': getattr(response, 'total_tokens', 0)
            }

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            # If it's a token limit error, try with just the first text truncated heavily
            if "max allowed tokens" in str(e) and texts:
                logger.warning("Attempting emergency fallback with single truncated text")
                try:
                    emergency_text = str(texts[0])[:10000] + "..."
                    response = self.voyage_client.embed(
                        texts=[emergency_text],
                        model=self.model_name,
                        input_type=input_type
                    )
                    return {
                        'success': True,
                        'embeddings': response.embeddings,
                        'token_count': getattr(response, 'total_tokens', 0)
                    }
                except Exception as e2:
                    logger.error(f"Emergency fallback also failed: {e2}")

            return {
                'success': False,
                'error': str(e),
                'embeddings': []
            }

    def _compute_embeddings_direct(self, texts: List[str], model: str = None, **kwargs) -> np.ndarray:
        """
        Direct computation method for emergency cache integration
        
        Args:
            texts: List of texts to embed
            model: Model name (optional, uses self.model_name if None)
            **kwargs: Additional arguments
            
        Returns:
            NumPy array of embeddings
        """
        if not self.voyage_available:
            logger.error("Voyage.ai client not available for direct computation")
            return np.array([])
            
        try:
            model_to_use = model or self.model_name
            input_type = kwargs.get('input_type', 'document')
            
            logger.info(f"🔄 Direct computation: {len(texts)} texts with {model_to_use}")
            
            # Process in batches if too many texts
            all_embeddings = []
            batch_size = 128  # Conservative batch size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                processed_batch = self._preprocess_texts(batch_texts)
                
                if not processed_batch:
                    continue
                    
                # Generate embeddings for batch
                batch_result = self._generate_batch_embeddings(
                    processed_batch,
                    input_type
                )
                
                if batch_result['success']:
                    all_embeddings.extend(batch_result['embeddings'])
                else:
                    logger.warning(f"Batch {i//batch_size + 1} failed: {batch_result.get('error', 'Unknown error')}")
                
                # Small delay between batches
                time.sleep(0.1)
            
            if all_embeddings:
                embeddings_array = np.array(all_embeddings)
                logger.info(f"✅ Direct computation successful: {embeddings_array.shape}")
                return embeddings_array
            else:
                logger.error("No embeddings generated in direct computation")
                return np.array([])
                
        except Exception as e:
            logger.error(f"❌ Error in direct computation: {e}")
            return np.array([])

    def analyze_text_similarity(
        self,
        df: pd.DataFrame,
        text_column: str = 'body_cleaned',
        reference_texts: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze text similarity using embeddings

        Args:
            df: DataFrame with texts to analyze
            text_column: Column containing text data
            reference_texts: Optional reference texts for comparison

        Returns:
            DataFrame with similarity scores and clusters
        """
        logger.info(f"Analyzing text similarity for {len(df)} records")

        # Extract texts
        texts = df[text_column].fillna('').astype(str).tolist()

        # Generate embeddings
        cache_key = self._generate_cache_key(texts, 'similarity_analysis')
        embedding_result = self.generate_embeddings(
            texts,
            input_type="document",
            cache_key=cache_key
        )

        if not embedding_result['embeddings']:
            logger.warning("No embeddings generated, returning original DataFrame")
            return df

        # Convert to numpy array for analysis
        embeddings_matrix = np.array(embedding_result['embeddings'])

        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings_matrix)

        # Add similarity metrics to DataFrame
        result_df = df.copy()

        # Average similarity to all other texts
        result_df['avg_similarity'] = np.mean(similarity_matrix, axis=1)

        # Maximum similarity to any other text
        np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
        result_df['max_similarity'] = np.max(similarity_matrix, axis=1)

        # Find most similar text indices
        result_df['most_similar_idx'] = np.argmax(similarity_matrix, axis=1)

        # Identify potential duplicates based on high similarity
        high_similarity_mask = result_df['max_similarity'] > self.similarity_threshold
        result_df['potential_duplicate'] = high_similarity_mask

        # Semantic clustering
        n_clusters = min(10, len(df) // 100 + 1)  # Adaptive cluster count
        if len(embeddings_matrix) > n_clusters:
            clusters = self._perform_semantic_clustering(embeddings_matrix, n_clusters)
            result_df['semantic_cluster'] = clusters

        # Reference comparison if provided
        if reference_texts:
            ref_similarities = self._compare_to_references(
                embeddings_matrix,
                reference_texts
            )
            result_df['reference_similarity'] = ref_similarities

        logger.info(f"Similarity analysis completed. Found {high_similarity_mask.sum()} potential duplicates")

        return result_df

    def _perform_semantic_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> List[int]:
        """Perform semantic clustering on embeddings"""
        try:
            # Use KMeans for clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            clusters = kmeans.fit_predict(embeddings)

            logger.info(f"Semantic clustering completed with {n_clusters} clusters")
            return clusters.tolist()

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return [0] * len(embeddings)

    def _compare_to_references(
        self,
        embeddings: np.ndarray,
        reference_texts: List[str]
    ) -> List[float]:
        """Compare embeddings to reference texts"""
        try:
            # Generate embeddings for reference texts
            ref_result = self.generate_embeddings(
                reference_texts,
                input_type="query"
            )

            if not ref_result['embeddings']:
                return [0.0] * len(embeddings)

            ref_embeddings = np.array(ref_result['embeddings'])

            # Calculate similarities to references
            similarities = cosine_similarity(embeddings, ref_embeddings)

            # Return maximum similarity to any reference
            return np.max(similarities, axis=1).tolist()

        except Exception as e:
            logger.error(f"Reference comparison failed: {e}")
            return [0.0] * len(embeddings)

    def extract_semantic_topics(
        self,
        df: pd.DataFrame,
        text_column: str = 'body_cleaned',
        n_topics: int = 10
    ) -> Dict[str, Any]:
        """
        Extract semantic topics using embedding-based clustering

        Args:
            df: DataFrame with texts
            text_column: Column containing text data
            n_topics: Number of topics to extract

        Returns:
            Dictionary with topic analysis results
        """
        logger.info(f"Extracting {n_topics} semantic topics from {len(df)} texts")

        # Filter out empty texts
        valid_texts = df[text_column].fillna('').astype(str)
        valid_mask = valid_texts.str.strip() != ''
        filtered_df = df[valid_mask].copy()
        texts = valid_texts[valid_mask].tolist()

        if len(texts) < n_topics:
            logger.warning(f"Not enough texts ({len(texts)}) for {n_topics} topics")
            n_topics = max(1, len(texts) // 2)

        # Generate embeddings
        cache_key = self._generate_cache_key(texts, f'topic_extraction_{n_topics}')
        embedding_result = self.generate_embeddings(
            texts,
            input_type="document",
            cache_key=cache_key
        )

        if not embedding_result['embeddings']:
            return {'topics': [], 'assignments': [], 'error': 'No embeddings generated'}

        embeddings_matrix = np.array(embedding_result['embeddings'])

        # Perform clustering for topic extraction
        clusters = self._perform_semantic_clustering(embeddings_matrix, n_topics)

        # Analyze topics
        topics = []
        for topic_id in range(n_topics):
            topic_mask = np.array(clusters) == topic_id
            topic_texts = [texts[i] for i in range(len(texts)) if topic_mask[i]]
            topic_embeddings = embeddings_matrix[topic_mask]

            if len(topic_texts) == 0:
                continue

            # Calculate topic centroid
            centroid = np.mean(topic_embeddings, axis=0)

            # Find representative texts (closest to centroid)
            similarities_to_centroid = cosine_similarity(
                topic_embeddings,
                centroid.reshape(1, -1)
            ).flatten()

            top_indices = np.argsort(similarities_to_centroid)[-3:][::-1]
            representative_texts = [topic_texts[i] for i in top_indices if i < len(topic_texts)]

            # Use Anthropic API for topic interpretation if available
            topic_description = self._interpret_topic_with_ai(representative_texts, topic_id)

            topics.append({
                'topic_id': topic_id,
                'document_count': len(topic_texts),
                'representative_texts': representative_texts[:3],
                'description': topic_description,
                'coherence_score': float(np.mean(similarities_to_centroid))
            })

        # Add topic assignments to DataFrame
        result_df = filtered_df.copy()
        result_df['topic_id'] = clusters

        return {
            'topics': topics,
            'topic_assignments': result_df,
            'n_topics_extracted': len(topics),
            'embedding_model': self.model_name,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _interpret_topic_with_ai(
        self,
        representative_texts: List[str],
        topic_id: int
    ) -> str:
        """Use AI to interpret and describe topic"""
        if not self.api_available or not representative_texts:
            return f"Topic {topic_id}"

        try:
            texts_sample = '\n'.join([f"- {text[:100]}..." for text in representative_texts[:3]])

            prompt = f"""
Analise os seguintes textos representativos de um tópico identificado em mensagens do Telegram brasileiro (2019-2023):

TEXTOS REPRESENTATIVOS:
{texts_sample}

Este é o tópico #{topic_id} identificado através de análise semântica de embeddings.

Forneça uma descrição concisa (máximo 2-3 palavras) que capture o tema principal destes textos.
Considere o contexto político brasileiro, movimento bolsonarista, e temas como:
- Política governamental
- Pandemia e saúde
- Eleições
- Economia
- Educação
- Mídia e comunicação
- Questões sociais

Responda apenas com a descrição do tópico:
"""

            response = self.create_message(
                prompt,
                stage="topic_interpretation",
                operation="semantic_topic_description",
                temperature=0.3
            )

            # Extract clean topic description
            description = response.strip().strip('"').strip("'")
            if len(description) > 50:
                description = description[:47] + "..."

            return description or f"Topic {topic_id}"

        except Exception as e:
            logger.warning(f"AI topic interpretation failed: {e}")
            return f"Topic {topic_id}"

    def find_semantic_duplicates(
        self,
        df: pd.DataFrame,
        text_column: str = 'body_cleaned',
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Find semantic duplicates using embeddings

        Args:
            df: DataFrame to analyze
            text_column: Column with text data
            threshold: Similarity threshold (default: self.similarity_threshold)

        Returns:
            Dictionary with duplicate analysis results
        """
        if threshold is None:
            threshold = self.similarity_threshold

        logger.info(f"Finding semantic duplicates with threshold {threshold}")

        # Analyze similarity
        similarity_df = self.analyze_text_similarity(df, text_column)

        # Identify duplicates
        duplicates_mask = similarity_df['max_similarity'] > threshold
        duplicates = similarity_df[duplicates_mask].copy()

        # Group duplicates
        duplicate_groups = []
        processed_indices = set()

        # Vectorized approach: process duplicates in batches
        for idx in duplicates.index:
            if idx in processed_indices:
                continue

            # Find all texts similar to this one (vectorized)
            similar_mask = (
                (similarity_df['max_similarity'] > threshold) &
                (similarity_df.index != idx)
            )
            similar_indices = similarity_df[similar_mask].index.tolist()

            if similar_indices:
                row = duplicates.loc[idx]
                group = {
                    'primary_index': idx,
                    'primary_text': str(row[text_column])[:100] + "...",
                    'similar_indices': similar_indices,
                    'group_size': len(similar_indices) + 1,
                    'max_similarity': float(row['max_similarity'])
                }
                duplicate_groups.append(group)

                processed_indices.add(idx)
                processed_indices.update(similar_indices)

        return {
            'total_potential_duplicates': len(duplicates),
            'duplicate_groups': duplicate_groups,
            'unique_texts_after_dedup': len(df) - len(duplicates),
            'deduplication_ratio': len(duplicates) / len(df) if len(df) > 0 else 0,
            'threshold_used': threshold,
            'embedding_model': self.model_name
        }

    def enhance_semantic_analysis(
        self,
        df: pd.DataFrame,
        text_column: str = 'body_cleaned'
    ) -> pd.DataFrame:
        """
        Enhance existing semantic analysis with embeddings (with cost optimization)

        Args:
            df: DataFrame with existing analysis
            text_column: Column with text data

        Returns:
            Enhanced DataFrame with embedding-based features
        """
        logger.info(f"Enhancing semantic analysis for {len(df)} records")

        # Apply cost optimization sampling if enabled
        working_df = self.apply_cost_optimized_sampling(df, text_column)

        if len(working_df) < len(df):
            logger.info(f"Amostragem aplicada: analisando {len(working_df)} de {len(df)} mensagens")

            # Generate analysis for sampled data
            enhanced_sample = self.analyze_text_similarity(working_df, text_column)

            # Extract semantic topics from sample
            topic_result = self.extract_semantic_topics(enhanced_sample, text_column)

            # Merge topic assignments for sample
            if 'topic_assignments' in topic_result:
                topic_assignments = topic_result['topic_assignments']
                enhanced_sample = enhanced_sample.merge(
                    topic_assignments[['topic_id']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )

            # Extend insights to full dataset using similarity patterns
            enhanced_df = self._extend_analysis_to_full_dataset(df, enhanced_sample, text_column)
        else:
            # Generate base similarity analysis for full dataset
            enhanced_df = self.analyze_text_similarity(working_df, text_column)

            # Extract semantic topics
            topic_result = self.extract_semantic_topics(enhanced_df, text_column)

            # Merge topic assignments
            if 'topic_assignments' in topic_result:
                topic_assignments = topic_result['topic_assignments']
                enhanced_df = enhanced_df.merge(
                    topic_assignments[['topic_id']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )

        # Add semantic quality scores
        enhanced_df['semantic_quality'] = self._calculate_semantic_quality(enhanced_df)

        # Add metadata
        enhanced_df['embedding_model'] = self.model_name
        enhanced_df['semantic_analysis_timestamp'] = datetime.now().isoformat()
        enhanced_df['cost_optimized'] = self.enable_sampling
        enhanced_df['sample_ratio'] = len(working_df) / len(df) if len(df) > 0 else 1.0

        logger.info("Semantic analysis enhancement completed")

        return enhanced_df

    def _extend_analysis_to_full_dataset(
        self,
        full_df: pd.DataFrame,
        analyzed_sample: pd.DataFrame,
        text_column: str
    ) -> pd.DataFrame:
        """
        Estende análise da amostra para o dataset completo usando inferência
        """
        logger.info(f"Estendendo análise de {len(analyzed_sample)} para {len(full_df)} mensagens")

        # Começar com dataset completo
        result_df = full_df.copy()

        # Inicializar colunas com valores padrão
        result_df['avg_similarity'] = 0.0
        result_df['max_similarity'] = 0.0
        result_df['potential_duplicate'] = False
        result_df['semantic_cluster'] = -1
        result_df['topic_id'] = -1

        # Copiar análises da amostra diretamente
        common_indices = set(analyzed_sample.index).intersection(set(result_df.index))
        for idx in common_indices:
            for col in ['avg_similarity', 'max_similarity', 'potential_duplicate', 'semantic_cluster', 'topic_id']:
                if col in analyzed_sample.columns:
                    result_df.loc[idx, col] = analyzed_sample.loc[idx, col]

        # Para mensagens não analisadas, inferir baseado em similaridade simples
        unanalyzed_indices = set(result_df.index) - common_indices

        if unanalyzed_indices and len(analyzed_sample) > 0:
            logger.info(f"Inferindo análise para {len(unanalyzed_indices)} mensagens não analisadas")

            # Inferência baseada em palavras-chave e padrões textuais
            for idx in list(unanalyzed_indices)[:1000]:  # Limitar para performance
                text = str(result_df.loc[idx, text_column]).lower()

                # Encontrar mensagem mais similar na amostra baseado em palavras-chave
                best_match_idx = self._find_text_similarity_match(text, analyzed_sample, text_column)

                if best_match_idx is not None:
                    # Copiar características da mensagem similar
                    for col in ['semantic_cluster', 'topic_id']:
                        if col in analyzed_sample.columns:
                            result_df.loc[idx, col] = analyzed_sample.loc[best_match_idx, col]

                    # Similaridade reduzida por ser inferência
                    result_df.loc[idx, 'avg_similarity'] = analyzed_sample.loc[best_match_idx, 'avg_similarity'] * 0.7
                    result_df.loc[idx, 'max_similarity'] = analyzed_sample.loc[best_match_idx, 'max_similarity'] * 0.7

        return result_df

    def _find_text_similarity_match(
        self,
        target_text: str,
        sample_df: pd.DataFrame,
        text_column: str
    ) -> Optional[int]:
        """
        Encontra a mensagem mais similar na amostra baseado em palavras-chave
        """
        target_words = set(target_text.split())
        best_similarity = 0
        best_match_idx = None

        # Comparar com até 100 mensagens da amostra para performance
        sample_indices = sample_df.index[:100]

        for idx in sample_indices:
            sample_text = str(sample_df.loc[idx, text_column]).lower()
            sample_words = set(sample_text.split())

            # Similaridade Jaccard simples
            if len(target_words) > 0 and len(sample_words) > 0:
                intersection = len(target_words.intersection(sample_words))
                union = len(target_words.union(sample_words))
                similarity = intersection / union if union > 0 else 0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx

        return best_match_idx if best_similarity > 0.1 else None

    def _calculate_semantic_quality(self, df: pd.DataFrame) -> List[float]:
        """Calculate semantic quality scores for texts (vectorized)"""
        # Base score for all rows
        quality_scores = pd.Series(0.5, index=df.index)

        # Higher score for texts with moderate similarity (vectorized)
        if 'avg_similarity' in df.columns:
            moderate_sim_mask = (df['avg_similarity'] >= 0.2) & (df['avg_similarity'] <= 0.7)
            quality_scores.loc[moderate_sim_mask] += 0.3

        # Lower score for potential duplicates (vectorized)
        if 'potential_duplicate' in df.columns:
            duplicate_mask = df['potential_duplicate'] == True
            quality_scores.loc[duplicate_mask] -= 0.2

        # Ensure scores are between 0 and 1 (vectorized)
        quality_scores = quality_scores.clip(0.0, 1.0)

        return quality_scores.tolist()

    def _generate_cache_key(self, texts: List[str], operation: str) -> str:
        """Generate cache key for embeddings"""
        # Create hash from texts and operation
        content = f"{operation}_{self.model_name}_{''.join(texts[:10])}"
        return hashlib.md5(content.encode()).hexdigest()

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save embeddings to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved embeddings to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load embeddings from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return None

    def _fallback_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Fallback when Voyage.ai is not available"""
        logger.warning("Using fallback embeddings (random vectors)")

        # Generate random embeddings as fallback
        embedding_size = 1024
        embeddings = np.random.normal(0, 1, (len(texts), embedding_size)).tolist()

        return {
            'embeddings': embeddings,
            'model': 'fallback_random',
            'embedding_size': embedding_size,
            'processing_stats': {
                'total_texts': len(texts),
                'successful_embeddings': len(texts),
                'failed_embeddings': 0
            },
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }

    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about available embedding models with cost analysis"""

        current_model_info = self.voyage_models.get(self.model_name, {})

        return {
            'current_model': self.model_name,
            'voyage_available': self.voyage_available,
            'model_details': current_model_info,
            'cost_optimization': {
                'enabled': self.enable_sampling,
                'max_messages_per_dataset': self.max_messages_per_dataset,
                'sampling_strategy': self.sampling_strategy,
                'estimated_cost_per_dataset': self._calculate_estimated_cost(),
                'free_quota_remaining': self._estimate_quota_usage()
            },
            'configuration': {
                'batch_size': self.batch_size,
                'max_tokens': self.max_tokens,
                'cache_embeddings': self.cache_embeddings,
                'similarity_threshold': self.similarity_threshold
            },
            'available_models': self.voyage_models
        }

    def _calculate_estimated_cost(self) -> Dict[str, Any]:
        """Calcula custo estimado por dataset"""

        model_info = self.voyage_models.get(self.model_name, {})
        price_per_1k = model_info.get('price_per_1k_tokens', 0)

        if self.enable_sampling:
            estimated_tokens = self.max_messages_per_dataset * 60  # ~60 tokens por mensagem
        else:
            estimated_tokens = 1300000 * 77  # Estimativa sem otimização

        estimated_cost = (estimated_tokens / 1000) * price_per_1k

        return {
            'estimated_tokens': estimated_tokens,
            'price_per_1k_tokens': price_per_1k,
            'estimated_cost_usd': round(estimated_cost, 4),
            'likely_free': estimated_tokens < model_info.get('free_quota', 0)
        }

    def _estimate_quota_usage(self) -> Dict[str, Any]:
        """Estima uso da cota gratuita"""

        model_info = self.voyage_models.get(self.model_name, {})
        free_quota = model_info.get('free_quota', 0)

        cost_info = self._calculate_estimated_cost()
        estimated_tokens = cost_info['estimated_tokens']

        if free_quota == 0:
            return {
                'has_free_quota': False,
                'message': 'No free quota available - all usage charged'
            }

        quota_usage_percent = (estimated_tokens / free_quota) * 100
        executions_possible = free_quota // estimated_tokens if estimated_tokens > 0 else 0

        return {
            'has_free_quota': True,
            'free_quota_tokens': free_quota,
            'estimated_usage_percent': round(quota_usage_percent, 2),
            'executions_possible': executions_possible,
            'tokens_remaining_after_execution': free_quota - estimated_tokens
        }


def create_voyage_embedding_analyzer(config: Dict[str, Any]) -> VoyageEmbeddingAnalyzer:
    """
    Factory function to create VoyageEmbeddingAnalyzer instance

    Args:
        config: Configuration dictionary

    Returns:
        VoyageEmbeddingAnalyzer instance
    """
    return VoyageEmbeddingAnalyzer(config)
