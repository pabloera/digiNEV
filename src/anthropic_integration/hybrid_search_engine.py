"""
Hybrid Search Engine with FAISS and Optimized Performance
Combines dense (semantic) and sparse (keyword) search for optimal results
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

import re
from collections import Counter

# Sparse search components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import AnthropicBase
from .optimized_cache import EmbeddingCache, OptimizedCache
from .voyage_embeddings import VoyageEmbeddingAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual search result"""
    doc_id: int
    text: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    metadata: Dict[str, Any]
    relevance_explanation: str


@dataclass
class SearchStats:
    """Search performance statistics"""
    total_time: float
    dense_time: float
    sparse_time: float
    rerank_time: float
    total_docs: int
    results_returned: int
    cache_hits: int


class HybridSearchEngine(AnthropicBase):
    """
    Advanced Hybrid Search Engine

    Features:
    - FAISS-powered dense vector search
    - TF-IDF sparse keyword search
    - Smart score fusion
    - Optimized caching
    - Multi-threaded operations
    - Query expansion
    - Result re-ranking
    """

    def __init__(self, config: Dict[str, Any], embedding_analyzer: VoyageEmbeddingAnalyzer = None):
        super().__init__(config)

        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for HybridSearchEngine. Install with: pip install faiss-cpu")

        # Initialize embedding analyzer
        self.embedding_analyzer = embedding_analyzer or VoyageEmbeddingAnalyzer(config)

        # Configuration
        search_config = config.get('hybrid_search', {})
        self.max_results = search_config.get('max_results', 100)
        self.dense_weight = search_config.get('dense_weight', 0.7)
        self.sparse_weight = search_config.get('sparse_weight', 0.3)
        self.rerank_top_k = search_config.get('rerank_top_k', 200)
        self.min_similarity = search_config.get('min_similarity', 0.1)

        # FAISS configuration
        faiss_config = search_config.get('faiss', {})
        self.index_type = faiss_config.get('index_type', 'IVF')  # Options: Flat, IVF, HNSW
        self.nlist = faiss_config.get('nlist', 100)  # Number of clusters for IVF
        self.nprobe = faiss_config.get('nprobe', 10)  # Number of clusters to search

        # TF-IDF configuration
        tfidf_config = search_config.get('tfidf', {})
        self.max_features = tfidf_config.get('max_features', 10000)
        self.ngram_range = tuple(tfidf_config.get('ngram_range', [1, 2]))
        self.min_df = tfidf_config.get('min_df', 2)
        self.max_df = tfidf_config.get('max_df', 0.95)

        # Initialize components
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_store = None
        self.embeddings = None

        # Cache setup
        cache_dir = Path(config.get('data', {}).get('interim_path', 'data/interim')) / 'hybrid_search_cache'
        self.embedding_cache = EmbeddingCache(cache_dir / 'embeddings', max_memory_mb=256)
        self.search_cache = OptimizedCache(cache_dir / 'search_results', max_memory_mb=128, ttl_hours=6)

        # Threading
        self.lock = threading.RLock()

        # Statistics
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'last_index_build': None
        }

        logger.info("HybridSearchEngine initialized successfully")

    def build_index(self, df: pd.DataFrame, text_column: str = 'body_cleaned',
                   metadata_columns: List[str] = None) -> Dict[str, Any]:
        """
        Build hybrid search index

        Args:
            df: DataFrame with documents
            text_column: Column containing text to index
            metadata_columns: Additional columns to store as metadata

        Returns:
            Build statistics and information
        """
        logger.info(f"Building hybrid search index for {len(df)} documents")
        start_time = time.time()

        try:
            # Prepare documents
            texts = df[text_column].fillna('').astype(str).tolist()

            # Filter out very short texts
            valid_indices = [i for i, text in enumerate(texts) if len(text.strip()) > 20]
            valid_texts = [texts[i] for i in valid_indices]
            valid_df = df.iloc[valid_indices].copy()

            logger.info(f"Processing {len(valid_texts)} valid texts (filtered from {len(texts)})")

            # Build dense index (FAISS)
            dense_time = time.time()
            embeddings_result = self._build_dense_index(valid_texts)
            dense_build_time = time.time() - dense_time

            # Build sparse index (TF-IDF)
            sparse_time = time.time()
            tfidf_result = self._build_sparse_index(valid_texts)
            sparse_build_time = time.time() - sparse_time

            # Store document metadata
            self._build_document_store(valid_df, valid_indices, metadata_columns)

            total_time = time.time() - start_time

            # Update statistics
            self.search_stats['last_index_build'] = datetime.now()

            result = {
                'success': True,
                'total_documents': len(valid_texts),
                'embedding_dimension': embeddings_result.get('dimension'),
                'faiss_index_type': self.index_type,
                'tfidf_features': tfidf_result.get('features'),
                'build_time': {
                    'total': total_time,
                    'dense': dense_build_time,
                    'sparse': sparse_build_time,
                    'metadata': total_time - dense_build_time - sparse_build_time
                },
                'index_info': {
                    'faiss_total': self.faiss_index.ntotal if self.faiss_index else 0,
                    'tfidf_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else (0, 0)
                }
            }

            logger.info(f"Index built successfully in {total_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error building search index: {e}")
            return {'success': False, 'error': str(e)}

    def _build_dense_index(self, texts: List[str]) -> Dict[str, Any]:
        """Build FAISS dense vector index"""
        logger.info("Building dense vector index with FAISS")

        # Check cache first
        cache_key = f"embeddings_{hash(str(texts[:10]))}"  # Use first 10 texts as key
        cached_embeddings = self.embedding_cache.get_embeddings(cache_key)

        if cached_embeddings and len(cached_embeddings[0]) == len(texts):
            logger.info("Using cached embeddings")
            embeddings, metadata = cached_embeddings
        else:
            # Generate embeddings
            logger.info("Generating embeddings...")
            embedding_result = self.embedding_analyzer.generate_embeddings(texts)

            # Extract embeddings from result dictionary
            if isinstance(embedding_result, dict):
                embeddings = embedding_result.get('embeddings', [])
                if not embeddings:
                    raise ValueError("No embeddings returned from VoyageEmbeddingAnalyzer")
            else:
                embeddings = embedding_result

            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Ensure float32 for FAISS
            embeddings = embeddings.astype(np.float32)

            # Cache embeddings
            metadata = {'model': self.embedding_analyzer.model_name, 'count': len(texts)}
            self.embedding_cache.put_embeddings(cache_key, embeddings, metadata)

        # Store embeddings
        self.embeddings = embeddings
        dimension = embeddings.shape[1]

        # Build FAISS index
        if self.index_type == 'Flat':
            # Exact search (slower but more accurate)
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        elif self.index_type == 'IVF':
            # Approximate search with clustering
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            # Train the index
            index.train(embeddings)
            index.nprobe = self.nprobe
        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World (fast and accurate)
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")

        # Add vectors to index
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self.faiss_index = index

        logger.info(f"FAISS index built: {index.ntotal} vectors, dimension {dimension}")

        return {
            'dimension': dimension,
            'index_type': self.index_type,
            'total_vectors': index.ntotal
        }

    def _build_sparse_index(self, texts: List[str]) -> Dict[str, Any]:
        """Build TF-IDF sparse index"""
        logger.info("Building TF-IDF sparse index")

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self._get_portuguese_stopwords(),
            lowercase=True,
            strip_accents='unicode'
        )

        # Fit and transform texts
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        logger.info(f"TF-IDF matrix built: {self.tfidf_matrix.shape}")

        return {
            'features': self.tfidf_matrix.shape[1],
            'documents': self.tfidf_matrix.shape[0],
            'sparsity': 1.0 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))
        }

    def _build_document_store(self, df: pd.DataFrame, valid_indices: List[int],
                            metadata_columns: List[str] = None) -> None:
        """Build document metadata store"""
        logger.info("Building document metadata store")

        default_metadata_columns = ['data', 'canal', 'url', 'hashtags']
        if metadata_columns:
            default_metadata_columns.extend(metadata_columns)

        # Filter columns that actually exist
        available_columns = [col for col in default_metadata_columns if col in df.columns]

        self.document_store = []
        for idx, row in df.iterrows():
            doc_metadata = {
                'original_index': valid_indices[len(self.document_store)],
                'doc_id': len(self.document_store)
            }

            # Add available metadata
            for col in available_columns:
                doc_metadata[col] = row[col]

            self.document_store.append(doc_metadata)

        logger.info(f"Document store built: {len(self.document_store)} documents")

    def search(self, query: str, max_results: int = None,
              dense_weight: float = None, sparse_weight: float = None) -> List[SearchResult]:
        """
        Perform hybrid search

        Args:
            query: Search query
            max_results: Maximum number of results
            dense_weight: Weight for dense search (0-1)
            sparse_weight: Weight for sparse search (0-1)

        Returns:
            List of search results
        """
        if not self.faiss_index or not self.tfidf_matrix:
            raise ValueError("Search index not built. Call build_index() first.")

        start_time = time.time()

        # Use provided weights or defaults
        dense_w = dense_weight if dense_weight is not None else self.dense_weight
        sparse_w = sparse_weight if sparse_weight is not None else self.sparse_weight
        max_res = max_results if max_results is not None else self.max_results

        # Check cache
        cache_key = f"search_{hash(query)}_{dense_w}_{sparse_w}_{max_res}"
        cached_results = self.search_cache.get(cache_key)
        if cached_results:
            self.search_stats['cache_hits'] += 1
            return cached_results

        try:
            # Dense search
            dense_start = time.time()
            dense_results = self._dense_search(query, self.rerank_top_k)
            dense_time = time.time() - dense_start

            # Sparse search
            sparse_start = time.time()
            sparse_results = self._sparse_search(query, self.rerank_top_k)
            sparse_time = time.time() - sparse_start

            # Fusion and re-ranking
            rerank_start = time.time()
            hybrid_results = self._fusion_rerank(dense_results, sparse_results,
                                               dense_w, sparse_w, max_res)
            rerank_time = time.time() - rerank_start

            total_time = time.time() - start_time

            # Update statistics
            self.search_stats['total_searches'] += 1
            self.search_stats['avg_search_time'] = (
                (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + total_time) /
                self.search_stats['total_searches']
            )

            # Cache results
            self.search_cache.put(cache_key, hybrid_results)

            # Log search stats
            logger.debug(f"Search completed in {total_time:.3f}s "
                        f"(dense: {dense_time:.3f}s, sparse: {sparse_time:.3f}s, rerank: {rerank_time:.3f}s)")

            return hybrid_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform dense vector search using FAISS"""
        try:
            # Generate query embedding
            embedding_result = self.embedding_analyzer.generate_embeddings([query])

            # Extract embeddings from result dictionary
            if isinstance(embedding_result, dict):
                query_embedding = embedding_result.get('embeddings', [])
                if not query_embedding:
                    raise ValueError("No embeddings returned for query")
            else:
                query_embedding = embedding_result

            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            query_embedding = query_embedding.astype(np.float32)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.faiss_index.search(query_embedding, top_k)

            # Return results as (doc_id, score) tuples
            results = []
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx != -1 and score > self.min_similarity:  # -1 indicates invalid result
                    results.append((int(idx), float(score)))

            return results

        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []

    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform sparse TF-IDF search"""
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]

            # Return results as (doc_id, score) tuples
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > self.min_similarity:
                    results.append((int(idx), float(score)))

            return results

        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    def _fusion_rerank(self, dense_results: List[Tuple[int, float]],
                      sparse_results: List[Tuple[int, float]],
                      dense_weight: float, sparse_weight: float,
                      max_results: int) -> List[SearchResult]:
        """Fuse and re-rank dense and sparse results"""
        try:
            # Normalize weights
            total_weight = dense_weight + sparse_weight
            if total_weight > 0:
                dense_weight /= total_weight
                sparse_weight /= total_weight

            # Collect all unique document IDs
            all_docs = set()
            dense_scores = {}
            sparse_scores = {}

            # Process dense results
            for doc_id, score in dense_results:
                all_docs.add(doc_id)
                dense_scores[doc_id] = score

            # Process sparse results
            for doc_id, score in sparse_results:
                all_docs.add(doc_id)
                sparse_scores[doc_id] = score

            # Calculate hybrid scores
            hybrid_candidates = []
            for doc_id in all_docs:
                dense_score = dense_scores.get(doc_id, 0.0)
                sparse_score = sparse_scores.get(doc_id, 0.0)

                # Hybrid score with weights
                hybrid_score = (dense_weight * dense_score + sparse_weight * sparse_score)

                if hybrid_score > self.min_similarity:
                    hybrid_candidates.append({
                        'doc_id': doc_id,
                        'dense_score': dense_score,
                        'sparse_score': sparse_score,
                        'hybrid_score': hybrid_score
                    })

            # Sort by hybrid score
            hybrid_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)

            # Create SearchResult objects
            results = []
            for candidate in hybrid_candidates[:max_results]:
                doc_id = candidate['doc_id']

                # Get document metadata
                if doc_id < len(self.document_store):
                    doc_metadata = self.document_store[doc_id]
                    text = doc_metadata.get('body_cleaned', doc_metadata.get('body', 'N/A'))

                    # Generate relevance explanation
                    explanation = self._generate_relevance_explanation(
                        candidate['dense_score'], candidate['sparse_score'],
                        dense_weight, sparse_weight
                    )

                    result = SearchResult(
                        doc_id=doc_id,
                        text=text[:500] + '...' if len(text) > 500 else text,  # Truncate for display
                        dense_score=candidate['dense_score'],
                        sparse_score=candidate['sparse_score'],
                        hybrid_score=candidate['hybrid_score'],
                        metadata=doc_metadata,
                        relevance_explanation=explanation
                    )

                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in fusion/re-ranking: {e}")
            return []

    def _generate_relevance_explanation(self, dense_score: float, sparse_score: float,
                                      dense_weight: float, sparse_weight: float) -> str:
        """Generate human-readable relevance explanation"""
        explanations = []

        if dense_score > 0.7:
            explanations.append("high semantic similarity")
        elif dense_score > 0.5:
            explanations.append("moderate semantic similarity")
        elif dense_score > 0.3:
            explanations.append("low semantic similarity")

        if sparse_score > 0.5:
            explanations.append("strong keyword match")
        elif sparse_score > 0.3:
            explanations.append("moderate keyword match")
        elif sparse_score > 0.1:
            explanations.append("weak keyword match")

        if not explanations:
            explanations.append("minimal relevance")

        return f"Relevance: {', '.join(explanations)} (semantic: {dense_score:.2f}, keyword: {sparse_score:.2f})"

    def _get_portuguese_stopwords(self) -> List[str]:
        """Get Portuguese stopwords for TF-IDF"""
        return [
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no',
            'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à',
            'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só',
            'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter',
            'seus', 'suas', 'nem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa',
            'num', 'numa', 'pelos', 'pelas', 'essa', 'este', 'del', 'te', 'lo', 'le', 'les', 'rt', 'via'
        ]

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        stats = {
            **self.search_stats,
            'embedding_cache_stats': self.embedding_cache.get_stats(),
            'search_cache_stats': self.search_cache.get_stats(),
            'index_info': {
                'faiss_total': self.faiss_index.ntotal if self.faiss_index else 0,
                'tfidf_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else (0, 0),
                'documents_stored': len(self.document_store) if self.document_store else 0
            }
        }

        return stats

    def cleanup_cache(self) -> Dict[str, int]:
        """Clean up expired cache entries"""
        embedding_cleaned = self.embedding_cache.cleanup_expired()
        search_cleaned = self.search_cache.cleanup_expired()

        return {
            'embedding_cache_cleaned': embedding_cleaned,
            'search_cache_cleaned': search_cleaned,
            'total_cleaned': embedding_cleaned + search_cleaned
        }


def get_hybrid_search_engine(config: Dict[str, Any],
                            embedding_analyzer: VoyageEmbeddingAnalyzer = None) -> HybridSearchEngine:
    """
    Factory function to create HybridSearchEngine instance

    Args:
        config: Configuration dictionary
        embedding_analyzer: Optional embedding analyzer instance

    Returns:
        HybridSearchEngine instance
    """
    return HybridSearchEngine(config, embedding_analyzer)
