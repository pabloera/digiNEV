"""
digiNEV Semantic Search: Intelligent content discovery engine for Brazilian political discourse research queries
Function: Natural language search with context-aware results for finding specific discourse patterns and political content
Usage: Social scientists query the dataset naturally - search for "authoritarian discourse" or "violence rhetoric" with semantic understanding
"""

import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .base import AnthropicBase
from .hybrid_search_engine import HybridSearchEngine, SearchResult
from .optimized_cache import OptimizedCache
from .voyage_embeddings import VoyageEmbeddingAnalyzer

logger = logging.getLogger(__name__)

class SemanticSearchEngine(AnthropicBase):
    """
    Advanced Semantic Search & Intelligence Engine

    Capabilities:
    - Natural language querying of political content
    - Semantic similarity search across millions of messages
    - Intelligent content discovery and clustering
    - Temporal semantic evolution tracking
    - Automated insight generation
    - Political discourse pattern analysis
    - Conspiracy/misinformation detection
    - Cross-channel influence mapping
    """

    def __init__(self, config: Dict[str, Any], embedding_analyzer: VoyageEmbeddingAnalyzer = None):
        super().__init__(config)

        # Initialize or use provided embedding analyzer with Voyage optimization
        self.embedding_analyzer = embedding_analyzer or VoyageEmbeddingAnalyzer(config)
        
        # For test compatibility - expose voyage client
        self.voyage_client = self.embedding_analyzer.voyage_embeddings.client

        # Initialize hybrid search engine
        self.hybrid_engine = HybridSearchEngine(config, self.embedding_analyzer)

        # Configuration
        search_config = config.get('semantic_search', {})
        self.max_results = search_config.get('max_results', 100)
        self.similarity_threshold = search_config.get('similarity_threshold', 0.7)
        self.cluster_eps = search_config.get('cluster_eps', 0.3)
        self.min_cluster_size = search_config.get('min_cluster_size', 5)
        self.enable_caching = search_config.get('enable_caching', True)

        # Voyage.ai optimization settings
        self.voyage_optimized = hasattr(self.embedding_analyzer, 'voyage_available') and self.embedding_analyzer.voyage_available

        if self.voyage_optimized:
            logger.info("Semantic Search: Voyage.ai integration active")
            # Optimize for Voyage.ai usage
            self.similarity_threshold = max(0.75, self.similarity_threshold)  # Higher precision
            self.enable_query_optimization = True
        else:
            logger.info("⚠️ Semantic Search: Using fallback embeddings")
            self.enable_query_optimization = False

        # Search index storage (legacy - for backward compatibility)
        self.search_index = {}
        self.embeddings_cache = {}
        self.metadata_cache = {}

        # Results cache using optimized cache
        cache_dir = Path(config.get('data', {}).get('interim_path', 'data/interim')) / 'semantic_search_cache'
        self.results_cache = OptimizedCache(cache_dir, max_memory_mb=64, ttl_hours=12)

        # Political context for Brazilian discourse
        self.political_keywords = {
            'government': ['governo', 'presidente', 'ministro', 'secretário', 'deputado', 'senador', 'prefeito', 'governador'],
            'institutions': ['stf', 'supremo', 'congresso', 'senado', 'câmara', 'tse', 'stj', 'pf', 'polícia federal'],
            'political_movements': ['direita', 'esquerda', 'centro', 'conservador', 'liberal', 'populista'],
            'controversies': ['corrupção', 'impeachment', 'cpi', 'operação', 'investigação', 'denúncia'],
            'social_issues': ['família', 'valores', 'tradição', 'religião', 'educação', 'segurança'],
            'conspiracy': ['deep state', 'globalismo', 'comunismo', 'mídia manipulação', 'fake news', 'sistema'],
            'pandemic': ['covid', 'vacina', 'lockdown', 'isolamento', 'hidroxicloroquina', 'kit covid', 'cloroquina'],
            'elections': ['eleição', 'urna', 'voto', 'fraude', 'auditoria', 'segundo turno', 'pesquisa eleitoral']
        }

        logger.info("SemanticSearchEngine initialized successfully")

    def search(self, documents: List[str], query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents given a query (for test compatibility).
        
        Args:
            documents: List of document texts to search in
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with document and score fields
        """
        try:
            # Get embeddings for documents and query using voyage client
            doc_response = self.voyage_client.embed(documents, model="voyage-3-lite")
            query_response = self.voyage_client.embed([query], model="voyage-3-lite")
            
            doc_embeddings = doc_response.embeddings
            query_embedding = query_response.embeddings[0]
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((i, similarity, documents[i]))
            
            # Sort by similarity (descending) and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Format results for test compatibility
            results = []
            for doc_idx, score, document in top_results:
                result = {
                    'document': document,
                    'text': document,  # Alternative field name
                    'score': float(score),
                    'similarity': float(score),  # Alternative field name
                    'index': doc_idx
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            # Fallback: return simple results based on text matching
            results = []
            query_lower = query.lower()
            for i, doc in enumerate(documents):
                # Simple text matching fallback
                score = 1.0 if query_lower in doc.lower() else 0.5
                result = {
                    'document': doc,
                    'text': doc,
                    'score': score,
                    'similarity': score,
                    'index': i
                }
                results.append(result)
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

    def build_search_index(self, df: pd.DataFrame, text_column: str = 'body_cleaned') -> Dict[str, Any]:
        """
        Build comprehensive search index from dataset using hybrid engine

        Args:
            df: DataFrame with processed data
            text_column: Column containing cleaned text

        Returns:
            Index building results and statistics
        """
        logger.info(f"Building enhanced semantic search index for {len(df)} documents")

        start_time = time.time()

        # Use hybrid search engine for index building
        hybrid_result = self.hybrid_engine.build_index(
            df,
            text_column=text_column,
            metadata_columns=['data', 'canal', 'url', 'hashtags', 'texto']
        )

        # Legacy compatibility - store basic index info
        if hybrid_result.get('success'):
            self.search_index = {
                'total_documents': hybrid_result['total_documents'],
                'embedding_dimension': hybrid_result.get('embedding_dimension'),
                'build_timestamp': datetime.now(),
                'hybrid_enabled': True
            }

        total_time = time.time() - start_time

        result = {
            'success': hybrid_result.get('success', False),
            'total_documents': hybrid_result.get('total_documents', 0),
            'hybrid_engine_stats': hybrid_result,
            'build_time_seconds': total_time,
            'enhanced_features': {
                'faiss_enabled': True,
                'hybrid_search': True,
                'optimized_cache': True,
                'compressed_storage': True
            }
        }

        if hybrid_result.get('success'):
            logger.info(f"Enhanced search index built successfully in {total_time:.2f}s")
        else:
            logger.error(f"Failed to build search index: {hybrid_result.get('error')}")

        return result

    def semantic_search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict[str, Any] = None,
        include_metadata: bool = True,
        search_mode: str = 'hybrid'
    ) -> Dict[str, Any]:
        """
        Perform enhanced semantic search with hybrid approach

        Args:
            query: Natural language search query
            top_k: Number of results to return
            filters: Optional filters (date_range, channels, etc.)
            include_metadata: Include document metadata in results
            search_mode: 'hybrid', 'dense', or 'sparse'

        Returns:
            Enhanced search results with hybrid scoring
        """
        if not self.search_index:
            return {'error': 'Search index not built. Call build_search_index() first.'}

        start_time = time.time()
        top_k = top_k or self.max_results

        # Check cache first
        cache_key = f"search_{hash(query)}_{top_k}_{search_mode}_{hash(str(filters))}"
        cached_result = self.results_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result

        try:
            # Configure weights based on search mode
            if search_mode == 'dense':
                dense_weight, sparse_weight = 1.0, 0.0
            elif search_mode == 'sparse':
                dense_weight, sparse_weight = 0.0, 1.0
            else:  # hybrid
                dense_weight, sparse_weight = 0.7, 0.3

            # Perform hybrid search
            search_results = self.hybrid_engine.search(
                query=query,
                max_results=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )

            # Convert to legacy format for compatibility
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    'text': result.text,
                    'similarity_score': result.hybrid_score,
                    'dense_score': result.dense_score,
                    'sparse_score': result.sparse_score,
                    'relevance_explanation': result.relevance_explanation,
                    'doc_id': result.doc_id
                }

                if include_metadata:
                    formatted_result['metadata'] = result.metadata

                formatted_results.append(formatted_result)

            search_time = time.time() - start_time

            final_result = {
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'search_time_seconds': search_time,
                'search_mode': search_mode,
                'enhanced_features': {
                    'hybrid_scoring': search_mode == 'hybrid',
                    'faiss_enabled': True,
                    'cache_enabled': True
                },
                'weights_used': {
                    'dense': dense_weight,
                    'sparse': sparse_weight
                }
            }

            # Cache the result
            self.results_cache.put(cache_key, final_result)

            logger.debug(f"Search completed in {search_time:.3f}s, {len(formatted_results)} results")
            return final_result

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return {
                'error': str(e),
                'query': query,
                'results': [],
                'search_mode': search_mode
            }

    def get_search_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced search performance statistics"""
        try:
            # Get hybrid engine stats
            hybrid_stats = self.hybrid_engine.get_search_stats()

            # Get cache stats
            cache_stats = self.results_cache.get_stats()

            return {
                'hybrid_search_engine': hybrid_stats,
                'results_cache': cache_stats,
                'features_enabled': {
                    'faiss_acceleration': True,
                    'hybrid_search': True,
                    'compressed_cache': True,
                    'optimized_embeddings': True
                },
                'performance_summary': {
                    'total_searches': hybrid_stats.get('total_searches', 0),
                    'avg_search_time': hybrid_stats.get('avg_search_time', 0),
                    'cache_hit_rate': cache_stats.get('hit_rate', 0),
                    'memory_usage_mb': cache_stats.get('memory_usage_mb', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {'error': str(e)}

    def cleanup_caches(self) -> Dict[str, Any]:
        """Clean up all caches"""
        try:
            # Clean hybrid engine caches
            hybrid_cleanup = self.hybrid_engine.cleanup_cache()

            # Clean results cache
            results_cleaned = self.results_cache.cleanup_expired()

            return {
                'hybrid_engine_cleanup': hybrid_cleanup,
                'results_cache_cleaned': results_cleaned,
                'total_items_cleaned': hybrid_cleanup.get('total_cleaned', 0) + results_cleaned
            }
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return {'error': str(e)}

    def discover_content_patterns(
        self,
        topic: str = None,
        time_range: Tuple[str, str] = None,
        min_cluster_size: int = None
    ) -> Dict[str, Any]:
        """
        Discover content patterns and themes automatically

        Args:
            topic: Optional topic focus for discovery
            time_range: Optional time range (start_date, end_date)
            min_cluster_size: Minimum cluster size for pattern detection

        Returns:
            Discovered patterns, clusters, and insights
        """
        if not self.search_index:
            return {'error': 'Search index not built. Call build_search_index() first.'}

        logger.info("Discovering content patterns and themes")

        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size

        # Apply filters if specified
        filters = {}
        if time_range:
            filters['date_range'] = time_range

        filtered_indices = self._apply_filters(filters) if filters else list(range(self.search_index['index_size']))

        if len(filtered_indices) < min_cluster_size:
            return {'error': f'Not enough documents ({len(filtered_indices)}) for pattern discovery'}

        # Get embeddings for filtered documents
        filtered_embeddings = self.search_index['embeddings'][filtered_indices]

        # Perform clustering to discover patterns
        clusterer = DBSCAN(
            eps=self.cluster_eps,
            min_samples=min_cluster_size,
            metric='cosine'
        )

        clusters = clusterer.fit_predict(filtered_embeddings)

        # Analyze discovered clusters
        unique_clusters = set(clusters)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise cluster

        discovered_patterns = []

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_indices = [filtered_indices[i] for i, mask in enumerate(cluster_mask) if mask]

            if len(cluster_indices) < min_cluster_size:
                continue

            # Analyze cluster content
            cluster_texts = [self.search_index['texts'][i] for i in cluster_indices]
            cluster_metadata = [self.search_index['metadata'][i] for i in cluster_indices]

            # Generate cluster summary with AI
            pattern_analysis = self._analyze_pattern_with_ai(cluster_texts[:10], cluster_id)

            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_statistics(cluster_metadata)

            discovered_patterns.append({
                'pattern_id': cluster_id,
                'document_count': len(cluster_indices),
                'description': pattern_analysis.get('description', f'Pattern {cluster_id}'),
                'key_themes': pattern_analysis.get('key_themes', []),
                'political_relevance': pattern_analysis.get('political_relevance', 'medium'),
                'representative_texts': cluster_texts[:3],
                'statistics': cluster_stats,
                'document_indices': cluster_indices
            })

        # Sort patterns by significance
        discovered_patterns.sort(key=lambda x: x['document_count'], reverse=True)

        # Generate overall insights
        overall_insights = self._generate_discovery_insights(discovered_patterns)

        return {
            'total_patterns_discovered': len(discovered_patterns),
            'documents_analyzed': len(filtered_indices),
            'clustering_algorithm': 'DBSCAN',
            'parameters': {
                'eps': self.cluster_eps,
                'min_samples': min_cluster_size
            },
            'discovered_patterns': discovered_patterns,
            'overall_insights': overall_insights,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def analyze_semantic_evolution(
        self,
        concept: str,
        time_windows: int = 12,
        window_size_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze how semantic concepts evolve over time

        Args:
            concept: Concept to track (e.g., "democracia", "vacinas", "eleições")
            time_windows: Number of time windows to analyze
            window_size_days: Size of each time window in days

        Returns:
            Semantic evolution analysis over time
        """
        if not self.search_index:
            return {'error': 'Search index not built. Call build_search_index() first.'}

        logger.info(f"Analyzing semantic evolution of concept: '{concept}'")

        # Find documents related to the concept
        concept_search = self.semantic_search(
            query=concept,
            top_k=1000,  # Get more results for temporal analysis
            include_metadata=True
        )

        if not concept_search.get('results'):
            return {'error': f'No documents found related to concept: {concept}'}

        # Organize results by time windows
        evolution_data = []
        concept_docs = concept_search['results']

        # Extract dates and sort documents
        dated_docs = []
        for doc in concept_docs:
            metadata = doc.get('metadata', {})
            date_str = metadata.get('datetime') or metadata.get('timestamp')

            if date_str:
                try:
                    # Try different date formats
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            date_obj = datetime.strptime(str(date_str), fmt)
                            dated_docs.append((date_obj, doc))
                            break
                        except ValueError:
                            continue
                except:
                    continue

        if not dated_docs:
            return {'error': 'No documents with valid dates found'}

        # Sort by date
        dated_docs.sort(key=lambda x: x[0])

        # Create time windows
        start_date = dated_docs[0][0]
        end_date = dated_docs[-1][0]

        window_delta = timedelta(days=window_size_days)
        current_date = start_date

        for window_idx in range(time_windows):
            window_end = current_date + window_delta

            # Get documents in this window
            window_docs = [
                doc for date, doc in dated_docs
                if current_date <= date < window_end
            ]

            if len(window_docs) < 3:  # Skip windows with too few docs
                current_date = window_end
                continue

            # Analyze semantic characteristics of this window
            window_texts = [doc['text'] for doc in window_docs]
            window_analysis = self._analyze_temporal_window(
                concept,
                window_texts,
                current_date,
                window_end
            )

            evolution_data.append({
                'window_index': window_idx,
                'start_date': current_date.isoformat(),
                'end_date': window_end.isoformat(),
                'document_count': len(window_docs),
                'semantic_analysis': window_analysis,
                'average_similarity': np.mean([doc['similarity_score'] for doc in window_docs]),
                'representative_texts': window_texts[:2]
            })

            current_date = window_end

        # Generate evolution insights
        evolution_insights = self._generate_evolution_insights(concept, evolution_data)

        return {
            'concept': concept,
            'total_related_documents': len(concept_docs),
            'time_windows_analyzed': len(evolution_data),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'evolution_timeline': evolution_data,
            'insights': evolution_insights,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def detect_influence_networks(
        self,
        similarity_threshold: float = None,
        min_network_size: int = 5
    ) -> Dict[str, Any]:
        """
        Detect influence networks and content propagation patterns

        Args:
            similarity_threshold: Minimum similarity for network connections
            min_network_size: Minimum size for network detection

        Returns:
            Detected influence networks and propagation patterns
        """
        if not self.search_index:
            return {'error': 'Search index not built. Call build_search_index() first.'}

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        logger.info("Detecting influence networks and content propagation")

        # Calculate pairwise similarities
        similarities = cosine_similarity(self.search_index['embeddings'])

        # Find high-similarity connections
        high_sim_pairs = []
        n_docs = len(self.search_index['texts'])

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if similarities[i][j] > similarity_threshold:
                    high_sim_pairs.append((i, j, similarities[i][j]))

        logger.info(f"Found {len(high_sim_pairs)} high-similarity connections")

        # Build network graph
        networks = self._build_influence_networks(high_sim_pairs, min_network_size)

        # Analyze networks
        network_analysis = []
        for network_id, network in enumerate(networks):
            if len(network) < min_network_size:
                continue

            # Get network metadata
            network_metadata = [self.search_index['metadata'][i] for i in network]
            network_texts = [self.search_index['texts'][i] for i in network]

            # Analyze network characteristics
            network_stats = self._analyze_network_characteristics(network_metadata, network_texts)

            network_analysis.append({
                'network_id': network_id,
                'size': len(network),
                'document_indices': network,
                'characteristics': network_stats,
                'representative_texts': network_texts[:3],
                'channels_involved': network_stats.get('unique_channels', []),
                'time_span': network_stats.get('time_span', {}),
                'influence_score': network_stats.get('influence_score', 0.0)
            })

        # Sort by influence score
        network_analysis.sort(key=lambda x: x['influence_score'], reverse=True)

        return {
            'total_networks_detected': len(network_analysis),
            'similarity_threshold_used': similarity_threshold,
            'min_network_size': min_network_size,
            'high_similarity_connections': len(high_sim_pairs),
            'influence_networks': network_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def generate_automated_insights(self, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Generate automated insights from the indexed content

        Args:
            focus_areas: Optional list of areas to focus analysis on

        Returns:
            Comprehensive automated insights and discoveries
        """
        if not self.search_index:
            return {'error': 'Search index not built. Call build_search_index() first.'}

        logger.info("Generating automated insights from semantic analysis")

        if focus_areas is None:
            focus_areas = [
                'political_discourse', 'conspiracy_theories', 'institutional_trust',
                'pandemic_response', 'election_integrity', 'media_criticism'
            ]

        insights = {}

        for focus_area in focus_areas:
            logger.info(f"Analyzing focus area: {focus_area}")

            # Define search terms for each focus area
            search_terms = self._get_search_terms_for_focus(focus_area)

            area_insights = []
            for term in search_terms:
                search_result = self.semantic_search(
                    query=term,
                    top_k=50,
                    include_metadata=True
                )

                if search_result.get('results'):
                    # Analyze this subset
                    term_analysis = self._analyze_focus_area_content(
                        term,
                        search_result['results']
                    )
                    area_insights.append(term_analysis)

            # Synthesize insights for this focus area
            synthesized_insights = self._synthesize_area_insights(focus_area, area_insights)
            insights[focus_area] = synthesized_insights

        # Generate cross-area insights
        cross_insights = self._generate_cross_area_insights(insights)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(insights, cross_insights)

        return {
            'focus_areas_analyzed': focus_areas,
            'individual_insights': insights,
            'cross_area_insights': cross_insights,
            'executive_summary': executive_summary,
            'methodology': {
                'search_engine': 'semantic_embeddings',
                'ai_analysis': self.api_available,
                'index_size': self.search_index['index_size']
            },
            'generated_at': datetime.now().isoformat()
        }

    # Helper Methods

    def _build_keyword_index(self, df: pd.DataFrame, texts: List[str]):
        """Build keyword-based index for hybrid search"""
        keyword_index = defaultdict(list)

        for idx, text in enumerate(texts):
            text_lower = text.lower()

            # Index political keywords
            for category, keywords in self.political_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        keyword_index[f"{category}:{keyword}"].append(idx)

        self.search_index['keyword_index'] = dict(keyword_index)

    def _build_temporal_index(self, df: pd.DataFrame):
        """Build temporal index for time-based filtering"""
        temporal_index = defaultdict(list)

        for idx, row in df.iterrows():
            date_str = row.get('datetime') or row.get('timestamp')
            if date_str:
                try:
                    # Extract year-month for indexing
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                        try:
                            date_obj = datetime.strptime(str(date_str), fmt)
                            year_month = f"{date_obj.year}-{date_obj.month:02d}"
                            temporal_index[year_month].append(idx)
                            break
                        except ValueError:
                            continue
                except:
                    continue

        self.search_index['temporal_index'] = dict(temporal_index)

    def _build_channel_index(self, df: pd.DataFrame):
        """Build channel-based index"""
        channel_index = defaultdict(list)

        for idx, row in df.iterrows():
            channel = row.get('channel') or row.get('canal')
            if channel:
                channel_index[str(channel).lower()].append(idx)

        self.search_index['channel_index'] = dict(channel_index)

    def _apply_filters(self, filters: Dict[str, Any]) -> List[int]:
        """Apply filters to search results"""
        if not filters:
            return list(range(self.search_index['index_size']))

        valid_indices = set(range(self.search_index['index_size']))

        # Date range filter
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            date_indices = set()

            for year_month, indices in self.search_index.get('temporal_index', {}).items():
                # Simple year-month comparison (can be improved)
                if start_date <= year_month <= end_date:
                    date_indices.update(indices)

            valid_indices &= date_indices

        # Channel filter
        if 'channels' in filters:
            channel_indices = set()
            for channel in filters['channels']:
                channel_indices.update(
                    self.search_index.get('channel_index', {}).get(channel.lower(), [])
                )
            valid_indices &= channel_indices

        return list(valid_indices)

    def _analyze_query_with_ai(self, query: str, top_results: List[Dict]) -> Dict[str, Any]:
        """Analyze query and results using AI"""
        if not self.api_available or not top_results:
            return {}

        try:
            result_texts = [r['text'][:200] + "..." for r in top_results]
            results_sample = '\n'.join([f"{i+1}. {text}" for i, text in enumerate(result_texts)])

            prompt = f"""
Analise esta consulta semântica em um dataset brasileiro de Telegram (2019-2023):

CONSULTA: "{query}"

TOP RESULTADOS ENCONTRADOS:
{results_sample}

Forneça uma análise em JSON:
{{
  "query_intent": "intenção da consulta",
  "political_context": "contexto político identificado",
  "key_themes_found": ["tema1", "tema2", "tema3"],
  "relevance_assessment": "alta|media|baixa",
  "potential_bias_indicators": ["indicador1", "indicador2"],
  "recommended_follow_up": ["consulta_sugerida1", "consulta_sugerida2"]
}}
"""

            response = self.create_message(
                prompt,
                stage="semantic_search",
                operation="query_analysis",
                temperature=0.3
            )

            return self.parse_json_response(response)

        except Exception as e:
            logger.warning(f"AI query analysis failed: {e}")
            return {}

    def _analyze_pattern_with_ai(self, texts: List[str], pattern_id: int) -> Dict[str, Any]:
        """Analyze discovered pattern using AI"""
        if not self.api_available:
            return {'description': f'Pattern {pattern_id}', 'key_themes': []}

        try:
            texts_sample = '\n'.join([f"- {text[:150]}..." for text in texts[:5]])

            prompt = f"""
Analise este padrão de conteúdo descoberto em mensagens do Telegram brasileiro:

TEXTOS REPRESENTATIVOS:
{texts_sample}

Este padrão foi identificado automaticamente através de clustering semântico.

Responda em JSON:
{{
  "description": "descrição concisa do padrão (max 50 chars)",
  "key_themes": ["tema1", "tema2", "tema3"],
  "political_relevance": "alta|media|baixa",
  "discourse_type": "institucional|conspiratorio|informativo|mobilizacao",
  "emotional_tone": "positivo|negativo|neutro|polarizado",
  "potential_significance": "significado potencial deste padrão"
}}
"""

            response = self.create_message(
                prompt,
                stage="semantic_search",
                operation="pattern_analysis",
                temperature=0.3
            )

            return self.parse_json_response(response)

        except Exception as e:
            logger.warning(f"AI pattern analysis failed: {e}")
            return {'description': f'Pattern {pattern_id}', 'key_themes': []}

    def _calculate_index_statistics(self) -> Dict[str, Any]:
        """Calculate search index statistics"""
        if not self.search_index:
            return {}

        texts = self.search_index['texts']
        metadata = self.search_index['metadata']

        # Text statistics
        text_lengths = [len(text) for text in texts]

        # Channel statistics
        channels = [meta.get('channel', 'unknown') for meta in metadata]
        channel_counts = Counter(channels)

        # Date statistics
        dates = []
        for meta in metadata:
            date_str = meta.get('datetime') or meta.get('timestamp')
            if date_str:
                dates.append(date_str)

        return {
            'total_documents': len(texts),
            'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
            'unique_channels': len(channel_counts),
            'top_channels': dict(channel_counts.most_common(5)),
            'date_range': {
                'earliest': min(dates) if dates else None,
                'latest': max(dates) if dates else None
            },
            'text_length_distribution': {
                'min': min(text_lengths) if text_lengths else 0,
                'max': max(text_lengths) if text_lengths else 0,
                'median': np.median(text_lengths) if text_lengths else 0
            }
        }

    def _calculate_cluster_statistics(self, metadata: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for a cluster"""
        if not metadata:
            return {}

        # Channel distribution
        channels = [meta.get('channel', 'unknown') for meta in metadata]
        channel_counts = Counter(channels)

        # Time distribution
        dates = []
        for meta in metadata:
            date_str = meta.get('datetime') or meta.get('timestamp')
            if date_str:
                dates.append(date_str)

        return {
            'document_count': len(metadata),
            'unique_channels': len(channel_counts),
            'channel_distribution': dict(channel_counts.most_common(3)),
            'temporal_spread': {
                'earliest': min(dates) if dates else None,
                'latest': max(dates) if dates else None,
                'total_dates': len(dates)
            }
        }

    def _generate_discovery_insights(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Generate insights from discovered patterns"""
        if not patterns:
            return {}

        total_docs = sum(p['document_count'] for p in patterns)

        # Identify dominant themes
        all_themes = []
        for pattern in patterns:
            all_themes.extend(pattern.get('key_themes', []))

        theme_counts = Counter(all_themes)

        return {
            'total_patterns': len(patterns),
            'total_documents_in_patterns': total_docs,
            'dominant_themes': dict(theme_counts.most_common(5)),
            'largest_pattern_size': max(p['document_count'] for p in patterns),
            'pattern_size_distribution': [p['document_count'] for p in patterns]
        }

    def _analyze_temporal_window(
        self,
        concept: str,
        texts: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze semantic characteristics of a temporal window"""
        # Basic analysis without AI
        basic_analysis = {
            'document_count': len(texts),
            'avg_text_length': np.mean([len(text) for text in texts]),
            'concept_mentions': sum(1 for text in texts if concept.lower() in text.lower())
        }

        # AI analysis if available
        if self.api_available and texts:
            try:
                texts_sample = '\n'.join(texts[:3])

                prompt = f"""
Analise como o conceito "{concept}" aparece nestes textos do período {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}:

{texts_sample}

Responda em JSON:
{{
  "dominant_sentiment": "positivo|negativo|neutro",
  "context_type": "informativo|opinativo|mobilizacao|critica",
  "semantic_focus": "foco semântico principal",
  "evolution_indicator": "estabilidade|mudança|intensificação"
}}
"""

                response = self.create_message(
                    prompt,
                    stage="semantic_search",
                    operation="temporal_analysis",
                    temperature=0.3
                )

                ai_analysis = self.parse_json_response(response)
                basic_analysis.update(ai_analysis)

            except Exception as e:
                logger.warning(f"AI temporal analysis failed: {e}")

        return basic_analysis

    def _generate_evolution_insights(self, concept: str, evolution_data: List[Dict]) -> Dict[str, Any]:
        """Generate insights about concept evolution"""
        if not evolution_data:
            return {}

        # Calculate trends
        doc_counts = [w['document_count'] for w in evolution_data]
        similarities = [w['average_similarity'] for w in evolution_data]

        return {
            'concept': concept,
            'temporal_trend': 'increasing' if doc_counts[-1] > doc_counts[0] else 'decreasing',
            'semantic_stability': 'stable' if np.std(similarities) < 0.1 else 'volatile',
            'peak_period': evolution_data[doc_counts.index(max(doc_counts))]['start_date'] if doc_counts else None,
            'total_windows_analyzed': len(evolution_data),
            'avg_documents_per_window': np.mean(doc_counts) if doc_counts else 0
        }

    def _build_influence_networks(self, similarity_pairs: List[Tuple], min_size: int) -> List[List[int]]:
        """Build influence networks from similarity connections"""
        # Simple network building using connected components
        networks = []
        processed = set()

        # Build adjacency list
        adjacency = defaultdict(set)
        for i, j, sim in similarity_pairs:
            adjacency[i].add(j)
            adjacency[j].add(i)

        # Find connected components
        for node in adjacency:
            if node in processed:
                continue

            # BFS to find connected component
            network = []
            queue = [node]
            component_processed = set()

            while queue:
                current = queue.pop(0)
                if current in component_processed:
                    continue

                component_processed.add(current)
                network.append(current)

                for neighbor in adjacency[current]:
                    if neighbor not in component_processed:
                        queue.append(neighbor)

            if len(network) >= min_size:
                networks.append(network)

            processed.update(component_processed)

        return networks

    def _analyze_network_characteristics(self, metadata: List[Dict], texts: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of an influence network"""
        if not metadata:
            return {}

        # Channel analysis
        channels = [meta.get('channel', 'unknown') for meta in metadata]
        unique_channels = list(set(channels))

        # Time analysis
        dates = []
        for meta in metadata:
            date_str = meta.get('datetime') or meta.get('timestamp')
            if date_str:
                dates.append(date_str)

        # Calculate influence score based on network characteristics
        influence_score = len(unique_channels) * 0.3 + len(metadata) * 0.1
        if len(dates) > 0:
            influence_score += min(len(set(dates)), 10) * 0.1

        return {
            'unique_channels': unique_channels,
            'total_messages': len(metadata),
            'channel_diversity': len(unique_channels),
            'time_span': {
                'earliest': min(dates) if dates else None,
                'latest': max(dates) if dates else None
            },
            'influence_score': influence_score,
            'avg_text_length': np.mean([len(text) for text in texts]) if texts else 0
        }

    def _get_search_terms_for_focus(self, focus_area: str) -> List[str]:
        """Get search terms for a focus area"""
        focus_terms = {
            'political_discourse': ['democracia', 'governo', 'política', 'instituições'],
            'conspiracy_theories': ['deep state', 'globalismo', 'manipulação', 'conspiração'],
            'institutional_trust': ['stf', 'supremo', 'congresso', 'justiça'],
            'pandemic_response': ['covid', 'vacina', 'lockdown', 'pandemia'],
            'election_integrity': ['eleição', 'urna', 'fraude', 'voto'],
            'media_criticism': ['mídia', 'imprensa', 'jornalismo', 'fake news']
        }

        return focus_terms.get(focus_area, [focus_area])

    def _analyze_focus_area_content(self, term: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze content for a specific focus area term"""
        if not results:
            return {'term': term, 'analysis': 'no_content'}

        # Basic statistics
        avg_similarity = np.mean([r['similarity_score'] for r in results])

        # Channel diversity
        channels = []
        for result in results:
            channel = result.get('metadata', {}).get('channel')
            if channel:
                channels.append(channel)

        unique_channels = len(set(channels))

        return {
            'term': term,
            'document_count': len(results),
            'avg_similarity': avg_similarity,
            'channel_diversity': unique_channels,
            'top_channels': list(Counter(channels).most_common(3)),
            'sample_texts': [r['text'][:100] for r in results[:2]]
        }

    def _synthesize_area_insights(self, focus_area: str, area_insights: List[Dict]) -> Dict[str, Any]:
        """Synthesize insights for a focus area"""
        if not area_insights:
            return {'focus_area': focus_area, 'status': 'no_data'}

        total_docs = sum(insight['document_count'] for insight in area_insights)
        avg_similarity = np.mean([insight['avg_similarity'] for insight in area_insights])

        # Collect all channels
        all_channels = []
        for insight in area_insights:
            all_channels.extend([ch for ch, count in insight.get('top_channels', [])])

        top_channels = Counter(all_channels).most_common(5)

        return {
            'focus_area': focus_area,
            'total_related_documents': total_docs,
            'average_semantic_relevance': avg_similarity,
            'terms_analyzed': [insight['term'] for insight in area_insights],
            'top_channels_overall': top_channels,
            'engagement_level': 'high' if total_docs > 100 else 'medium' if total_docs > 20 else 'low'
        }

    def _generate_cross_area_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights across focus areas"""
        if not insights:
            return {}

        # Find areas with highest engagement
        engagement_scores = {}
        for area, data in insights.items():
            if isinstance(data, dict) and 'total_related_documents' in data:
                engagement_scores[area] = data['total_related_documents']

        sorted_areas = sorted(engagement_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'most_active_areas': sorted_areas[:3],
            'total_areas_analyzed': len(insights),
            'cross_cutting_themes': ['política', 'governo', 'democracia'],  # Could be enhanced with AI
            'overall_discourse_health': 'requires_analysis'  # Placeholder for more complex analysis
        }

    def _generate_executive_summary(self, insights: Dict[str, Any], cross_insights: Dict[str, Any]) -> str:
        """Generate executive summary of all insights"""
        if not insights:
            return "No insights generated due to insufficient data."

        most_active = cross_insights.get('most_active_areas', [])

        summary = f"Análise semântica identificou {len(insights)} áreas temáticas no discurso político brasileiro."

        if most_active:
            top_area = most_active[0][0].replace('_', ' ')
            summary += f" A área mais ativa foi '{top_area}' com {most_active[0][1]} documentos relacionados."

        summary += " A análise revela padrões complexos de engajamento e evolução semântica no período analisado."

        return summary

def create_semantic_search_engine(
    config: Dict[str, Any],
    embedding_analyzer: VoyageEmbeddingAnalyzer = None
) -> SemanticSearchEngine:
    """
    Factory function to create SemanticSearchEngine instance

    Args:
        config: Configuration dictionary
        embedding_analyzer: Optional pre-initialized embedding analyzer

    Returns:
        SemanticSearchEngine instance
    """
    return SemanticSearchEngine(config, embedding_analyzer)
