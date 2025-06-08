"""
Content Discovery Engine for Automated Political Discourse Insights
Automatically discovers patterns, trends, and important content in political discourse
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import re

from .base import AnthropicBase
from .semantic_search_engine import SemanticSearchEngine
from .voyage_embeddings import VoyageEmbeddingAnalyzer

logger = logging.getLogger(__name__)


class ContentDiscoveryEngine(AnthropicBase):
    """
    Automated Content Discovery Engine for Political Discourse Analysis
    
    Capabilities:
    - Automated trend detection and analysis
    - Anomaly detection in discourse patterns
    - Influence network discovery
    - Conspiracy theory and misinformation detection
    - Coordinated behavior identification
    - Emerging narrative tracking
    - Cross-platform content analysis
    - Automated insight generation
    - Real-time monitoring capabilities
    """
    
    def __init__(self, config: Dict[str, Any], search_engine: SemanticSearchEngine = None):
        super().__init__(config)
        
        # Initialize search engine
        if search_engine:
            self.search_engine = search_engine
        else:
            embedding_analyzer = VoyageEmbeddingAnalyzer(config)
            self.search_engine = SemanticSearchEngine(config, embedding_analyzer)
        
        # Discovery configuration
        discovery_config = config.get('content_discovery', {})
        self.min_trend_documents = discovery_config.get('min_trend_documents', 10)
        self.anomaly_threshold = discovery_config.get('anomaly_threshold', 2.0)
        self.coordination_threshold = discovery_config.get('coordination_threshold', 0.85)
        self.min_network_size = discovery_config.get('min_network_size', 5)
        self.enable_real_time = discovery_config.get('enable_real_time', False)
        
        # Discovery state
        self.discovered_patterns = {}
        self.trend_history = []
        self.anomaly_alerts = []
        self.coordination_networks = []
        
        # Brazilian political context patterns
        self.political_indicators = {
            'conspiracy_patterns': [
                r'\b(deep state|estado profundo|globalismo|nova ordem)\b',
                r'\b(manipulação|controle|dominação) (mundial|global)\b',
                r'\b(elite|elites) (globalista|mundial|controladora)\b',
                r'\b(agenda|plano) (oculto|secreto|global)\b'
            ],
            'misinformation_markers': [
                r'\b(fake news|mídia mentirosa|imprensa golpista)\b',
                r'\b(censura|silenciamento|perseguição)\b',
                r'\b(verdade|realidade) (escondida|censurada)\b',
                r'\b(mídia|imprensa) (corrupta|vendida|comprada)\b'
            ],
            'institutional_attack': [
                r'\b(stf|supremo) (ilegítimo|corrupto|golpista)\b',
                r'\b(urna|eleição) (fraudada|roubada|manipulada)\b',
                r'\b(justiça|sistema) (corrupto|aparelhado|ilegítimo)\b',
                r'\b(congresso|senado) (vendido|corrupto|golpista)\b'
            ],
            'mobilization_calls': [
                r'\b(ocupar|tomar|invadir) (brasília|congresso|supremo)\b',
                r'\b(intervenção|militar|forças armadas)\b',
                r'\b(revolução|resistência|luta) (patriótica|conservadora)\b',
                r'\b(mobilização|manifestação|protesto) (nacional|geral)\b'
            ]
        }
        
        logger.info("ContentDiscoveryEngine initialized successfully")
    
    def discover_content_patterns(self) -> Dict[str, Any]:
        """
        Discover general content patterns in the discourse
        
        Returns:
            Comprehensive content pattern analysis
        """
        logger.info("Discovering general content patterns")
        
        if not self.search_engine or not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        try:
            # Get all texts from search index
            hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
            if not document_store:
                return {'error': 'Document store not available'}
            
            texts = []
            metadata = []
            for doc in document_store:
                texts.append(doc.get('body_cleaned', doc.get('body', '')))
                metadata.append(doc)
            
            if len(texts) < 10:
                return {'error': 'Insufficient data for pattern discovery'}
            
            # Analyze different types of patterns
            patterns_result = {
                'total_documents': len(texts),
                'analysis_timestamp': datetime.now().isoformat(),
                'patterns': {}
            }
            
            # 1. Frequent topics/themes
            topic_patterns = self._discover_topic_patterns(texts, metadata)
            patterns_result['patterns']['topics'] = topic_patterns
            
            # 2. Linguistic patterns
            linguistic_patterns = self._discover_linguistic_patterns(texts)
            patterns_result['patterns']['linguistic'] = linguistic_patterns
            
            # 3. Temporal patterns
            temporal_patterns = self._discover_temporal_patterns(texts, metadata)
            patterns_result['patterns']['temporal'] = temporal_patterns
            
            # 4. Channel/source patterns
            source_patterns = self._discover_source_patterns(metadata)
            patterns_result['patterns']['sources'] = source_patterns
            
            # 5. AI-enhanced pattern interpretation
            if self.api_available:
                ai_insights = self._get_ai_pattern_insights(patterns_result)
                patterns_result['ai_insights'] = ai_insights
            
            return patterns_result
            
        except Exception as e:
            logger.error(f"Error in content pattern discovery: {e}")
            return {
                'error': str(e),
                'fallback_analysis': self._basic_pattern_analysis(texts if 'texts' in locals() else [])
            }
    
    def _discover_topic_patterns(self, texts: List[str], metadata: List[Dict]) -> Dict[str, Any]:
        """Discover topic-level patterns"""
        # Simple keyword frequency analysis
        from collections import Counter
        import re
        
        # Extract keywords (simple approach)
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            words = [w for w in words if len(w) > 3]  # Filter short words
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_topics = word_freq.most_common(20)
        
        return {
            'top_keywords': top_topics,
            'total_unique_words': len(word_freq),
            'total_words': len(all_words)
        }
    
    def _discover_linguistic_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Discover linguistic patterns"""
        # Basic linguistic analysis
        patterns = {
            'avg_text_length': np.mean([len(text) for text in texts]),
            'short_texts': sum(1 for text in texts if len(text) < 50),
            'long_texts': sum(1 for text in texts if len(text) > 500),
            'question_count': sum(1 for text in texts if '?' in text),
            'exclamation_count': sum(1 for text in texts if '!' in text),
            'uppercase_ratio': np.mean([sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts])
        }
        
        return patterns
    
    def _discover_temporal_patterns(self, texts: List[str], metadata: List[Dict]) -> Dict[str, Any]:
        """Discover temporal patterns"""
        timestamps = []
        for meta in metadata:
            timestamp_str = meta.get('datetime') or meta.get('timestamp')
            if timestamp_str:
                try:
                    # Try multiple formats
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            ts = datetime.strptime(str(timestamp_str), fmt)
                            timestamps.append(ts)
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        if not timestamps:
            return {'error': 'No valid timestamps found'}
        
        # Basic temporal analysis
        timestamps.sort()
        time_span = (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0
        
        # Hour distribution
        hour_dist = Counter([ts.hour for ts in timestamps])
        
        return {
            'time_span_days': time_span,
            'total_timestamped': len(timestamps),
            'earliest': timestamps[0].isoformat() if timestamps else None,
            'latest': timestamps[-1].isoformat() if timestamps else None,
            'peak_hours': hour_dist.most_common(5)
        }
    
    def _discover_source_patterns(self, metadata: List[Dict]) -> Dict[str, Any]:
        """Discover source/channel patterns"""
        channels = []
        senders = []
        
        for meta in metadata:
            if 'channel' in meta and meta['channel']:
                channels.append(meta['channel'])
            if 'sender' in meta and meta['sender']:
                senders.append(meta['sender'])
        
        channel_dist = Counter(channels) if channels else Counter()
        sender_dist = Counter(senders) if senders else Counter()
        
        return {
            'unique_channels': len(channel_dist),
            'unique_senders': len(sender_dist),
            'top_channels': channel_dist.most_common(10),
            'top_senders': sender_dist.most_common(10),
            'total_messages': len(metadata)
        }
    
    def _get_ai_pattern_insights(self, patterns_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI insights on discovered patterns"""
        try:
            prompt = f"""
            Analyze these content patterns from political discourse data:
            
            Topics: {patterns_data['patterns'].get('topics', {})}
            Linguistic: {patterns_data['patterns'].get('linguistic', {})}
            Temporal: {patterns_data['patterns'].get('temporal', {})}
            Sources: {patterns_data['patterns'].get('sources', {})}
            
            Provide insights on:
            1. Main themes and narratives
            2. Communication patterns
            3. Potential coordinated behavior
            4. Notable anomalies or trends
            
            Keep response concise but insightful.
            """
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'ai_analysis': response.content[0].text,
                'model_used': self.model
            }
            
        except Exception as e:
            logger.error(f"Failed to get AI insights: {e}")
            return {'error': 'AI analysis unavailable'}
    
    def _basic_pattern_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Basic fallback pattern analysis"""
        if not texts:
            return {'error': 'No texts provided'}
        
        return {
            'total_texts': len(texts),
            'avg_length': np.mean([len(text) for text in texts]),
            'non_empty_texts': sum(1 for text in texts if text.strip()),
            'analysis_type': 'basic_fallback'
        }
    
    def discover_emerging_trends(
        self, 
        time_window_days: int = 7,
        min_growth_rate: float = 2.0
    ) -> Dict[str, Any]:
        """
        Discover emerging trends in political discourse
        
        Args:
            time_window_days: Time window for trend analysis
            min_growth_rate: Minimum growth rate to consider as trending
            
        Returns:
            Discovered trends with analysis
        """
        logger.info(f"Discovering emerging trends in {time_window_days}-day window")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        # Get all documents with timestamps
        documents = []
        hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
        
        if not document_store:
            logger.warning("Document store not available for emerging trends")
            return {'error': 'Document store not available'}
            
        for i, doc_metadata in enumerate(document_store):
            timestamp_str = doc_metadata.get('data') or doc_metadata.get('datetime') or doc_metadata.get('timestamp')
            if timestamp_str:
                try:
                    # Parse timestamp
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            timestamp = datetime.strptime(str(timestamp_str), fmt)
                            documents.append({
                                'index': i,
                                'timestamp': timestamp,
                                'text': doc_metadata.get('body_cleaned', doc_metadata.get('body', '')),
                                'metadata': doc_metadata
                            })
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        if len(documents) < self.min_trend_documents:
            return {'error': f'Insufficient documents with timestamps: {len(documents)}'}
        
        # Sort by timestamp
        documents.sort(key=lambda x: x['timestamp'])
        
        # Divide into time windows
        end_time = documents[-1]['timestamp']
        start_time = end_time - timedelta(days=time_window_days * 2)  # Look back 2 windows
        
        # Filter recent documents
        recent_docs = [doc for doc in documents if doc['timestamp'] >= start_time]
        
        if len(recent_docs) < self.min_trend_documents:
            return {'error': 'Insufficient recent documents'}
        
        # Analyze trends
        trends_discovered = self._analyze_temporal_trends(recent_docs, time_window_days, min_growth_rate)
        
        # Enhance with AI analysis
        enhanced_trends = self._enhance_trends_with_ai(trends_discovered)
        
        return {
            'time_window_days': time_window_days,
            'documents_analyzed': len(recent_docs),
            'trends_discovered': enhanced_trends,
            'analysis_timestamp': datetime.now().isoformat(),
            'methodology': 'temporal_clustering_with_ai_enhancement'
        }
    
    def detect_coordination_patterns(
        self, 
        similarity_threshold: float = None,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Detect coordinated messaging patterns
        
        Args:
            similarity_threshold: Threshold for message similarity
            time_window_minutes: Time window for coordination detection
            
        Returns:
            Detected coordination patterns
        """
        if similarity_threshold is None:
            similarity_threshold = self.coordination_threshold
        
        logger.info("Detecting coordination patterns in messaging")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        # Prepare documents with timestamps
        timed_documents = self._prepare_timed_documents()
        
        if len(timed_documents) < 10:
            return {'error': 'Insufficient timed documents for coordination analysis'}
        
        # Detect high-similarity message clusters
        similarity_clusters = self._find_similarity_clusters(timed_documents, similarity_threshold)
        
        # Analyze temporal patterns for coordination
        coordination_patterns = []
        
        for cluster in similarity_clusters:
            if len(cluster) < 3:  # Skip small clusters
                continue
            
            # Analyze temporal distribution
            timestamps = [doc['timestamp'] for doc in cluster]
            time_analysis = self._analyze_temporal_coordination(timestamps, time_window_minutes)
            
            if time_analysis['is_coordinated']:
                # Analyze channels involved
                channels = [doc['metadata'].get('channel', 'unknown') for doc in cluster]
                channel_analysis = self._analyze_channel_coordination(channels)
                
                # Generate AI insights about coordination
                coordination_insights = self._analyze_coordination_with_ai(cluster)
                
                coordination_patterns.append({
                    'pattern_id': len(coordination_patterns),
                    'message_count': len(cluster),
                    'unique_channels': len(set(channels)),
                    'time_span_minutes': time_analysis['time_span_minutes'],
                    'coordination_score': time_analysis['coordination_score'],
                    'channel_analysis': channel_analysis,
                    'sample_messages': [doc['text'][:200] for doc in cluster[:3]],
                    'timestamps': [doc['timestamp'].isoformat() for doc in cluster],
                    'ai_insights': coordination_insights
                })
        
        # Rank by coordination score
        coordination_patterns.sort(key=lambda x: x['coordination_score'], reverse=True)
        
        return {
            'total_patterns_detected': len(coordination_patterns),
            'coordination_threshold': similarity_threshold,
            'time_window_minutes': time_window_minutes,
            'documents_analyzed': len(timed_documents),
            'coordination_patterns': coordination_patterns,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def detect_misinformation_campaigns(self) -> Dict[str, Any]:
        """
        Detect potential misinformation campaigns and conspiracy theories
        
        Returns:
            Detected misinformation patterns and campaigns
        """
        logger.info("Detecting misinformation campaigns and conspiracy theories")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        misinformation_findings = {
            'conspiracy_content': [],
            'misinformation_markers': [],
            'institutional_attacks': [],
            'mobilization_calls': [],
            'coordinated_campaigns': []
        }
        
        # Analyze each pattern type
        for pattern_type, patterns in self.political_indicators.items():
            findings = self._detect_pattern_type(patterns, pattern_type)
            misinformation_findings[pattern_type] = findings
        
        # Cross-reference with coordination patterns
        coordination_data = self.detect_coordination_patterns()
        coordinated_misinfo = self._find_coordinated_misinformation(
            misinformation_findings, 
            coordination_data.get('coordination_patterns', [])
        )
        
        misinformation_findings['coordinated_campaigns'] = coordinated_misinfo
        
        # Generate comprehensive analysis
        campaign_analysis = self._analyze_misinformation_campaigns(misinformation_findings)
        
        return {
            'patterns_detected': misinformation_findings,
            'campaign_analysis': campaign_analysis,
            'risk_assessment': self._assess_misinformation_risk(misinformation_findings),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def discover_influence_networks(self, min_influence_score: float = 0.5) -> Dict[str, Any]:
        """
        Discover influence networks and information propagation patterns
        
        Args:
            min_influence_score: Minimum influence score for network inclusion
            
        Returns:
            Discovered influence networks and propagation patterns
        """
        logger.info("Discovering influence networks and propagation patterns")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        # Build influence graph
        influence_graph = self._build_influence_graph()
        
        # Detect communities
        communities = self._detect_influence_communities(influence_graph)
        
        # Analyze each community
        network_analysis = []
        
        for community_id, community in enumerate(communities):
            if len(community) < self.min_network_size:
                continue
            
            # Calculate community metrics
            community_metrics = self._calculate_community_metrics(community, influence_graph)
            
            if community_metrics['influence_score'] < min_influence_score:
                continue
            
            # Analyze content patterns in community
            content_analysis = self._analyze_community_content(community)
            
            # Generate AI insights
            ai_insights = self._analyze_network_with_ai(community, content_analysis)
            
            network_analysis.append({
                'network_id': community_id,
                'size': len(community),
                'channels': list(community),
                'metrics': community_metrics,
                'content_analysis': content_analysis,
                'ai_insights': ai_insights,
                'influence_rank': community_metrics['influence_score']
            })
        
        # Sort by influence score
        network_analysis.sort(key=lambda x: x['influence_rank'], reverse=True)
        
        return {
            'total_networks_discovered': len(network_analysis),
            'min_influence_threshold': min_influence_score,
            'network_analysis': network_analysis,
            'propagation_patterns': self._analyze_propagation_patterns(influence_graph),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_automated_insights(self, focus_period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive automated insights about the discourse
        
        Args:
            focus_period_days: Period to focus analysis on
            
        Returns:
            Comprehensive automated insights
        """
        logger.info(f"Generating automated insights for {focus_period_days}-day period")
        
        insights_report = {
            'analysis_period_days': focus_period_days,
            'generated_at': datetime.now().isoformat(),
            'insights': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Discover trends
        trends = self.discover_emerging_trends(time_window_days=focus_period_days)
        insights_report['insights']['emerging_trends'] = trends
        
        # Detect coordination
        coordination = self.detect_coordination_patterns()
        insights_report['insights']['coordination_patterns'] = coordination
        
        # Detect misinformation
        misinformation = self.detect_misinformation_campaigns()
        insights_report['insights']['misinformation_analysis'] = misinformation
        
        # Discover networks
        networks = self.discover_influence_networks()
        insights_report['insights']['influence_networks'] = networks
        
        # Generate alerts
        insights_report['alerts'] = self._generate_automated_alerts(insights_report['insights'])
        
        # Generate recommendations
        insights_report['recommendations'] = self._generate_research_recommendations(insights_report['insights'])
        
        # Generate executive summary
        insights_report['executive_summary'] = self._generate_executive_summary(insights_report)
        
        return insights_report
    
    def monitor_real_time_patterns(self, check_interval_minutes: int = 15) -> Dict[str, Any]:
        """
        Monitor real-time patterns and anomalies (simulated)
        
        Args:
            check_interval_minutes: Interval between checks
            
        Returns:
            Real-time monitoring results
        """
        if not self.enable_real_time:
            return {'error': 'Real-time monitoring not enabled'}
        
        logger.info(f"Starting real-time pattern monitoring (interval: {check_interval_minutes}m)")
        
        # This is a simplified simulation of real-time monitoring
        # In production, this would connect to live data streams
        
        monitoring_results = {
            'monitoring_start': datetime.now().isoformat(),
            'check_interval_minutes': check_interval_minutes,
            'alerts_generated': [],
            'patterns_detected': [],
            'anomalies_found': []
        }
        
        # Simulate recent activity analysis
        recent_activity = self._simulate_recent_activity_analysis()
        monitoring_results['recent_activity'] = recent_activity
        
        # Check for anomalies
        anomalies = self._detect_anomalies(recent_activity)
        monitoring_results['anomalies_found'] = anomalies
        
        return monitoring_results
    
    # Helper Methods
    
    def _get_hybrid_engine_data(self):
        """Get data from hybrid engine safely"""
        hybrid_engine = getattr(self.search_engine, 'hybrid_engine', None)
        if not hybrid_engine:
            return None, None, None
            
        document_store = getattr(hybrid_engine, 'document_store', None)
        embeddings = getattr(hybrid_engine, 'embeddings', None)
        
        return hybrid_engine, document_store, embeddings
    
    def _prepare_timed_documents(self) -> List[Dict]:
        """Prepare documents with timestamp information"""
        timed_docs = []
        
        # Access document store from hybrid engine
        hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
        
        if not document_store or embeddings is None:
            logger.warning("Document store or embeddings not available")
            return timed_docs
        
        for i, doc_metadata in enumerate(document_store):
            if i >= len(embeddings):
                break
                
            timestamp_str = doc_metadata.get('data') or doc_metadata.get('datetime') or doc_metadata.get('timestamp')
            if timestamp_str:
                try:
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            timestamp = datetime.strptime(str(timestamp_str), fmt)
                            timed_docs.append({
                                'index': i,
                                'timestamp': timestamp,
                                'text': doc_metadata.get('body_cleaned', doc_metadata.get('body', '')),
                                'metadata': doc_metadata,
                                'embedding': embeddings[i]
                            })
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        return timed_docs
    
    def _analyze_temporal_trends(
        self, 
        documents: List[Dict], 
        window_days: int, 
        min_growth: float
    ) -> List[Dict]:
        """Analyze temporal trends in document clusters"""
        
        # Create time windows
        end_time = documents[-1]['timestamp']
        window1_start = end_time - timedelta(days=window_days * 2)
        window2_start = end_time - timedelta(days=window_days)
        
        # Split documents into windows
        window1_docs = [d for d in documents if window1_start <= d['timestamp'] < window2_start]
        window2_docs = [d for d in documents if d['timestamp'] >= window2_start]
        
        if len(window1_docs) < 5 or len(window2_docs) < 5:
            return []
        
        # Extract topics from each window using simple keyword analysis
        window1_topics = self._extract_window_topics(window1_docs)
        window2_topics = self._extract_window_topics(window2_docs)
        
        # Find trending topics
        trends = []
        for topic, window2_count in window2_topics.items():
            window1_count = window1_topics.get(topic, 0)
            if window1_count > 0:
                growth_rate = window2_count / window1_count
                if growth_rate >= min_growth and window2_count >= self.min_trend_documents:
                    trends.append({
                        'topic': topic,
                        'window1_count': window1_count,
                        'window2_count': window2_count,
                        'growth_rate': growth_rate,
                        'trend_strength': min(growth_rate / min_growth, 5.0)
                    })
            elif window2_count >= self.min_trend_documents:
                # New topic
                trends.append({
                    'topic': topic,
                    'window1_count': 0,
                    'window2_count': window2_count,
                    'growth_rate': float('inf'),
                    'trend_strength': 5.0,
                    'new_topic': True
                })
        
        # Sort by trend strength
        trends.sort(key=lambda x: x['trend_strength'], reverse=True)
        
        return trends
    
    def _extract_window_topics(self, documents: List[Dict]) -> Counter:
        """Extract topics from a time window using keyword frequency"""
        
        # Simple keyword extraction
        keywords = []
        for doc in documents:
            text = doc['text'].lower()
            # Extract political keywords
            for category, terms in self.search_engine.political_keywords.items():
                for term in terms:
                    if term in text:
                        keywords.append(term)
        
        return Counter(keywords)
    
    def _enhance_trends_with_ai(self, trends: List[Dict]) -> List[Dict]:
        """Enhance trend analysis with AI insights"""
        
        if not self.api_available or not trends:
            return trends
        
        enhanced_trends = []
        
        for trend in trends:
            try:
                # Get sample texts for this trend
                trend_topic = trend['topic']
                search_result = self.search_engine.semantic_search(
                    query=trend_topic,
                    top_k=5,
                    include_metadata=True
                )
                
                if search_result.get('results'):
                    sample_texts = [r['text'][:200] for r in search_result['results'][:3]]
                    
                    # Analyze with AI
                    ai_analysis = self._analyze_trend_with_ai(trend_topic, sample_texts, trend)
                    
                    enhanced_trend = trend.copy()
                    enhanced_trend['ai_analysis'] = ai_analysis
                    enhanced_trends.append(enhanced_trend)
                else:
                    enhanced_trends.append(trend)
                    
            except Exception as e:
                logger.warning(f"AI trend enhancement failed for {trend['topic']}: {e}")
                enhanced_trends.append(trend)
        
        return enhanced_trends
    
    def _analyze_trend_with_ai(self, topic: str, sample_texts: List[str], trend_data: Dict) -> Dict[str, Any]:
        """Analyze a trend using AI"""
        
        texts_sample = '\n'.join([f"- {text}..." for text in sample_texts])
        
        prompt = f"""
Analise esta tendência emergente no discurso político brasileiro:

TÓPICO: {topic}
CRESCIMENTO: {trend_data.get('growth_rate', 'N/A')}x
DOCUMENTOS RECENTES: {trend_data.get('window2_count', 0)}

TEXTOS REPRESENTATIVOS:
{texts_sample}

Responda em JSON:
{{
  "trend_significance": "alta|media|baixa",
  "political_context": "contexto político desta tendência",
  "potential_impact": "impacto potencial no discurso",
  "related_events": ["evento1", "evento2"],
  "discourse_type": "informativo|mobilização|crítica|conspiratório",
  "monitoring_priority": "alta|media|baixa",
  "key_actors": ["ator1", "ator2"],
  "narrative_direction": "direção da narrativa"
}}
"""
        
        try:
            response = self.create_message(
                prompt,
                stage="content_discovery",
                operation="trend_analysis",
                temperature=0.3
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"AI trend analysis failed: {e}")
            return {'analysis_failed': True, 'error': str(e)}
    
    def _find_similarity_clusters(self, documents: List[Dict], threshold: float) -> List[List[Dict]]:
        """Find clusters of highly similar documents"""
        
        if len(documents) < 2:
            return []
        
        # Extract embeddings
        embeddings = np.array([doc['embedding'] for doc in documents])
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find connected components of similar documents
        clusters = []
        processed = set()
        
        for i in range(len(documents)):
            if i in processed:
                continue
            
            # Start new cluster
            cluster = [documents[i]]
            queue = [i]
            cluster_processed = {i}
            
            while queue:
                current_idx = queue.pop(0)
                
                # Find similar documents
                for j in range(len(documents)):
                    if j not in cluster_processed and similarities[current_idx][j] >= threshold:
                        cluster.append(documents[j])
                        queue.append(j)
                        cluster_processed.add(j)
                
            if len(cluster) >= 2:  # Only keep clusters with multiple documents
                clusters.append(cluster)
            
            processed.update(cluster_processed)
        
        return clusters
    
    def _analyze_temporal_coordination(self, timestamps: List[datetime], window_minutes: int) -> Dict[str, Any]:
        """Analyze if timestamps show coordination patterns"""
        
        if len(timestamps) < 2:
            return {'is_coordinated': False}
        
        timestamps_sorted = sorted(timestamps)
        time_diffs = []
        
        for i in range(1, len(timestamps_sorted)):
            diff_minutes = (timestamps_sorted[i] - timestamps_sorted[i-1]).total_seconds() / 60
            time_diffs.append(diff_minutes)
        
        # Calculate coordination indicators
        max_diff = max(time_diffs) if time_diffs else 0
        avg_diff = np.mean(time_diffs) if time_diffs else 0
        
        # Messages within time window are considered coordinated
        is_coordinated = max_diff <= window_minutes and len(timestamps) >= 3
        
        # Calculate coordination score
        if is_coordinated:
            # Higher score for tighter timing
            coordination_score = max(0, 1 - (max_diff / window_minutes))
        else:
            coordination_score = 0
        
        total_span = (timestamps_sorted[-1] - timestamps_sorted[0]).total_seconds() / 60
        
        return {
            'is_coordinated': is_coordinated,
            'coordination_score': coordination_score,
            'time_span_minutes': total_span,
            'max_gap_minutes': max_diff,
            'avg_gap_minutes': avg_diff,
            'message_count': len(timestamps)
        }
    
    def _analyze_channel_coordination(self, channels: List[str]) -> Dict[str, Any]:
        """Analyze coordination patterns across channels"""
        
        channel_counts = Counter(channels)
        unique_channels = len(channel_counts)
        total_messages = len(channels)
        
        # Calculate distribution entropy (lower = more concentrated)
        if unique_channels > 1:
            probs = [count/total_messages for count in channel_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(unique_channels)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0
        
        return {
            'unique_channels': unique_channels,
            'total_messages': total_messages,
            'channel_distribution': dict(channel_counts),
            'distribution_entropy': normalized_entropy,
            'coordination_type': 'single_channel' if unique_channels == 1 else 'cross_channel'
        }
    
    def _analyze_coordination_with_ai(self, cluster: List[Dict]) -> Dict[str, Any]:
        """Analyze coordination cluster with AI"""
        
        if not self.api_available or len(cluster) < 2:
            return {}
        
        try:
            # Prepare sample data
            sample_messages = [doc['text'][:150] for doc in cluster[:5]]
            channels = [doc['metadata'].get('channel', 'unknown') for doc in cluster]
            timestamps = [doc['timestamp'].strftime('%Y-%m-%d %H:%M') for doc in cluster]
            
            messages_text = '\n'.join([f"{i+1}. [{timestamps[i]}] {msg}..." for i, msg in enumerate(sample_messages)])
            
            prompt = f"""
Analise este padrão de coordenação em mensagens do Telegram brasileiro:

MENSAGENS SIMILARES:
{messages_text}

CANAIS ENVOLVIDOS: {set(channels)}
TOTAL DE MENSAGENS: {len(cluster)}

Avalie se há evidências de coordenação:

Responda em JSON:
{{
  "coordination_likelihood": "alta|media|baixa",
  "coordination_type": "orgânica|suspeita|artificial",
  "evidence_indicators": ["indicador1", "indicador2"],
  "content_analysis": "análise do conteúdo",
  "risk_level": "alto|medio|baixo",
  "recommended_action": "ação recomendada"
}}
"""
            
            response = self.create_message(
                prompt,
                stage="content_discovery",
                operation="coordination_analysis",
                temperature=0.3
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"AI coordination analysis failed: {e}")
            return {}
    
    def _detect_pattern_type(self, patterns: List[str], pattern_type: str) -> List[Dict]:
        """Detect specific patterns in content"""
        
        findings = []
        
        for pattern in patterns:
            # Search for pattern in texts
            matching_docs = []
            
            hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
            if document_store:
                for i, doc_metadata in enumerate(document_store):
                    text = doc_metadata.get('body_cleaned', doc_metadata.get('body', ''))
                    if re.search(pattern, text, re.IGNORECASE):
                        matching_docs.append({
                            'index': i,
                            'text': text[:200] + "...",
                            'metadata': doc_metadata,
                            'pattern_matched': pattern
                        })
            
            if matching_docs:
                findings.append({
                    'pattern': pattern,
                    'pattern_type': pattern_type,
                    'matches_found': len(matching_docs),
                    'sample_matches': matching_docs[:5],
                    'channels_affected': len(set(doc['metadata'].get('channel', 'unknown') for doc in matching_docs))
                })
        
        return findings
    
    def _find_coordinated_misinformation(
        self, 
        misinformation_findings: Dict, 
        coordination_patterns: List[Dict]
    ) -> List[Dict]:
        """Find overlap between misinformation and coordination patterns"""
        
        coordinated_campaigns = []
        
        for coord_pattern in coordination_patterns:
            if coord_pattern['coordination_score'] < 0.7:  # Focus on high coordination
                continue
            
            # Check if coordinated messages contain misinformation patterns
            coord_messages = coord_pattern.get('sample_messages', [])
            
            misinfo_matches = []
            for pattern_type, findings in misinformation_findings.items():
                if pattern_type == 'coordinated_campaigns':
                    continue
                
                for finding in findings:
                    pattern = finding['pattern']
                    for message in coord_messages:
                        if re.search(pattern, message, re.IGNORECASE):
                            misinfo_matches.append({
                                'pattern_type': pattern_type,
                                'pattern': pattern,
                                'message': message[:100] + "..."
                            })
            
            if misinfo_matches:
                coordinated_campaigns.append({
                    'coordination_pattern_id': coord_pattern.get('pattern_id'),
                    'coordination_score': coord_pattern['coordination_score'],
                    'message_count': coord_pattern['message_count'],
                    'channels_involved': coord_pattern['unique_channels'],
                    'misinformation_patterns': misinfo_matches,
                    'risk_level': 'high' if len(misinfo_matches) > 3 else 'medium'
                })
        
        return coordinated_campaigns
    
    def _analyze_misinformation_campaigns(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze misinformation campaign characteristics"""
        
        total_patterns = sum(len(pattern_findings) for pattern_type, pattern_findings in findings.items() if pattern_type != 'coordinated_campaigns')
        total_coordinated = len(findings.get('coordinated_campaigns', []))
        
        # Calculate risk metrics
        risk_factors = []
        if total_coordinated > 0:
            risk_factors.append(f"Coordinated misinformation campaigns detected: {total_coordinated}")
        
        pattern_distribution = {}
        for pattern_type, pattern_findings in findings.items():
            if pattern_type != 'coordinated_campaigns' and pattern_findings:
                pattern_distribution[pattern_type] = len(pattern_findings)
                if len(pattern_findings) > 5:
                    risk_factors.append(f"High {pattern_type} activity: {len(pattern_findings)} patterns")
        
        return {
            'total_patterns_detected': total_patterns,
            'coordinated_campaigns': total_coordinated,
            'pattern_distribution': pattern_distribution,
            'risk_factors': risk_factors,
            'threat_level': 'high' if total_coordinated > 2 else 'medium' if total_patterns > 10 else 'low'
        }
    
    def _assess_misinformation_risk(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall misinformation risk level"""
        
        risk_score = 0
        risk_indicators = []
        
        # Count different types of patterns
        for pattern_type, pattern_findings in findings.items():
            if pattern_type == 'coordinated_campaigns':
                # Coordinated campaigns are high risk
                risk_score += len(pattern_findings) * 3
                if pattern_findings:
                    risk_indicators.append(f"Coordinated misinformation campaigns: {len(pattern_findings)}")
            else:
                risk_score += len(pattern_findings)
                if len(pattern_findings) > 3:
                    risk_indicators.append(f"High {pattern_type} activity")
        
        # Determine risk level
        if risk_score >= 15:
            risk_level = 'critical'
        elif risk_score >= 10:
            risk_level = 'high'
        elif risk_score >= 5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_indicators': risk_indicators,
            'monitoring_recommendation': 'immediate' if risk_level == 'critical' else 'regular'
        }
    
    def _build_influence_graph(self) -> nx.Graph:
        """Build influence graph from channel interactions"""
        
        graph = nx.Graph()
        
        # Add nodes (channels)
        channels = set()
        hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
        if not document_store:
            return graph
        for metadata in document_store:
            channel = metadata.get('channel')
            if channel:
                channels.add(channel)
        
        graph.add_nodes_from(channels)
        
        # Add edges based on content similarity (simplified)
        channel_docs = defaultdict(list)
        for i, metadata in enumerate(document_store):
            channel = metadata.get('channel')
            if channel:
                channel_docs[channel].append(i)
        
        # Calculate channel-to-channel similarities
        for ch1 in channels:
            for ch2 in channels:
                if ch1 != ch2:
                    # Calculate similarity between channels
                    similarity = self._calculate_channel_similarity(
                        channel_docs[ch1], 
                        channel_docs[ch2]
                    )
                    if similarity > 0.3:  # Threshold for connection
                        graph.add_edge(ch1, ch2, weight=similarity)
        
        return graph
    
    def _calculate_channel_similarity(self, docs1: List[int], docs2: List[int]) -> float:
        """Calculate similarity between two channels based on their content"""
        
        if not docs1 or not docs2:
            return 0.0
        
        # Get embeddings from hybrid engine
        hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
        if embeddings is None:
            return 0.0
            
        # Sample documents from each channel
        sample1 = docs1[:min(10, len(docs1))]
        sample2 = docs2[:min(10, len(docs2))]
        
        # Get embeddings
        try:
            embeddings1 = embeddings[sample1]
            embeddings2 = embeddings[sample2]
            
            # Calculate average similarity
            similarities = cosine_similarity(embeddings1, embeddings2)
            return float(np.mean(similarities))
        except Exception as e:
            logger.warning(f"Error calculating channel similarity: {e}")
            return 0.0
    
    def _detect_influence_communities(self, graph: nx.Graph) -> List[List[str]]:
        """Detect communities in the influence graph"""
        
        if len(graph.nodes) < 3:
            return []
        
        try:
            # Use simple connected components as communities
            communities = list(nx.connected_components(graph))
            return [list(community) for community in communities if len(community) >= self.min_network_size]
        except:
            return []
    
    def _calculate_community_metrics(self, community: List[str], graph: nx.Graph) -> Dict[str, Any]:
        """Calculate metrics for an influence community"""
        
        subgraph = graph.subgraph(community)
        
        # Basic network metrics
        density = nx.density(subgraph) if len(community) > 1 else 0
        
        # Calculate influence score based on network structure
        influence_score = density * len(community) / 10  # Normalized
        
        # Centrality measures
        centrality = {}
        if len(community) > 2:
            try:
                betweenness = nx.betweenness_centrality(subgraph)
                centrality['most_central'] = max(betweenness, key=betweenness.get) if betweenness else None
                centrality['centrality_scores'] = dict(betweenness)
            except:
                centrality['most_central'] = community[0]
        
        return {
            'size': len(community),
            'density': density,
            'influence_score': influence_score,
            'centrality_analysis': centrality
        }
    
    def _analyze_community_content(self, community: List[str]) -> Dict[str, Any]:
        """Analyze content patterns within a community"""
        
        # Collect all documents from community channels
        community_docs = []
        hybrid_engine, document_store, embeddings = self._get_hybrid_engine_data()
        if not document_store:
            return {}
        for metadata in document_store:
            if metadata.get('channel') in community:
                community_docs.append(metadata)
        
        if not community_docs:
            return {}
        
        # Analyze temporal patterns
        timestamps = []
        for doc in community_docs:
            timestamp_str = doc.get('datetime') or doc.get('timestamp')
            if timestamp_str:
                try:
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                        try:
                            timestamp = datetime.strptime(str(timestamp_str), fmt)
                            timestamps.append(timestamp)
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        temporal_analysis = {}
        if timestamps:
            timestamps.sort()
            temporal_analysis = {
                'earliest': timestamps[0].isoformat(),
                'latest': timestamps[-1].isoformat(),
                'span_days': (timestamps[-1] - timestamps[0]).days,
                'message_count': len(timestamps)
            }
        
        return {
            'total_documents': len(community_docs),
            'temporal_analysis': temporal_analysis,
            'channels_in_community': community
        }
    
    def _analyze_network_with_ai(self, community: List[str], content_analysis: Dict) -> Dict[str, Any]:
        """Analyze influence network using AI"""
        
        if not self.api_available:
            return {}
        
        try:
            community_info = f"Canais: {', '.join(community[:5])}"
            if len(community) > 5:
                community_info += f" (e mais {len(community)-5})"
            
            doc_count = content_analysis.get('total_documents', 0)
            temporal = content_analysis.get('temporal_analysis', {})
            
            prompt = f"""
Analise esta rede de influência no Telegram brasileiro:

{community_info}
Total de documentos: {doc_count}
Período de atividade: {temporal.get('span_days', 'N/A')} dias

Avalie as características desta rede:

Responda em JSON:
{{
  "network_type": "orgânica|artificial|mista",
  "influence_pattern": "centralizada|distribuída|hierárquica",
  "content_focus": "foco principal do conteúdo",
  "coordination_level": "alta|media|baixa",
  "political_alignment": "alinhamento político detectado",
  "risk_assessment": "alto|medio|baixo",
  "monitoring_priority": "alta|media|baixa"
}}
"""
            
            response = self.create_message(
                prompt,
                stage="content_discovery",
                operation="network_analysis",
                temperature=0.3
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"AI network analysis failed: {e}")
            return {}
    
    def _analyze_propagation_patterns(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze information propagation patterns"""
        
        if len(graph.nodes) < 2:
            return {}
        
        # Find key propagation metrics
        propagation_metrics = {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'network_density': nx.density(graph),
            'connected_components': nx.number_connected_components(graph)
        }
        
        # Find most connected nodes (potential hubs)
        if len(graph.nodes) > 0:
            degree_centrality = nx.degree_centrality(graph)
            top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            propagation_metrics['top_influence_hubs'] = top_hubs
        
        return propagation_metrics
    
    def _generate_automated_alerts(self, insights: Dict[str, Any]) -> List[Dict]:
        """Generate automated alerts based on insights"""
        
        alerts = []
        
        # Check for coordination alerts
        coordination_data = insights.get('coordination_patterns', {})
        if coordination_data.get('total_patterns_detected', 0) > 5:
            alerts.append({
                'type': 'coordination_alert',
                'severity': 'high',
                'message': f"High coordination activity detected: {coordination_data['total_patterns_detected']} patterns",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for misinformation alerts
        misinfo_data = insights.get('misinformation_analysis', {})
        risk_assessment = misinfo_data.get('risk_assessment', {})
        if risk_assessment.get('risk_level') in ['high', 'critical']:
            alerts.append({
                'type': 'misinformation_alert',
                'severity': risk_assessment['risk_level'],
                'message': f"Misinformation risk level: {risk_assessment['risk_level']}",
                'indicators': risk_assessment.get('risk_indicators', []),
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for trending alerts
        trends_data = insights.get('emerging_trends', {})
        if trends_data.get('trends_discovered'):
            high_growth_trends = [
                t for t in trends_data['trends_discovered'] 
                if t.get('growth_rate', 0) > 5.0
            ]
            if high_growth_trends:
                alerts.append({
                    'type': 'trending_alert',
                    'severity': 'medium',
                    'message': f"Rapid trend growth detected: {len(high_growth_trends)} topics",
                    'trending_topics': [t['topic'] for t in high_growth_trends[:3]],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _generate_research_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate research recommendations based on insights"""
        
        recommendations = []
        
        # Based on trending topics
        trends_data = insights.get('emerging_trends', {})
        if trends_data.get('trends_discovered'):
            top_trends = trends_data['trends_discovered'][:3]
            for trend in top_trends:
                recommendations.append(
                    f"Investigar evolução do tópico '{trend['topic']}' (crescimento {trend.get('growth_rate', 'N/A')}x)"
                )
        
        # Based on coordination patterns
        coordination_data = insights.get('coordination_patterns', {})
        if coordination_data.get('coordination_patterns'):
            high_coord = [p for p in coordination_data['coordination_patterns'] if p['coordination_score'] > 0.8]
            if high_coord:
                recommendations.append(
                    f"Analisar {len(high_coord)} padrões de coordenação com alta suspeição"
                )
        
        # Based on influence networks
        networks_data = insights.get('influence_networks', {})
        if networks_data.get('network_analysis'):
            top_networks = networks_data['network_analysis'][:2]
            for network in top_networks:
                recommendations.append(
                    f"Monitorar rede de influência #{network['network_id']} com {network['size']} canais"
                )
        
        return recommendations
    
    def _generate_executive_summary(self, insights_report: Dict[str, Any]) -> str:
        """Generate executive summary of all insights"""
        
        insights = insights_report.get('insights', {})
        alerts = insights_report.get('alerts', [])
        
        summary_parts = []
        
        # Trends summary
        trends = insights.get('emerging_trends', {})
        if trends.get('trends_discovered'):
            trend_count = len(trends['trends_discovered'])
            summary_parts.append(f"Identificadas {trend_count} tendências emergentes no discurso político")
        
        # Coordination summary
        coordination = insights.get('coordination_patterns', {})
        coord_count = coordination.get('total_patterns_detected', 0)
        if coord_count > 0:
            summary_parts.append(f"Detectados {coord_count} padrões de coordenação suspeita")
        
        # Misinformation summary
        misinfo = insights.get('misinformation_analysis', {})
        risk_level = misinfo.get('risk_assessment', {}).get('risk_level', 'baixo')
        summary_parts.append(f"Nível de risco de desinformação: {risk_level}")
        
        # Alerts summary
        if alerts:
            high_severity_alerts = [a for a in alerts if a.get('severity') in ['high', 'critical']]
            if high_severity_alerts:
                summary_parts.append(f"Gerados {len(high_severity_alerts)} alertas de alta prioridade")
        
        # Combine into summary
        if summary_parts:
            summary = "Análise automatizada do período: " + ". ".join(summary_parts) + "."
        else:
            summary = "Análise automatizada concluída sem identificação de padrões significativos."
        
        return summary
    
    def _simulate_recent_activity_analysis(self) -> Dict[str, Any]:
        """Simulate recent activity analysis for real-time monitoring"""
        
        # This is a simulation - in production, this would analyze actual recent data
        return {
            'simulated': True,
            'recent_message_volume': np.random.randint(100, 1000),
            'channel_activity_change': np.random.uniform(-0.3, 0.5),
            'new_patterns_detected': np.random.randint(0, 3),
            'anomaly_score': np.random.uniform(0, 1)
        }
    
    def _detect_anomalies(self, recent_activity: Dict[str, Any]) -> List[Dict]:
        """Detect anomalies in recent activity"""
        
        anomalies = []
        
        # Check for volume anomalies
        volume = recent_activity.get('recent_message_volume', 0)
        if volume > 800:  # Arbitrary threshold
            anomalies.append({
                'type': 'volume_spike',
                'severity': 'medium',
                'description': f"Unusual message volume: {volume}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for pattern anomalies
        anomaly_score = recent_activity.get('anomaly_score', 0)
        if anomaly_score > self.anomaly_threshold / 3:  # Scaled threshold
            anomalies.append({
                'type': 'pattern_anomaly',
                'severity': 'high' if anomaly_score > 0.8 else 'medium',
                'description': f"Anomalous pattern detected (score: {anomaly_score:.2f})",
                'timestamp': datetime.now().isoformat()
            })
        
        return anomalies


def create_content_discovery_engine(
    config: Dict[str, Any], 
    search_engine: SemanticSearchEngine = None
) -> ContentDiscoveryEngine:
    """
    Factory function to create ContentDiscoveryEngine instance
    
    Args:
        config: Configuration dictionary
        search_engine: Optional pre-initialized search engine
        
    Returns:
        ContentDiscoveryEngine instance
    """
    return ContentDiscoveryEngine(config, search_engine)