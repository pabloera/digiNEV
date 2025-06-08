"""
Temporal Semantic Evolution Tracker
Tracks how semantic concepts and political discourse evolve over time in Brazilian Telegram data
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from .base import AnthropicBase
from .semantic_search_engine import SemanticSearchEngine
from .voyage_embeddings import VoyageEmbeddingAnalyzer

logger = logging.getLogger(__name__)


class TemporalEvolutionTracker(AnthropicBase):
    """
    Temporal Semantic Evolution Tracker for Political Discourse Analysis
    
    Capabilities:
    - Track semantic evolution of political concepts over time
    - Detect narrative shifts and discourse changes
    - Analyze concept emergence and decay patterns
    - Monitor political polarization evolution
    - Track misinformation campaign development
    - Detect coordinated narrative changes
    - Analyze event-driven discourse shifts
    - Generate evolution predictions and forecasts
    """
    
    def __init__(self, config: Dict[str, Any], search_engine: SemanticSearchEngine = None):
        super().__init__(config)
        
        # Initialize search engine
        if search_engine:
            self.search_engine = search_engine
        else:
            embedding_analyzer = VoyageEmbeddingAnalyzer(config)
            self.search_engine = SemanticSearchEngine(config, embedding_analyzer)
        
        # Evolution tracking configuration
        evolution_config = config.get('temporal_evolution', {})
        self.default_time_window_days = evolution_config.get('time_window_days', 30)
        self.min_documents_per_window = evolution_config.get('min_documents_per_window', 10)
        self.evolution_threshold = evolution_config.get('evolution_threshold', 0.3)
        self.max_concepts_tracked = evolution_config.get('max_concepts_tracked', 50)
        
        # Brazilian political timeline and events
        self.political_timeline = {
            '2019': {
                'major_events': ['Posse Bolsonaro', 'Início governo', 'Sergio Moro ministro'],
                'key_themes': ['novo governo', 'mudanças', 'expectativas']
            },
            '2020': {
                'major_events': ['Início pandemia', 'Saída Sergio Moro', 'Auxílio emergencial'],
                'key_themes': ['covid', 'crise sanitária', 'lockdown', 'economia']
            },
            '2021': {
                'major_events': ['Vacinação', 'CPI da Covid', 'Crise política'],
                'key_themes': ['vacinas', 'cpi', 'investigação', 'polarização']
            },
            '2022': {
                'major_events': ['Eleições', 'Debates', 'Segundo turno'],
                'key_themes': ['eleição', 'urnas', 'democracia', 'resultado']
            },
            '2023': {
                'major_events': ['Posse Lula', 'Mudança governo', '8 de janeiro'],
                'key_themes': ['transição', 'golpe', 'democracia', 'polarização']
            }
        }
        
        # Concept categories for tracking
        self.concept_categories = {
            'political_actors': ['bolsonaro', 'lula', 'moro', 'doria', 'ciro'],
            'institutions': ['stf', 'congresso', 'tse', 'pf', 'exercito'],
            'democratic_concepts': ['democracia', 'liberdade', 'constituição', 'direitos'],
            'pandemic_terms': ['covid', 'vacina', 'lockdown', 'isolamento', 'hidroxicloroquina'],
            'electoral_terms': ['eleição', 'urna', 'voto', 'fraude', 'auditoria'],
            'conspiracy_terms': ['deep state', 'globalismo', 'comunismo', 'manipulação'],
            'media_terms': ['mídia', 'fake news', 'imprensa', 'censura']
        }
        
        # Evolution tracking state
        self.tracked_concepts = {}
        self.evolution_history = []
        self.narrative_shifts = []
        
        logger.info("TemporalEvolutionTracker initialized successfully")
    
    def track_concept_evolution(
        self, 
        concept: str, 
        start_date: str = None, 
        end_date: str = None,
        time_window_days: int = None
    ) -> Dict[str, Any]:
        """
        Track how a specific concept evolves semantically over time
        
        Args:
            concept: Political concept to track (e.g., "democracia", "eleições")
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            time_window_days: Size of time windows for analysis
            
        Returns:
            Comprehensive concept evolution analysis
        """
        logger.info(f"Tracking semantic evolution of concept: '{concept}'")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        if time_window_days is None:
            time_window_days = self.default_time_window_days
        
        # Find all documents related to the concept
        concept_search = self.search_engine.semantic_search(
            query=concept,
            top_k=2000,  # Get many results for temporal analysis
            include_metadata=True
        )
        
        if not concept_search.get('results'):
            return {'error': f'No documents found for concept: {concept}'}
        
        # Extract and validate timestamps
        timed_documents = self._extract_timed_documents(concept_search['results'])
        
        if len(timed_documents) < self.min_documents_per_window:
            return {'error': f'Insufficient timed documents: {len(timed_documents)}'}
        
        # Filter by date range if specified
        if start_date or end_date:
            timed_documents = self._filter_by_date_range(timed_documents, start_date, end_date)
        
        # Create temporal windows
        temporal_windows = self._create_temporal_windows(timed_documents, time_window_days)
        
        # Analyze evolution across windows
        evolution_analysis = self._analyze_concept_evolution_across_windows(
            concept, 
            temporal_windows
        )
        
        # Detect major shifts and events
        narrative_shifts = self._detect_narrative_shifts(evolution_analysis)
        
        # Generate evolution insights
        evolution_insights = self._generate_evolution_insights(
            concept, 
            evolution_analysis, 
            narrative_shifts
        )
        
        # Create evolution visualization data
        visualization_data = self._create_evolution_visualization_data(evolution_analysis)
        
        return {
            'concept': concept,
            'analysis_period': {
                'start': timed_documents[0]['timestamp'].isoformat() if timed_documents else None,
                'end': timed_documents[-1]['timestamp'].isoformat() if timed_documents else None,
                'total_documents': len(timed_documents),
                'time_windows': len(temporal_windows)
            },
            'evolution_analysis': evolution_analysis,
            'narrative_shifts': narrative_shifts,
            'insights': evolution_insights,
            'visualization_data': visualization_data,
            'methodology': {
                'time_window_days': time_window_days,
                'min_documents_per_window': self.min_documents_per_window,
                'semantic_model': self.search_engine.search_index.get('embedding_model', 'unknown')
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def track_multiple_concepts(
        self, 
        concepts: List[str],
        comparative_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Track evolution of multiple concepts for comparative analysis
        
        Args:
            concepts: List of concepts to track
            comparative_analysis: Whether to perform comparative analysis
            
        Returns:
            Multi-concept evolution analysis
        """
        logger.info(f"Tracking evolution of {len(concepts)} concepts: {concepts}")
        
        multi_concept_results = {
            'concepts_analyzed': concepts,
            'individual_analyses': {},
            'comparative_analysis': {},
            'cross_concept_insights': {}
        }
        
        # Track each concept individually
        for concept in concepts:
            try:
                concept_analysis = self.track_concept_evolution(concept)
                multi_concept_results['individual_analyses'][concept] = concept_analysis
            except Exception as e:
                logger.error(f"Failed to analyze concept '{concept}': {e}")
                multi_concept_results['individual_analyses'][concept] = {'error': str(e)}
        
        # Perform comparative analysis if requested
        if comparative_analysis:
            multi_concept_results['comparative_analysis'] = self._perform_comparative_analysis(
                multi_concept_results['individual_analyses']
            )
            
            multi_concept_results['cross_concept_insights'] = self._generate_cross_concept_insights(
                multi_concept_results['individual_analyses'],
                multi_concept_results['comparative_analysis']
            )
        
        return multi_concept_results
    
    def detect_discourse_shifts(
        self, 
        time_period_months: int = 12,
        shift_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Detect major discourse shifts across the entire dataset
        
        Args:
            time_period_months: Period to analyze for shifts
            shift_threshold: Threshold for detecting significant shifts
            
        Returns:
            Detected discourse shifts and analysis
        """
        logger.info(f"Detecting discourse shifts over {time_period_months} months")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        if shift_threshold is None:
            shift_threshold = self.evolution_threshold
        
        # Get all documents with timestamps
        all_documents = self._get_all_timed_documents()
        
        if len(all_documents) < 100:
            return {'error': 'Insufficient documents for discourse shift analysis'}
        
        # Divide into monthly windows
        monthly_windows = self._create_monthly_windows(all_documents, time_period_months)
        
        # Analyze discourse characteristics in each window
        discourse_evolution = self._analyze_discourse_evolution(monthly_windows)
        
        # Detect significant shifts
        detected_shifts = self._detect_significant_shifts(discourse_evolution, shift_threshold)
        
        # Correlate with political events
        event_correlations = self._correlate_with_political_events(detected_shifts)
        
        # Generate shift analysis
        shift_analysis = self._generate_shift_analysis(
            detected_shifts, 
            event_correlations,
            discourse_evolution
        )
        
        return {
            'analysis_period_months': time_period_months,
            'total_documents_analyzed': len(all_documents),
            'monthly_windows_created': len(monthly_windows),
            'discourse_evolution': discourse_evolution,
            'detected_shifts': detected_shifts,
            'event_correlations': event_correlations,
            'shift_analysis': shift_analysis,
            'methodology': {
                'shift_threshold': shift_threshold,
                'window_type': 'monthly',
                'analysis_method': 'semantic_similarity_tracking'
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def analyze_polarization_evolution(self) -> Dict[str, Any]:
        """
        Analyze how political polarization evolves over time
        
        Returns:
            Polarization evolution analysis
        """
        logger.info("Analyzing polarization evolution over time")
        
        if not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        # Define polarization indicators
        polarization_terms = {
            'left_terms': ['lula', 'pt', 'esquerda', 'socialismo', 'direitos'],
            'right_terms': ['bolsonaro', 'direita', 'conservador', 'tradição', 'família'],
            'institutional_terms': ['stf', 'democracia', 'constituição', 'justiça'],
            'anti_institutional': ['golpe', 'ditadura', 'censura', 'perseguição']
        }
        
        # Track each category over time
        polarization_tracking = {}
        
        for category, terms in polarization_terms.items():
            category_evolution = []
            
            for term in terms:
                term_evolution = self.track_concept_evolution(term)
                if not term_evolution.get('error'):
                    category_evolution.append(term_evolution)
            
            polarization_tracking[category] = category_evolution
        
        # Analyze polarization patterns
        polarization_patterns = self._analyze_polarization_patterns(polarization_tracking)
        
        # Calculate polarization metrics
        polarization_metrics = self._calculate_polarization_metrics(polarization_patterns)
        
        # Generate polarization insights
        polarization_insights = self._generate_polarization_insights(
            polarization_patterns, 
            polarization_metrics
        )
        
        return {
            'polarization_categories': list(polarization_terms.keys()),
            'tracking_results': polarization_tracking,
            'polarization_patterns': polarization_patterns,
            'polarization_metrics': polarization_metrics,
            'insights': polarization_insights,
            'generated_at': datetime.now().isoformat()
        }
    
    def predict_concept_trajectory(
        self, 
        concept: str, 
        prediction_days: int = 30
    ) -> Dict[str, Any]:
        """
        Predict future trajectory of a concept based on historical evolution
        
        Args:
            concept: Concept to predict trajectory for
            prediction_days: Number of days to predict forward
            
        Returns:
            Trajectory prediction analysis
        """
        logger.info(f"Predicting trajectory for concept '{concept}' over {prediction_days} days")
        
        # First get historical evolution
        historical_evolution = self.track_concept_evolution(concept)
        
        if historical_evolution.get('error'):
            return historical_evolution
        
        # Analyze historical patterns
        historical_patterns = self._extract_historical_patterns(historical_evolution)
        
        # Generate predictions
        trajectory_prediction = self._generate_trajectory_prediction(
            concept,
            historical_patterns,
            prediction_days
        )
        
        # Calculate prediction confidence
        prediction_confidence = self._calculate_prediction_confidence(historical_patterns)
        
        # Generate prediction insights
        prediction_insights = self._generate_prediction_insights(
            concept,
            trajectory_prediction,
            prediction_confidence
        )
        
        return {
            'concept': concept,
            'prediction_period_days': prediction_days,
            'historical_analysis': historical_evolution,
            'historical_patterns': historical_patterns,
            'trajectory_prediction': trajectory_prediction,
            'prediction_confidence': prediction_confidence,
            'insights': prediction_insights,
            'methodology': 'historical_pattern_extrapolation',
            'generated_at': datetime.now().isoformat()
        }
    
    # Helper Methods
    
    def _extract_timed_documents(self, search_results: List[Dict]) -> List[Dict]:
        """Extract documents with valid timestamps"""
        
        timed_docs = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            timestamp_str = metadata.get('datetime') or metadata.get('timestamp')
            
            if timestamp_str:
                try:
                    # Try different date formats
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            timestamp = datetime.strptime(str(timestamp_str), fmt)
                            timed_docs.append({
                                'timestamp': timestamp,
                                'text': result['text'],
                                'similarity_score': result['similarity_score'],
                                'metadata': metadata
                            })
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        # Sort by timestamp
        timed_docs.sort(key=lambda x: x['timestamp'])
        
        return timed_docs
    
    def _filter_by_date_range(
        self, 
        documents: List[Dict], 
        start_date: str = None, 
        end_date: str = None
    ) -> List[Dict]:
        """Filter documents by date range"""
        
        filtered_docs = documents
        
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_docs = [doc for doc in filtered_docs if doc['timestamp'] >= start_dt]
            except ValueError:
                logger.warning(f"Invalid start_date format: {start_date}")
        
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_docs = [doc for doc in filtered_docs if doc['timestamp'] <= end_dt]
            except ValueError:
                logger.warning(f"Invalid end_date format: {end_date}")
        
        return filtered_docs
    
    def _create_temporal_windows(
        self, 
        documents: List[Dict], 
        window_days: int
    ) -> List[Dict]:
        """Create temporal windows from documents"""
        
        if not documents:
            return []
        
        windows = []
        start_date = documents[0]['timestamp']
        end_date = documents[-1]['timestamp']
        
        current_date = start_date
        window_delta = timedelta(days=window_days)
        
        while current_date < end_date:
            window_end = current_date + window_delta
            
            # Get documents in this window
            window_docs = [
                doc for doc in documents 
                if current_date <= doc['timestamp'] < window_end
            ]
            
            if len(window_docs) >= self.min_documents_per_window:
                windows.append({
                    'window_id': len(windows),
                    'start_date': current_date,
                    'end_date': window_end,
                    'documents': window_docs,
                    'document_count': len(window_docs)
                })
            
            current_date = window_end
        
        return windows
    
    def _analyze_concept_evolution_across_windows(
        self, 
        concept: str, 
        windows: List[Dict]
    ) -> List[Dict]:
        """Analyze how concept evolves across temporal windows"""
        
        evolution_analysis = []
        
        for i, window in enumerate(windows):
            # Analyze semantic characteristics of this window
            window_analysis = self._analyze_window_semantics(concept, window)
            
            # Compare with previous window if available
            if i > 0:
                prev_window = windows[i-1]
                evolution_metrics = self._calculate_evolution_metrics(
                    concept, 
                    prev_window, 
                    window
                )
                window_analysis['evolution_metrics'] = evolution_metrics
            
            # Add contextual analysis
            window_analysis['contextual_analysis'] = self._analyze_window_context(
                concept, 
                window
            )
            
            evolution_analysis.append(window_analysis)
        
        return evolution_analysis
    
    def _analyze_window_semantics(self, concept: str, window: Dict) -> Dict[str, Any]:
        """Analyze semantic characteristics of a temporal window"""
        
        window_docs = window['documents']
        
        # Basic statistics
        avg_similarity = np.mean([doc['similarity_score'] for doc in window_docs])
        
        # Extract key themes from window texts
        window_texts = [doc['text'] for doc in window_docs]
        key_themes = self._extract_key_themes_from_texts(window_texts)
        
        # Channel analysis
        channels = [doc['metadata'].get('channel', 'unknown') for doc in window_docs]
        channel_distribution = Counter(channels)
        
        # Sentiment estimation (simplified)
        sentiment_indicators = self._estimate_window_sentiment(window_texts)
        
        return {
            'window_id': window['window_id'],
            'period': {
                'start': window['start_date'].isoformat(),
                'end': window['end_date'].isoformat()
            },
            'document_count': len(window_docs),
            'avg_similarity_to_concept': avg_similarity,
            'key_themes': key_themes,
            'channel_distribution': dict(channel_distribution.most_common(5)),
            'sentiment_indicators': sentiment_indicators,
            'dominant_sentiment': sentiment_indicators.get('dominant', 'neutral')
        }
    
    def _calculate_evolution_metrics(
        self, 
        concept: str, 
        prev_window: Dict, 
        current_window: Dict
    ) -> Dict[str, Any]:
        """Calculate evolution metrics between windows"""
        
        prev_docs = prev_window['documents']
        curr_docs = current_window['documents']
        
        # Calculate similarity change
        prev_avg_sim = np.mean([doc['similarity_score'] for doc in prev_docs])
        curr_avg_sim = np.mean([doc['similarity_score'] for doc in curr_docs])
        similarity_change = curr_avg_sim - prev_avg_sim
        
        # Calculate volume change
        volume_change = len(curr_docs) - len(prev_docs)
        volume_change_pct = (volume_change / len(prev_docs)) * 100 if len(prev_docs) > 0 else 0
        
        # Calculate thematic evolution
        prev_themes = self._extract_key_themes_from_texts([doc['text'] for doc in prev_docs])
        curr_themes = self._extract_key_themes_from_texts([doc['text'] for doc in curr_docs])
        
        # Theme overlap analysis
        prev_theme_set = set(prev_themes)
        curr_theme_set = set(curr_themes)
        theme_overlap = len(prev_theme_set.intersection(curr_theme_set))
        theme_novelty = len(curr_theme_set - prev_theme_set)
        
        return {
            'similarity_change': similarity_change,
            'volume_change': volume_change,
            'volume_change_percent': volume_change_pct,
            'theme_overlap': theme_overlap,
            'theme_novelty': theme_novelty,
            'evolution_magnitude': abs(similarity_change) + (abs(volume_change_pct) / 100),
            'evolution_direction': 'increasing' if similarity_change > 0 else 'decreasing'
        }
    
    def _extract_key_themes_from_texts(self, texts: List[str]) -> List[str]:
        """Extract key themes from a collection of texts"""
        
        # Simple keyword extraction based on frequency
        all_words = []
        for text in texts:
            # Clean and tokenize
            words = text.lower().split()
            # Filter relevant political keywords
            for category, terms in self.concept_categories.items():
                for term in terms:
                    if term in text.lower():
                        all_words.append(term)
        
        # Count frequencies
        word_counts = Counter(all_words)
        
        # Return top themes
        return [word for word, count in word_counts.most_common(5)]
    
    def _estimate_window_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Estimate sentiment characteristics of window texts"""
        
        # Simple sentiment indicators based on keywords
        positive_indicators = ['sucesso', 'vitória', 'progresso', 'melhoria', 'avanço']
        negative_indicators = ['crise', 'problema', 'fracasso', 'corrupção', 'mentira']
        neutral_indicators = ['análise', 'debate', 'discussão', 'questão', 'tema']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for text in texts:
            text_lower = text.lower()
            if any(word in text_lower for word in positive_indicators):
                positive_count += 1
            elif any(word in text_lower for word in negative_indicators):
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(texts)
        if total == 0:
            return {'dominant': 'neutral', 'distribution': {}}
        
        sentiment_dist = {
            'positive': positive_count / total,
            'negative': negative_count / total,
            'neutral': neutral_count / total
        }
        
        dominant = max(sentiment_dist.items(), key=lambda x: x[1])[0]
        
        return {
            'dominant': dominant,
            'distribution': sentiment_dist,
            'polarization_score': abs(sentiment_dist['positive'] - sentiment_dist['negative'])
        }
    
    def _analyze_window_context(self, concept: str, window: Dict) -> Dict[str, Any]:
        """Analyze contextual factors for a window"""
        
        window_start = window['start_date']
        year = str(window_start.year)
        month = window_start.month
        
        # Find relevant political events
        relevant_events = []
        if year in self.political_timeline:
            timeline_data = self.political_timeline[year]
            relevant_events = timeline_data.get('major_events', [])
        
        # AI-powered contextual analysis if available
        ai_context = {}
        if self.api_available:
            ai_context = self._analyze_context_with_ai(concept, window)
        
        return {
            'period_year': year,
            'period_month': month,
            'relevant_political_events': relevant_events,
            'ai_contextual_analysis': ai_context
        }
    
    def _analyze_context_with_ai(self, concept: str, window: Dict) -> Dict[str, Any]:
        """Analyze window context using AI"""
        
        try:
            sample_texts = [doc['text'][:200] for doc in window['documents'][:3]]
            period_str = f"{window['start_date'].strftime('%Y-%m-%d')} to {window['end_date'].strftime('%Y-%m-%d')}"
            
            texts_sample = '\n'.join([f"- {text}..." for text in sample_texts])
            
            prompt = f"""
Analise o contexto político brasileiro para o conceito "{concept}" no período {period_str}:

TEXTOS REPRESENTATIVOS:
{texts_sample}

Considerando o contexto político brasileiro, analise:

Responda em JSON:
{{
  "political_context": "contexto político do período",
  "discourse_characteristics": "características do discurso",
  "external_influences": ["influência1", "influência2"],
  "narrative_focus": "foco narrativo principal",
  "significance_level": "alta|media|baixa"
}}
"""
            
            response = self.create_message(
                prompt,
                stage="temporal_evolution",
                operation="context_analysis",
                temperature=0.3
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"AI context analysis failed: {e}")
            return {}
    
    def _detect_narrative_shifts(self, evolution_analysis: List[Dict]) -> List[Dict]:
        """Detect significant narrative shifts in evolution analysis"""
        
        shifts = []
        
        for i, analysis in enumerate(evolution_analysis):
            if i == 0:  # Skip first window
                continue
            
            evolution_metrics = analysis.get('evolution_metrics', {})
            
            # Check for significant changes
            similarity_change = abs(evolution_metrics.get('similarity_change', 0))
            volume_change_pct = abs(evolution_metrics.get('volume_change_percent', 0))
            theme_novelty = evolution_metrics.get('theme_novelty', 0)
            
            # Determine if this represents a significant shift
            shift_score = similarity_change * 2 + (volume_change_pct / 100) + (theme_novelty / 10)
            
            if shift_score > self.evolution_threshold:
                shift_type = self._classify_shift_type(evolution_metrics, analysis)
                
                shifts.append({
                    'shift_id': len(shifts),
                    'window_id': analysis['window_id'],
                    'period': analysis['period'],
                    'shift_score': shift_score,
                    'shift_type': shift_type,
                    'evolution_metrics': evolution_metrics,
                    'contextual_factors': analysis.get('contextual_analysis', {})
                })
        
        return shifts
    
    def _classify_shift_type(self, evolution_metrics: Dict, analysis: Dict) -> str:
        """Classify the type of narrative shift"""
        
        similarity_change = evolution_metrics.get('similarity_change', 0)
        volume_change_pct = evolution_metrics.get('volume_change_percent', 0)
        theme_novelty = evolution_metrics.get('theme_novelty', 0)
        
        if theme_novelty > 3:
            return 'thematic_shift'
        elif volume_change_pct > 50:
            return 'volume_surge' if volume_change_pct > 0 else 'volume_decline'
        elif abs(similarity_change) > 0.2:
            return 'semantic_shift'
        else:
            return 'gradual_evolution'
    
    def _generate_evolution_insights(
        self, 
        concept: str, 
        evolution_analysis: List[Dict], 
        narrative_shifts: List[Dict]
    ) -> Dict[str, Any]:
        """Generate insights about concept evolution"""
        
        if not evolution_analysis:
            return {}
        
        # Calculate overall trends
        similarity_trend = self._calculate_overall_trend([
            a.get('avg_similarity_to_concept', 0) for a in evolution_analysis
        ])
        
        volume_trend = self._calculate_overall_trend([
            a.get('document_count', 0) for a in evolution_analysis
        ])
        
        # Identify most significant periods
        most_active_period = max(evolution_analysis, key=lambda x: x.get('document_count', 0))
        
        # Sentiment evolution
        sentiment_evolution = [a.get('sentiment_indicators', {}).get('dominant', 'neutral') for a in evolution_analysis]
        
        return {
            'overall_similarity_trend': similarity_trend,
            'overall_volume_trend': volume_trend,
            'total_narrative_shifts': len(narrative_shifts),
            'most_active_period': {
                'period': most_active_period.get('period'),
                'document_count': most_active_period.get('document_count')
            },
            'sentiment_evolution_pattern': self._analyze_sentiment_pattern(sentiment_evolution),
            'concept_stability': 'stable' if len(narrative_shifts) < 2 else 'volatile',
            'evolution_summary': self._generate_evolution_summary(concept, evolution_analysis, narrative_shifts)
        }
    
    def _calculate_overall_trend(self, values: List[float]) -> str:
        """Calculate overall trend from a series of values"""
        
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.3:
            return 'increasing'
        elif correlation < -0.3:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_sentiment_pattern(self, sentiment_evolution: List[str]) -> str:
        """Analyze sentiment evolution pattern"""
        
        if not sentiment_evolution:
            return 'no_data'
        
        sentiment_counts = Counter(sentiment_evolution)
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Check for changes
        unique_sentiments = len(set(sentiment_evolution))
        
        if unique_sentiments == 1:
            return f'consistently_{dominant_sentiment}'
        elif unique_sentiments == 2:
            return 'polarized'
        else:
            return 'volatile'
    
    def _generate_evolution_summary(
        self, 
        concept: str, 
        evolution_analysis: List[Dict], 
        narrative_shifts: List[Dict]
    ) -> str:
        """Generate a textual summary of concept evolution"""
        
        total_periods = len(evolution_analysis)
        total_shifts = len(narrative_shifts)
        
        if total_shifts == 0:
            return f"O conceito '{concept}' manteve-se relativamente estável ao longo de {total_periods} períodos analisados."
        elif total_shifts == 1:
            return f"O conceito '{concept}' apresentou uma mudança significativa durante os {total_periods} períodos analisados."
        else:
            return f"O conceito '{concept}' passou por {total_shifts} mudanças significativas em {total_periods} períodos, indicando alta volatilidade no discurso."
    
    def _create_evolution_visualization_data(self, evolution_analysis: List[Dict]) -> Dict[str, Any]:
        """Create data structure for evolution visualization"""
        
        if not evolution_analysis:
            return {}
        
        # Prepare time series data
        time_series = []
        
        for analysis in evolution_analysis:
            time_series.append({
                'period_start': analysis['period']['start'],
                'period_end': analysis['period']['end'],
                'document_count': analysis['document_count'],
                'avg_similarity': analysis['avg_similarity_to_concept'],
                'dominant_sentiment': analysis.get('dominant_sentiment', 'neutral'),
                'key_themes': analysis.get('key_themes', [])
            })
        
        return {
            'time_series_data': time_series,
            'chart_config': {
                'x_axis': 'period_start',
                'y_axes': ['document_count', 'avg_similarity'],
                'color_coding': 'dominant_sentiment'
            },
            'data_format': 'temporal_evolution_visualization'
        }
    
    def _get_all_timed_documents(self) -> List[Dict]:
        """Get all documents with timestamps from the search index"""
        
        if not self.search_engine.search_index:
            return []
        
        timed_docs = []
        
        for i, metadata in enumerate(self.search_engine.search_index['metadata']):
            timestamp_str = metadata.get('datetime') or metadata.get('timestamp')
            if timestamp_str:
                try:
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                        try:
                            timestamp = datetime.strptime(str(timestamp_str), fmt)
                            timed_docs.append({
                                'index': i,
                                'timestamp': timestamp,
                                'text': self.search_engine.search_index['texts'][i],
                                'metadata': metadata,
                                'embedding': self.search_engine.search_index['embeddings'][i]
                            })
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        timed_docs.sort(key=lambda x: x['timestamp'])
        return timed_docs
    
    def _create_monthly_windows(self, documents: List[Dict], months: int) -> List[Dict]:
        """Create monthly windows from documents"""
        
        if not documents:
            return []
        
        windows = []
        start_date = documents[0]['timestamp']
        end_date = documents[-1]['timestamp']
        
        # Ensure we don't exceed the available data range
        actual_months = min(months, (end_date - start_date).days // 30)
        
        current_date = start_date
        
        for month_idx in range(actual_months):
            month_end = current_date + timedelta(days=30)  # Approximate month
            
            month_docs = [
                doc for doc in documents 
                if current_date <= doc['timestamp'] < month_end
            ]
            
            if len(month_docs) >= 10:  # Minimum documents per month
                windows.append({
                    'month_id': month_idx,
                    'start_date': current_date,
                    'end_date': month_end,
                    'documents': month_docs,
                    'document_count': len(month_docs)
                })
            
            current_date = month_end
        
        return windows


def create_temporal_evolution_tracker(
    config: Dict[str, Any], 
    search_engine: SemanticSearchEngine = None
) -> TemporalEvolutionTracker:
    """
    Factory function to create TemporalEvolutionTracker instance
    
    Args:
        config: Configuration dictionary
        search_engine: Optional pre-initialized search engine
        
    Returns:
        TemporalEvolutionTracker instance
    """
    return TemporalEvolutionTracker(config, search_engine)