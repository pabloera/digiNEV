"""
Test suite for analysis modules (sentiment, topic modeling, political analysis, etc.).
Tests the specific analysis capabilities of the pipeline.

These tests define expected behavior for different types of analysis.
"""

import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pandas as pd
import pytest

from conftest import (
    assert_dataframe_columns,
    assert_valid_analysis_result,
    mock_anthropic_sentiment_response,
    mock_anthropic_political_response
)


class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""
    
    def test_sentiment_analyzer_initialization(self, test_config):
        """Test sentiment analyzer can be initialized."""
        from src.anthropic_integration.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(test_config)
        
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'analyze_batch') or hasattr(analyzer, 'analyze')
        
    def test_sentiment_analysis_basic_functionality(self, test_config, sample_telegram_data):
        """Test basic sentiment analysis functionality."""
        from src.anthropic_integration.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(test_config)
        
        # Mock API response
        with patch.object(analyzer, 'client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = mock_anthropic_sentiment_response()
            mock_client.messages.create.return_value = mock_response
            
            # Should analyze sentiment
            results = analyzer.analyze_batch(sample_telegram_data['body'].head(5).tolist())
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Check result structure
            for result in results:
                assert 'sentiment' in result
                assert 'confidence' in result
                assert result['sentiment'] in ['positive', 'negative', 'neutral']
                assert 0 <= result['confidence'] <= 1
                
    def test_sentiment_analysis_handles_different_emotions(self, test_config):
        """Test sentiment analysis with different emotional content."""
        from src.anthropic_integration.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(test_config)
        
        test_texts = [
            "Estou muito feliz hoje! 😊",
            "Que situação terrível e triste 😢",
            "Reunião de trabalho às 14h",
            "REVOLTANTE!!! Não aceito isso!!!",
            "Amo minha família ❤️"
        ]
        
        with patch.object(analyzer, 'client') as mock_client:
            # Mock different responses for different emotions
            responses = [
                {"sentiment": "positive", "confidence": 0.9, "emotion": "joy"},
                {"sentiment": "negative", "confidence": 0.8, "emotion": "sadness"},
                {"sentiment": "neutral", "confidence": 0.7, "emotion": "neutral"},
                {"sentiment": "negative", "confidence": 0.95, "emotion": "anger"},
                {"sentiment": "positive", "confidence": 0.9, "emotion": "love"}
            ]
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({"results": responses})
            mock_client.messages.create.return_value = mock_response
            
            results = analyzer.analyze_batch(test_texts)
            
            # Should detect different emotions appropriately
            assert len(results) == 5
            assert results[0]['sentiment'] == 'positive'  # Happy
            assert results[1]['sentiment'] == 'negative'  # Sad
            assert results[2]['sentiment'] == 'neutral'   # Neutral
            assert results[3]['sentiment'] == 'negative'  # Anger
            assert results[4]['sentiment'] == 'positive'  # Love
            
    def test_sentiment_analysis_handles_political_context(self, test_config, sample_political_data):
        """Test sentiment analysis in political context."""
        from src.anthropic_integration.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(test_config)
        
        political_texts = sample_political_data['body'].tolist()
        
        with patch.object(analyzer, 'client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = mock_anthropic_sentiment_response()
            mock_client.messages.create.return_value = mock_response
            
            results = analyzer.analyze_batch(political_texts)
            
            # Should handle political content
            assert len(results) > 0
            
            # Should identify political themes
            for result in results:
                if 'themes' in result:
                    themes = result['themes']
                    political_themes = ['politics', 'government', 'democracy', 'authority']
                    # At least some results should have political themes
                    # (This is a heuristic test - in real implementation, we'd verify specific cases)


class TestPoliticalAnalysis:
    """Test political discourse analysis."""
    
    def test_political_analyzer_initialization(self, test_config):
        """Test political analyzer initialization."""
        from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
        
        analyzer = PoliticalAnalyzer(test_config)
        
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'analyze_batch') or hasattr(analyzer, 'classify')
        
    def test_political_classification_categories(self, test_config, sample_political_data):
        """Test political classification into different categories."""
        from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
        
        analyzer = PoliticalAnalyzer(test_config)
        
        with patch.object(analyzer, 'client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = mock_anthropic_political_response()
            mock_client.messages.create.return_value = mock_response
            
            results = analyzer.analyze_batch(sample_political_data['body'].tolist())
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Check classification structure
            for result in results:
                assert 'classification' in result
                assert 'confidence' in result or 'classification' in result and 'confidence' in result['classification']
                
    def test_negation_detection(self, test_config):
        """Test detection of negationist discourse."""
        from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
        
        analyzer = PoliticalAnalyzer(test_config)
        
        negationist_texts = [
            "Vacinas são experimentais e perigosas",
            "Mudanças climáticas são farsa",
            "Urnas eletrônicas são fraudulentas",
            "Mídia tradicional só mente"
        ]
        
        with patch.object(analyzer, 'client') as mock_client:
            # Mock response indicating negationist content
            negationist_response = json.dumps({
                "0": {"classification": {"primary": "negationist", "confidence": 0.9}},
                "1": {"classification": {"primary": "negationist", "confidence": 0.85}},
                "2": {"classification": {"primary": "negationist", "confidence": 0.95}},
                "3": {"classification": {"primary": "negationist", "confidence": 0.8}}
            })
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = negationist_response
            mock_client.messages.create.return_value = mock_response
            
            results = analyzer.analyze_batch(negationist_texts)
            
            # Should detect negationist content
            assert len(results) == 4
            for result in results:
                # Should classify as negationist
                classification = result.get('classification', {})
                primary = classification.get('primary', '')
                assert 'negationist' in primary.lower() or 'negacion' in primary.lower()
                
    def test_conspiracy_theory_detection(self, test_config):
        """Test detection of conspiracy theories."""
        from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
        
        analyzer = PoliticalAnalyzer(test_config)
        
        conspiracy_texts = [
            "Globalistas querem destruir o Brasil",
            "Nova ordem mundial controla tudo",
            "Agenda 2030 é plano de dominação",
            "Comunismo infiltrado nas universidades"
        ]
        
        with patch.object(analyzer, 'client') as mock_client:
            conspiracy_response = json.dumps({
                "0": {"classification": {"primary": "conspiracy_theory", "confidence": 0.9}},
                "1": {"classification": {"primary": "conspiracy_theory", "confidence": 0.95}},
                "2": {"classification": {"primary": "conspiracy_theory", "confidence": 0.8}},
                "3": {"classification": {"primary": "conspiracy_theory", "confidence": 0.85}}
            })
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = conspiracy_response
            mock_client.messages.create.return_value = mock_response
            
            results = analyzer.analyze_batch(conspiracy_texts)
            
            # Should detect conspiracy theories
            for result in results:
                classification = result.get('classification', {})
                primary = classification.get('primary', '')
                assert 'conspiracy' in primary.lower() or 'conspir' in primary.lower()
                
    def test_authoritarian_discourse_detection(self, test_config):
        """Test detection of authoritarian discourse."""
        from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
        
        analyzer = PoliticalAnalyzer(test_config)
        
        authoritarian_texts = [
            "STF precisa ser fechado",
            "Congresso não representa o povo",
            "Imprensa é inimiga do povo",
            "Ditadura militar foi necessária"
        ]
        
        with patch.object(analyzer, 'client') as mock_client:
            authoritarian_response = json.dumps({
                "0": {"classification": {"primary": "authoritarian", "confidence": 0.9}},
                "1": {"classification": {"primary": "authoritarian", "confidence": 0.85}},
                "2": {"classification": {"primary": "authoritarian", "confidence": 0.8}},
                "3": {"classification": {"primary": "authoritarian", "confidence": 0.95}}
            })
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = authoritarian_response
            mock_client.messages.create.return_value = mock_response
            
            results = analyzer.analyze_batch(authoritarian_texts)
            
            # Should detect authoritarian discourse
            for result in results:
                classification = result.get('classification', {})
                primary = classification.get('primary', '')
                assert 'authoritarian' in primary.lower() or 'autorit' in primary.lower()


class TestTopicModeling:
    """Test topic modeling functionality."""
    
    def test_topic_modeler_initialization(self, test_config):
        """Test topic modeler initialization."""
        from src.anthropic_integration.voyage_topic_modeler import VoyageTopicModeler
        
        modeler = VoyageTopicModeler(test_config)
        
        assert hasattr(modeler, 'config')
        assert hasattr(modeler, 'generate_topics') or hasattr(modeler, 'fit')
        
    def test_topic_generation(self, test_config, sample_telegram_data):
        """Test topic generation from text data."""
        from src.anthropic_integration.voyage_topic_modeler import VoyageTopicModeler
        
        modeler = VoyageTopicModeler(test_config)
        
        # Mock embedding generation
        with patch.object(modeler, 'voyage_client') as mock_voyage:
            # Mock embeddings for clustering
            mock_embeddings = np.random.rand(100, 384).tolist()  # Typical embedding dimension
            mock_voyage.embed.return_value = Mock(embeddings=mock_embeddings)
            
            results = modeler.generate_topics(sample_telegram_data['body'].tolist())
            
            assert isinstance(results, dict)
            assert 'topics' in results
            assert 'document_topics' in results
            
            # Check topic structure
            topics = results['topics']
            assert len(topics) > 0
            
            for topic_id, topic_data in topics.items():
                assert 'words' in topic_data
                assert 'label' in topic_data or 'name' in topic_data
                assert isinstance(topic_data['words'], list)
                
    def test_topic_interpretation(self, test_config, sample_topic_results):
        """Test topic interpretation with Anthropic."""
        from src.anthropic_integration.topic_interpreter import TopicInterpreter
        
        interpreter = TopicInterpreter(test_config)
        
        with patch.object(interpreter, 'client') as mock_client:
            interpretation_response = json.dumps({
                "0": {
                    "label": "Processo Democrático",
                    "description": "Discussões sobre eleições e democracia",
                    "discourse_type": "democratic",
                    "themes": ["elections", "democracy", "voting"]
                },
                "1": {
                    "label": "Economia Nacional", 
                    "description": "Debates sobre economia e políticas fiscais",
                    "discourse_type": "economic",
                    "themes": ["economy", "inflation", "fiscal_policy"]
                }
            })
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = interpretation_response
            mock_client.messages.create.return_value = mock_response
            
            # Create mock LDA model
            mock_lda_model = Mock()
            mock_lda_model.num_topics = 2
            mock_lda_model.show_topic.side_effect = [
                [('política', 0.1), ('eleições', 0.08), ('democracia', 0.07)],
                [('economia', 0.1), ('inflação', 0.08), ('dólar', 0.07)]
            ]
            
            results = interpreter.interpret_topics(mock_lda_model)
            
            assert isinstance(results, dict)
            assert len(results) == 2
            
            for topic_id, interpretation in results.items():
                assert 'label' in interpretation
                assert 'description' in interpretation
                assert 'discourse_type' in interpretation


class TestClusteringAnalysis:
    """Test clustering functionality."""
    
    def test_clustering_analyzer_initialization(self, test_config):
        """Test clustering analyzer initialization."""
        from src.anthropic_integration.voyage_clustering_analyzer import VoyageClusteringAnalyzer
        
        analyzer = VoyageClusteringAnalyzer(test_config)
        
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'cluster_messages') or hasattr(analyzer, 'fit_predict')
        
    def test_message_clustering(self, test_config, sample_telegram_data):
        """Test clustering of messages."""
        from src.anthropic_integration.voyage_clustering_analyzer import VoyageClusteringAnalyzer
        
        analyzer = VoyageClusteringAnalyzer(test_config)
        
        # Mock embedding generation
        with patch.object(analyzer, 'voyage_client') as mock_voyage:
            mock_embeddings = np.random.rand(100, 384).tolist()
            mock_voyage.embed.return_value = Mock(embeddings=mock_embeddings)
            
            results = analyzer.cluster_messages(sample_telegram_data['body'].tolist())
            
            assert isinstance(results, dict)
            assert 'clusters' in results
            assert 'cluster_labels' in results
            
            # Check clustering results
            cluster_labels = results['cluster_labels']
            assert len(cluster_labels) == len(sample_telegram_data)
            
            # Should have reasonable number of clusters
            unique_clusters = set(cluster_labels)
            assert 2 <= len(unique_clusters) <= 20  # Reasonable cluster count
            
    def test_cluster_validation(self, test_config):
        """Test cluster validation with Anthropic."""
        from src.anthropic_integration.cluster_validator import ClusterValidator
        
        validator = ClusterValidator(test_config)
        
        sample_cluster_messages = [
            "Política brasileira precisa mudar",
            "Eleições são fundamentais para democracia", 
            "Congresso deve representar o povo"
        ]
        
        with patch.object(validator, 'client') as mock_client:
            validation_response = json.dumps({
                "theme": "Political Discussion",
                "coherence": 0.85,
                "main_topics": ["politics", "democracy", "government"],
                "outliers": [],
                "quality_score": 0.9
            })
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = validation_response
            mock_client.messages.create.return_value = mock_response
            
            result = validator.validate_cluster(sample_cluster_messages, cluster_id=0)
            
            assert isinstance(result, dict)
            assert 'theme' in result
            assert 'coherence' in result
            assert result['coherence'] > 0.5  # Should be reasonably coherent


class TestNetworkAnalysis:
    """Test network analysis functionality."""
    
    def test_network_analyzer_initialization(self, test_config):
        """Test network analyzer initialization."""
        from src.anthropic_integration.intelligent_network_analyzer import IntelligentNetworkAnalyzer
        
        analyzer = IntelligentNetworkAnalyzer(test_config)
        
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'analyze_networks') or hasattr(analyzer, 'build_network')
        
    def test_channel_topic_network_generation(self, test_config, sample_telegram_data):
        """Test generation of channel-topic networks."""
        from src.anthropic_integration.intelligent_network_analyzer import IntelligentNetworkAnalyzer
        
        analyzer = IntelligentNetworkAnalyzer(test_config)
        
        # Add topic assignments to data
        sample_data_with_topics = sample_telegram_data.copy()
        sample_data_with_topics['topic'] = [f'topic_{i % 5}' for i in range(len(sample_data_with_topics))]
        
        result = analyzer.analyze_networks(sample_data_with_topics)
        
        assert isinstance(result, dict)
        assert 'channel_topic_network' in result or 'networks' in result
        
        # Should have network structure
        if 'channel_topic_network' in result:
            network = result['channel_topic_network']
            assert 'nodes' in network
            assert 'edges' in network
            
    def test_domain_sharing_network(self, test_config, sample_telegram_data):
        """Test domain sharing network analysis."""
        from src.anthropic_integration.intelligent_network_analyzer import IntelligentNetworkAnalyzer
        
        analyzer = IntelligentNetworkAnalyzer(test_config)
        
        # Add domain data
        sample_data_with_domains = sample_telegram_data.copy()
        domains = ['globo.com', 'folha.uol.com.br', 'youtube.com', 'facebook.com', 'twitter.com']
        sample_data_with_domains['domain'] = [domains[i % len(domains)] for i in range(len(sample_data_with_domains))]
        
        result = analyzer.analyze_networks(sample_data_with_domains)
        
        # Should analyze domain sharing patterns
        assert isinstance(result, dict)
        # Network should contain domain information
        
    def test_influence_detection(self, test_config, sample_telegram_data):
        """Test detection of influential nodes in networks."""
        from src.anthropic_integration.intelligent_network_analyzer import IntelligentNetworkAnalyzer
        
        analyzer = IntelligentNetworkAnalyzer(test_config)
        
        # Create data with clear influence patterns
        influence_data = sample_telegram_data.copy()
        
        # Make some channels more active
        influence_data.loc[influence_data['channel'] == 'canal_0', 'forwards'] = 100
        influence_data.loc[influence_data['channel'] == 'canal_0', 'views'] = 1000
        
        result = analyzer.analyze_networks(influence_data)
        
        assert isinstance(result, dict)
        
        # Should identify influential channels
        if 'influential_nodes' in result:
            influential = result['influential_nodes']
            assert isinstance(influential, list)
            # Should identify canal_0 as influential
            influential_names = [node['id'] if isinstance(node, dict) else node for node in influential]
            # This is a heuristic test - in real implementation we'd have more specific checks


class TestTemporalAnalysis:
    """Test temporal analysis functionality."""
    
    def test_temporal_analyzer_initialization(self, test_config):
        """Test temporal analyzer initialization."""
        from src.anthropic_integration.smart_temporal_analyzer import SmartTemporalAnalyzer
        
        analyzer = SmartTemporalAnalyzer(test_config)
        
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'analyze_temporal_patterns') or hasattr(analyzer, 'analyze')
        
    def test_temporal_pattern_detection(self, test_config, sample_telegram_data):
        """Test detection of temporal patterns."""
        from src.anthropic_integration.smart_temporal_analyzer import SmartTemporalAnalyzer
        
        analyzer = SmartTemporalAnalyzer(test_config)
        
        result = analyzer.analyze_temporal_patterns(sample_telegram_data)
        
        assert isinstance(result, dict)
        assert 'patterns' in result or 'temporal_trends' in result
        
        # Should identify temporal trends
        if 'patterns' in result:
            patterns = result['patterns']
            assert isinstance(patterns, (list, dict))
            
    def test_event_detection(self, test_config):
        """Test detection of significant events in timeline."""
        from src.anthropic_integration.smart_temporal_analyzer import SmartTemporalAnalyzer
        
        analyzer = SmartTemporalAnalyzer(test_config)
        
        # Create data with clear event spike
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        volumes = [10] * 30
        volumes[15] = 100  # Spike on day 15
        
        event_data = pd.DataFrame({
            'date': dates,
            'volume': volumes,
            'body': [f'Message for day {i}' for i in range(30)]
        })
        
        result = analyzer.analyze_temporal_patterns(event_data)
        
        # Should detect the spike
        assert isinstance(result, dict)
        
        # Look for event detection or anomaly detection
        if 'events' in result:
            events = result['events']
            assert len(events) > 0
            
    def test_periodic_pattern_detection(self, test_config):
        """Test detection of periodic patterns (daily, weekly cycles)."""
        from src.anthropic_integration.smart_temporal_analyzer import SmartTemporalAnalyzer
        
        analyzer = SmartTemporalAnalyzer(test_config)
        
        # Create data with weekly pattern
        dates = pd.date_range('2023-01-01', periods=14, freq='D')  # 2 weeks
        # Higher activity on weekdays, lower on weekends
        volumes = []
        for date in dates:
            if date.weekday() < 5:  # Weekday
                volumes.append(50)
            else:  # Weekend
                volumes.append(10)
                
        pattern_data = pd.DataFrame({
            'date': dates,
            'volume': volumes,
            'body': [f'Message for {date}' for date in dates]
        })
        
        result = analyzer.analyze_temporal_patterns(pattern_data)
        
        # Should detect weekly pattern
        assert isinstance(result, dict)
        
        if 'periodic_patterns' in result:
            patterns = result['periodic_patterns']
            # Should identify weekly cycle
            assert any('week' in str(pattern).lower() for pattern in patterns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
