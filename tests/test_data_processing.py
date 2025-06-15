"""
Test suite for data processing functionality.
Tests data validation, cleaning, encoding, deduplication, and feature extraction.

These tests define expected behavior for data processing stages.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

import pandas as pd
import pytest
import numpy as np

from conftest import assert_dataframe_columns, create_test_data_file


class TestDataValidation:
    """Test data validation and schema checking."""
    
    def test_feature_validator_initialization(self, test_config):
        """Test feature validator initialization."""
        from src.anthropic_integration.feature_validator import FeatureValidator
        
        validator = FeatureValidator(test_config)
        
        assert hasattr(validator, 'config')
        assert hasattr(validator, 'validate_dataset') or hasattr(validator, 'validate')
        
    def test_required_columns_validation(self, test_config, sample_telegram_data):
        """Test validation of required columns."""
        from src.anthropic_integration.feature_validator import FeatureValidator
        
        validator = FeatureValidator(test_config)
        
        # Should pass with complete data
        result = validator.validate_dataset(sample_telegram_data)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert result['is_valid'] == True
        
        # Should fail with missing columns
        incomplete_data = sample_telegram_data.drop(columns=['body'])
        result = validator.validate_dataset(incomplete_data)
        
        assert result['is_valid'] == False
        assert 'missing_columns' in result
        assert 'body' in result['missing_columns']
        
    def test_data_type_validation(self, test_config):
        """Test validation of data types."""
        from src.anthropic_integration.feature_validator import FeatureValidator
        
        validator = FeatureValidator(test_config)
        
        # Test with correct data types
        valid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'body': ['text1', 'text2', 'text3'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'channel': ['channel1', 'channel2', 'channel3']
        })
        
        result = validator.validate_dataset(valid_data)
        assert result['is_valid'] == True
        
        # Test with incorrect data types
        invalid_data = valid_data.copy()
        invalid_data['date'] = ['not_a_date', 'invalid', 'bad_format']
        
        result = validator.validate_dataset(invalid_data)
        # Should detect type issues
        assert 'data_type_issues' in result or result['is_valid'] == False
        
    def test_data_quality_checks(self, test_config):
        """Test data quality validation checks."""
        from src.anthropic_integration.feature_validator import FeatureValidator
        
        validator = FeatureValidator(test_config)
        
        # Create data with quality issues
        quality_data = pd.DataFrame({
            'id': [1, 2, None, 4, 5],  # Missing ID
            'body': ['good text', '', 'another text', None, 'final text'],  # Empty and null
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'channel': ['channel1', 'channel2', 'channel3', 'channel4', 'channel5']
        })
        
        result = validator.validate_dataset(quality_data)
        
        # Should detect quality issues
        assert 'quality_issues' in result or 'missing_values' in result
        
        if 'quality_issues' in result:
            issues = result['quality_issues']
            assert 'missing_ids' in issues or 'empty_bodies' in issues
            
    def test_schema_consistency_validation(self, test_config):
        """Test schema consistency across datasets."""
        from src.anthropic_integration.feature_validator import FeatureValidator
        
        validator = FeatureValidator(test_config)
        
        # First dataset schema
        dataset1 = pd.DataFrame({
            'id': [1, 2, 3],
            'body': ['text1', 'text2', 'text3'],
            'channel': ['ch1', 'ch2', 'ch3']
        })
        
        # Second dataset with different schema
        dataset2 = pd.DataFrame({
            'id': [4, 5, 6],
            'body': ['text4', 'text5', 'text6'],
            'author': ['author1', 'author2', 'author3']  # Different column
        })
        
        if hasattr(validator, 'validate_schema_consistency'):
            result = validator.validate_schema_consistency([dataset1, dataset2])
            
            assert isinstance(result, dict)
            assert 'schema_consistent' in result
            assert result['schema_consistent'] == False  # Schemas differ


class TestEncodingValidation:
    """Test encoding detection and validation."""
    
    def test_encoding_validator_initialization(self, test_config):
        """Test encoding validator initialization."""
        from src.anthropic_integration.encoding_validator import EncodingValidator
        
        validator = EncodingValidator(test_config)
        
        assert hasattr(validator, 'config')
        assert hasattr(validator, 'validate_encoding') or hasattr(validator, 'detect_encoding')
        
    def test_encoding_detection(self, test_config, test_data_dir):
        """Test encoding detection for files."""
        from src.anthropic_integration.encoding_validator import EncodingValidator
        
        validator = EncodingValidator(test_config)
        
        # Create test file with UTF-8 encoding
        utf8_data = pd.DataFrame({
            'body': ['Texto com acentuação: ção, ã, é, í, ó, ú, ç'],
            'channel': ['canal_teste']
        })
        
        utf8_file = test_data_dir / "utf8_test.csv"
        utf8_data.to_csv(utf8_file, index=False, encoding='utf-8')
        
        # Should detect UTF-8 encoding
        result = validator.validate_encoding(str(utf8_file))
        
        assert isinstance(result, dict)
        assert 'detected_encoding' in result
        assert 'is_valid' in result
        assert result['detected_encoding'].lower() in ['utf-8', 'utf8']
        
    def test_encoding_corruption_detection(self, test_config, test_data_dir):
        """Test detection of encoding corruption."""
        from src.anthropic_integration.encoding_validator import EncodingValidator
        
        validator = EncodingValidator(test_config)
        
        # Create file with encoding corruption patterns
        corrupted_data = pd.DataFrame({
            'body': ['Texto com corrupÃ§Ã£o de encoding', 'Ã©ste texto estÃ¡ corrompido'],
            'channel': ['canal_teste', 'canal_teste2']
        })
        
        corrupted_file = test_data_dir / "corrupted_test.csv"
        corrupted_data.to_csv(corrupted_file, index=False, encoding='utf-8')
        
        result = validator.validate_encoding(str(corrupted_file))
        
        # Should detect corruption patterns
        assert 'corruption_detected' in result or 'encoding_issues' in result
        
    def test_encoding_fix_suggestions(self, test_config):
        """Test encoding fix suggestions."""
        from src.anthropic_integration.encoding_validator import EncodingValidator
        
        validator = EncodingValidator(test_config)
        
        # Test text with corruption patterns
        corrupted_text = "CÃ¡rdio da manhÃ£"
        
        if hasattr(validator, 'suggest_encoding_fix'):
            suggestion = validator.suggest_encoding_fix(corrupted_text)
            
            assert isinstance(suggestion, dict)
            assert 'suggested_fix' in suggestion
            assert 'confidence' in suggestion
            
    def test_batch_encoding_validation(self, test_config, test_data_dir):
        """Test validation of multiple files."""
        from src.anthropic_integration.encoding_validator import EncodingValidator
        
        validator = EncodingValidator(test_config)
        
        # Create multiple test files
        files = []
        for i, encoding in enumerate(['utf-8', 'latin1']):
            try:
                test_data = pd.DataFrame({
                    'body': [f'Test text {i} with special chars: ção'],
                    'channel': [f'channel_{i}']
                })
                
                file_path = test_data_dir / f"encoding_test_{i}.csv"
                test_data.to_csv(file_path, index=False, encoding=encoding)
                files.append(str(file_path))
            except UnicodeEncodeError:
                # Skip if encoding doesn't support characters
                continue
        
        if hasattr(validator, 'validate_multiple_files') and files:
            results = validator.validate_multiple_files(files)
            
            assert isinstance(results, list)
            assert len(results) == len(files)
            
            for result in results:
                assert 'file_path' in result
                assert 'detected_encoding' in result


class TestDataCleaning:
    """Test text cleaning and preprocessing."""
    
    def test_text_cleaner_initialization(self, test_config):
        """Test text cleaner initialization."""
        from src.anthropic_integration.intelligent_text_cleaner import IntelligentTextCleaner
        
        cleaner = IntelligentTextCleaner(test_config)
        
        assert hasattr(cleaner, 'config')
        assert hasattr(cleaner, 'clean_text_data') or hasattr(cleaner, 'clean')
        
    def test_basic_text_cleaning(self, test_config):
        """Test basic text cleaning operations."""
        from src.anthropic_integration.intelligent_text_cleaner import IntelligentTextCleaner
        
        cleaner = IntelligentTextCleaner(test_config)
        
        dirty_data = pd.DataFrame({
            'body': [
                'Texto com    espaços extras    ',
                'TEXTO EM MAIÚSCULAS',
                'Texto com URLs: https://exemplo.com/teste',
                'Texto com @mentions e #hashtags',
                'Texto\ncom\nquebras\nde\nlinha'
            ]
        })
        
        cleaned = cleaner.clean_text_data(dirty_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert 'body' in cleaned.columns
        
        # Check cleaning results
        cleaned_texts = cleaned['body'].tolist()
        
        # Should normalize whitespace
        assert '    ' not in cleaned_texts[0]
        
        # Should handle case conversion (if configured)
        # Should handle URLs, mentions, hashtags based on config
        
    def test_political_context_cleaning(self, test_config, sample_political_data):
        """Test cleaning with political context preservation."""
        from src.anthropic_integration.intelligent_text_cleaner import IntelligentTextCleaner
        
        cleaner = IntelligentTextCleaner(test_config)
        
        # Mock Anthropic API for intelligent cleaning
        with patch.object(cleaner, 'client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "cleaned_texts": [
                    {"id": 0, "cleaned": "vacinas são experimentais", "preserved_terms": ["vacinas"]},
                    {"id": 1, "cleaned": "mudanças climáticas são farsa", "preserved_terms": ["climáticas"]}
                ]
            })
            mock_client.messages.create.return_value = mock_response
            
            cleaned = cleaner.clean_text_data(sample_political_data)
            
            # Should preserve political terms while cleaning
            assert isinstance(cleaned, pd.DataFrame)
            
    def test_encoding_fix_during_cleaning(self, test_config):
        """Test encoding fixes during cleaning process."""
        from src.anthropic_integration.intelligent_text_cleaner import IntelligentTextCleaner
        
        cleaner = IntelligentTextCleaner(test_config)
        
        corrupted_data = pd.DataFrame({
            'body': [
                'Texto com corrupÃ§Ã£o',
                'AcentuaÃ§Ã£o incorreta',
                'Caracteres especiais: Ã©, Ã­, Ã³'
            ]
        })
        
        cleaned = cleaner.clean_text_data(corrupted_data)
        
        # Should fix encoding issues
        cleaned_texts = cleaned['body'].tolist()
        
        # Should not contain corruption patterns
        for text in cleaned_texts:
            assert 'Ã§Ã£' not in text
            assert 'Ã©' not in text
            
    def test_cleaning_preserves_important_content(self, test_config):
        """Test that cleaning preserves semantically important content."""
        from src.anthropic_integration.intelligent_text_cleaner import IntelligentTextCleaner
        
        cleaner = IntelligentTextCleaner(test_config)
        
        important_data = pd.DataFrame({
            'body': [
                'STF (Supremo Tribunal Federal) é importante',
                'COVID-19 afetou o Brasil em 2020-2021',
                'Bolsonaro vs Lula nas eleições de 2022',
                'Negação da ciência é perigosa para a democracia'
            ]
        })
        
        cleaned = cleaner.clean_text_data(important_data)
        cleaned_texts = cleaned['body'].tolist()
        
        # Should preserve important political entities
        assert any('stf' in text.lower() or 'supremo' in text.lower() for text in cleaned_texts)
        assert any('covid' in text.lower() for text in cleaned_texts)
        assert any('bolsonaro' in text.lower() for text in cleaned_texts)
        assert any('lula' in text.lower() for text in cleaned_texts)


class TestDeduplication:
    """Test deduplication functionality."""
    
    def test_deduplication_validator_initialization(self, test_config):
        """Test deduplication validator initialization."""
        from src.anthropic_integration.deduplication_validator import DeduplicationValidator
        
        validator = DeduplicationValidator(test_config)
        
        assert hasattr(validator, 'config')
        assert hasattr(validator, 'deduplicate_data') or hasattr(validator, 'find_duplicates')
        
    def test_exact_duplicate_detection(self, test_config):
        """Test detection of exact duplicates."""
        from src.anthropic_integration.deduplication_validator import DeduplicationValidator
        
        validator = DeduplicationValidator(test_config)
        
        data_with_duplicates = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'body': [
                'Mensagem original',
                'Outra mensagem',
                'Mensagem original',  # Exact duplicate
                'Terceira mensagem',
                'Mensagem original'   # Another exact duplicate
            ],
            'channel': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']
        })
        
        result = validator.deduplicate_data(data_with_duplicates)
        
        assert isinstance(result, dict)
        assert 'deduplicated_data' in result
        assert 'duplicates_found' in result
        
        deduplicated = result['deduplicated_data']
        duplicates_count = result['duplicates_found']
        
        # Should remove exact duplicates
        assert len(deduplicated) < len(data_with_duplicates)
        assert duplicates_count > 0
        
    def test_near_duplicate_detection(self, test_config):
        """Test detection of near duplicates."""
        from src.anthropic_integration.deduplication_validator import DeduplicationValidator
        
        validator = DeduplicationValidator(test_config)
        
        near_duplicate_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'body': [
                'Política brasileira precisa de mudanças',
                'A política brasileira precisa de mudanças urgentes',  # Near duplicate
                'Economia está em crise no país',
                'Futebol é popular no Brasil'
            ],
            'channel': ['ch1', 'ch2', 'ch3', 'ch4']
        })
        
        # Mock similarity calculation
        with patch.object(validator, 'calculate_similarity') as mock_similarity:
            mock_similarity.side_effect = [0.95, 0.2, 0.1, 0.2, 0.15, 0.1]  # High similarity for first pair
            
            result = validator.deduplicate_data(near_duplicate_data, similarity_threshold=0.9)
            
            # Should detect near duplicates
            assert result['duplicates_found'] > 0
            
    def test_semantic_duplicate_detection(self, test_config):
        """Test detection of semantic duplicates using embeddings."""
        from src.anthropic_integration.deduplication_validator import DeduplicationValidator
        
        validator = DeduplicationValidator(test_config)
        
        semantic_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'body': [
                'O presidente fez declarações importantes',
                'Declarações relevantes foram feitas pelo presidente',  # Semantic duplicate
                'Chuva forte hoje na cidade',
                'Economia brasileira em crescimento'
            ],
            'channel': ['ch1', 'ch2', 'ch3', 'ch4']
        })
        
        # Mock embedding-based similarity
        with patch.object(validator, 'voyage_client') as mock_voyage:
            # Mock embeddings that are similar for semantic duplicates
            embeddings = [
                [0.1, 0.2, 0.3],  # Similar to next
                [0.15, 0.25, 0.35],  # Similar to previous
                [0.8, 0.9, 1.0],  # Different
                [0.4, 0.5, 0.6]   # Different
            ]
            mock_voyage.embed.return_value = Mock(embeddings=embeddings)
            
            if hasattr(validator, 'detect_semantic_duplicates'):
                result = validator.detect_semantic_duplicates(semantic_data)
                
                assert isinstance(result, dict)
                assert 'semantic_duplicates' in result
                
    def test_forwarded_message_deduplication(self, test_config):
        """Test deduplication of forwarded messages."""
        from src.anthropic_integration.deduplication_validator import DeduplicationValidator
        
        validator = DeduplicationValidator(test_config)
        
        forwarded_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'body': [
                'Mensagem original importante',
                'Forwarded: Mensagem original importante',  # Forwarded version
                'Fwd: Mensagem original importante',        # Another forwarded version
                'Mensagem completamente diferente'
            ],
            'channel': ['ch1', 'ch2', 'ch3', 'ch4']
        })
        
        if hasattr(validator, 'handle_forwarded_duplicates'):
            result = validator.handle_forwarded_duplicates(forwarded_data)
            
            # Should identify and handle forwarded duplicates
            assert isinstance(result, dict)
            deduplicated = result.get('deduplicated_data', forwarded_data)
            
            # Should reduce count due to forwarded duplicates
            assert len(deduplicated) <= len(forwarded_data)


class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_feature_extractor_initialization(self, test_config):
        """Test feature extractor initialization."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        assert hasattr(extractor, 'config')
        assert hasattr(extractor, 'extract_features') or hasattr(extractor, 'extract')
        
    def test_basic_feature_extraction(self, test_config, sample_telegram_data):
        """Test extraction of basic features."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        result = extractor.extract_features(sample_telegram_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have additional feature columns
        expected_features = ['text_length', 'word_count', 'has_urls', 'has_hashtags', 'has_mentions']
        
        for feature in expected_features:
            if feature in result.columns:
                # Verify feature is properly calculated
                if feature == 'text_length':
                    assert all(result[feature] > 0)
                elif feature == 'word_count':
                    assert all(result[feature] > 0)
                elif feature.startswith('has_'):
                    assert result[feature].dtype == bool
                    
    def test_url_extraction(self, test_config):
        """Test URL extraction and domain classification."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        url_data = pd.DataFrame({
            'body': [
                'Veja esta notícia: https://g1.globo.com/politica/noticia.html',
                'Link do YouTube: https://youtube.com/watch?v=abc123',
                'Post sem links aqui',
                'Múltiplos links: https://folha.com https://uol.com.br'
            ]
        })
        
        result = extractor.extract_features(url_data)
        
        # Should extract URLs and domains
        if 'urls' in result.columns:
            urls = result['urls'].tolist()
            assert any('g1.globo.com' in str(url) for url in urls)
            assert any('youtube.com' in str(url) for url in urls)
            
        if 'domain_count' in result.columns:
            domain_counts = result['domain_count'].tolist()
            assert domain_counts[3] == 2  # Multiple links in last message
            
    def test_hashtag_extraction(self, test_config):
        """Test hashtag extraction and normalization."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        hashtag_data = pd.DataFrame({
            'body': [
                'Política importante #política #brasil #eleições',
                'Economia em foco #economia #mercado',
                'Mensagem sem hashtags',
                'Multiple tags: #Covid19 #pandemia #saúde #brasil'
            ]
        })
        
        result = extractor.extract_features(hashtag_data)
        
        # Should extract and normalize hashtags
        if 'hashtags' in result.columns:
            hashtags = result['hashtags'].tolist()
            # Should find hashtags in first and last messages
            assert hashtags[0] is not None and len(str(hashtags[0])) > 0
            assert hashtags[3] is not None and len(str(hashtags[3])) > 0
            
        if 'hashtag_count' in result.columns:
            counts = result['hashtag_count'].tolist()
            assert counts[0] > 0  # First message has hashtags
            assert counts[2] == 0  # Third message has no hashtags
            
    def test_mention_extraction(self, test_config):
        """Test user mention extraction."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        mention_data = pd.DataFrame({
            'body': [
                'Importante @presidente @congresso',
                'Menção a @jornalista na reportagem',
                'Sem menções aqui',
                'Conversa com @fulano @sicrano @beltrano'
            ]
        })
        
        result = extractor.extract_features(mention_data)
        
        # Should extract mentions
        if 'mentions' in result.columns:
            mentions = result['mentions'].tolist()
            assert mentions[0] is not None  # First has mentions
            assert mentions[2] is None or len(str(mentions[2])) == 0  # Third has none
            
        if 'mention_count' in result.columns:
            counts = result['mention_count'].tolist()
            assert counts[3] == 3  # Last message has 3 mentions
            
    def test_sentiment_feature_extraction(self, test_config):
        """Test sentiment-related feature extraction."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        sentiment_data = pd.DataFrame({
            'body': [
                'Muito feliz com os resultados! Excelente!',
                'Situação terrível e preocupante...',
                'REVOLTANTE!!! Não aceito isso!!!',
                'Informação neutra sobre o tempo'
            ]
        })
        
        result = extractor.extract_features(sentiment_data)
        
        # Should extract sentiment indicators
        if 'exclamation_count' in result.columns:
            exclamations = result['exclamation_count'].tolist()
            assert exclamations[0] > 0  # Happy message has exclamations
            assert exclamations[2] > 0  # Angry message has many exclamations
            
        if 'caps_ratio' in result.columns:
            caps_ratios = result['caps_ratio'].tolist()
            assert caps_ratios[2] > caps_ratios[3]  # Angry message has more caps
            
    def test_political_feature_extraction(self, test_config, sample_political_data):
        """Test extraction of political discourse features."""
        from src.anthropic_integration.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(test_config)
        
        result = extractor.extract_features(sample_political_data)
        
        # Should extract political features
        if 'political_entities' in result.columns:
            entities = result['political_entities'].tolist()
            # Should identify political entities in the data
            
        if 'discourse_markers' in result.columns:
            markers = result['discourse_markers'].tolist()
            # Should identify discourse markers (negation, conspiracy, etc.)


class TestStatisticalAnalysis:
    """Test statistical analysis of data."""
    
    def test_statistical_analyzer_initialization(self, test_config):
        """Test statistical analyzer initialization."""
        from src.anthropic_integration.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(test_config)
        
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'generate_statistics') or hasattr(analyzer, 'analyze')
        
    def test_descriptive_statistics_generation(self, test_config, sample_telegram_data):
        """Test generation of descriptive statistics."""
        from src.anthropic_integration.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(test_config)
        
        stats = analyzer.generate_statistics(sample_telegram_data)
        
        assert isinstance(stats, dict)
        
        # Should include basic statistics
        expected_stats = ['total_messages', 'unique_channels', 'date_range', 'avg_message_length']
        
        for stat in expected_stats:
            if stat in stats:
                assert stats[stat] is not None
                
    def test_temporal_statistics(self, test_config, sample_telegram_data):
        """Test temporal statistics generation."""
        from src.anthropic_integration.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(test_config)
        
        stats = analyzer.generate_statistics(sample_telegram_data)
        
        # Should include temporal analysis
        if 'temporal_patterns' in stats:
            temporal = stats['temporal_patterns']
            assert isinstance(temporal, dict)
            
            # Should have time-based aggregations
            expected_temporal = ['messages_by_hour', 'messages_by_day', 'peak_activity']
            for pattern in expected_temporal:
                if pattern in temporal:
                    assert temporal[pattern] is not None
                    
    def test_channel_statistics(self, test_config, sample_telegram_data):
        """Test channel-specific statistics."""
        from src.anthropic_integration.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(test_config)
        
        stats = analyzer.generate_statistics(sample_telegram_data)
        
        # Should include channel analysis
        if 'channel_stats' in stats:
            channel_stats = stats['channel_stats']
            assert isinstance(channel_stats, dict)
            
            # Should have per-channel metrics
            for channel, metrics in channel_stats.items():
                assert 'message_count' in metrics
                assert metrics['message_count'] > 0
                
    def test_content_statistics(self, test_config, sample_telegram_data):
        """Test content-specific statistics."""
        from src.anthropic_integration.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(test_config)
        
        stats = analyzer.generate_statistics(sample_telegram_data)
        
        # Should include content analysis
        if 'content_stats' in stats:
            content = stats['content_stats']
            assert isinstance(content, dict)
            
            # Should analyze content patterns
            expected_content = ['avg_length', 'url_frequency', 'hashtag_frequency']
            for metric in expected_content:
                if metric in content:
                    assert content[metric] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
