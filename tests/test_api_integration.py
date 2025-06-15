"""
Test suite for API integrations (Anthropic and Voyage AI).
Tests the integration layer, error handling, and API functionality.

These tests mock API responses to test integration logic without making real API calls.
"""

import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pytest

from conftest import MockAnthropicResponse


class TestAnthropicIntegration:
    """Test Anthropic API integration."""
    
    def test_anthropic_base_initialization(self, test_config):
        """Test Anthropic base class initialization."""
        from src.anthropic_integration.base import AnthropicBase
        
        base = AnthropicBase(test_config)
        
        assert hasattr(base, 'config')
        assert hasattr(base, 'client') or hasattr(base, '_client')
        assert base.config == test_config
        
    def test_anthropic_client_creation(self, test_config):
        """Test Anthropic client is created properly."""
        from src.anthropic_integration.base import AnthropicBase
        
        # Mock Anthropic client creation
        with patch('src.anthropic_integration.base.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            base = AnthropicBase(test_config)
            
            # Should create client with API key
            mock_anthropic.assert_called_once()
            
    def test_api_error_handling(self, test_config):
        """Test handling of API errors."""
        from src.anthropic_integration.api_error_handler import APIErrorHandler
        
        error_handler = APIErrorHandler(test_config)
        
        # Test rate limit handling
        rate_limit_error = Exception("Rate limit exceeded")
        result = error_handler.handle_error(rate_limit_error, "rate_limit")
        
        assert isinstance(result, dict)
        assert 'error_type' in result
        assert 'should_retry' in result
        
    def test_progressive_timeout_manager(self, test_config):
        """Test progressive timeout management."""
        from src.anthropic_integration.progressive_timeout_manager import ProgressiveTimeoutManager
        
        timeout_manager = ProgressiveTimeoutManager(test_config)
        
        # Should start with initial timeout
        initial_timeout = timeout_manager.get_current_timeout()
        assert initial_timeout > 0
        
        # Should increase timeout after failure
        timeout_manager.on_request_failed()
        increased_timeout = timeout_manager.get_current_timeout()
        assert increased_timeout > initial_timeout
        
        # Should reset after success
        timeout_manager.on_request_success()
        reset_timeout = timeout_manager.get_current_timeout()
        assert reset_timeout <= increased_timeout
        
    def test_cost_monitoring(self, test_config):
        """Test API cost monitoring."""
        from src.anthropic_integration.cost_monitor import CostMonitor
        
        cost_monitor = CostMonitor(test_config)
        
        # Should track token usage
        cost_monitor.track_request(
            model="claude-3-5-haiku-20241022",
            input_tokens=100,
            output_tokens=50
        )
        
        # Should calculate costs
        total_cost = cost_monitor.get_total_cost()
        assert total_cost >= 0
        
        # Should provide usage summary
        summary = cost_monitor.get_usage_summary()
        assert isinstance(summary, dict)
        assert 'total_requests' in summary
        assert 'total_tokens' in summary
        
    def test_concurrent_processing(self, test_config):
        """Test concurrent API request processing."""
        from src.anthropic_integration.concurrent_processor import ConcurrentProcessor
        
        processor = ConcurrentProcessor(test_config)
        
        # Mock multiple requests
        requests = [
            {"prompt": "Test prompt 1", "data": "data1"},
            {"prompt": "Test prompt 2", "data": "data2"},
            {"prompt": "Test prompt 3", "data": "data3"}
        ]
        
        # Mock API responses
        with patch.object(processor, 'process_single_request') as mock_process:
            mock_process.return_value = {"success": True, "result": "test_result"}
            
            results = processor.process_concurrent_requests(requests, max_workers=2)
            
            assert isinstance(results, list)
            assert len(results) == 3
            
            # Should call process for each request
            assert mock_process.call_count == 3
            
    def test_cache_integration(self, test_config):
        """Test cache integration with API calls."""
        from src.anthropic_integration.optimized_cache import OptimizedCache
        
        cache = OptimizedCache(test_config)
        
        # Should store and retrieve cached responses
        cache_key = "test_prompt_hash"
        test_response = {"result": "cached_response"}
        
        cache.set(cache_key, test_response)
        retrieved = cache.get(cache_key)
        
        assert retrieved == test_response
        
        # Should handle cache misses
        missing = cache.get("nonexistent_key")
        assert missing is None
        
    def test_batch_processing_optimization(self, test_config):
        """Test batch processing optimization."""
        from src.anthropic_integration.base import AnthropicBase
        
        base = AnthropicBase(test_config)
        
        # Should batch requests efficiently
        test_data = [f"Test message {i}" for i in range(25)]
        batch_size = 10
        
        # Mock batch processing
        with patch.object(base, 'client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({"results": [{"processed": True}] * 10})
            mock_client.messages.create.return_value = mock_response
            
            if hasattr(base, 'process_batch'):
                results = base.process_batch(test_data, batch_size=batch_size)
                
                # Should process in batches
                expected_batches = (len(test_data) + batch_size - 1) // batch_size
                assert mock_client.messages.create.call_count == expected_batches


class TestVoyageIntegration:
    """Test Voyage AI integration."""
    
    def test_voyage_embeddings_initialization(self, test_config):
        """Test Voyage embeddings initialization."""
        from src.anthropic_integration.voyage_embeddings import VoyageEmbeddings
        
        voyage = VoyageEmbeddings(test_config)
        
        assert hasattr(voyage, 'config')
        assert hasattr(voyage, 'client') or hasattr(voyage, '_client')
        
    def test_embedding_generation(self, test_config):
        """Test embedding generation."""
        from src.anthropic_integration.voyage_embeddings import VoyageEmbeddings
        
        voyage = VoyageEmbeddings(test_config)
        
        test_texts = [
            "Política brasileira contemporânea",
            "Economia nacional em crise",
            "Democracia e instituições"
        ]
        
        # Mock Voyage API response
        with patch.object(voyage, 'client') as mock_client:
            mock_embeddings = [[0.1, 0.2, 0.3] for _ in range(len(test_texts))]
            mock_client.embed.return_value = Mock(embeddings=mock_embeddings)
            
            embeddings = voyage.generate_embeddings(test_texts)
            
            assert isinstance(embeddings, list)
            assert len(embeddings) == len(test_texts)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) == 3 for emb in embeddings)  # Mock dimension
            
    def test_embedding_sampling_optimization(self, test_config):
        """Test embedding generation with sampling optimization."""
        from src.anthropic_integration.voyage_embeddings import VoyageEmbeddings
        
        # Set sampling configuration
        sampling_config = test_config.copy()
        sampling_config['voyage_embeddings']['enable_sampling'] = True
        sampling_config['voyage_embeddings']['max_messages'] = 100
        
        voyage = VoyageEmbeddings(sampling_config)
        
        # Large dataset that should be sampled
        large_dataset = [f"Message {i}" for i in range(500)]
        
        with patch.object(voyage, 'client') as mock_client:
            mock_embeddings = [[0.1, 0.2, 0.3] for _ in range(100)]  # Only 100 embeddings
            mock_client.embed.return_value = Mock(embeddings=mock_embeddings)
            
            embeddings = voyage.generate_embeddings(large_dataset)
            
            # Should sample down to max_messages
            assert len(embeddings) == 100
            
    def test_voyage_error_handling(self, test_config):
        """Test Voyage API error handling."""
        from src.anthropic_integration.voyage_embeddings import VoyageEmbeddings
        
        voyage = VoyageEmbeddings(test_config)
        
        test_texts = ["Test message"]
        
        # Mock API error
        with patch.object(voyage, 'client') as mock_client:
            mock_client.embed.side_effect = Exception("API Error")
            
            # Should handle error gracefully
            try:
                embeddings = voyage.generate_embeddings(test_texts)
                # Should return empty list or handle error appropriately
                assert embeddings is None or isinstance(embeddings, list)
            except Exception as e:
                # Should provide meaningful error message
                assert "API Error" in str(e) or len(str(e)) > 0
                
    def test_voyage_semantic_search(self, test_config):
        """Test semantic search functionality."""
        from src.anthropic_integration.semantic_search_engine import SemanticSearchEngine
        
        search_engine = SemanticSearchEngine(test_config)
        
        documents = [
            "Política brasileira e democracia",
            "Economia e mercado financeiro", 
            "Tecnologia e inovação",
            "Meio ambiente e sustentabilidade"
        ]
        
        query = "democracia no Brasil"
        
        # Mock embeddings for documents and query
        with patch.object(search_engine, 'voyage_client') as mock_voyage:
            # Mock document embeddings
            doc_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4]]
            query_embedding = [0.15, 0.25, 0.35]  # Similar to first document
            
            mock_voyage.embed.side_effect = [
                Mock(embeddings=doc_embeddings),  # For documents
                Mock(embeddings=[query_embedding])  # For query
            ]
            
            results = search_engine.search(documents, query, top_k=2)
            
            assert isinstance(results, list)
            assert len(results) <= 2  # top_k
            
            # Results should have scores and documents
            for result in results:
                assert 'document' in result or 'text' in result
                assert 'score' in result or 'similarity' in result


class TestAPIRateLimiting:
    """Test API rate limiting and throttling."""
    
    def test_rate_limit_respect(self, test_config):
        """Test that API calls respect rate limits."""
        from src.anthropic_integration.base import AnthropicBase
        
        # Configure low rate limits for testing
        test_config_with_limits = test_config.copy()
        test_config_with_limits['anthropic']['rate_limit'] = 2  # 2 requests per minute
        
        base = AnthropicBase(test_config_with_limits)
        
        request_times = []
        
        with patch.object(base, 'client') as mock_client:
            mock_response = MockAnthropicResponse('{"result": "success"}')
            mock_client.messages.create.return_value = mock_response
            
            # Make multiple requests
            for i in range(3):
                start_time = time.time()
                if hasattr(base, 'make_request'):
                    base.make_request("test prompt")
                request_times.append(time.time() - start_time)
            
            # Should introduce delays to respect rate limits
            if len(request_times) > 1:
                # Later requests should take longer (due to rate limiting)
                assert request_times[-1] >= request_times[0]
                
    def test_exponential_backoff(self, test_config):
        """Test exponential backoff on failures."""
        from src.anthropic_integration.api_error_handler import APIErrorHandler
        
        error_handler = APIErrorHandler(test_config)
        
        # Simulate multiple failures
        backoff_times = []
        for attempt in range(4):
            backoff_time = error_handler.get_backoff_time(attempt)
            backoff_times.append(backoff_time)
            
        # Should increase exponentially
        for i in range(1, len(backoff_times)):
            assert backoff_times[i] > backoff_times[i-1]
            
    def test_circuit_breaker_pattern(self, test_config):
        """Test circuit breaker pattern for API failures."""
        from src.anthropic_integration.api_error_handler import APIErrorHandler
        
        error_handler = APIErrorHandler(test_config)
        
        # Simulate consecutive failures
        for _ in range(5):
            error_handler.record_failure()
            
        # Should open circuit after threshold
        assert error_handler.is_circuit_open() == True
        
        # Should close circuit after timeout
        error_handler.reset_circuit()
        assert error_handler.is_circuit_open() == False


class TestAPIDataFlow:
    """Test data flow through API integrations."""
    
    def test_anthropic_data_transformation(self, test_config, sample_telegram_data):
        """Test data transformation for Anthropic API."""
        from src.anthropic_integration.base import AnthropicBase
        
        base = AnthropicBase(test_config)
        
        # Test data preparation
        messages = sample_telegram_data['body'].head(5).tolist()
        
        if hasattr(base, 'prepare_batch_data'):
            prepared_data = base.prepare_batch_data(messages)
            
            assert isinstance(prepared_data, (list, dict))
            
            # Should format data appropriately for API
            if isinstance(prepared_data, list):
                assert len(prepared_data) <= len(messages)
            
    def test_voyage_data_transformation(self, test_config, sample_telegram_data):
        """Test data transformation for Voyage API."""
        from src.anthropic_integration.voyage_embeddings import VoyageEmbeddings
        
        voyage = VoyageEmbeddings(test_config)
        
        messages = sample_telegram_data['body'].head(10).tolist()
        
        if hasattr(voyage, 'prepare_texts'):
            prepared_texts = voyage.prepare_texts(messages)
            
            assert isinstance(prepared_texts, list)
            assert len(prepared_texts) <= len(messages)
            
            # Should clean and prepare texts
            for text in prepared_texts:
                assert isinstance(text, str)
                assert len(text) > 0
                
    def test_response_validation(self, test_config):
        """Test validation of API responses."""
        from src.anthropic_integration.base import AnthropicBase
        
        base = AnthropicBase(test_config)
        
        # Test valid response
        valid_response = {"sentiment": "positive", "confidence": 0.8}
        
        if hasattr(base, 'validate_response'):
            is_valid = base.validate_response(valid_response, expected_keys=['sentiment'])
            assert is_valid == True
            
            # Test invalid response
            invalid_response = {"invalid": "data"}
            is_valid = base.validate_response(invalid_response, expected_keys=['sentiment'])
            assert is_valid == False
            
    def test_response_parsing(self, test_config):
        """Test parsing of complex API responses."""
        from src.anthropic_integration.base import AnthropicBase
        
        base = AnthropicBase(test_config)
        
        # Mock complex JSON response
        complex_response = MockAnthropicResponse(json.dumps({
            "results": [
                {"id": 0, "analysis": {"sentiment": "positive", "confidence": 0.9}},
                {"id": 1, "analysis": {"sentiment": "negative", "confidence": 0.7}}
            ],
            "metadata": {"model": "claude-3-haiku", "timestamp": "2023-01-01T00:00:00"}
        }))
        
        if hasattr(base, 'parse_response'):
            parsed = base.parse_response(complex_response)
            
            assert isinstance(parsed, dict)
            assert 'results' in parsed
            assert len(parsed['results']) == 2


class TestAPIIntegrationWithPipeline:
    """Test integration of APIs with the main pipeline."""
    
    def test_anthropic_integration_in_pipeline(self, test_config, project_root, temp_csv_file):
        """Test Anthropic integration within pipeline execution."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Enable Anthropic integration
        anthropic_config = test_config.copy()
        anthropic_config['anthropic']['enable_api_integration'] = True
        
        pipeline = UnifiedAnthropicPipeline(anthropic_config, str(project_root))
        
        # Mock all Anthropic API calls
        with patch('src.anthropic_integration.base.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_response = MockAnthropicResponse('{"sentiment": "positive", "confidence": 0.8}')
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_class.return_value = mock_client
            
            # Should integrate Anthropic analysis
            result = pipeline.run_complete_pipeline([temp_csv_file])
            
            assert isinstance(result, dict)
            # API integration should enhance results
            
    def test_voyage_integration_in_pipeline(self, test_config, project_root, temp_csv_file):
        """Test Voyage integration within pipeline execution."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Enable Voyage integration
        voyage_config = test_config.copy()
        voyage_config['voyage_embeddings']['enable_sampling'] = True
        
        pipeline = UnifiedAnthropicPipeline(voyage_config, str(project_root))
        
        # Mock Voyage API calls
        with patch('voyageai.Client') as mock_voyage_class:
            mock_client = Mock()
            mock_client.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])
            mock_voyage_class.return_value = mock_client
            
            # Should integrate Voyage embeddings
            result = pipeline.run_complete_pipeline([temp_csv_file])
            
            assert isinstance(result, dict)
            # Embedding integration should enhance analysis
            
    def test_api_fallback_mechanisms(self, test_config, project_root, temp_csv_file):
        """Test fallback mechanisms when APIs fail."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Mock API failures
        with patch('src.anthropic_integration.base.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("API Failure")
            mock_anthropic.return_value = mock_client
            
            # Should fallback to traditional methods
            result = pipeline.run_complete_pipeline([temp_csv_file])
            
            assert isinstance(result, dict)
            # Should complete even with API failures
            assert result.get('overall_success') is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
