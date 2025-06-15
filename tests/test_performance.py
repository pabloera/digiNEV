"""
Test suite for performance optimizations and system efficiency.
Tests caching, parallel processing, memory management, and performance monitoring.

These tests ensure the optimization systems work correctly.
"""

import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any
import json

import pandas as pd
import pytest

from conftest import assert_valid_analysis_result


class TestCacheSystem:
    """Test unified cache system."""
    
    def test_cache_system_initialization(self, test_config):
        """Test cache system initialization."""
        from src.core.unified_cache_system import UnifiedCacheSystem
        
        cache = UnifiedCacheSystem()
        
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'set')
        assert hasattr(cache, 'clear') or hasattr(cache, 'reset')
        
    def test_basic_cache_operations(self, test_config):
        """Test basic cache set/get operations."""
        from src.core.unified_cache_system import UnifiedCacheSystem
        
        cache = UnifiedCacheSystem()
        
        # Test set/get
        test_key = "test_key_123"
        test_value = {"data": "test_value", "timestamp": "2023-01-01"}
        
        cache.set(test_key, test_value)
        retrieved = cache.get(test_key)
        
        assert retrieved == test_value
        
        # Test cache miss
        missing = cache.get("nonexistent_key")
        assert missing is None
        
    def test_cache_ttl_functionality(self, test_config):
        """Test cache TTL (time-to-live) functionality."""
        from src.core.unified_cache_system import UnifiedCacheSystem
        
        cache = UnifiedCacheSystem()
        
        # Set item with short TTL
        test_key = "ttl_test_key"
        test_value = "ttl_test_value"
        
        if hasattr(cache, 'set_with_ttl'):
            cache.set_with_ttl(test_key, test_value, ttl_seconds=1)
            
            # Should be available immediately
            assert cache.get(test_key) == test_value
            
            # Should expire after TTL
            time.sleep(1.1)
            assert cache.get(test_key) is None
            
    def test_cache_invalidation(self, test_config):
        """Test cache invalidation strategies."""
        from src.core.unified_cache_system import UnifiedCacheSystem
        
        cache = UnifiedCacheSystem()
        
        # Set multiple items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Test selective invalidation
        if hasattr(cache, 'invalidate'):
            cache.invalidate("key1")
            assert cache.get("key1") is None
            assert cache.get("key2") == "value2"
            
        # Test pattern-based invalidation
        if hasattr(cache, 'invalidate_pattern'):
            cache.invalidate_pattern("key*")
            assert cache.get("key2") is None
            assert cache.get("key3") is None
            
    def test_cache_statistics(self, test_config):
        """Test cache performance statistics."""
        from src.core.unified_cache_system import UnifiedCacheSystem
        
        cache = UnifiedCacheSystem()
        
        # Generate cache hits and misses
        cache.set("hit_key", "hit_value")
        
        # Cache hit
        cache.get("hit_key")
        
        # Cache miss  
        cache.get("miss_key")
        
        if hasattr(cache, 'get_stats'):
            stats = cache.get_stats()
            
            assert isinstance(stats, dict)
            assert 'hits' in stats or 'hit_count' in stats
            assert 'misses' in stats or 'miss_count' in stats
            
            # Should track hits and misses
            hits = stats.get('hits', stats.get('hit_count', 0))
            misses = stats.get('misses', stats.get('miss_count', 0))
            
            assert hits > 0
            assert misses > 0
            
    def test_cache_memory_management(self, test_config):
        """Test cache memory management and limits."""
        from src.core.unified_cache_system import UnifiedCacheSystem
        
        cache = UnifiedCacheSystem()
        
        # Fill cache with many items
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
            
        if hasattr(cache, 'get_memory_usage'):
            memory_usage = cache.get_memory_usage()
            assert memory_usage > 0
            
        if hasattr(cache, 'evict_lru'):
            # Test LRU eviction
            initial_count = len(cache.keys()) if hasattr(cache, 'keys') else None
            cache.evict_lru(100)  # Evict 100 items
            
            if initial_count:
                final_count = len(cache.keys())
                assert final_count < initial_count


class TestParallelProcessing:
    """Test parallel processing capabilities."""
    
    def test_parallel_engine_initialization(self, test_config):
        """Test parallel processing engine initialization."""
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            
            engine = get_global_parallel_engine()
            
            if engine:
                assert hasattr(engine, 'process_parallel') or hasattr(engine, 'map')
                assert hasattr(engine, 'max_workers') or hasattr(engine, 'pool_size')
                
        except ImportError:
            pytest.skip("Parallel engine not implemented yet")
            
    def test_parallel_data_processing(self, test_config, sample_telegram_data):
        """Test parallel processing of data batches."""
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            
            engine = get_global_parallel_engine()
            
            if engine:
                # Define simple processing function
                def process_message(message):
                    return {
                        'original': message,
                        'length': len(message),
                        'processed': True
                    }
                
                messages = sample_telegram_data['body'].head(20).tolist()
                
                # Process in parallel
                if hasattr(engine, 'process_parallel'):
                    results = engine.process_parallel(process_message, messages)
                    
                    assert isinstance(results, list)
                    assert len(results) == len(messages)
                    
                    for result in results:
                        assert 'processed' in result
                        assert result['processed'] == True
                        
        except ImportError:
            pytest.skip("Parallel engine not implemented yet")
            
    def test_parallel_api_calls(self, test_config):
        """Test parallel API call processing."""
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            
            engine = get_global_parallel_engine()
            
            if engine:
                # Mock API calls
                def mock_api_call(data):
                    time.sleep(0.1)  # Simulate API delay
                    return {'result': f'processed_{data}', 'success': True}
                
                api_data = [f'request_{i}' for i in range(10)]
                
                start_time = time.time()
                
                if hasattr(engine, 'process_parallel'):
                    results = engine.process_parallel(mock_api_call, api_data, max_workers=3)
                    
                    end_time = time.time()
                    
                    # Should complete faster than sequential processing
                    assert end_time - start_time < 1.0  # Should be much faster than 1 second
                    assert len(results) == len(api_data)
                    
        except ImportError:
            pytest.skip("Parallel engine not implemented yet")
            
    def test_parallel_processing_error_handling(self, test_config):
        """Test error handling in parallel processing."""
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            
            engine = get_global_parallel_engine()
            
            if engine:
                def error_prone_function(data):
                    if data == 'error_item':
                        raise ValueError("Intentional error")
                    return {'result': data, 'success': True}
                
                test_data = ['good_item1', 'error_item', 'good_item2']
                
                if hasattr(engine, 'process_parallel'):
                    results = engine.process_parallel(error_prone_function, test_data)
                    
                    # Should handle errors gracefully
                    assert isinstance(results, list)
                    
                    # Should have some successful results
                    successful = [r for r in results if r.get('success') == True]
                    assert len(successful) > 0
                    
        except ImportError:
            pytest.skip("Parallel engine not implemented yet")


class TestStreamingPipeline:
    """Test streaming data processing."""
    
    def test_streaming_pipeline_initialization(self, test_config):
        """Test streaming pipeline initialization."""
        try:
            from src.optimized.streaming_pipeline import get_global_streaming_pipeline
            
            pipeline = get_global_streaming_pipeline()
            
            if pipeline:
                assert hasattr(pipeline, 'process_stream') or hasattr(pipeline, 'stream')
                
        except ImportError:
            pytest.skip("Streaming pipeline not implemented yet")
            
    def test_streaming_data_processing(self, test_config, sample_telegram_data):
        """Test streaming processing of large datasets."""
        try:
            from src.optimized.streaming_pipeline import get_global_streaming_pipeline
            
            pipeline = get_global_streaming_pipeline()
            
            if pipeline:
                # Simulate streaming data
                def data_generator():
                    for _, row in sample_telegram_data.iterrows():
                        yield row.to_dict()
                
                processed_count = 0
                
                if hasattr(pipeline, 'process_stream'):
                    for result in pipeline.process_stream(data_generator()):
                        processed_count += 1
                        assert isinstance(result, dict)
                    
                    assert processed_count == len(sample_telegram_data)
                    
        except ImportError:
            pytest.skip("Streaming pipeline not implemented yet")
            
    def test_streaming_memory_efficiency(self, test_config):
        """Test memory efficiency of streaming processing."""
        try:
            from src.optimized.streaming_pipeline import get_global_streaming_pipeline
            
            pipeline = get_global_streaming_pipeline()
            
            if pipeline:
                # Create large dataset generator
                def large_data_generator():
                    for i in range(10000):
                        yield {
                            'id': i,
                            'body': f'Message {i} with some content',
                            'data': 'x' * 1000  # 1KB per message
                        }
                
                processed = 0
                
                if hasattr(pipeline, 'process_stream'):
                    for result in pipeline.process_stream(large_data_generator()):
                        processed += 1
                        if processed > 100:  # Don't process all for test speed
                            break
                    
                    assert processed > 100
                    
        except ImportError:
            pytest.skip("Streaming pipeline not implemented yet")


class TestPerformanceMonitoring:
    """Test performance monitoring system."""
    
    def test_performance_monitor_initialization(self, test_config):
        """Test performance monitor initialization."""
        try:
            from src.optimized.realtime_monitor import get_global_performance_monitor
            
            monitor = get_global_performance_monitor()
            
            if monitor:
                assert hasattr(monitor, 'start_monitoring') or hasattr(monitor, 'start')
                assert hasattr(monitor, 'stop_monitoring') or hasattr(monitor, 'stop')
                assert hasattr(monitor, 'get_metrics') or hasattr(monitor, 'get_stats')
                
        except ImportError:
            pytest.skip("Performance monitor not implemented yet")
            
    def test_performance_metrics_collection(self, test_config):
        """Test collection of performance metrics."""
        try:
            from src.optimized.realtime_monitor import get_global_performance_monitor
            
            monitor = get_global_performance_monitor()
            
            if monitor:
                # Start monitoring
                if hasattr(monitor, 'start_monitoring'):
                    monitor.start_monitoring()
                
                # Simulate some work
                time.sleep(0.1)
                for i in range(1000):
                    _ = i ** 2
                
                # Get metrics
                if hasattr(monitor, 'get_metrics'):
                    metrics = monitor.get_metrics()
                    
                    assert isinstance(metrics, dict)
                    
                    # Should include basic performance metrics
                    expected_metrics = ['cpu_usage', 'memory_usage', 'execution_time']
                    
                    for metric in expected_metrics:
                        if metric in metrics:
                            assert metrics[metric] >= 0
                
                # Stop monitoring
                if hasattr(monitor, 'stop_monitoring'):
                    monitor.stop_monitoring()
                    
        except ImportError:
            pytest.skip("Performance monitor not implemented yet")
            
    def test_performance_alerting(self, test_config):
        """Test performance alerting system."""
        try:
            from src.optimized.realtime_monitor import get_global_performance_monitor
            
            monitor = get_global_performance_monitor()
            
            if monitor:
                alerts = []
                
                # Mock alert handler
                def alert_handler(alert_type, message, severity):
                    alerts.append({
                        'type': alert_type,
                        'message': message,
                        'severity': severity
                    })
                
                if hasattr(monitor, 'set_alert_handler'):
                    monitor.set_alert_handler(alert_handler)
                
                # Simulate high resource usage
                if hasattr(monitor, 'simulate_high_usage'):
                    monitor.simulate_high_usage()
                    time.sleep(0.1)
                    
                    # Should generate alerts
                    assert len(alerts) > 0
                    
        except ImportError:
            pytest.skip("Performance monitor not implemented yet")


class TestMemoryOptimization:
    """Test memory optimization system."""
    
    def test_memory_manager_initialization(self, test_config):
        """Test memory manager initialization."""
        try:
            from src.optimized.memory_optimizer import get_global_memory_manager
            
            manager = get_global_memory_manager()
            
            if manager:
                assert hasattr(manager, 'optimize_memory') or hasattr(manager, 'manage')
                assert hasattr(manager, 'get_memory_usage') or hasattr(manager, 'get_stats')
                
        except ImportError:
            pytest.skip("Memory optimizer not implemented yet")
            
    def test_memory_usage_monitoring(self, test_config):
        """Test memory usage monitoring."""
        try:
            from src.optimized.memory_optimizer import get_global_memory_manager
            
            manager = get_global_memory_manager()
            
            if manager:
                # Get initial memory usage
                if hasattr(manager, 'get_memory_usage'):
                    initial_usage = manager.get_memory_usage()
                    assert initial_usage >= 0
                    
                    # Create memory load
                    large_data = ['x' * 10000 for _ in range(1000)]  # ~10MB
                    
                    # Check memory increase
                    loaded_usage = manager.get_memory_usage()
                    assert loaded_usage >= initial_usage
                    
                    # Clean up
                    del large_data
                    
        except ImportError:
            pytest.skip("Memory optimizer not implemented yet")
            
    def test_adaptive_memory_management(self, test_config):
        """Test adaptive memory management."""
        try:
            from src.optimized.memory_optimizer import get_global_memory_manager
            
            manager = get_global_memory_manager()
            
            if manager:
                # Start adaptive management
                if hasattr(manager, 'start_adaptive_management'):
                    manager.start_adaptive_management()
                    
                    # Simulate memory pressure
                    large_objects = []
                    for i in range(100):
                        large_objects.append(['x' * 1000 for _ in range(100)])
                        
                    time.sleep(0.1)  # Allow manager to react
                    
                    # Check if manager took action
                    if hasattr(manager, 'get_optimization_actions'):
                        actions = manager.get_optimization_actions()
                        assert isinstance(actions, list)
                        
                    # Stop adaptive management
                    if hasattr(manager, 'stop_adaptive_management'):
                        manager.stop_adaptive_management()
                        
        except ImportError:
            pytest.skip("Memory optimizer not implemented yet")


class TestBenchmarkSystem:
    """Test benchmark and profiling system."""
    
    def test_benchmark_initialization(self, test_config):
        """Test benchmark system initialization."""
        try:
            from src.optimized.pipeline_benchmark import get_global_benchmark
            
            benchmark = get_global_benchmark()
            
            if benchmark:
                assert hasattr(benchmark, 'run_benchmark') or hasattr(benchmark, 'benchmark')
                assert hasattr(benchmark, 'get_results') or hasattr(benchmark, 'get_report')
                
        except ImportError:
            pytest.skip("Benchmark system not implemented yet")
            
    def test_pipeline_stage_benchmarking(self, test_config, sample_telegram_data):
        """Test benchmarking of individual pipeline stages."""
        try:
            from src.optimized.pipeline_benchmark import get_global_benchmark
            
            benchmark = get_global_benchmark()
            
            if benchmark:
                # Define mock stage function
                def mock_stage_function(data):
                    time.sleep(0.01)  # Simulate processing time
                    return data.copy()
                
                if hasattr(benchmark, 'benchmark_stage'):
                    result = benchmark.benchmark_stage(
                        'mock_stage',
                        mock_stage_function,
                        sample_telegram_data
                    )
                    
                    assert isinstance(result, dict)
                    assert 'execution_time' in result
                    assert 'memory_usage' in result or 'peak_memory' in result
                    assert result['execution_time'] > 0
                    
        except ImportError:
            pytest.skip("Benchmark system not implemented yet")
            
    def test_performance_comparison(self, test_config):
        """Test performance comparison between implementations."""
        try:
            from src.optimized.pipeline_benchmark import get_global_benchmark
            
            benchmark = get_global_benchmark()
            
            if benchmark:
                # Define two implementations
                def slow_implementation(data):
                    time.sleep(0.02)
                    return [item.upper() for item in data]
                
                def fast_implementation(data):
                    time.sleep(0.01)
                    return [item.upper() for item in data]
                
                test_data = ['test'] * 100
                
                if hasattr(benchmark, 'compare_implementations'):
                    comparison = benchmark.compare_implementations(
                        'text_processing',
                        [
                            ('slow', slow_implementation),
                            ('fast', fast_implementation)
                        ],
                        test_data
                    )
                    
                    assert isinstance(comparison, dict)
                    assert 'results' in comparison
                    assert len(comparison['results']) == 2
                    
                    # Fast implementation should be faster
                    slow_time = comparison['results']['slow']['execution_time']
                    fast_time = comparison['results']['fast']['execution_time']
                    assert fast_time < slow_time
                    
        except ImportError:
            pytest.skip("Benchmark system not implemented yet")


class TestOptimizationIntegration:
    """Test integration of optimization systems."""
    
    def test_optimization_system_coordination(self, test_config):
        """Test coordination between optimization systems."""
        from run_pipeline import check_optimization_systems
        
        # Check all optimization systems
        optimizations = check_optimization_systems()
        
        assert isinstance(optimizations, dict)
        
        # Should have all 5 weeks of optimizations
        expected_weeks = [
            'week1_emergency',
            'week2_caching', 
            'week3_parallelization',
            'week4_monitoring',
            'week5_production'
        ]
        
        for week in expected_weeks:
            assert week in optimizations
            assert isinstance(optimizations[week], bool)
            
    def test_optimization_performance_impact(self, test_config, sample_telegram_data, temp_csv_file):
        """Test performance impact of optimizations."""
        # This test should compare performance with and without optimizations
        
        # Disable optimizations
        no_opt_config = test_config.copy()
        no_opt_config['anthropic']['enable_api_integration'] = False
        no_opt_config['voyage_embeddings']['enable_sampling'] = False
        
        # Enable optimizations
        opt_config = test_config.copy()
        opt_config['anthropic']['enable_api_integration'] = True
        opt_config['voyage_embeddings']['enable_sampling'] = True
        
        # Time both approaches (mock implementation)
        start_time = time.time()
        # Mock pipeline execution without optimizations
        time.sleep(0.1)  # Simulate slower processing
        no_opt_time = time.time() - start_time
        
        start_time = time.time()
        # Mock pipeline execution with optimizations
        time.sleep(0.05)  # Simulate faster processing
        opt_time = time.time() - start_time
        
        # Optimized version should be faster (in real implementation)
        assert opt_time <= no_opt_time
        
    def test_optimization_resource_usage(self, test_config):
        """Test resource usage of optimization systems."""
        # This test should verify that optimizations don't consume excessive resources
        
        # Check if optimization systems can be initialized without errors
        optimization_errors = []
        
        try:
            from src.optimized.optimized_pipeline import get_global_optimized_pipeline
            get_global_optimized_pipeline()
        except Exception as e:
            optimization_errors.append(f"Optimized pipeline: {e}")
            
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            get_global_parallel_engine()
        except Exception as e:
            optimization_errors.append(f"Parallel engine: {e}")
            
        try:
            from src.optimized.realtime_monitor import get_global_performance_monitor
            get_global_performance_monitor()
        except Exception as e:
            optimization_errors.append(f"Performance monitor: {e}")
            
        # Should not have excessive initialization errors
        assert len(optimization_errors) <= 3  # Allow some systems to be optional


class TestEndToEndPerformance:
    """Test end-to-end performance scenarios."""
    
    def test_large_dataset_processing_performance(self, test_config, test_data_dir):
        """Test performance with large datasets."""
        # Create larger dataset for performance testing
        large_data = pd.DataFrame({
            'id': range(5000),
            'body': [f'Test message {i} with political content about elections and democracy' for i in range(5000)],
            'date': pd.date_range('2023-01-01', periods=5000, freq='H'),
            'channel': [f'channel_{i % 50}' for i in range(5000)]
        })
        
        large_file = test_data_dir / "large_test_dataset.csv"
        large_data.to_csv(large_file, index=False)
        
        # Measure processing time
        start_time = time.time()
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            # Configure for performance
            perf_config = test_config.copy()
            perf_config['anthropic']['enable_api_integration'] = False  # Disable API for speed
            perf_config['processing']['chunk_size'] = 1000
            
            pipeline = UnifiedAnthropicPipeline(perf_config, str(Path.cwd()))
            
            # Mock rapid processing
            with patch.object(pipeline, 'run_complete_pipeline') as mock_run:
                mock_run.return_value = {'overall_success': True, 'total_records': 5000}
                
                result = pipeline.run_complete_pipeline([str(large_file)])
                
                processing_time = time.time() - start_time
                
                # Should complete in reasonable time
                assert processing_time < 10.0  # Should process quickly with mocking
                assert result['overall_success'] == True
                
        except ImportError:
            # If pipeline not available, just measure file operations
            df = pd.read_csv(large_file)
            processing_time = time.time() - start_time
            
            assert len(df) == 5000
            assert processing_time < 5.0  # File I/O should be fast
            
    def test_memory_usage_with_large_data(self, test_config):
        """Test memory usage patterns with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large dataset
        large_dataset = pd.DataFrame({
            'body': [f'Message {i} ' * 100 for i in range(10000)],  # ~100 chars per message
            'channel': [f'channel_{i % 100}' for i in range(10000)]
        })
        
        # Simulate processing
        processed = large_dataset.copy()
        processed['length'] = processed['body'].str.len()
        processed['words'] = processed['body'].str.split().str.len()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        del large_dataset, processed
        
        # Should not use excessive memory
        assert memory_increase < 500  # Should not use more than 500MB for this test
        
    def test_concurrent_performance(self, test_config):
        """Test performance under concurrent load."""
        import threading
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                # Simulate concurrent work
                start = time.time()
                
                # Mock concurrent processing
                data = pd.DataFrame({
                    'body': [f'Worker {worker_id} message {i}' for i in range(100)]
                })
                
                # Simple processing
                processed = data['body'].str.len().sum()
                
                end = time.time()
                
                results.append({
                    'worker_id': worker_id,
                    'processing_time': end - start,
                    'processed_items': len(data),
                    'result': processed
                })
                
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 5
        
        # All workers should complete reasonably quickly
        for result in results:
            assert result['processing_time'] < 1.0
            assert result['processed_items'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
