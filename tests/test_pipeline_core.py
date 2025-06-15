"""
Test suite for the core pipeline functionality.
Tests the main pipeline execution, stage management, and data flow.

This follows TDD principles - tests are written first to define expected behavior.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pandas as pd
import pytest

from conftest import (
    assert_dataframe_columns, 
    assert_valid_analysis_result,
    MockAnthropicResponse
)


class TestPipelineCore:
    """Test core pipeline functionality."""
    
    def test_pipeline_initialization(self, test_config, project_root):
        """Test that pipeline can be initialized with valid configuration."""
        # This test should fail initially since we haven't implemented the full pipeline yet
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Should initialize without errors
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Should have basic attributes
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'project_root')
        assert pipeline.config == test_config
        
        # Should have stage definitions
        assert hasattr(pipeline, 'stages') or hasattr(pipeline, '_stages')
        
    def test_pipeline_stage_count(self, test_config, project_root):
        """Test that pipeline has correct number of stages (22)."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Get stages (implementation may vary)
        if hasattr(pipeline, 'stages'):
            stages = pipeline.stages
        elif hasattr(pipeline, 'get_all_stages'):
            stages = pipeline.get_all_stages()
        else:
            # Fallback - check for stage methods
            stage_methods = [method for method in dir(pipeline) 
                           if method.startswith('stage_') or 'stage' in method.lower()]
            stages = stage_methods
        
        # Should have 22 stages
        assert len(stages) == 22, f"Expected 22 stages, got {len(stages)}"
    
    def test_pipeline_can_process_sample_data(self, test_config, project_root, sample_telegram_data, temp_csv_file):
        """Test that pipeline can process sample data without errors."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Should be able to process data
        result = pipeline.run_complete_pipeline([temp_csv_file])
        
        # Should return valid results
        assert isinstance(result, dict)
        assert 'overall_success' in result
        assert 'stage_results' in result or 'stages_completed' in result
        
    def test_pipeline_handles_empty_data(self, test_config, project_root, test_data_dir):
        """Test pipeline behavior with empty dataset."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Create empty CSV
        empty_df = pd.DataFrame(columns=['id', 'body', 'date', 'channel'])
        empty_file = test_data_dir / "empty.csv"
        empty_df.to_csv(empty_file, index=False)
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Should handle empty data gracefully
        result = pipeline.run_complete_pipeline([str(empty_file)])
        
        # Should not crash and return meaningful result
        assert isinstance(result, dict)
        assert 'error' in result or 'overall_success' in result
        
    def test_pipeline_stage_execution_order(self, test_config, project_root, temp_csv_file):
        """Test that pipeline stages execute in correct order."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Track stage execution order
        executed_stages = []
        
        # Mock stage execution to track order
        original_method = getattr(pipeline, 'execute_stage', None)
        if original_method:
            def track_execution(stage_id, *args, **kwargs):
                executed_stages.append(stage_id)
                return original_method(stage_id, *args, **kwargs)
            
            pipeline.execute_stage = track_execution
        
        # Run pipeline
        pipeline.run_complete_pipeline([temp_csv_file])
        
        # Check execution order
        expected_early_stages = ['01_chunk_processing', '02_encoding_validation', '03_deduplication']
        expected_late_stages = ['20_pipeline_validation']
        
        if executed_stages:
            # Early stages should come first
            for early_stage in expected_early_stages:
                if early_stage in executed_stages:
                    early_index = executed_stages.index(early_stage)
                    for late_stage in expected_late_stages:
                        if late_stage in executed_stages:
                            late_index = executed_stages.index(late_stage)
                            assert early_index < late_index, f"{early_stage} should execute before {late_stage}"


class TestStageManagement:
    """Test stage management functionality."""
    
    def test_checkpoint_loading(self, mock_checkpoints, project_root):
        """Test checkpoint loading functionality."""
        # This should be implemented in run_pipeline.py
        from run_pipeline import load_checkpoints
        
        # Mock checkpoint file
        checkpoint_file = project_root / "checkpoints" / "checkpoints.json"
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(mock_checkpoints, f)
        
        # Should load checkpoints correctly
        checkpoints = load_checkpoints()
        
        assert checkpoints is not None
        assert checkpoints['execution_summary']['completed_stages'] == 5
        assert checkpoints['execution_summary']['total_stages'] == 22
        
    def test_protection_checklist_loading(self, mock_protection_checklist, project_root):
        """Test protection checklist loading."""
        from run_pipeline import load_protection_checklist
        
        # Mock checklist file
        checklist_file = project_root / "checkpoints" / "checklist.json"
        checklist_file.parent.mkdir(exist_ok=True)
        
        with open(checklist_file, 'w') as f:
            json.dump(mock_protection_checklist, f)
        
        # Should load checklist correctly
        checklist = load_protection_checklist()
        
        assert checklist is not None
        assert checklist['statistics']['total_stages'] == 22
        assert checklist['statistics']['locked_stages'] == 1
        
    def test_stage_protection_check(self, mock_protection_checklist, project_root):
        """Test stage protection checking."""
        from run_pipeline import check_stage_protection
        
        # Test protected stage
        result = check_stage_protection('01_chunk_processing', mock_protection_checklist)
        assert result['can_overwrite'] == False
        assert result['protection_level'] == 'high'
        
        # Test locked stage
        result = check_stage_protection('02_encoding_validation', mock_protection_checklist)
        assert result['can_overwrite'] == False
        assert result['requires_override'] == True
        
    def test_resume_point_determination(self, mock_checkpoints):
        """Test resume point determination."""
        from run_pipeline import get_resume_point
        
        # Should return correct resume point
        resume_point = get_resume_point(mock_checkpoints)
        assert resume_point == '06_text_cleaning'
        
        # Should handle no checkpoints
        resume_point = get_resume_point(None)
        assert resume_point == '01_chunk_processing'
        
    def test_stage_skipping_logic(self, mock_checkpoints, mock_protection_checklist):
        """Test logic for skipping completed/protected stages."""
        from run_pipeline import should_skip_stage, should_skip_protected_stage
        
        # Should skip completed stages
        should_skip = should_skip_stage('01_chunk_processing', mock_checkpoints)
        assert should_skip == True
        
        # Should skip protected stages
        should_skip = should_skip_protected_stage('01_chunk_processing', mock_protection_checklist)
        assert should_skip == True
        
        # Should not skip uncompleted stages
        should_skip = should_skip_stage('06_text_cleaning', mock_checkpoints)
        assert should_skip == False


class TestDataProcessing:
    """Test data processing functionality."""
    
    def test_dataset_discovery(self, test_data_dir, sample_telegram_data):
        """Test dataset discovery functionality."""
        from run_pipeline import discover_datasets
        
        # Create test dataset
        test_file = test_data_dir / "test_dataset.csv"
        sample_telegram_data.to_csv(test_file, index=False)
        
        # Should discover datasets
        datasets = discover_datasets([str(test_data_dir)])
        
        assert len(datasets) > 0
        assert str(test_file) in datasets
        
    def test_dataset_validation(self, test_data_dir):
        """Test dataset validation (size, format)."""
        from run_pipeline import discover_datasets
        
        # Create empty file (should be ignored)
        empty_file = test_data_dir / "empty.csv"
        empty_file.touch()
        
        # Create valid file
        valid_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        valid_file = test_data_dir / "valid.csv"
        valid_df.to_csv(valid_file, index=False)
        
        datasets = discover_datasets([str(test_data_dir)])
        
        # Should only include valid datasets
        assert str(valid_file) in datasets
        assert str(empty_file) not in datasets
        
    def test_configuration_loading(self):
        """Test configuration loading from YAML files."""
        from run_pipeline import load_configuration
        
        # Should load configuration (with defaults if files not found)
        config = load_configuration()
        
        assert isinstance(config, dict)
        assert 'anthropic' in config
        assert 'processing' in config
        assert 'data' in config


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_pipeline_handles_corrupted_data(self, test_config, project_root, test_data_dir):
        """Test pipeline behavior with corrupted data."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Create corrupted CSV
        corrupted_file = test_data_dir / "corrupted.csv"
        with open(corrupted_file, 'w') as f:
            f.write("invalid,csv,data\n")
            f.write("missing,columns\n")
            f.write("inconsistent,row,count,extra\n")
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Should handle corrupted data gracefully
        result = pipeline.run_complete_pipeline([str(corrupted_file)])
        
        # Should not crash
        assert isinstance(result, dict)
        
    def test_pipeline_handles_missing_files(self, test_config, project_root):
        """Test pipeline behavior with missing files."""
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
        
        # Should handle missing files gracefully
        result = pipeline.run_complete_pipeline(["/nonexistent/file.csv"])
        
        # Should not crash and indicate error
        assert isinstance(result, dict)
        assert 'error' in result or result.get('overall_success') == False
        
    def test_pipeline_handles_memory_constraints(self, test_config, project_root, test_data_dir):
        """Test pipeline behavior with memory constraints."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'body': [f'Test message {i}' for i in range(10000)],
            'date': pd.date_range('2023-01-01', periods=10000, freq='H'),
            'channel': ['test_channel'] * 10000
        })
        
        large_file = test_data_dir / "large_dataset.csv"
        large_data.to_csv(large_file, index=False)
        
        # Set low memory limit in config
        low_memory_config = test_config.copy()
        low_memory_config['processing']['memory_limit'] = '100MB'
        low_memory_config['processing']['chunk_size'] = 100
        
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        pipeline = UnifiedAnthropicPipeline(low_memory_config, str(project_root))
        
        # Should handle large data with chunking
        result = pipeline.run_complete_pipeline([str(large_file)])
        
        # Should complete without memory errors
        assert isinstance(result, dict)


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_optimization_systems_detection(self):
        """Test detection of optimization systems."""
        from run_pipeline import check_optimization_systems
        
        # Should detect available optimization systems
        optimizations = check_optimization_systems()
        
        assert isinstance(optimizations, dict)
        assert 'week1_emergency' in optimizations
        assert 'week3_parallelization' in optimizations
        assert 'week4_monitoring' in optimizations
        assert 'week5_production' in optimizations
        
        # All values should be boolean
        for key, value in optimizations.items():
            assert isinstance(value, bool)
            
    def test_cache_system_functionality(self, test_config, project_root):
        """Test cache system functionality."""
        # Test should verify cache operations
        try:
            from src.core.unified_cache_system import UnifiedCacheSystem
            
            cache_system = UnifiedCacheSystem()
            
            # Should be able to store and retrieve data
            test_data = {"test": "data"}
            cache_key = "test_key"
            
            cache_system.set(cache_key, test_data)
            retrieved_data = cache_system.get(cache_key)
            
            assert retrieved_data == test_data
            
        except ImportError:
            # Cache system not implemented yet - test should fail in TDD
            pytest.fail("Cache system not implemented")
            
    def test_parallel_processing_capability(self):
        """Test parallel processing capabilities."""
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            
            parallel_engine = get_global_parallel_engine()
            
            if parallel_engine:
                # Should have parallel processing methods
                assert hasattr(parallel_engine, 'process_parallel') or hasattr(parallel_engine, 'map')
                
        except ImportError:
            # Parallel engine not implemented yet - expected in TDD
            pytest.fail("Parallel engine not implemented")


class TestIntegrationPoints:
    """Test integration between different components."""
    
    def test_anthropic_integration_initialization(self, test_config):
        """Test Anthropic API integration initialization."""
        try:
            from src.anthropic_integration.base import AnthropicBase
            
            # Should initialize with config
            base = AnthropicBase(test_config)
            
            assert hasattr(base, 'config')
            assert hasattr(base, 'client') or hasattr(base, '_client')
            
        except ImportError:
            pytest.fail("Anthropic base integration not found")
            
    def test_voyage_integration_initialization(self, test_config):
        """Test Voyage AI integration initialization."""
        try:
            from src.anthropic_integration.voyage_embeddings import VoyageEmbeddings
            
            # Should initialize with config
            voyage = VoyageEmbeddings(test_config)
            
            assert hasattr(voyage, 'config')
            assert hasattr(voyage, 'client') or hasattr(voyage, '_client')
            
        except ImportError:
            pytest.fail("Voyage embeddings integration not found")
            
    def test_dashboard_integration_setup(self, test_config, project_root):
        """Test dashboard integration setup."""
        from run_pipeline import setup_dashboard_integration
        
        # Should setup dashboard directories
        result = setup_dashboard_integration(test_config)
        
        # Should create necessary directories
        dashboard_path = Path(test_config['data']['dashboard_path'])
        assert dashboard_path.exists() or result == True  # Either created or setup successful
        
    def test_results_integration_with_dashboard(self, test_config):
        """Test results integration with dashboard."""
        from run_pipeline import integrate_with_dashboard
        
        mock_results = {
            'overall_success': True,
            'final_outputs': ['test_output.csv'],
            'stages_completed': {'01_chunk_processing': True}
        }
        
        # Should integrate results without errors
        result = integrate_with_dashboard(mock_results, test_config)
        
        # Should return success status
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
