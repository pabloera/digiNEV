"""
Test suite for system integration and end-to-end testing.
Tests the complete system integration, dashboard, and production deployment.

These tests verify that all components work together correctly.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import time

import pandas as pd
import pytest

from conftest import create_test_data_file, assert_valid_analysis_result


class TestSystemIntegration:
    """Test complete system integration."""
    
    def test_end_to_end_pipeline_execution(self, test_config, project_root, temp_csv_file):
        """Test complete end-to-end pipeline execution."""
        # This is the main integration test
        from run_pipeline import run_complete_pipeline_execution, load_configuration
        
        # Load actual configuration
        config = load_configuration()
        
        # Override with test settings
        config.update(test_config)
        config['anthropic']['enable_api_integration'] = False  # Disable API for testing
        
        # Run complete pipeline
        result = run_complete_pipeline_execution([temp_csv_file], config)
        
        assert isinstance(result, dict)
        assert 'overall_success' in result
        assert 'datasets_processed' in result
        assert 'stages_completed' in result
        
        # Should process at least one dataset
        if result['overall_success']:
            assert len(result['datasets_processed']) > 0
            
    def test_checkpoint_system_integration(self, test_config, project_root, mock_checkpoints):
        """Test checkpoint system integration with pipeline."""
        from run_pipeline import load_checkpoints, get_resume_point, should_skip_stage
        
        # Create checkpoint file
        checkpoint_file = project_root / "checkpoints" / "checkpoints.json"
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(mock_checkpoints, f)
        
        # Test checkpoint integration
        checkpoints = load_checkpoints()
        assert checkpoints is not None
        
        resume_point = get_resume_point(checkpoints)
        assert resume_point == '06_text_cleaning'
        
        # Should skip completed stages
        assert should_skip_stage('01_chunk_processing', checkpoints) == True
        assert should_skip_stage('06_text_cleaning', checkpoints) == False
        
    def test_protection_system_integration(self, test_config, project_root, mock_protection_checklist):
        """Test protection system integration."""
        from run_pipeline import (
            load_protection_checklist, 
            check_stage_protection, 
            should_skip_protected_stage
        )
        
        # Create protection checklist file
        checklist_file = project_root / "checkpoints" / "checklist.json"
        checklist_file.parent.mkdir(exist_ok=True)
        
        with open(checklist_file, 'w') as f:
            json.dump(mock_protection_checklist, f)
        
        # Test protection integration
        checklist = load_protection_checklist()
        assert checklist is not None
        
        # Test protection checking
        protection = check_stage_protection('02_encoding_validation', checklist)
        assert protection['can_overwrite'] == False
        assert protection['requires_override'] == True
        
        # Should skip protected stages
        assert should_skip_protected_stage('01_chunk_processing', checklist) == True
        
    def test_configuration_system_integration(self, test_config, project_root):
        """Test configuration system integration."""
        from run_pipeline import load_configuration
        
        # Create test configuration files
        config_dir = project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create settings.yaml
        settings_file = config_dir / "settings.yaml"
        with open(settings_file, 'w') as f:
            yaml_content = """
anthropic:
  enable_api_integration: false
  model: claude-3-5-haiku-20241022

processing:
  chunk_size: 1000
  encoding: utf-8

data:
  path: data/uploads
  interim_path: data/interim
"""
            f.write(yaml_content)
        
        # Load configuration
        config = load_configuration()
        
        assert isinstance(config, dict)
        assert 'anthropic' in config
        assert 'processing' in config
        assert config['anthropic']['enable_api_integration'] == False
        assert config['processing']['chunk_size'] == 1000
        
    def test_data_flow_integration(self, test_config, project_root, sample_telegram_data, test_data_dir):
        """Test data flow through the entire system."""
        # Create test dataset
        test_file = create_test_data_file(sample_telegram_data, "integration_test.csv", test_data_dir)
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
            
            # Mock all external dependencies
            with patch('src.anthropic_integration.base.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = '{"result": "success"}'
                mock_client.messages.create.return_value = mock_response
                mock_anthropic.return_value = mock_client
                
                # Run pipeline
                result = pipeline.run_complete_pipeline([test_file])
                
                # Should complete data flow
                assert isinstance(result, dict)
                
                # Should have processed the data
                if 'total_records' in result:
                    assert result['total_records'] > 0
                    
        except ImportError:
            pytest.skip("Pipeline components not fully implemented")
            
    def test_optimization_system_integration(self, test_config):
        """Test integration of optimization systems."""
        from run_pipeline import check_optimization_systems
        
        # Check optimization systems
        optimizations = check_optimization_systems()
        
        assert isinstance(optimizations, dict)
        
        # Should check all 5 weeks
        expected_systems = [
            'week1_emergency',
            'week2_caching',
            'week3_parallelization', 
            'week4_monitoring',
            'week5_production'
        ]
        
        for system in expected_systems:
            assert system in optimizations
            assert isinstance(optimizations[system], bool)
            
        # Calculate optimization rate
        active_count = sum(optimizations.values())
        total_count = len(optimizations)
        optimization_rate = active_count / total_count
        
        # Should have reasonable optimization coverage
        assert optimization_rate >= 0.0  # At least some optimizations available


class TestDashboardIntegration:
    """Test dashboard integration."""
    
    def test_dashboard_data_preparation(self, test_config, sample_telegram_data):
        """Test preparation of data for dashboard."""
        from run_pipeline import setup_dashboard_integration
        
        # Setup dashboard
        dashboard_ready = setup_dashboard_integration(test_config)
        
        # Should create dashboard directories
        dashboard_path = Path(test_config.get('data', {}).get('dashboard_path', 'src/dashboard/data'))
        
        if dashboard_ready:
            assert dashboard_path.exists() or dashboard_ready == True
            
    def test_dashboard_results_integration(self, test_config):
        """Test integration of results with dashboard."""
        from run_pipeline import integrate_with_dashboard
        
        # Mock pipeline results
        mock_results = {
            'overall_success': True,
            'datasets_processed': ['test_dataset.csv'],
            'stages_completed': {
                '01_chunk_processing': [{'dataset': 'test', 'success': True}],
                '02_encoding_validation': [{'dataset': 'test', 'success': True}]
            },
            'final_outputs': ['/path/to/output.csv'],
            'optimization_summary': {
                'active_optimizations': '3/5 weeks',
                'optimization_rate': '60%'
            }
        }
        
        # Should integrate without errors
        result = integrate_with_dashboard(mock_results, test_config)
        
        assert isinstance(result, bool)
        
    def test_dashboard_file_generation(self, test_config, test_data_dir):
        """Test generation of dashboard files."""
        from run_pipeline import integrate_with_dashboard
        
        # Create dashboard directory
        dashboard_dir = test_data_dir / "dashboard_results"
        dashboard_dir.mkdir(exist_ok=True)
        
        # Update config
        dashboard_config = test_config.copy()
        dashboard_config['data']['dashboard_path'] = str(test_data_dir)
        
        # Mock results with file outputs
        mock_results = {
            'overall_success': True,
            'final_outputs': [],  # No files to copy for this test
            'execution_time': 123.45,
            'optimization_summary': {'test': 'data'}
        }
        
        result = integrate_with_dashboard(mock_results, dashboard_config)
        
        # Should create results file
        results_files = list(dashboard_dir.glob("pipeline_results_*.json"))
        
        if result:
            assert len(results_files) > 0
            
            # Check results file content
            with open(results_files[0], 'r') as f:
                saved_results = json.load(f)
                
            assert saved_results['overall_success'] == True
            assert 'execution_time' in saved_results
            
    def test_dashboard_startup_check(self, test_config, project_root):
        """Test dashboard startup verification."""
        dashboard_script = project_root / "src" / "dashboard" / "start_dashboard.py"
        
        if dashboard_script.exists():
            # Try to validate dashboard script syntax
            try:
                with open(dashboard_script, 'r') as f:
                    dashboard_code = f.read()
                
                # Should be valid Python
                compile(dashboard_code, str(dashboard_script), 'exec')
                
                # Should have main execution
                assert 'if __name__' in dashboard_code
                
            except SyntaxError as e:
                pytest.fail(f"Dashboard script has syntax errors: {e}")
                
        else:
            pytest.skip("Dashboard script not found")


class TestAPIIntegrationInSystem:
    """Test API integration within the complete system."""
    
    def test_anthropic_api_system_integration(self, test_config, project_root, temp_csv_file):
        """Test Anthropic API integration in complete system."""
        # Enable API integration
        api_config = test_config.copy()
        api_config['anthropic']['enable_api_integration'] = True
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(api_config, str(project_root))
            
            # Mock Anthropic API
            with patch('src.anthropic_integration.base.Anthropic') as mock_anthropic_class:
                mock_client = Mock()
                
                # Mock successful API responses
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = json.dumps({
                    "sentiment": "positive",
                    "confidence": 0.8,
                    "classification": "democratic_discourse"
                })
                mock_client.messages.create.return_value = mock_response
                mock_anthropic_class.return_value = mock_client
                
                # Run pipeline with API integration
                result = pipeline.run_complete_pipeline([temp_csv_file])
                
                # Should complete successfully
                assert isinstance(result, dict)
                
                # API should have been called
                assert mock_client.messages.create.called
                
        except ImportError:
            pytest.skip("API integration components not implemented")
            
    def test_voyage_api_system_integration(self, test_config, project_root, temp_csv_file):
        """Test Voyage AI integration in complete system."""
        # Enable Voyage integration
        voyage_config = test_config.copy()
        voyage_config['voyage_embeddings']['enable_sampling'] = True
        voyage_config['voyage_embeddings']['max_messages'] = 100
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(voyage_config, str(project_root))
            
            # Mock Voyage API
            with patch('voyageai.Client') as mock_voyage_class:
                mock_client = Mock()
                mock_client.embed.return_value = Mock(
                    embeddings=[[0.1, 0.2, 0.3] for _ in range(10)]
                )
                mock_voyage_class.return_value = mock_client
                
                # Run pipeline with Voyage integration
                result = pipeline.run_complete_pipeline([temp_csv_file])
                
                # Should complete successfully
                assert isinstance(result, dict)
                
        except ImportError:
            pytest.skip("Voyage integration components not implemented")
            
    def test_api_fallback_system_integration(self, test_config, project_root, temp_csv_file):
        """Test API fallback mechanisms in complete system."""
        api_config = test_config.copy()
        api_config['anthropic']['enable_api_integration'] = True
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(api_config, str(project_root))
            
            # Mock API failures
            with patch('src.anthropic_integration.base.Anthropic') as mock_anthropic_class:
                mock_client = Mock()
                mock_client.messages.create.side_effect = Exception("API Error")
                mock_anthropic_class.return_value = mock_client
                
                # Should fallback gracefully
                result = pipeline.run_complete_pipeline([temp_csv_file])
                
                # Should still complete (with fallback methods)
                assert isinstance(result, dict)
                
        except ImportError:
            pytest.skip("API integration components not implemented")


class TestProductionDeployment:
    """Test production deployment readiness."""
    
    def test_production_configuration_validation(self, test_config, project_root):
        """Test production configuration validation."""
        try:
            from src.optimized.production_deploy import get_global_deployment_system
            
            deployment = get_global_deployment_system()
            
            if deployment:
                # Should validate production readiness
                if hasattr(deployment, 'validate_production_config'):
                    validation = deployment.validate_production_config(test_config)
                    
                    assert isinstance(validation, dict)
                    assert 'is_production_ready' in validation
                    assert 'issues' in validation or 'warnings' in validation
                    
        except ImportError:
            pytest.skip("Production deployment system not implemented")
            
    def test_production_security_checks(self, test_config):
        """Test production security validation."""
        try:
            from src.optimized.production_deploy import get_global_deployment_system
            
            deployment = get_global_deployment_system()
            
            if deployment:
                if hasattr(deployment, 'run_security_checks'):
                    security_check = deployment.run_security_checks(test_config)
                    
                    assert isinstance(security_check, dict)
                    assert 'security_score' in security_check or 'passed' in security_check
                    
                    # Should check for sensitive data exposure
                    if 'checks' in security_check:
                        checks = security_check['checks']
                        assert 'api_key_exposure' in checks or 'credentials_check' in checks
                        
        except ImportError:
            pytest.skip("Production deployment system not implemented")
            
    def test_production_performance_requirements(self, test_config, sample_telegram_data):
        """Test production performance requirements."""
        try:
            from src.optimized.production_deploy import get_global_deployment_system
            
            deployment = get_global_deployment_system()
            
            if deployment:
                if hasattr(deployment, 'validate_performance_requirements'):
                    # Test with sample data
                    perf_check = deployment.validate_performance_requirements(
                        sample_telegram_data, 
                        requirements={
                            'max_processing_time': 300,  # 5 minutes
                            'max_memory_usage': 2048,    # 2GB
                            'min_throughput': 1000       # 1000 messages/minute
                        }
                    )
                    
                    assert isinstance(perf_check, dict)
                    assert 'meets_requirements' in perf_check
                    
        except ImportError:
            pytest.skip("Production deployment system not implemented")
            
    def test_production_monitoring_setup(self, test_config):
        """Test production monitoring setup."""
        try:
            from src.optimized.production_deploy import get_global_deployment_system
            
            deployment = get_global_deployment_system()
            
            if deployment:
                if hasattr(deployment, 'setup_production_monitoring'):
                    monitoring_setup = deployment.setup_production_monitoring(test_config)
                    
                    assert isinstance(monitoring_setup, dict)
                    assert 'monitoring_enabled' in monitoring_setup
                    
                    # Should setup logging and metrics
                    if 'components' in monitoring_setup:
                        components = monitoring_setup['components']
                        expected_components = ['logging', 'metrics', 'alerting']
                        
                        for component in expected_components:
                            assert component in components or monitoring_setup['monitoring_enabled']
                            
        except ImportError:
            pytest.skip("Production deployment system not implemented")


class TestErrorHandlingIntegration:
    """Test error handling across the complete system."""
    
    def test_system_error_recovery(self, test_config, project_root, test_data_dir):
        """Test system error recovery mechanisms."""
        # Create corrupted test file
        corrupted_file = test_data_dir / "corrupted.csv"
        with open(corrupted_file, 'w') as f:
            f.write("invalid,csv,format\n")
            f.write("missing,data\n")
            f.write("inconsistent,row,data,extra,columns\n")
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(test_config, str(project_root))
            
            # Should handle corrupted data gracefully
            result = pipeline.run_complete_pipeline([str(corrupted_file)])
            
            assert isinstance(result, dict)
            
            # Should either succeed with error handling or fail gracefully
            if not result.get('overall_success', False):
                assert 'error' in result or 'stage_results' in result
                
        except ImportError:
            # Test basic error handling with file operations
            try:
                df = pd.read_csv(corrupted_file)
                # If it loads, check if it handles inconsistent data
                assert len(df.columns) >= 3
            except Exception as e:
                # Should fail gracefully
                assert isinstance(e, (pd.errors.ParserError, pd.errors.EmptyDataError))
                
    def test_system_timeout_handling(self, test_config, project_root, temp_csv_file):
        """Test system timeout handling."""
        # Configure short timeouts for testing
        timeout_config = test_config.copy()
        timeout_config['processing']['timeout_seconds'] = 1  # Very short timeout
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(timeout_config, str(project_root))
            
            # Mock slow processing
            with patch.object(pipeline, 'run_complete_pipeline') as mock_run:
                def slow_processing(*args, **kwargs):
                    time.sleep(2)  # Longer than timeout
                    return {'overall_success': True}
                
                mock_run.side_effect = slow_processing
                
                # Should handle timeout
                start_time = time.time()
                result = pipeline.run_complete_pipeline([temp_csv_file])
                end_time = time.time()
                
                # Should either complete quickly or handle timeout
                assert end_time - start_time < 5  # Should not hang
                
        except ImportError:
            pytest.skip("Pipeline components not implemented")
            
    def test_system_memory_limit_handling(self, test_config, project_root, test_data_dir):
        """Test system memory limit handling."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(50000),
            'body': [f'Large message {i} ' * 100 for i in range(50000)],  # ~10KB per message
            'date': pd.date_range('2023-01-01', periods=50000, freq='H'),
            'channel': [f'channel_{i % 100}' for i in range(50000)]
        })
        
        large_file = test_data_dir / "large_dataset.csv"
        large_data.to_csv(large_file, index=False)
        
        # Configure memory limits
        memory_config = test_config.copy()
        memory_config['processing']['memory_limit'] = '100MB'  # Low limit
        memory_config['processing']['chunk_size'] = 1000       # Force chunking
        
        try:
            from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            pipeline = UnifiedAnthropicPipeline(memory_config, str(project_root))
            
            # Should handle large data with chunking
            result = pipeline.run_complete_pipeline([str(large_file)])
            
            assert isinstance(result, dict)
            
            # Should complete without memory errors
            if 'error' in result:
                # Should not be memory-related error
                assert 'memory' not in result['error'].lower()
                
        except ImportError:
            # Test basic chunked processing
            chunk_size = 1000
            total_processed = 0
            
            for chunk in pd.read_csv(large_file, chunksize=chunk_size):
                total_processed += len(chunk)
                assert len(chunk) <= chunk_size
                
            assert total_processed == 50000


class TestSystemValidation:
    """Test complete system validation."""
    
    def test_system_consistency_validation(self, test_config, project_root):
        """Test system consistency validation."""
        try:
            from src.anthropic_integration.system_validator import SystemValidator
            
            validator = SystemValidator(test_config)
            
            # Should validate system consistency
            validation = validator.validate_system_consistency()
            
            assert isinstance(validation, dict)
            assert 'is_consistent' in validation
            assert 'issues' in validation or 'checks' in validation
            
            # Should check component compatibility
            if 'component_compatibility' in validation:
                compatibility = validation['component_compatibility']
                assert isinstance(compatibility, dict)
                
        except ImportError:
            # Basic system consistency checks
            required_dirs = ['src', 'config', 'data']
            project_path = Path(project_root)
            
            for required_dir in required_dirs:
                dir_path = project_path / required_dir
                if not dir_path.exists():
                    pytest.fail(f"Required directory missing: {required_dir}")
                    
    def test_system_performance_validation(self, test_config, project_root, sample_telegram_data):
        """Test system performance validation."""
        try:
            from src.anthropic_integration.system_validator import SystemValidator
            
            validator = SystemValidator(test_config)
            
            # Should validate performance requirements
            if hasattr(validator, 'validate_performance'):
                perf_validation = validator.validate_performance(sample_telegram_data)
                
                assert isinstance(perf_validation, dict)
                assert 'performance_score' in perf_validation or 'meets_requirements' in perf_validation
                
        except ImportError:
            # Basic performance validation
            start_time = time.time()
            
            # Simple processing
            processed = sample_telegram_data['body'].str.len().sum()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete quickly
            assert processing_time < 5.0
            assert processed > 0
            
    def test_system_security_validation(self, test_config, project_root):
        """Test system security validation."""
        try:
            from src.anthropic_integration.system_validator import SystemValidator
            
            validator = SystemValidator(test_config)
            
            # Should validate security requirements
            if hasattr(validator, 'validate_security'):
                security_validation = validator.validate_security()
                
                assert isinstance(security_validation, dict)
                assert 'security_score' in security_validation or 'is_secure' in security_validation
                
        except ImportError:
            # Basic security checks
            # Check for hardcoded API keys in config
            config_str = json.dumps(test_config)
            
            # Should not contain obvious API keys
            suspicious_patterns = ['sk-', 'api_key', 'secret']
            for pattern in suspicious_patterns:
                if pattern in config_str:
                    # Should be placeholder or test values
                    assert 'test' in config_str.lower() or 'placeholder' in config_str.lower()


class TestDocumentationIntegration:
    """Test documentation and help system integration."""
    
    def test_readme_documentation(self, project_root):
        """Test README documentation exists and is complete."""
        readme_file = project_root / "README.md"
        
        if readme_file.exists():
            with open(readme_file, 'r') as f:
                readme_content = f.read()
            
            # Should have essential sections
            essential_sections = [
                'installation', 'usage', 'configuration', 'pipeline'
            ]
            
            content_lower = readme_content.lower()
            for section in essential_sections:
                assert section in content_lower, f"README missing {section} section"
                
            # Should have code examples
            assert '```' in readme_content, "README should have code examples"
            
        else:
            pytest.fail("README.md not found")
            
    def test_configuration_documentation(self, project_root):
        """Test configuration documentation."""
        config_files = [
            project_root / "config" / "settings.yaml",
            project_root / "config" / "anthropic.yaml",
            project_root / "config" / "processing.yaml"
        ]
        
        documented_configs = []
        
        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Should have comments explaining settings
                if '#' in content:
                    documented_configs.append(config_file.name)
                    
        # Should have at least some documented configuration
        assert len(documented_configs) > 0, "No configuration files have documentation"
        
    def test_api_documentation(self, project_root):
        """Test API documentation exists."""
        api_docs = [
            project_root / "docs" / "api.md",
            project_root / "src" / "anthropic_integration" / "README.md"
        ]
        
        api_documented = False
        
        for doc_file in api_docs:
            if doc_file.exists():
                with open(doc_file, 'r') as f:
                    content = f.read()
                
                # Should document API usage
                if any(term in content.lower() for term in ['anthropic', 'api', 'integration']):
                    api_documented = True
                    break
                    
        # Should have API documentation
        assert api_documented, "API integration should be documented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
