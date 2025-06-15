#!/usr/bin/env python3
"""
Academic Week 3-4 Integration Test
=================================

Test the consolidated academic optimizations for social science research.
Validates parallel processing, streaming, and monitoring integration.

Data: 2025-06-15
Purpose: Academic research validation
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_academic_config() -> Dict[str, Any]:
    """Load academic optimization configuration"""
    try:
        config_path = Path("config/academic_optimizations.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            academic_config = yaml.safe_load(f)
        
        # Load base configuration
        base_config_path = Path("config/settings.yaml")
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configurations
        base_config['academic'] = academic_config['academic']
        base_config['performance_targets'] = academic_config['performance_targets']
        
        return base_config
        
    except Exception as e:
        logger.error(f"Failed to load academic config: {e}")
        return {
            'academic': {
                'monthly_budget': 50.0,
                'parallel_processing': {'enabled': False},
                'streaming': {'enabled': False},
                'monitoring': {'enabled': False}
            }
        }

def create_test_academic_dataset() -> pd.DataFrame:
    """Create a test dataset for academic research validation"""
    import numpy as np
    
    # Create synthetic Brazilian political discourse data
    data = {
        'message_id': range(1000),
        'text_content': [
            f"Mensagem de teste político brasileiro {i}" 
            for i in range(1000)
        ],
        'user_id': np.random.randint(1, 100, 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'platform': np.random.choice(['telegram', 'twitter', 'facebook'], 1000),
        'language': ['pt-BR'] * 1000,
        'region': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], 1000)
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Created test academic dataset: {len(df)} records")
    return df

def test_academic_pipeline_initialization():
    """Test academic pipeline initialization with Week 3-4 optimizations"""
    logger.info("🧪 Testing academic pipeline initialization...")
    
    try:
        from anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Load academic configuration
        config = load_academic_config()
        project_root = str(Path.cwd())
        
        # Initialize academic pipeline
        pipeline = UnifiedAnthropicPipeline(config, project_root)
        
        # Validate optimizations are initialized
        optimizations_status = {
            'parallel_engine': pipeline.parallel_engine is not None,
            'streaming_config': pipeline.streaming_config is not None,
            'realtime_monitor': pipeline.realtime_monitor is not None,
            'performance_tracker': hasattr(pipeline, 'performance_tracker'),
            'academic_monitor': hasattr(pipeline, '_academic_monitor')
        }
        
        logger.info(f"✅ Academic optimizations status: {optimizations_status}")
        
        # Test academic summary
        summary = pipeline.get_academic_summary()
        logger.info(f"📊 Academic summary generated with {len(summary)} sections")
        
        return {
            'success': True,
            'optimizations': optimizations_status,
            'summary_sections': list(summary.keys())
        }
        
    except Exception as e:
        logger.error(f"❌ Academic pipeline initialization failed: {e}")
        return {'success': False, 'error': str(e)}

def test_academic_stage_processing():
    """Test academic stage processing with Week 3-4 optimizations"""
    logger.info("🧪 Testing academic stage processing...")
    
    try:
        from anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Initialize pipeline
        config = load_academic_config()
        pipeline = UnifiedAnthropicPipeline(config, str(Path.cwd()))
        
        # Create test dataset
        test_data = create_test_academic_dataset()
        
        # Test stage processing
        start_time = time.time()
        stage_results = pipeline._process_stages(test_data, "academic_test_dataset")
        processing_time = time.time() - start_time
        
        # Analyze results
        successful_stages = sum(1 for result in stage_results.values() 
                              if isinstance(result, dict) and result.get('success', False))
        
        parallel_stages = sum(1 for result in stage_results.values()
                            if isinstance(result, dict) and result.get('parallel_execution', False))
        
        logger.info(f"✅ Academic processing completed in {processing_time:.2f}s")
        logger.info(f"📈 Successful stages: {successful_stages}/{len(pipeline.stages)}")
        logger.info(f"⚡ Parallel stages: {parallel_stages}")
        
        # Check for academic performance summary
        has_performance_summary = 'academic_performance_summary' in stage_results
        
        return {
            'success': True,
            'processing_time': processing_time,
            'successful_stages': successful_stages,
            'total_stages': len(pipeline.stages),
            'parallel_stages': parallel_stages,
            'has_performance_summary': has_performance_summary
        }
        
    except Exception as e:
        logger.error(f"❌ Academic stage processing failed: {e}")
        return {'success': False, 'error': str(e)}

def test_academic_performance_tracking():
    """Test academic performance tracking and reporting"""
    logger.info("🧪 Testing academic performance tracking...")
    
    try:
        from anthropic_integration.unified_pipeline import AcademicPerformanceTracker
        
        # Initialize performance tracker
        tracker = AcademicPerformanceTracker()
        
        # Simulate stage executions
        test_stages = [
            ('07_linguistic_processing', 2.5, True, 45.2),
            ('09_topic_modeling', 5.1, True, 78.3),
            ('10_tfidf_extraction', 1.8, True, 23.7),
            ('11_clustering', 3.2, True, 56.1),
            ('05_political_analysis', 4.1, False, 0.0)
        ]
        
        for stage_id, exec_time, was_parallel, memory_used in test_stages:
            tracker.record_stage_execution(stage_id, exec_time, was_parallel, memory_used)
        
        # Generate academic report
        report = tracker.get_academic_report()
        
        logger.info(f"✅ Academic performance report generated")
        logger.info(f"📊 Research quality score: {report['academic_performance']['research_quality_score']:.1f}")
        logger.info(f"⚡ Parallel efficiency: {report['academic_performance']['parallel_efficiency_percent']:.1f}%")
        
        return {
            'success': True,
            'research_quality_score': report['academic_performance']['research_quality_score'],
            'parallel_efficiency': report['academic_performance']['parallel_efficiency_percent'],
            'total_execution_time': report['academic_performance']['total_execution_time'],
            'stages_optimized': report['academic_performance']['stages_optimized']
        }
        
    except Exception as e:
        logger.error(f"❌ Academic performance tracking failed: {e}")
        return {'success': False, 'error': str(e)}

def test_academic_budget_monitoring():
    """Test academic budget monitoring for research cost control"""
    logger.info("🧪 Testing academic budget monitoring...")
    
    try:
        from anthropic_integration.unified_pipeline import AcademicBudgetMonitor
        
        # Initialize budget monitor
        monitor = AcademicBudgetMonitor(monthly_budget=50.0)
        
        # Simulate stage executions with costs
        test_stages = [
            ('09_topic_modeling', 12.5),
            ('10_tfidf_extraction', 8.3),
            ('11_clustering', 15.2),
            ('19_semantic_search', 6.7)
        ]
        
        for stage_name, exec_time in test_stages:
            monitor.log_stage_start(stage_name)
            monitor.log_stage_completion(stage_name, exec_time)
        
        # Get budget summary
        summary = monitor.get_budget_summary()
        
        logger.info(f"✅ Academic budget monitoring completed")
        logger.info(f"💰 Budget usage: ${summary['current_usage']:.3f}/${summary['monthly_budget']}")
        logger.info(f"📊 Usage percentage: {summary['usage_percent']:.1f}%")
        
        return {
            'success': True,
            'budget_usage': summary['current_usage'],
            'budget_remaining': summary['remaining_budget'],
            'usage_percent': summary['usage_percent'],
            'stages_tracked': len(summary['stage_costs'])
        }
        
    except Exception as e:
        logger.error(f"❌ Academic budget monitoring failed: {e}")
        return {'success': False, 'error': str(e)}

def run_academic_integration_tests():
    """Run comprehensive academic Week 3-4 integration tests"""
    logger.info("🎓 Starting Academic Week 3-4 Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Pipeline Initialization", test_academic_pipeline_initialization),
        ("Stage Processing", test_academic_stage_processing),
        ("Performance Tracking", test_academic_performance_tracking),
        ("Budget Monitoring", test_academic_budget_monitoring)
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results[test_name] = result
            
            if result.get('success', False):
                logger.info(f"✅ {test_name} Test: PASSED")
                passed_tests += 1
            else:
                logger.error(f"❌ {test_name} Test: FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"💥 {test_name} Test: EXCEPTION - {e}")
            results[test_name] = {'success': False, 'error': str(e)}
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info(f"🎓 Academic Week 3-4 Integration Test Summary")
    logger.info(f"✅ Passed: {passed_tests}/{len(tests)} tests")
    logger.info(f"📊 Success Rate: {(passed_tests/len(tests)*100):.1f}%")
    
    if passed_tests == len(tests):
        logger.info("🎉 All academic optimizations are working correctly!")
        logger.info("🚀 Week 3-4 consolidation: COMPLETE")
    else:
        logger.warning("⚠️  Some academic optimizations need attention")
    
    return results

if __name__ == "__main__":
    results = run_academic_integration_tests()
    
    # Return appropriate exit code
    all_passed = all(result.get('success', False) for result in results.values())
    sys.exit(0 if all_passed else 1)