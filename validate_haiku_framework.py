#!/usr/bin/env python3
"""
Haiku Framework Validation: Test claude-3-5-haiku-20241022 unified model configuration
Function: Validate smart fallback strategy and stage-specific optimizations for academic research
Usage: poetry run python validate_haiku_framework.py
"""

import sys
import os
import yaml
from pathlib import Path
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load academic configuration"""
    config_path = Path(__file__).parent / "config" / "anthropic.yaml.example"
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return {}

def validate_model_configuration(config: Dict[str, Any]) -> bool:
    """Validate unified model configuration"""
    logger.info("\n🔍 Validating Model Configuration...")
    
    anthropic_config = config.get('anthropic', {})
    
    # Check primary model
    primary_model = anthropic_config.get('primary_model')
    if primary_model != "claude-3-5-haiku-20241022":
        logger.error(f"❌ Primary model should be claude-3-5-haiku-20241022, got: {primary_model}")
        return False
    logger.info(f"✅ Primary model: {primary_model}")
    
    # Check stage configurations
    stage_configs = anthropic_config.get('stage_configs', {})
    haiku_model = "claude-3-5-haiku-20241022"
    
    required_stages = [
        'political_analysis', 'sentiment_analysis', 'topic_interpretation',
        'text_processing', 'qualitative_analysis', 'network_analysis'
    ]
    
    for stage in required_stages:
        stage_config = stage_configs.get(stage, {})
        stage_model = stage_config.get('model')
        
        if stage_model != haiku_model:
            logger.error(f"❌ Stage {stage} should use {haiku_model}, got: {stage_model}")
            return False
        
        temperature = stage_config.get('temperature')
        max_tokens = stage_config.get('max_tokens')
        
        logger.info(f"✅ {stage}: model={stage_model}, temp={temperature}, tokens={max_tokens}")
    
    return True

def validate_fallback_strategy(config: Dict[str, Any]) -> bool:
    """Validate smart fallback strategy configuration"""
    logger.info("\n🛡️ Validating Smart Fallback Strategy...")
    
    anthropic_config = config.get('anthropic', {})
    fallback_config = anthropic_config.get('fallback_strategy', {})
    
    # Check fallback settings
    smart_fallback = fallback_config.get('enable_smart_fallback')
    if not smart_fallback:
        logger.error("❌ Smart fallback should be enabled")
        return False
    logger.info("✅ Smart fallback enabled")
    
    # Check budget threshold
    budget_threshold = fallback_config.get('budget_exceeded_threshold')
    if budget_threshold != 0.8:
        logger.warning(f"⚠️ Budget threshold is {budget_threshold}, recommended: 0.8")
    else:
        logger.info(f"✅ Budget threshold: {budget_threshold}")
    
    # Check quality threshold
    quality_threshold = fallback_config.get('quality_threshold')
    if quality_threshold != 0.7:
        logger.warning(f"⚠️ Quality threshold is {quality_threshold}, recommended: 0.7")
    else:
        logger.info(f"✅ Quality threshold: {quality_threshold}")
    
    return True

def validate_academic_settings(config: Dict[str, Any]) -> bool:
    """Validate academic research settings"""
    logger.info("\n🎓 Validating Academic Settings...")
    
    anthropic_config = config.get('anthropic', {})
    
    # Check budget controls
    monthly_budget = anthropic_config.get('monthly_budget_usd')
    if monthly_budget != 50.0:
        logger.warning(f"⚠️ Monthly budget is ${monthly_budget}, academic default: $50")
    else:
        logger.info(f"✅ Monthly budget: ${monthly_budget}")
    
    # Check rate limiting
    requests_per_minute = anthropic_config.get('requests_per_minute')
    if requests_per_minute != 100:
        logger.warning(f"⚠️ Rate limit is {requests_per_minute}/min, haiku supports 100/min")
    else:
        logger.info(f"✅ Rate limit: {requests_per_minute} requests/minute")
    
    # Check Portuguese optimization
    portuguese_opt = anthropic_config.get('portuguese_optimization')
    if not portuguese_opt:
        logger.error("❌ Portuguese optimization should be enabled")
        return False
    logger.info("✅ Portuguese optimization enabled")
    
    # Check category preservation
    preserve_categories = anthropic_config.get('preserve_original_categories')
    if not preserve_categories:
        logger.error("❌ Original category preservation should be enabled")
        return False
    logger.info("✅ Original categories preserved")
    
    return True

def validate_temperature_settings(config: Dict[str, Any]) -> bool:
    """Validate stage-specific temperature settings"""
    logger.info("\n🌡️ Validating Temperature Settings...")
    
    stage_configs = config.get('anthropic', {}).get('stage_configs', {})
    
    # Expected temperature ranges for different analysis types
    expected_temps = {
        'political_analysis': 0.1,      # Deterministic
        'sentiment_analysis': 0.2,      # Consistent
        'topic_interpretation': 0.4,    # Creative
        'text_processing': 0.2,         # Consistent
        'qualitative_analysis': 0.3,    # Balanced
        'network_analysis': 0.2         # Pattern detection
    }
    
    for stage, expected_temp in expected_temps.items():
        stage_config = stage_configs.get(stage, {})
        actual_temp = stage_config.get('temperature')
        
        if actual_temp != expected_temp:
            logger.warning(f"⚠️ {stage}: temp={actual_temp}, recommended={expected_temp}")
        else:
            logger.info(f"✅ {stage}: temperature={actual_temp} (optimal)")
    
    return True

def test_smart_fallback_imports() -> bool:
    """Test if smart fallback strategy can be imported"""
    logger.info("\n📦 Testing Smart Fallback Strategy Import...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from src.core.smart_fallback_strategy import SmartFallbackStrategy
        logger.info("✅ SmartFallbackStrategy imported successfully")
        
        # Test initialization
        test_config = {
            'monthly_budget_usd': 50.0,
            'fallback_strategy': {
                'enable_smart_fallback': True,
                'budget_exceeded_threshold': 0.8,
                'quality_threshold': 0.7
            },
            'stage_configs': {
                'political_analysis': {
                    'model': 'claude-3-5-haiku-20241022',
                    'temperature': 0.1,
                    'max_tokens': 2500
                }
            }
        }
        
        strategy = SmartFallbackStrategy(test_config)
        logger.info("✅ SmartFallbackStrategy initialized successfully")
        
        # Test stage configuration
        stage_config = strategy.get_stage_config('political_analysis')
        if stage_config['model'] == 'claude-3-5-haiku-20241022':
            logger.info("✅ Stage configuration working correctly")
        else:
            logger.error(f"❌ Stage configuration error: {stage_config}")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def test_anthropic_base_integration() -> bool:
    """Test AnthropicBase integration with smart fallback"""
    logger.info("\n🔗 Testing AnthropicBase Integration...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from src.anthropic_integration.base import AnthropicBase
        logger.info("✅ AnthropicBase imported successfully")
        
        # Test configuration
        test_config = {
            'anthropic': {
                'api_key': 'test_key',
                'primary_model': 'claude-3-5-haiku-20241022',
                'monthly_budget_usd': 50.0,
                'fallback_strategy': {
                    'enable_smart_fallback': True
                },
                'stage_configs': {
                    'political_analysis': {
                        'model': 'claude-3-5-haiku-20241022',
                        'temperature': 0.1,
                        'max_tokens': 2500
                    }
                }
            }
        }
        
        # Initialize base with stage operation
        base = AnthropicBase(test_config, stage_operation='political_analysis')
        logger.info("✅ AnthropicBase initialized with smart fallback")
        
        # Test temperature and token settings
        temp = base._get_stage_temperature('political_analysis')
        tokens = base._get_stage_max_tokens('political_analysis')
        
        if temp == 0.1 and tokens == 2500:
            logger.info(f"✅ Stage settings: temp={temp}, tokens={tokens}")
        else:
            logger.error(f"❌ Stage settings error: temp={temp}, tokens={tokens}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def generate_summary_report(results: Dict[str, bool]) -> None:
    """Generate validation summary report"""
    logger.info("\n" + "="*60)
    logger.info("📋 HAIKU FRAMEWORK VALIDATION SUMMARY")
    logger.info("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info("-"*60)
    logger.info(f"📊 Results: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Haiku framework ready for academic research!")
        logger.info("🚀 System optimized for: claude-3-5-haiku-20241022 unified model")
        logger.info("💰 Budget-efficient Brazilian political discourse analysis enabled")
    else:
        logger.warning("⚠️ Some tests failed - review configuration before deployment")
    
    logger.info("="*60)

def main():
    """Main validation function"""
    logger.info("🚀 Starting Haiku Framework Validation...")
    logger.info("🎯 Testing claude-3-5-haiku-20241022 unified configuration")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("❌ Cannot proceed without configuration")
        return False
    
    # Run validation tests
    results = {
        "Model Configuration": validate_model_configuration(config),
        "Fallback Strategy": validate_fallback_strategy(config),
        "Academic Settings": validate_academic_settings(config),
        "Temperature Settings": validate_temperature_settings(config),
        "Smart Fallback Import": test_smart_fallback_imports(),
        "AnthropicBase Integration": test_anthropic_base_integration()
    }
    
    # Generate summary
    generate_summary_report(results)
    
    # Return overall success
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)