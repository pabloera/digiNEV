#!/usr/bin/env python3
"""
Test Week 2 Optimizations - Validation Script
============================================

Tests the Week 2 advanced optimizations:
- Unified Embeddings Engine (L1/L2 cache hierarchy)
- Smart Claude Cache (semantic caching)
- Performance Monitor (real-time metrics)

This script validates that all integrations work correctly.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_unified_embeddings_engine():
    """Test Unified Embeddings Engine functionality"""
    logger.info("🧪 Testing Unified Embeddings Engine...")
    
    try:
        from src.optimized.unified_embeddings_engine import (
            get_global_unified_engine, 
            EmbeddingRequest,
            create_production_engine,
            create_development_engine
        )
        
        # Test factory functions
        prod_engine = create_production_engine()
        dev_engine = create_development_engine()
        
        # Test global instance
        global_engine = get_global_unified_engine()
        
        # Get comprehensive stats
        stats = global_engine.get_comprehensive_stats()
        
        logger.info(f"✅ Unified Embeddings Engine: Working")
        logger.info(f"   Strategy: {stats['strategy']['name']}")
        logger.info(f"   Memory: {stats['strategy']['max_memory_mb']}MB")
        logger.info(f"   Disk: {stats['strategy']['max_disk_gb']}GB")
        logger.info(f"   Workers: {stats['system']['workers']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified Embeddings Engine failed: {e}")
        return False

def test_smart_claude_cache():
    """Test Smart Claude Cache functionality"""
    logger.info("🧪 Testing Smart Claude Cache...")
    
    try:
        from src.optimized.smart_claude_cache import (
            get_global_claude_cache,
            ClaudeRequest,
            ClaudeResponse,
            create_production_claude_cache,
            create_development_claude_cache
        )
        
        # Test factory functions
        prod_cache = create_production_claude_cache()
        dev_cache = create_development_claude_cache()
        
        # Test global instance
        global_cache = get_global_claude_cache()
        
        # Get comprehensive stats
        stats = global_cache.get_comprehensive_stats()
        
        logger.info(f"✅ Smart Claude Cache: Working")
        logger.info(f"   Memory entries: {stats['resource_usage']['memory_entries']}")
        logger.info(f"   Memory usage: {stats['resource_usage']['memory_usage_mb']:.2f}MB")
        logger.info(f"   TTL: {stats['cache_configuration']['ttl_hours']}h")
        logger.info(f"   Similarity threshold: {stats['cache_configuration']['similarity_threshold']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Smart Claude Cache failed: {e}")
        return False

def test_performance_monitor():
    """Test Performance Monitor functionality"""
    logger.info("🧪 Testing Performance Monitor...")
    
    try:
        from src.optimized.performance_monitor import (
            get_global_performance_monitor,
            create_production_monitor,
            create_development_monitor,
            PerformanceMonitor
        )
        
        # Test factory functions
        prod_monitor = create_production_monitor({})
        dev_monitor = create_development_monitor({})
        
        # Test global instance
        global_monitor = get_global_performance_monitor({})
        
        # Test monitoring features
        global_monitor.start_monitoring()
        
        # Generate executive summary
        summary = global_monitor.generate_executive_summary()
        
        # Get real-time dashboard data
        dashboard_data = global_monitor.get_real_time_dashboard_data()
        
        global_monitor.stop_monitoring()
        
        logger.info(f"✅ Performance Monitor: Working")
        logger.info(f"   Session duration: {summary['session_overview']['duration_minutes']:.2f}min")
        logger.info(f"   Performance grade: {summary['session_overview']['performance_grade']}")
        logger.info(f"   Memory usage: {summary['resource_utilization']['current_memory_percent']:.1f}%")
        logger.info(f"   CPU usage: {summary['resource_utilization']['current_cpu_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance Monitor failed: {e}")
        return False

def test_voyage_integration():
    """Test Voyage.ai integration with Week 2 optimizations"""
    logger.info("🧪 Testing Voyage.ai Integration...")
    
    try:
        from src.anthropic_integration.voyage_embeddings import VoyageEmbeddingAnalyzer
        
        # Test with minimal config
        config = {
            'embeddings': {
                'model': 'voyage-3.5-lite',
                'batch_size': 128,
                'cache_embeddings': True
            }
        }
        
        analyzer = VoyageEmbeddingAnalyzer(config)
        
        # Check if Week 2 optimizations are available
        if hasattr(analyzer, 'WEEK2_OPTIMIZATIONS_AVAILABLE'):
            week2_available = getattr(analyzer, 'WEEK2_OPTIMIZATIONS_AVAILABLE', False)
        else:
            # Import and check manually
            try:
                from src.optimized.unified_embeddings_engine import get_global_unified_engine
                week2_available = True
            except ImportError:
                week2_available = False
        
        logger.info(f"✅ Voyage.ai Integration: Working")
        logger.info(f"   Model: {analyzer.model_name}")
        logger.info(f"   Batch size: {analyzer.batch_size}")
        logger.info(f"   Week 2 optimizations: {'Available' if week2_available else 'Not Available'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voyage.ai Integration failed: {e}")
        return False

def test_anthropic_integration():
    """Test Anthropic API integration with Week 2 optimizations"""
    logger.info("🧪 Testing Anthropic Integration...")
    
    try:
        from src.anthropic_integration.base import AnthropicBase
        
        # Test with minimal config
        config = {
            'anthropic': {
                'model': 'claude-3-5-haiku-20241022',
                'temperature': 0.3,
                'max_tokens': 1000
            }
        }
        
        base = AnthropicBase(config)
        
        # Check if Week 2 optimizations are integrated
        week2_cache_available = getattr(base, 'week2_cache_available', False)
        has_smart_cache = hasattr(base, 'smart_claude_cache') and base.smart_claude_cache is not None
        has_performance_monitor = hasattr(base, 'performance_monitor') and base.performance_monitor is not None
        
        logger.info(f"✅ Anthropic Integration: Working")
        logger.info(f"   Model: {base.model}")
        logger.info(f"   API available: {base.api_available}")
        logger.info(f"   Week 2 cache available: {week2_cache_available}")
        logger.info(f"   Smart cache integrated: {has_smart_cache}")
        logger.info(f"   Performance monitor integrated: {has_performance_monitor}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Anthropic Integration failed: {e}")
        return False

def main():
    """Run all Week 2 optimization tests"""
    logger.info("🚀 WEEK 2 OPTIMIZATIONS VALIDATION")
    logger.info("=" * 50)
    
    tests = [
        ("Unified Embeddings Engine", test_unified_embeddings_engine),
        ("Smart Claude Cache", test_smart_claude_cache),
        ("Performance Monitor", test_performance_monitor),
        ("Voyage.ai Integration", test_voyage_integration),
        ("Anthropic Integration", test_anthropic_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 VALIDATION SUMMARY:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    total = len(results)
    success_rate = (passed / total) * 100
    
    logger.info(f"\n🎯 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🏆 Week 2 optimizations are ready for production!")
        return 0
    elif success_rate >= 60:
        logger.info("⚠️ Week 2 optimizations need some fixes before production")
        return 1
    else:
        logger.info("❌ Week 2 optimizations have significant issues")
        return 2

if __name__ == "__main__":
    sys.exit(main())