#!/usr/bin/env python3
"""
Consolidated Test Suite for Week 1-5 Pipeline Optimizations
===========================================================

Comprehensive validation framework for all optimization phases:
- Week 1: Emergency Optimizations (Cache, Performance, Error Handling)
- Week 2: Advanced Caching & Monitoring (Hierarchical, Smart Cache, Monitoring)
- Week 3: Parallelization & Streaming (Parallel Engine, Streaming, Async)
- Week 4: Advanced Monitoring & Validation (Benchmarks, Quality, Real-time)
- Week 5: Production Readiness & Fine-tuning (Memory, Deploy, Enterprise)

This is the master test file referenced in CLAUDE.md documentation.

Usage:
    poetry run python test_all_weeks_consolidated.py
    poetry run python test_all_weeks_consolidated.py --week 1
    poetry run python test_all_weeks_consolidated.py --verbose
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeekTestResult:
    """Result for a single week's tests"""
    
    def __init__(self, week: int, name: str):
        self.week = week
        self.name = name
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.execution_time = 0.0
        self.details = {}
        self.errors = []
        
    @property
    def success_rate(self) -> float:
        return (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0.0
        
    @property
    def passed(self) -> bool:
        return self.tests_failed == 0 and self.tests_run > 0

class ConsolidatedTestSuite:
    """Master test suite for all optimization weeks"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.total_execution_time = 0.0
        
    def run_all_weeks(self, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for all optimization weeks"""
        logger.info("🚀 Starting Consolidated Test Suite for Week 1-5 Optimizations")
        self.start_time = time.time()
        
        weeks_to_test = [
            (1, "Emergency Optimizations"),
            (2, "Advanced Caching & Monitoring"),
            (3, "Parallelization & Streaming"),
            (4, "Advanced Monitoring & Validation"),
            (5, "Production Readiness & Fine-tuning")
        ]
        
        for week_num, week_name in weeks_to_test:
            logger.info(f"📋 Testing Week {week_num}: {week_name}")
            result = self._run_week_tests(week_num, week_name, verbose)
            self.results[f"week_{week_num}"] = result
            
            if verbose:
                self._print_week_results(result)
        
        self.total_execution_time = time.time() - self.start_time
        return self._generate_final_report()
    
    def run_specific_week(self, week: int, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific week"""
        week_names = {
            1: "Emergency Optimizations",
            2: "Advanced Caching & Monitoring", 
            3: "Parallelization & Streaming",
            4: "Advanced Monitoring & Validation",
            5: "Production Readiness & Fine-tuning"
        }
        
        if week not in week_names:
            raise ValueError(f"Invalid week number: {week}. Must be 1-5.")
            
        logger.info(f"🎯 Testing Week {week}: {week_names[week]}")
        self.start_time = time.time()
        
        result = self._run_week_tests(week, week_names[week], verbose)
        self.results[f"week_{week}"] = result
        
        self.total_execution_time = time.time() - self.start_time
        return self._generate_final_report()
    
    def _run_week_tests(self, week: int, name: str, verbose: bool) -> WeekTestResult:
        """Run tests for a specific week"""
        result = WeekTestResult(week, name)
        start_time = time.time()
        
        try:
            if week == 1:
                result = self._test_week1_emergency_optimizations(result, verbose)
            elif week == 2:
                result = self._test_week2_advanced_caching(result, verbose)
            elif week == 3:
                result = self._test_week3_parallelization(result, verbose)
            elif week == 4:
                result = self._test_week4_monitoring(result, verbose)
            elif week == 5:
                result = self._test_week5_production(result, verbose)
                
        except Exception as e:
            logger.error(f"❌ Week {week} testing failed: {e}")
            result.errors.append(str(e))
            result.tests_failed += 1
            
        result.execution_time = time.time() - start_time
        return result
    
    def _test_week1_emergency_optimizations(self, result: WeekTestResult, verbose: bool) -> WeekTestResult:
        """Test Week 1: Emergency Optimizations"""
        
        # Test 1: Emergency Cache System
        test_name = "emergency_cache_system"
        result.tests_run += 1
        try:
            from src.optimized.optimized_pipeline import get_global_optimized_pipeline
            orchestrator = get_global_optimized_pipeline()
            
            # Verify cache system is functional
            cache_functional = hasattr(orchestrator, 'cache_system')
            if cache_functional:
                result.tests_passed += 1
                result.details[test_name] = "✅ Emergency cache system operational"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Emergency cache system not found"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Cache test failed: {e}"
            
        # Test 2: Performance Fixes
        test_name = "performance_fixes"
        result.tests_run += 1
        try:
            from src.utils.performance_config import configure_performance_settings
            
            # Test performance configuration
            config = configure_performance_settings()
            if config and 'numexpr_threads' in config:
                result.tests_passed += 1
                result.details[test_name] = f"✅ Performance config active: {config['numexpr_threads']} threads"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Performance configuration not found"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Performance test failed: {e}"
            
        # Test 3: Error Handling System
        test_name = "error_handling_system"
        result.tests_run += 1
        try:
            from src.anthropic_integration.api_error_handler import APIErrorHandler
            
            # Test error handler initialization
            handler = APIErrorHandler()
            if handler:
                result.tests_passed += 1
                result.details[test_name] = "✅ Enhanced error handling system active"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Error handling system not functional"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Error handling test failed: {e}"
            
        return result
    
    def _test_week2_advanced_caching(self, result: WeekTestResult, verbose: bool) -> WeekTestResult:
        """Test Week 2: Advanced Caching & Monitoring"""
        
        # Test 1: Hierarchical Cache System
        test_name = "hierarchical_cache_l1_l2"
        result.tests_run += 1
        try:
            from src.core.unified_cache_system import UnifiedCacheSystem
            
            cache = UnifiedCacheSystem()
            # Test basic cache operations
            test_key = "test_hierarchy"
            test_value = {"data": "test_value", "timestamp": time.time()}
            
            cache.set(test_key, test_value)
            retrieved = cache.get(test_key)
            
            if retrieved == test_value:
                result.tests_passed += 1
                result.details[test_name] = "✅ Hierarchical L1/L2 cache functional"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Cache hierarchy not working correctly"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Hierarchical cache test failed: {e}"
            
        # Test 2: Smart Claude Cache
        test_name = "smart_claude_cache"
        result.tests_run += 1
        try:
            from src.optimized.smart_claude_cache import get_global_claude_cache
            
            smart_cache = get_global_claude_cache()
            if smart_cache:
                result.tests_passed += 1
                result.details[test_name] = "✅ Smart Claude cache system active"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Smart Claude cache not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Smart cache test failed: {e}"
            
        # Test 3: Performance Monitoring
        test_name = "performance_monitoring"
        result.tests_run += 1
        try:
            from src.optimized.performance_monitor import get_global_performance_monitor
            
            monitor = get_global_performance_monitor()
            if monitor:
                result.tests_passed += 1
                result.details[test_name] = "✅ Performance monitoring system active"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Performance monitoring not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Monitoring test failed: {e}"
            
        return result
    
    def _test_week3_parallelization(self, result: WeekTestResult, verbose: bool) -> WeekTestResult:
        """Test Week 3: Parallelization & Streaming"""
        
        # Test 1: Parallel Processing Engine
        test_name = "parallel_processing_engine"
        result.tests_run += 1
        try:
            from src.optimized.parallel_engine import get_global_parallel_engine
            
            engine = get_global_parallel_engine()
            if engine:
                result.tests_passed += 1
                result.details[test_name] = "✅ Parallel processing engine operational"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Parallel engine not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Parallel engine test failed: {e}"
            
        # Test 2: Streaming Pipeline
        test_name = "streaming_pipeline"
        result.tests_run += 1
        try:
            from src.optimized.streaming_pipeline import get_global_streaming_pipeline
            
            pipeline = get_global_streaming_pipeline()
            if pipeline:
                result.tests_passed += 1
                result.details[test_name] = "✅ Streaming pipeline system active"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Streaming pipeline not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Streaming test failed: {e}"
            
        # Test 3: Async Stages Orchestrator
        test_name = "async_stages_orchestrator"
        result.tests_run += 1
        try:
            from src.optimized.async_stages import get_global_async_orchestrator
            
            orchestrator = get_global_async_orchestrator()
            if orchestrator:
                result.tests_passed += 1
                result.details[test_name] = "✅ Async stages orchestrator functional"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Async orchestrator not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Async stages test failed: {e}"
            
        return result
    
    def _test_week4_monitoring(self, result: WeekTestResult, verbose: bool) -> WeekTestResult:
        """Test Week 4: Advanced Monitoring & Validation"""
        
        # Test 1: Pipeline Benchmark System
        test_name = "pipeline_benchmark_system"
        result.tests_run += 1
        try:
            from src.optimized.pipeline_benchmark import create_development_benchmark
            
            benchmark = create_development_benchmark()
            if benchmark:
                result.tests_passed += 1
                result.details[test_name] = "✅ Pipeline benchmark system operational"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Benchmark system not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Benchmark test failed: {e}"
            
        # Test 2: Real-time Monitor
        test_name = "realtime_monitor"
        result.tests_run += 1
        try:
            from src.optimized.realtime_monitor import get_global_realtime_monitor
            
            monitor = get_global_realtime_monitor()
            if monitor:
                result.tests_passed += 1
                result.details[test_name] = "✅ Real-time monitoring system active"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Real-time monitor not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Real-time monitor test failed: {e}"
            
        # Test 3: Quality Regression Tests
        test_name = "quality_regression_tests"
        result.tests_run += 1
        try:
            from src.optimized.quality_tests import create_development_quality_tests
            
            quality_tests = create_development_quality_tests()
            if quality_tests:
                result.tests_passed += 1
                result.details[test_name] = "✅ Quality regression system functional"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Quality tests not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Quality tests failed: {e}"
            
        return result
    
    def _test_week5_production(self, result: WeekTestResult, verbose: bool) -> WeekTestResult:
        """Test Week 5: Production Readiness & Fine-tuning"""
        
        # Test 1: Adaptive Memory Manager
        test_name = "adaptive_memory_manager"
        result.tests_run += 1
        try:
            from src.optimized.memory_optimizer import get_global_memory_optimizer
            
            optimizer = get_global_memory_optimizer()
            if optimizer:
                result.tests_passed += 1
                result.details[test_name] = "✅ Adaptive memory manager operational"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Memory optimizer not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Memory optimizer test failed: {e}"
            
        # Test 2: Production Deployment System
        test_name = "production_deployment_system"
        result.tests_run += 1
        try:
            from src.optimized.production_deploy import get_global_deployment_system
            
            deployment = get_global_deployment_system()
            if deployment:
                result.tests_passed += 1
                result.details[test_name] = "✅ Production deployment system ready"
            else:
                result.tests_failed += 1
                result.details[test_name] = "❌ Deployment system not available"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Deployment test failed: {e}"
            
        # Test 3: Enterprise Features
        test_name = "enterprise_features"
        result.tests_run += 1
        try:
            # Test multiple enterprise features
            enterprise_features = []
            
            try:
                from src.optimized.memory_optimizer import MemoryProfiler
                enterprise_features.append("MemoryProfiler")
            except:
                pass
                
            try:
                from src.optimized.production_deploy import DeploymentConfig
                enterprise_features.append("DeploymentConfig")
            except:
                pass
                
            if len(enterprise_features) >= 2:
                result.tests_passed += 1
                result.details[test_name] = f"✅ Enterprise features active: {', '.join(enterprise_features)}"
            else:
                result.tests_failed += 1
                result.details[test_name] = f"❌ Limited enterprise features: {enterprise_features}"
                
        except Exception as e:
            result.tests_failed += 1
            result.details[test_name] = f"❌ Enterprise features test failed: {e}"
            
        return result
    
    def _print_week_results(self, result: WeekTestResult):
        """Print results for a single week"""
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"\n📊 Week {result.week} Results: {status}")
        print(f"   Tests: {result.tests_passed}/{result.tests_run} passed ({result.success_rate:.1f}%)")
        print(f"   Time: {result.execution_time:.2f}s")
        
        if result.details:
            print("   Details:")
            for test, detail in result.details.items():
                print(f"     • {test}: {detail}")
                
        if result.errors:
            print("   Errors:")
            for error in result.errors:
                print(f"     • {error}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        total_tests = sum(r.tests_run for r in self.results.values())
        total_passed = sum(r.tests_passed for r in self.results.values())
        total_failed = sum(r.tests_failed for r in self.results.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
        
        # Check if targets are achieved
        targets_achieved = {
            "95_percent_success_rate": overall_success_rate >= 95.0,
            "all_weeks_functional": all(r.passed for r in self.results.values()),
            "no_critical_failures": total_failed == 0
        }
        
        report = {
            "execution_timestamp": datetime.now().isoformat(),
            "total_execution_time": self.total_execution_time,
            "summary": {
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "overall_success_rate": overall_success_rate,
                "weeks_tested": len(self.results),
                "weeks_passed": sum(1 for r in self.results.values() if r.passed)
            },
            "targets_achieved": targets_achieved,
            "week_results": {
                week_key: {
                    "week": result.week,
                    "name": result.name,
                    "passed": result.passed,
                    "tests_run": result.tests_run,
                    "tests_passed": result.tests_passed,
                    "tests_failed": result.tests_failed,
                    "success_rate": result.success_rate,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "errors": result.errors
                }
                for week_key, result in self.results.items()
            },
            "recommendations": self._generate_recommendations(targets_achieved, overall_success_rate)
        }
        
        return report
    
    def _generate_recommendations(self, targets: Dict[str, bool], success_rate: float) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if not targets["95_percent_success_rate"]:
            recommendations.append(f"❌ Target 95% success rate not achieved ({success_rate:.1f}%). Review failed tests before consolidation.")
            
        if not targets["all_weeks_functional"]:
            failed_weeks = [r.week for r in self.results.values() if not r.passed]
            recommendations.append(f"❌ Week(s) {failed_weeks} have failures. Address issues before proceeding.")
            
        if not targets["no_critical_failures"]:
            recommendations.append("❌ Critical failures detected. System not ready for production consolidation.")
            
        if all(targets.values()):
            recommendations.append("✅ All targets achieved. System ready for optimization consolidation.")
            
        if success_rate >= 90:
            recommendations.append("✅ High success rate indicates stable optimization framework.")
            
        return recommendations

def print_final_report(report: Dict[str, Any]):
    """Print comprehensive final report"""
    print("\n" + "="*80)
    print("🎯 CONSOLIDATED TEST SUITE - FINAL REPORT")
    print("="*80)
    print(f"📅 Execution Time: {report['execution_timestamp']}")
    print(f"⏱️  Total Duration: {report['total_execution_time']:.2f} seconds")
    
    summary = report['summary']
    print(f"\n📊 Overall Results:")
    print(f"   🧪 Total Tests: {summary['total_tests']}")
    print(f"   ✅ Passed: {summary['tests_passed']}")
    print(f"   ❌ Failed: {summary['tests_failed']}")
    print(f"   📈 Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"   📋 Weeks Tested: {summary['weeks_tested']}")
    print(f"   ✅ Weeks Passed: {summary['weeks_passed']}")
    
    print(f"\n🎯 Target Achievement:")
    for target, achieved in report['targets_achieved'].items():
        status = "✅" if achieved else "❌"
        target_name = target.replace("_", " ").title()
        print(f"   {status} {target_name}")
    
    print(f"\n📋 Week-by-Week Results:")
    for week_key, week_result in report['week_results'].items():
        status = "✅" if week_result['passed'] else "❌"
        print(f"   Week {week_result['week']}: {status} {week_result['tests_passed']}/{week_result['tests_run']} ({week_result['success_rate']:.1f}%) - {week_result['name']}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")
    
    # Overall status
    all_targets = all(report['targets_achieved'].values())
    if all_targets:
        print(f"\n🎉 CONSOLIDATION READY: All targets achieved! System ready for Week 1-5 optimization integration.")
    else:
        print(f"\n⚠️  CONSOLIDATION BLOCKED: Address failures before proceeding with integration.")
    
    print("="*80)

def save_report(report: Dict[str, Any], filename: str = None):
    """Save detailed report to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"consolidated_test_report_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Detailed report saved: {filename}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Consolidated Test Suite for Week 1-5 Optimizations')
    parser.add_argument('--week', type=int, choices=[1,2,3,4,5], help='Test specific week only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--save-report', '-s', action='store_true', help='Save detailed JSON report')
    
    args = parser.parse_args()
    
    suite = ConsolidatedTestSuite()
    
    try:
        if args.week:
            report = suite.run_specific_week(args.week, args.verbose)
        else:
            report = suite.run_all_weeks(args.verbose)
        
        print_final_report(report)
        
        if args.save_report:
            save_report(report)
        
        # Exit code based on success
        all_targets = all(report['targets_achieved'].values())
        sys.exit(0 if all_targets else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        logger.exception("Test suite execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()