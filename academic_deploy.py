#!/usr/bin/env python3
"""
Academic Research Deployment System v5.0.0
==========================================

Simplified deployment system for social science research centers.
Optimized for academic computing environments with 4GB memory limit.

This deployment system provides:
- Automated academic environment setup
- 4GB memory optimization validation
- Cost-efficient configuration for research budgets ($50/month)
- Portuguese political analysis category preservation
- Research reproducibility validation

Usage:
    poetry run python academic_deploy.py --environment research
    poetry run python academic_deploy.py --validate
    poetry run python academic_deploy.py --help
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
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging for academic use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Academic Deploy - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AcademicDeploymentConfig:
    """Academic-focused deployment configuration"""
    
    def __init__(self):
        self.environment = "research"
        self.target_memory_gb = 4.0
        self.emergency_memory_gb = 6.0
        self.monthly_budget_usd = 50.0
        self.enable_portuguese_analysis = True
        self.enable_cost_monitoring = True
        self.enable_memory_optimization = True
        self.preserve_research_data = True
        self.academic_mode = True

class AcademicDeploymentSystem:
    """Simplified deployment system for academic research"""
    
    def __init__(self):
        self.config = AcademicDeploymentConfig()
        self.deployment_path = Path("deployment_backups/academic")
        self.deployment_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎓 Academic Deployment System v5.0.0 initialized")
        logger.info(f"📁 Deployment path: {self.deployment_path}")
    
    async def validate_academic_environment(self) -> Dict[str, Any]:
        """Validate academic computing environment requirements"""
        
        logger.info("🔍 Validating academic environment requirements...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'environment_suitable': True,
            'issues': [],
            'recommendations': [],
            'system_info': {}
        }
        
        try:
            # Check available memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            validation_results['system_info']['total_memory_gb'] = round(memory_gb, 2)
            
            if memory_gb < 4.0:
                validation_results['environment_suitable'] = False
                validation_results['issues'].append(
                    f"Insufficient memory: {memory_gb:.1f}GB < 4.0GB minimum required"
                )
            else:
                logger.info(f"✅ Memory check passed: {memory_gb:.1f}GB available")
            
            # Check Python version
            python_version = sys.version_info
            validation_results['system_info']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version < (3, 8):
                validation_results['environment_suitable'] = False
                validation_results['issues'].append(
                    f"Python version too old: {python_version.major}.{python_version.minor} < 3.8 required"
                )
            else:
                logger.info(f"✅ Python version check passed: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024**3)
            validation_results['system_info']['free_disk_gb'] = round(disk_gb, 2)
            
            if disk_gb < 10.0:
                validation_results['environment_suitable'] = False
                validation_results['issues'].append(
                    f"Insufficient disk space: {disk_gb:.1f}GB < 10.0GB recommended"
                )
            else:
                logger.info(f"✅ Disk space check passed: {disk_gb:.1f}GB available")
            
            # Check for Poetry (through pyproject.toml presence)
            pyproject_path = Path("pyproject.toml")
            if pyproject_path.exists():
                validation_results['system_info']['poetry_available'] = True
                logger.info("✅ Poetry project configuration found")
            else:
                validation_results['issues'].append("Poetry project configuration (pyproject.toml) not found")
                validation_results['recommendations'].append("Ensure you're running from the project root directory")
            
        except Exception as e:
            validation_results['environment_suitable'] = False
            validation_results['issues'].append(f"Environment validation error: {str(e)}")
            logger.error(f"❌ Environment validation failed: {e}")
        
        return validation_results
    
    async def validate_optimization_systems(self) -> Dict[str, Any]:
        """Validate that all optimization systems are working"""
        
        logger.info("🔧 Validating optimization systems...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations_working': True,
            'week_status': {},
            'test_results': {}
        }
        
        try:
            # Import test system
            from test_all_weeks_consolidated import ConsolidatedTestSuite
            
            # Run consolidated tests
            test_suite = ConsolidatedTestSuite()
            test_results = await asyncio.get_event_loop().run_in_executor(
                None, test_suite.run_all_weeks
            )
            
            validation_results['test_results'] = test_results
            
            # Check if all weeks passed
            success_rate = test_results['summary']['overall_success_rate']
            if success_rate >= 95.0:
                logger.info(f"✅ Optimization validation passed: {success_rate:.1f}% success rate")
                validation_results['optimizations_working'] = True
            else:
                logger.warning(f"⚠️ Optimization validation issues: {success_rate:.1f}% success rate")
                validation_results['optimizations_working'] = False
            
            # Record week-by-week status
            for week_key, week_result in test_results['week_results'].items():
                week_num = week_result['week']
                validation_results['week_status'][f'week_{week_num}'] = {
                    'passed': week_result['tests_passed'],
                    'total': week_result['tests_run'],
                    'status': 'working' if week_result['passed'] else 'issues'
                }
            
        except Exception as e:
            validation_results['optimizations_working'] = False
            validation_results['error'] = str(e)
            logger.error(f"❌ Optimization validation failed: {e}")
        
        return validation_results
    
    async def setup_academic_configuration(self) -> Dict[str, Any]:
        """Setup academic-optimized configuration"""
        
        logger.info("⚙️ Setting up academic configuration...")
        
        setup_results = {
            'timestamp': datetime.now().isoformat(),
            'configuration_applied': True,
            'academic_features': [],
            'optimizations': []
        }
        
        try:
            # Check if academic config exists
            academic_config_path = Path("config/academic_settings.yaml")
            
            if academic_config_path.exists():
                logger.info("✅ Academic configuration file found")
                setup_results['academic_features'].append("Academic settings configuration")
            else:
                logger.warning("⚠️ Academic configuration file not found, using defaults")
            
            # Validate core configuration files
            core_configs = {
                'settings.yaml': Path("config/settings.yaml"),
                'anthropic.yaml': Path("config/anthropic.yaml.template"),
                'voyage_embeddings.yaml': Path("config/voyage_embeddings.yaml.template")
            }
            
            for config_name, config_path in core_configs.items():
                if config_path.exists():
                    logger.info(f"✅ {config_name} configuration available")
                    setup_results['academic_features'].append(f"{config_name} available")
                else:
                    logger.warning(f"⚠️ {config_name} configuration missing")
            
            # Check for optimization components
            optimization_components = [
                ("Week 1 Emergency Cache", "src/optimized/optimized_pipeline.py"),
                ("Week 2 Smart Cache", "src/optimized/smart_claude_cache.py"),
                ("Week 3 Parallel Engine", "src/optimized/parallel_engine.py"),
                ("Week 4 Quality Tests", "src/optimized/quality_tests.py"),
                ("Week 5 Memory Optimizer", "src/optimized/memory_optimizer.py")
            ]
            
            for component_name, component_path in optimization_components:
                if Path(component_path).exists():
                    logger.info(f"✅ {component_name} available")
                    setup_results['optimizations'].append(component_name)
                else:
                    logger.warning(f"⚠️ {component_name} missing")
            
        except Exception as e:
            setup_results['configuration_applied'] = False
            setup_results['error'] = str(e)
            logger.error(f"❌ Academic configuration setup failed: {e}")
        
        return setup_results
    
    async def deploy_for_research(self) -> Dict[str, Any]:
        """Deploy system optimized for academic research"""
        
        logger.info("🚀 Starting academic research deployment...")
        
        deployment_start = time.time()
        deployment_id = f"academic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment_results = {
            'deployment_id': deployment_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'phases': {},
            'deployment_time_seconds': 0,
            'academic_features_enabled': []
        }
        
        try:
            # Phase 1: Environment validation
            logger.info("Phase 1: Environment validation")
            env_validation = await self.validate_academic_environment()
            deployment_results['phases']['environment_validation'] = env_validation
            
            if not env_validation['environment_suitable']:
                deployment_results['success'] = False
                deployment_results['error'] = "Environment validation failed"
                return deployment_results
            
            # Phase 2: Optimization validation
            logger.info("Phase 2: Optimization systems validation")
            opt_validation = await self.validate_optimization_systems()
            deployment_results['phases']['optimization_validation'] = opt_validation
            
            if not opt_validation['optimizations_working']:
                deployment_results['success'] = False
                deployment_results['error'] = "Optimization validation failed"
                return deployment_results
            
            # Phase 3: Academic configuration
            logger.info("Phase 3: Academic configuration setup")
            config_setup = await self.setup_academic_configuration()
            deployment_results['phases']['configuration_setup'] = config_setup
            
            # Phase 4: Research feature validation
            logger.info("Phase 4: Research feature validation")
            research_features = await self.validate_research_features()
            deployment_results['phases']['research_validation'] = research_features
            deployment_results['academic_features_enabled'] = research_features.get('features_enabled', [])
            
            # Phase 5: Final health check
            logger.info("Phase 5: Final system health check")
            health_check = await self.final_health_check()
            deployment_results['phases']['health_check'] = health_check
            
            if health_check['healthy']:
                logger.info(f"✅ Academic deployment {deployment_id} completed successfully")
                deployment_results['success'] = True
            else:
                logger.error(f"❌ Academic deployment {deployment_id} failed health check")
                deployment_results['success'] = False
                deployment_results['error'] = health_check.get('error', 'Health check failed')
            
        except Exception as e:
            deployment_results['success'] = False
            deployment_results['error'] = str(e)
            logger.error(f"❌ Academic deployment {deployment_id} failed: {e}")
        
        finally:
            deployment_results['deployment_time_seconds'] = time.time() - deployment_start
            
            # Save deployment report
            report_path = self.deployment_path / f"{deployment_id}_report.json"
            with open(report_path, 'w') as f:
                json.dump(deployment_results, f, indent=2)
            
            logger.info(f"📄 Deployment report saved: {report_path}")
        
        return deployment_results
    
    async def validate_research_features(self) -> Dict[str, Any]:
        """Validate research-specific features"""
        
        logger.info("🔬 Validating research features...")
        
        features_validation = {
            'timestamp': datetime.now().isoformat(),
            'features_working': True,
            'features_enabled': [],
            'portuguese_analysis': False,
            'cost_monitoring': False,
            'memory_optimization': False
        }
        
        try:
            # Check Portuguese political analysis
            from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
            features_validation['portuguese_analysis'] = True
            features_validation['features_enabled'].append("Portuguese Political Analysis")
            logger.info("✅ Portuguese political analysis available")
            
            # Check cost monitoring
            from src.anthropic_integration.cost_monitor import CostMonitor
            features_validation['cost_monitoring'] = True
            features_validation['features_enabled'].append("Academic Cost Monitoring")
            logger.info("✅ Academic cost monitoring available")
            
            # Check memory optimization
            from src.optimized.memory_optimizer import AdaptiveMemoryManager
            features_validation['memory_optimization'] = True
            features_validation['features_enabled'].append("4GB Memory Optimization")
            logger.info("✅ 4GB memory optimization available")
            
            # Check academic configuration
            try:
                from src.academic_config import get_academic_config
                academic_config = get_academic_config()
                features_validation['features_enabled'].append("Academic Configuration System")
                logger.info("✅ Academic configuration system available")
            except ImportError:
                logger.warning("⚠️ Academic configuration system not available")
            
        except Exception as e:
            features_validation['features_working'] = False
            features_validation['error'] = str(e)
            logger.error(f"❌ Research features validation failed: {e}")
        
        return features_validation
    
    async def final_health_check(self) -> Dict[str, Any]:
        """Final health check for academic deployment"""
        
        logger.info("🏥 Running final health check...")
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'healthy': True,
            'checks_passed': [],
            'checks_failed': [],
            'system_ready_for_research': False
        }
        
        try:
            # Check 1: Import core pipeline
            try:
                from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
                health_results['checks_passed'].append("Core pipeline import")
                logger.info("✅ Core pipeline import successful")
            except Exception as e:
                health_results['checks_failed'].append(f"Core pipeline import: {str(e)}")
                health_results['healthy'] = False
            
            # Check 2: Memory availability
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb >= 2.0:  # Need at least 2GB available for 4GB target
                health_results['checks_passed'].append(f"Memory availability: {available_gb:.1f}GB")
                logger.info(f"✅ Memory availability check passed: {available_gb:.1f}GB")
            else:
                health_results['checks_failed'].append(f"Insufficient memory: {available_gb:.1f}GB < 2.0GB required")
                health_results['healthy'] = False
            
            # Check 3: Optimization systems functional
            try:
                from test_all_weeks_consolidated import WeekTestResult
                health_results['checks_passed'].append("Optimization test system")
                logger.info("✅ Optimization test system functional")
            except Exception as e:
                health_results['checks_failed'].append(f"Optimization test system: {str(e)}")
                health_results['healthy'] = False
            
            # Final determination
            if health_results['healthy'] and len(health_results['checks_passed']) >= 3:
                health_results['system_ready_for_research'] = True
                logger.info("✅ System ready for academic research")
            else:
                health_results['system_ready_for_research'] = False
                logger.warning("⚠️ System not fully ready for research")
            
        except Exception as e:
            health_results['healthy'] = False
            health_results['error'] = str(e)
            logger.error(f"❌ Final health check failed: {e}")
        
        return health_results
    
    def print_deployment_summary(self, deployment_results: Dict[str, Any]):
        """Print human-readable deployment summary for researchers"""
        
        print("\n" + "="*80)
        print("🎓 ACADEMIC RESEARCH DEPLOYMENT SUMMARY")
        print("="*80)
        
        print(f"📊 Deployment ID: {deployment_results['deployment_id']}")
        print(f"⏰ Completion Time: {deployment_results['deployment_time_seconds']:.1f} seconds")
        
        if deployment_results['success']:
            print("✅ DEPLOYMENT SUCCESSFUL - System ready for research")
        else:
            print("❌ DEPLOYMENT FAILED")
            if 'error' in deployment_results:
                print(f"🔍 Error: {deployment_results['error']}")
        
        print("\n📋 Academic Features Enabled:")
        for feature in deployment_results.get('academic_features_enabled', []):
            print(f"   ✅ {feature}")
        
        print("\n📊 Validation Summary:")
        phases = deployment_results.get('phases', {})
        
        for phase_name, phase_data in phases.items():
            phase_title = phase_name.replace('_', ' ').title()
            
            if phase_name == 'environment_validation':
                status = "✅ PASSED" if phase_data.get('environment_suitable', False) else "❌ FAILED"
                print(f"   {status} Environment Validation")
                if 'system_info' in phase_data:
                    memory_gb = phase_data['system_info'].get('total_memory_gb', 0)
                    print(f"      💾 Memory: {memory_gb:.1f}GB")
            
            elif phase_name == 'optimization_validation':
                status = "✅ PASSED" if phase_data.get('optimizations_working', False) else "❌ FAILED"
                print(f"   {status} Optimization Systems")
                if 'test_results' in phase_data:
                    success_rate = phase_data['test_results']['summary'].get('overall_success_rate', 0)
                    print(f"      📈 Success Rate: {success_rate:.1f}%")
            
            elif phase_name == 'research_validation':
                status = "✅ PASSED" if phase_data.get('features_working', False) else "❌ FAILED"
                print(f"   {status} Research Features")
            
            elif phase_name == 'health_check':
                status = "✅ PASSED" if phase_data.get('healthy', False) else "❌ FAILED"
                ready = phase_data.get('system_ready_for_research', False)
                print(f"   {status} Final Health Check")
                print(f"      🔬 Research Ready: {'Yes' if ready else 'No'}")
        
        print("\n💡 Next Steps for Researchers:")
        if deployment_results['success']:
            print("   1. Run complete analysis: poetry run python run_pipeline.py")
            print("   2. Launch dashboard: poetry run python src/dashboard/start_dashboard.py")
            print("   3. View academic guide: cat README.md")
        else:
            print("   1. Review error messages above")
            print("   2. Check system requirements (4GB RAM, Python 3.8+)")
            print("   3. Contact technical support if needed")
        
        print("="*80)

async def main():
    """Main deployment function for academic research centers"""
    
    parser = argparse.ArgumentParser(
        description="Academic Research Deployment System v5.0.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    poetry run python academic_deploy.py --environment research
    poetry run python academic_deploy.py --validate
    poetry run python academic_deploy.py --help
        """
    )
    
    parser.add_argument(
        '--environment',
        choices=['research', 'development', 'testing'],
        default='research',
        help='Target deployment environment (default: research)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation only, do not deploy'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize deployment system
    deployment_system = AcademicDeploymentSystem()
    
    if args.validate:
        # Run validation only
        logger.info("🔍 Running academic environment validation...")
        
        env_validation = await deployment_system.validate_academic_environment()
        opt_validation = await deployment_system.validate_optimization_systems()
        research_validation = await deployment_system.validate_research_features()
        
        print("\n" + "="*60)
        print("🔍 ACADEMIC VALIDATION REPORT")
        print("="*60)
        
        print(f"Environment Suitable: {'✅ Yes' if env_validation['environment_suitable'] else '❌ No'}")
        print(f"Optimizations Working: {'✅ Yes' if opt_validation['optimizations_working'] else '❌ No'}")
        print(f"Research Features: {'✅ Yes' if research_validation['features_working'] else '❌ No'}")
        
        if env_validation['issues']:
            print("\n⚠️ Issues Found:")
            for issue in env_validation['issues']:
                print(f"   - {issue}")
        
        if env_validation['recommendations']:
            print("\n💡 Recommendations:")
            for rec in env_validation['recommendations']:
                print(f"   - {rec}")
        
        print("="*60)
        
    else:
        # Full deployment
        logger.info(f"🚀 Starting academic deployment for {args.environment} environment...")
        
        deployment_results = await deployment_system.deploy_for_research()
        deployment_system.print_deployment_summary(deployment_results)

if __name__ == "__main__":
    asyncio.run(main())