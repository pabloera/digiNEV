#!/usr/bin/env python3
"""
BACKUP EXECUTION ORCHESTRATOR
Digital Discourse Monitor v5.0.0

Main script to execute comprehensive backup and validation before consolidation

Created: 2025-06-15
Author: Backup & Rollback Specialist
Purpose: Orchestrate complete backup and validation process
"""

import sys
import json
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, List

def print_banner():
    """Print system banner"""
    print("=" * 80)
    print("🛡️  BACKUP & ROLLBACK SYSTEM")
    print("Digital Discourse Monitor v5.0.0")
    print("Enterprise-Grade Backup for Optimization Consolidation")
    print("=" * 80)
    print(f"Execution Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def execute_backup_system() -> Dict[str, Any]:
    """Execute the backup system"""
    print("\n🔄 STEP 1: CREATING COMPREHENSIVE BACKUP")
    print("-" * 50)
    
    try:
        # Import and run backup system
        from backup_system import main as backup_main
        
        # Capture backup execution
        print("Executing backup system...")
        backup_main()
        
        return {
            'success': True,
            'message': 'Backup system executed successfully',
            'step': 'backup_creation'
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Backup system failed: {e}',
            'step': 'backup_creation',
            'error': str(e)
        }

def execute_checkpoint_framework() -> Dict[str, Any]:
    """Execute checkpoint validation"""
    print("\n✅ STEP 2: EXECUTING VALIDATION CHECKPOINTS")
    print("-" * 50)
    
    try:
        # Import and run checkpoint framework
        from checkpoint_framework import main as checkpoint_main
        
        # Execute checkpoints
        print("Running validation checkpoints...")
        can_proceed = checkpoint_main()
        
        return {
            'success': can_proceed,
            'message': 'Checkpoint validation completed' if can_proceed else 'Critical checkpoints failed',
            'step': 'checkpoint_validation',
            'can_proceed': can_proceed
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Checkpoint framework failed: {e}',
            'step': 'checkpoint_validation',
            'error': str(e),
            'can_proceed': False
        }

def execute_backup_validation() -> Dict[str, Any]:
    """Execute backup validation"""
    print("\n🔍 STEP 3: VALIDATING BACKUP INTEGRITY")
    print("-" * 50)
    
    try:
        # Import and run backup validator
        from backup_validator import main as validator_main
        
        # Execute validation
        print("Running backup integrity validation...")
        validation_success = validator_main()
        
        return {
            'success': validation_success,
            'message': 'Backup validation completed' if validation_success else 'Backup validation failed',
            'step': 'backup_validation',
            'validated': validation_success
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Backup validation failed: {e}',
            'step': 'backup_validation',
            'error': str(e),
            'validated': False
        }

def check_system_prerequisites() -> Dict[str, Any]:
    """Check system prerequisites"""
    print("\n🔧 STEP 0: CHECKING SYSTEM PREREQUISITES")
    print("-" * 50)
    
    prerequisites = {
        'poetry_env': False,
        'backup_directory': False,
        'optimization_files': False,
        'python_version': False
    }
    
    issues = []
    
    try:
        # Check Poetry environment
        result = subprocess.run(['poetry', 'env', 'info'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            prerequisites['poetry_env'] = True
            print("✅ Poetry environment active")
        else:
            issues.append("Poetry environment not active")
            print("❌ Poetry environment not active")
            
    except Exception as e:
        issues.append(f"Poetry check failed: {e}")
        print(f"❌ Poetry check failed: {e}")
    
    # Check backup directory
    backup_dir = Path("backup")
    if backup_dir.exists():
        prerequisites['backup_directory'] = True
        print("✅ Backup directory exists")
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)
        prerequisites['backup_directory'] = True
        print("✅ Created backup directory")
    
    # Check optimization files
    optimization_files = [
        "src/optimized/__init__.py",
        "src/optimized/optimized_pipeline.py",
        "src/optimized/memory_optimizer.py",
        "src/optimized/production_deploy.py"
    ]
    
    missing_files = []
    for file_path in optimization_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            
    if not missing_files:
        prerequisites['optimization_files'] = True
        print("✅ All optimization files present")
    else:
        issues.append(f"Missing optimization files: {missing_files}")
        print(f"❌ Missing optimization files: {missing_files}")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        prerequisites['python_version'] = True
        print(f"✅ Python version {python_version.major}.{python_version.minor} compatible")
    else:
        issues.append(f"Python version {python_version.major}.{python_version.minor} not supported")
        print(f"❌ Python version {python_version.major}.{python_version.minor} not supported")
    
    all_ready = all(prerequisites.values())
    
    return {
        'success': all_ready,
        'message': 'System prerequisites met' if all_ready else f'Prerequisites issues: {len(issues)}',
        'step': 'prerequisites',
        'prerequisites': prerequisites,
        'issues': issues
    }

def generate_execution_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive execution report"""
    
    # Calculate summary
    total_steps = len(results)
    successful_steps = sum(1 for r in results if r.get('success', False))
    failed_steps = total_steps - successful_steps
    
    # Determine overall status  
    overall_success = failed_steps == 0
    can_proceed_consolidation = all(
        r.get('can_proceed', True) and r.get('validated', True) 
        for r in results if r.get('success', False)
    )
    
    # Create report
    report = {
        'execution_summary': {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'overall_success': overall_success,
            'can_proceed_consolidation': can_proceed_consolidation
        },
        'step_results': results,
        'recommendations': []
    }
    
    # Add recommendations
    if overall_success and can_proceed_consolidation:
        report['recommendations'].append("✅ All systems ready - Consolidation can proceed safely")
        report['recommendations'].append("🚀 Execute consolidation with confidence")
        report['recommendations'].append("📋 Rollback scripts are ready if needed")
    else:
        report['recommendations'].append("❌ System not ready for consolidation")
        report['recommendations'].append("🛠️ Resolve failed steps before proceeding")
        report['recommendations'].append("📞 Review error messages and logs")
        
        # Specific recommendations based on failures
        for result in results:
            if not result.get('success', False):
                step = result.get('step', 'unknown')
                if step == 'prerequisites':
                    report['recommendations'].append("🔧 Fix system prerequisites first")
                elif step == 'backup_creation':
                    report['recommendations'].append("💾 Resolve backup creation issues")
                elif step == 'checkpoint_validation':
                    report['recommendations'].append("✅ Address checkpoint validation failures")
                elif step == 'backup_validation':
                    report['recommendations'].append("🔍 Fix backup integrity issues")
    
    return report

def save_execution_report(report: Dict[str, Any]):
    """Save execution report to file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"backup/execution_report_{timestamp}.json")
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n📄 Execution report saved: {report_file}")

def print_final_summary(report: Dict[str, Any]):
    """Print final execution summary"""
    summary = report['execution_summary']
    
    print("\n" + "=" * 80)
    print("🎯 BACKUP SYSTEM EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Successful: {summary['successful_steps']}")
    print(f"Failed: {summary['failed_steps']}")
    print(f"Overall Success: {'✅ YES' if summary['overall_success'] else '❌ NO'}")
    print(f"Ready for Consolidation: {'✅ YES' if summary['can_proceed_consolidation'] else '❌ NO'}")
    
    print("\n📋 RECOMMENDATIONS:")
    for recommendation in report['recommendations']:
        print(f"  {recommendation}")
    
    print("\n" + "=" * 80)
    
    if summary['can_proceed_consolidation']:
        print("🎉 BACKUP SYSTEM READY - CONSOLIDATION CAN PROCEED")
        print("💡 NEXT STEPS:")
        print("   1. Review execution report")
        print("   2. Begin optimization consolidation")
        print("   3. Keep rollback scripts ready")
        print("   4. Monitor system during consolidation")
    else:
        print("🚨 BACKUP SYSTEM NOT READY - RESOLVE ISSUES FIRST")
        print("💡 REQUIRED ACTIONS:")
        print("   1. Review failed steps above")
        print("   2. Fix identified issues")
        print("   3. Re-run backup system")
        print("   4. Do not proceed with consolidation")
    
    print("=" * 80)

def main():
    """Main backup execution orchestration"""
    
    # Print banner
    print_banner()
    
    # Execute all steps
    results = []
    
    # Step 0: Prerequisites
    prereq_result = check_system_prerequisites()
    results.append(prereq_result)
    
    if not prereq_result['success']:
        print("\n🚨 PREREQUISITES FAILED - CANNOT CONTINUE")
        report = generate_execution_report(results)
        save_execution_report(report)
        print_final_summary(report)
        return False
    
    # Step 1: Backup System
    backup_result = execute_backup_system()
    results.append(backup_result)
    
    if not backup_result['success']:
        print("\n🚨 BACKUP CREATION FAILED - STOPPING EXECUTION")
        report = generate_execution_report(results)
        save_execution_report(report)
        print_final_summary(report)
        return False
    
    # Step 2: Checkpoint Framework
    checkpoint_result = execute_checkpoint_framework()
    results.append(checkpoint_result)
    
    # Step 3: Backup Validation (even if checkpoints failed)
    validation_result = execute_backup_validation()
    results.append(validation_result)
    
    # Generate and save report
    report = generate_execution_report(results)
    save_execution_report(report)
    print_final_summary(report)
    
    # Return success status
    return report['execution_summary']['can_proceed_consolidation']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)