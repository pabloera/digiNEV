#!/usr/bin/env python3
"""
CHECKPOINT FRAMEWORK for Optimization Consolidation
Digital Discourse Monitor v5.0.0

Enterprise-grade checkpoint system for safe consolidation process

Created: 2025-06-15
Author: Backup & Rollback Specialist
Purpose: Validation checkpoints for consolidation process
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import subprocess
import sys

class CheckpointStatus(Enum):
    """Status of checkpoint validation"""
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class CheckpointResult:
    """Result of a checkpoint validation"""
    checkpoint_id: str
    name: str
    status: CheckpointStatus
    timestamp: str
    duration_seconds: float
    message: str
    details: Dict[str, Any]
    errors: List[str]

class ConsolidationCheckpoint:
    """
    Individual checkpoint for consolidation validation
    """
    
    def __init__(self, checkpoint_id: str, name: str, description: str, 
                 validation_func: Callable, critical: bool = True):
        self.checkpoint_id = checkpoint_id
        self.name = name
        self.description = description
        self.validation_func = validation_func
        self.critical = critical
        self.logger = logging.getLogger(f'Checkpoint.{checkpoint_id}')
        
    def execute(self) -> CheckpointResult:
        """Execute checkpoint validation"""
        start_time = datetime.datetime.now()
        
        try:
            self.logger.info(f"Starting checkpoint: {self.name}")
            
            # Execute validation function
            result = self.validation_func()
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.get('success', False):
                status = CheckpointStatus.PASSED
                message = result.get('message', 'Checkpoint passed')
                errors = []
            else:
                status = CheckpointStatus.FAILED
                message = result.get('message', 'Checkpoint failed')
                errors = result.get('errors', [])
                
            return CheckpointResult(
                checkpoint_id=self.checkpoint_id,
                name=self.name,
                status=status,
                timestamp=start_time.isoformat(),
                duration_seconds=duration,
                message=message,
                details=result.get('details', {}),
                errors=errors
            )
            
        except Exception as e:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Checkpoint {self.checkpoint_id} failed with exception: {e}")
            
            return CheckpointResult(
                checkpoint_id=self.checkpoint_id,
                name=self.name,
                status=CheckpointStatus.FAILED,
                timestamp=start_time.isoformat(),
                duration_seconds=duration,
                message=f"Exception: {str(e)}",
                details={},
                errors=[str(e)]
            )

class CheckpointFramework:
    """
    Framework for managing consolidation checkpoints
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_root = self.project_root / "backup"
        self.checkpoint_dir = self.backup_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize checkpoints
        self.checkpoints: List[ConsolidationCheckpoint] = []
        self.results: List[CheckpointResult] = []
        
        self.logger.info("Checkpoint framework initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.checkpoint_dir / f"checkpoints_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CheckpointFramework')
        
    def register_checkpoint(self, checkpoint: ConsolidationCheckpoint):
        """Register a checkpoint for execution"""
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Registered checkpoint: {checkpoint.checkpoint_id} - {checkpoint.name}")
        
    def execute_all_checkpoints(self) -> Dict[str, Any]:
        """Execute all registered checkpoints"""
        self.logger.info(f"Executing {len(self.checkpoints)} checkpoints...")
        
        execution_summary = {
            'total_checkpoints': len(self.checkpoints),
            'passed': 0,
            'failed': 0,
            'critical_failures': 0,
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': 0,
            'can_proceed': True
        }
        
        start_time = datetime.datetime.now()
        
        for checkpoint in self.checkpoints:
            result = checkpoint.execute()
            self.results.append(result)
            
            if result.status == CheckpointStatus.PASSED:
                execution_summary['passed'] += 1
            elif result.status == CheckpointStatus.FAILED:
                execution_summary['failed'] += 1
                if checkpoint.critical:
                    execution_summary['critical_failures'] += 1
                    
        end_time = datetime.datetime.now()
        execution_summary['end_time'] = end_time.isoformat()
        execution_summary['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Determine if consolidation can proceed
        if execution_summary['critical_failures'] > 0:
            execution_summary['can_proceed'] = False
            self.logger.error(f"Critical failures detected: {execution_summary['critical_failures']}")
        else:
            self.logger.info("All critical checkpoints passed")
            
        # Save results
        self.save_checkpoint_results(execution_summary)
        
        return execution_summary
        
    def save_checkpoint_results(self, summary: Dict[str, Any]):
        """Save checkpoint results to file"""
        results_data = {
            'summary': summary,
            'results': [asdict(result) for result in self.results]
        }
        
        results_file = self.checkpoint_dir / f"checkpoint_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        self.logger.info(f"Checkpoint results saved to: {results_file}")

def create_consolidation_checkpoints() -> CheckpointFramework:
    """Create all checkpoints for optimization consolidation"""
    
    framework = CheckpointFramework()
    
    # Checkpoint 1: Backup Integrity
    def validate_backup_integrity():
        """Validate that backup was created successfully"""
        backup_root = Path("backup")
        
        # Check if backup directories exist
        required_dirs = ['optimization_files', 'metadata', 'checksums', 'scripts', 'archives']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not (backup_root / dir_name).exists():
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            return {
                'success': False,
                'message': f"Missing backup directories: {missing_dirs}",
                'errors': [f"Directory not found: {d}" for d in missing_dirs]
            }
            
        # Check for recent backup session
        session_dirs = list(backup_root.glob("session_*"))
        if not session_dirs:
            return {
                'success': False,
                'message': "No backup sessions found",
                'errors': ["No session directories found"]
            }
            
        # Check latest session
        latest_session = max(session_dirs, key=lambda x: x.name)
        manifest_file = latest_session / "backup_manifest.json"
        
        if not manifest_file.exists():
            return {
                'success': False,
                'message': "Backup manifest not found",
                'errors': ["backup_manifest.json not found in latest session"]
            }
            
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                
            return {
                'success': True,
                'message': f"Backup integrity validated - {manifest['summary']['success_count']} files backed up",
                'details': {
                    'backup_id': manifest['backup_id'],
                    'total_files': manifest['total_files'],
                    'success_count': manifest['summary']['success_count'],
                    'error_count': manifest['summary']['error_count']
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error reading backup manifest: {e}",
                'errors': [str(e)]
            }
    
    framework.register_checkpoint(ConsolidationCheckpoint(
        checkpoint_id="CP001",
        name="Backup Integrity Validation",
        description="Validate that backup was created successfully",
        validation_func=validate_backup_integrity,
        critical=True
    ))
    
    # Checkpoint 2: Rollback Scripts
    def validate_rollback_scripts():
        """Validate rollback scripts are in place"""
        scripts_dir = Path("backup/scripts")
        
        required_scripts = ["rollback.sh", "rollback.py"]
        missing_scripts = []
        
        for script in required_scripts:
            script_path = scripts_dir / script
            if not script_path.exists():
                missing_scripts.append(script)
            elif not script_path.is_file():
                missing_scripts.append(f"{script} (not a file)")
                
        if missing_scripts:
            return {
                'success': False,
                'message': f"Missing rollback scripts: {missing_scripts}",
                'errors': [f"Script not found: {s}" for s in missing_scripts]
            }
            
        # Check if scripts are executable
        for script in required_scripts:
            script_path = scripts_dir / script
            if not script_path.stat().st_mode & 0o111:  # Check execute permission
                return {
                    'success': False,
                    'message': f"Script not executable: {script}",
                    'errors': [f"Missing execute permission: {script}"]
                }
                
        return {
            'success': True,
            'message': "Rollback scripts validated",
            'details': {
                'scripts_found': required_scripts,
                'scripts_directory': str(scripts_dir)
            }
        }
    
    framework.register_checkpoint(ConsolidationCheckpoint(
        checkpoint_id="CP002",
        name="Rollback Scripts Validation",
        description="Validate rollback scripts are in place and executable",
        validation_func=validate_rollback_scripts,
        critical=True
    ))
    
    # Checkpoint 3: File Checksums
    def validate_file_checksums():
        """Validate file checksums match backup"""
        optimization_files = [
            "src/optimized/__init__.py",
            "src/optimized/async_stages.py",
            "src/optimized/emergency_embeddings.py", 
            "src/optimized/memory_optimizer.py",
            "src/optimized/optimized_pipeline.py",
            "src/optimized/parallel_engine.py",
            "src/optimized/performance_monitor.py",
            "src/optimized/pipeline_benchmark.py",
            "src/optimized/production_deploy.py",
            "src/optimized/quality_tests.py",
            "src/optimized/realtime_monitor.py",
            "src/optimized/smart_claude_cache.py",
            "src/optimized/streaming_pipeline.py",
            "src/optimized/unified_embeddings_engine.py",
            "test_all_weeks_consolidated.py"
        ]
        
        checksum_mismatches = []
        
        for file_path in optimization_files:
            file_full_path = Path(file_path)
            
            if not file_full_path.exists():
                checksum_mismatches.append(f"File not found: {file_path}")
                continue
                
            # Calculate current checksum
            sha256_hash = hashlib.sha256()
            try:
                with open(file_full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                current_checksum = sha256_hash.hexdigest()
                
                # Find corresponding backup checksum
                checksums_dir = Path("backup/checksums")
                filename = file_full_path.name
                
                # Look for checksum file
                checksum_files = list(checksums_dir.glob(f"{filename}_*.sha256"))
                
                if not checksum_files:
                    checksum_mismatches.append(f"No backup checksum found for: {file_path}")
                    continue
                    
                # Get latest checksum
                latest_checksum_file = max(checksum_files, key=lambda x: x.name)
                
                with open(latest_checksum_file, 'r') as f:
                    backup_checksum = f.read().split()[0]
                    
                if current_checksum != backup_checksum:
                    checksum_mismatches.append(f"Checksum mismatch: {file_path}")
                    
            except Exception as e:
                checksum_mismatches.append(f"Error checking {file_path}: {e}")
                
        if checksum_mismatches:
            return {
                'success': False,
                'message': f"Checksum validation failed for {len(checksum_mismatches)} files",
                'errors': checksum_mismatches
            }
            
        return {
            'success': True,
            'message': f"Checksum validation passed for {len(optimization_files)} files",
            'details': {
                'files_validated': len(optimization_files),
                'files_checked': optimization_files
            }
        }
    
    framework.register_checkpoint(ConsolidationCheckpoint(
        checkpoint_id="CP003",
        name="File Checksum Validation",
        description="Validate current files match backup checksums",
        validation_func=validate_file_checksums,
        critical=True
    ))
    
    # Checkpoint 4: System Tests
    def validate_system_functionality():
        """Validate system is functional before consolidation"""
        try:
            # Test if we can import the optimized modules
            sys.path.insert(0, str(Path.cwd()))
            
            test_imports = [
                "src.optimized.optimized_pipeline",
                "src.optimized.memory_optimizer", 
                "src.optimized.production_deploy"
            ]
            
            import_errors = []
            
            for module_name in test_imports:
                try:
                    __import__(module_name)
                except Exception as e:
                    import_errors.append(f"Import failed for {module_name}: {e}")
                    
            if import_errors:
                return {
                    'success': False,
                    'message': f"Import validation failed for {len(import_errors)} modules",
                    'errors': import_errors
                }
                
            return {
                'success': True,
                'message': f"System functionality validated - {len(test_imports)} modules imported successfully",
                'details': {
                    'modules_tested': test_imports,
                    'import_success': True
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"System validation failed: {e}",
                'errors': [str(e)]
            }
    
    framework.register_checkpoint(ConsolidationCheckpoint(
        checkpoint_id="CP004",
        name="System Functionality Validation",
        description="Validate system functionality before consolidation",
        validation_func=validate_system_functionality,
        critical=True
    ))
    
    # Checkpoint 5: Environment Check
    def validate_environment():
        """Validate environment is ready for consolidation"""
        try:
            # Check Poetry environment
            result = subprocess.run(['poetry', 'env', 'info'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'message': "Poetry environment not active",
                    'errors': ["Poetry environment check failed"]
                }
                
            # Check for required directories
            required_dirs = ['src/optimized', 'backup', 'tests']
            missing_dirs = []
            
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing_dirs.append(dir_name)
                    
            if missing_dirs:
                return {
                    'success': False,
                    'message': f"Missing required directories: {missing_dirs}",
                    'errors': [f"Directory not found: {d}" for d in missing_dirs]
                }
                
            return {
                'success': True,
                'message': "Environment validation passed",
                'details': {
                    'poetry_active': True,
                    'required_directories': required_dirs
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Environment validation failed: {e}",
                'errors': [str(e)]
            }
    
    framework.register_checkpoint(ConsolidationCheckpoint(
        checkpoint_id="CP005",
        name="Environment Validation",
        description="Validate environment is ready for consolidation",
        validation_func=validate_environment,
        critical=True
    ))
    
    return framework

def main():
    """Main checkpoint execution"""
    print("=" * 70)
    print("CHECKPOINT FRAMEWORK")
    print("Digital Discourse Monitor v5.0.0")
    print("Consolidation Validation Checkpoints")
    print("=" * 70)
    
    # Create checkpoint framework
    framework = create_consolidation_checkpoints()
    
    # Execute all checkpoints
    summary = framework.execute_all_checkpoints()
    
    # Print results
    print("\n" + "=" * 70)
    print("CHECKPOINT EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total checkpoints: {summary['total_checkpoints']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Critical failures: {summary['critical_failures']}")
    print(f"Duration: {summary['duration_seconds']:.2f} seconds")
    
    if summary['can_proceed']:
        print("\n✅ ALL CRITICAL CHECKPOINTS PASSED")
        print("🚀 CONSOLIDATION CAN PROCEED SAFELY")
    else:
        print("\n❌ CRITICAL CHECKPOINTS FAILED")
        print("🛑 CONSOLIDATION CANNOT PROCEED")
        print("⚠️  RESOLVE ISSUES BEFORE CONTINUING")
        
    # Print individual results
    print("\n" + "=" * 70)
    print("INDIVIDUAL CHECKPOINT RESULTS")
    print("=" * 70)
    
    for result in framework.results:
        status_emoji = "✅" if result.status == CheckpointStatus.PASSED else "❌"
        print(f"{status_emoji} {result.checkpoint_id}: {result.name}")
        print(f"   Status: {result.status.value}")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Message: {result.message}")
        
        if result.errors:
            print("   Errors:")
            for error in result.errors:
                print(f"     - {error}")
        print()
        
    print("=" * 70)
    
    return summary['can_proceed']

if __name__ == "__main__":
    can_proceed = main()
    sys.exit(0 if can_proceed else 1)