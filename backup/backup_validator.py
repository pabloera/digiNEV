#!/usr/bin/env python3
"""
BACKUP VALIDATION SYSTEM
Digital Discourse Monitor v5.0.0

Enterprise-grade backup validation and integrity checking system

Created: 2025-06-15
Author: Backup & Rollback Specialist
Purpose: Comprehensive backup validation and integrity verification
"""

import os
import json
import hashlib
import logging
import datetime
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ValidationResult:
    """Result of backup validation"""
    validation_type: str
    success: bool
    message: str
    details: Dict[str, Any]
    errors: List[str]
    timestamp: str

class BackupValidator:
    """
    Comprehensive backup validation system
    
    Features:
    - File integrity validation
    - Checksum verification
    - Archive validation
    - Metadata consistency checks
    - Recovery simulation
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_root = self.project_root / "backup"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.setup_logging()
        
        # Validation results
        self.validation_results: List[ValidationResult] = []
        
        self.logger.info("Backup validator initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.backup_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"validation_{self.timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BackupValidator')
        
    def find_latest_backup_session(self) -> Optional[Path]:
        """Find the latest backup session directory"""
        session_dirs = list(self.backup_root.glob("session_*"))
        
        if not session_dirs:
            self.logger.error("No backup sessions found")
            return None
            
        latest_session = max(session_dirs, key=lambda x: x.name)
        self.logger.info(f"Latest backup session: {latest_session}")
        
        return latest_session
        
    def validate_backup_structure(self) -> ValidationResult:
        """Validate backup directory structure"""
        self.logger.info("Validating backup directory structure...")
        
        required_dirs = [
            'optimization_files',
            'metadata', 
            'checksums',
            'scripts',
            'archives',
            'rollback'
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.backup_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                existing_dirs.append(dir_name)
            else:
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            return ValidationResult(
                validation_type="backup_structure",
                success=False,
                message=f"Missing backup directories: {missing_dirs}",
                details={
                    'required_dirs': required_dirs,
                    'existing_dirs': existing_dirs,
                    'missing_dirs': missing_dirs
                },
                errors=[f"Directory not found: {d}" for d in missing_dirs],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        return ValidationResult(
            validation_type="backup_structure",
            success=True,
            message="Backup directory structure validated",
            details={
                'required_dirs': required_dirs,
                'existing_dirs': existing_dirs,
                'all_present': True
            },
            errors=[],
            timestamp=datetime.datetime.now().isoformat()
        )
        
    def validate_backup_manifest(self, session_dir: Path) -> ValidationResult:
        """Validate backup manifest file"""
        self.logger.info("Validating backup manifest...")
        
        manifest_file = session_dir / "backup_manifest.json"
        
        if not manifest_file.exists():
            return ValidationResult(
                validation_type="backup_manifest",
                success=False,
                message="Backup manifest file not found",
                details={'manifest_file': str(manifest_file)},
                errors=["backup_manifest.json not found"],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                
            # Validate required fields
            required_fields = ['backup_id', 'timestamp', 'project_root', 'total_files', 'files', 'summary']
            missing_fields = []
            
            for field in required_fields:
                if field not in manifest:
                    missing_fields.append(field)
                    
            if missing_fields:
                return ValidationResult(
                    validation_type="backup_manifest",
                    success=False,
                    message=f"Missing required fields in manifest: {missing_fields}",
                    details={'missing_fields': missing_fields},
                    errors=[f"Missing field: {f}" for f in missing_fields],
                    timestamp=datetime.datetime.now().isoformat()
                )
                
            # Validate summary data
            summary = manifest['summary']
            expected_summary_fields = ['total_size_bytes', 'total_lines', 'success_count', 'error_count']
            
            for field in expected_summary_fields:
                if field not in summary:
                    return ValidationResult(
                        validation_type="backup_manifest",
                        success=False,
                        message=f"Missing summary field: {field}",
                        details={'missing_summary_field': field},
                        errors=[f"Missing summary field: {field}"],
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    
            return ValidationResult(
                validation_type="backup_manifest",
                success=True,
                message="Backup manifest validated successfully",
                details={
                    'backup_id': manifest['backup_id'],
                    'total_files': manifest['total_files'],
                    'success_count': summary['success_count'],
                    'error_count': summary['error_count'],
                    'total_size_bytes': summary['total_size_bytes']
                },
                errors=[],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                validation_type="backup_manifest",
                success=False,
                message=f"Invalid JSON in manifest file: {e}",
                details={'json_error': str(e)},
                errors=[f"JSON decode error: {e}"],
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            return ValidationResult(
                validation_type="backup_manifest",
                success=False,
                message=f"Error reading manifest: {e}",
                details={'error': str(e)},
                errors=[str(e)],
                timestamp=datetime.datetime.now().isoformat()
            )
            
    def validate_file_integrity(self, session_dir: Path) -> ValidationResult:
        """Validate integrity of backed up files"""
        self.logger.info("Validating file integrity...")
        
        manifest_file = session_dir / "backup_manifest.json"
        
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                
            integrity_issues = []
            verified_files = []
            
            for file_path, metadata in manifest['files'].items():
                # Check if backup file exists
                backup_file = session_dir / file_path
                
                if not backup_file.exists():
                    integrity_issues.append(f"Backup file missing: {file_path}")
                    continue
                    
                # Verify file size
                actual_size = backup_file.stat().st_size
                expected_size = metadata['size_bytes']
                
                if actual_size != expected_size:
                    integrity_issues.append(f"Size mismatch for {file_path}: expected {expected_size}, got {actual_size}")
                    continue
                    
                # Verify checksum
                sha256_hash = hashlib.sha256()
                with open(backup_file, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                        
                actual_checksum = sha256_hash.hexdigest()
                expected_checksum = metadata['checksum']
                
                if actual_checksum != expected_checksum:
                    integrity_issues.append(f"Checksum mismatch for {file_path}")
                    continue
                    
                verified_files.append(file_path)
                
            if integrity_issues:
                return ValidationResult(
                    validation_type="file_integrity",
                    success=False,
                    message=f"File integrity issues found: {len(integrity_issues)} files",
                    details={
                        'total_files': len(manifest['files']),
                        'verified_files': len(verified_files),
                        'failed_files': len(integrity_issues)
                    },
                    errors=integrity_issues,
                    timestamp=datetime.datetime.now().isoformat()
                )
                
            return ValidationResult(
                validation_type="file_integrity",
                success=True,
                message=f"File integrity validated: {len(verified_files)} files verified",
                details={
                    'total_files': len(manifest['files']),
                    'verified_files': len(verified_files),
                    'all_verified': True
                },
                errors=[],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except Exception as e:
            return ValidationResult(
                validation_type="file_integrity",
                success=False,
                message=f"Error validating file integrity: {e}",
                details={'error': str(e)},
                errors=[str(e)],
                timestamp=datetime.datetime.now().isoformat()
            )
            
    def validate_checksums(self) -> ValidationResult:
        """Validate checksum files"""
        self.logger.info("Validating checksum files...")
        
        checksums_dir = self.backup_root / "checksums"
        
        if not checksums_dir.exists():
            return ValidationResult(
                validation_type="checksums",
                success=False,
                message="Checksums directory not found",
                details={'checksums_dir': str(checksums_dir)},
                errors=["Checksums directory missing"],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        checksum_files = list(checksums_dir.glob("*.sha256"))
        
        if not checksum_files:
            return ValidationResult(
                validation_type="checksums",
                success=False,
                message="No checksum files found",
                details={'checksums_dir': str(checksums_dir)},
                errors=["No .sha256 files found"],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        invalid_checksums = []
        valid_checksums = []
        
        for checksum_file in checksum_files:
            try:
                with open(checksum_file, 'r') as f:
                    content = f.read().strip()
                    
                # Validate format: checksum  filename
                parts = content.split('  ')
                if len(parts) != 2:
                    invalid_checksums.append(f"Invalid format in {checksum_file.name}")
                    continue
                    
                checksum, filename = parts
                
                # Validate checksum length (SHA256 = 64 hex chars)
                if len(checksum) != 64 or not all(c in '0123456789abcdef' for c in checksum.lower()):
                    invalid_checksums.append(f"Invalid checksum in {checksum_file.name}")
                    continue
                    
                valid_checksums.append(checksum_file.name)
                
            except Exception as e:
                invalid_checksums.append(f"Error reading {checksum_file.name}: {e}")
                
        if invalid_checksums:
            return ValidationResult(
                validation_type="checksums",
                success=False,
                message=f"Invalid checksum files found: {len(invalid_checksums)}",
                details={
                    'total_checksum_files': len(checksum_files),
                    'valid_checksums': len(valid_checksums),
                    'invalid_checksums': len(invalid_checksums)
                },
                errors=invalid_checksums,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        return ValidationResult(
            validation_type="checksums",
            success=True,
            message=f"Checksum files validated: {len(valid_checksums)} files",
            details={
                'total_checksum_files': len(checksum_files),
                'valid_checksums': len(valid_checksums),
                'all_valid': True
            },
            errors=[],
            timestamp=datetime.datetime.now().isoformat()
        )
        
    def validate_archive_integrity(self) -> ValidationResult:
        """Validate compressed archive integrity"""
        self.logger.info("Validating archive integrity...")
        
        archives_dir = self.backup_root / "archives"
        
        if not archives_dir.exists():
            return ValidationResult(
                validation_type="archive_integrity",
                success=False,
                message="Archives directory not found",
                details={'archives_dir': str(archives_dir)},
                errors=["Archives directory missing"],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        archive_files = list(archives_dir.glob("*.tar.gz"))
        
        if not archive_files:
            return ValidationResult(
                validation_type="archive_integrity",
                success=False,  
                message="No archive files found",
                details={'archives_dir': str(archives_dir)},
                errors=["No .tar.gz files found"],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        archive_issues = []
        valid_archives = []
        
        for archive_file in archive_files:
            try:
                # Test archive integrity
                with tarfile.open(archive_file, "r:gz") as tar:
                    members = tar.getmembers()
                    
                    # Verify we can read all members
                    for member in members[:5]:  # Test first 5 members
                        if member.isfile():
                            tar.extractfile(member).read(1024)  # Read first 1KB
                            
                valid_archives.append(archive_file.name)
                
            except Exception as e:
                archive_issues.append(f"Archive {archive_file.name} failed validation: {e}")
                
        if archive_issues:
            return ValidationResult(
                validation_type="archive_integrity",
                success=False,
                message=f"Archive validation issues: {len(archive_issues)}",
                details={
                    'total_archives': len(archive_files),
                    'valid_archives': len(valid_archives),
                    'failed_archives': len(archive_issues)
                },
                errors=archive_issues,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        return ValidationResult(
            validation_type="archive_integrity",
            success=True,
            message=f"Archive integrity validated: {len(valid_archives)} archives",
            details={
                'total_archives': len(archive_files),
                'valid_archives': len(valid_archives),
                'all_valid': True
            },
            errors=[],
            timestamp=datetime.datetime.now().isoformat()
        )
        
    def validate_rollback_scripts(self) -> ValidationResult:
        """Validate rollback scripts"""
        self.logger.info("Validating rollback scripts...")
        
        scripts_dir = self.backup_root / "scripts"
        
        if not scripts_dir.exists():
            return ValidationResult(
                validation_type="rollback_scripts",
                success=False,
                message="Scripts directory not found",
                details={'scripts_dir': str(scripts_dir)},
                errors=["Scripts directory missing"],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        required_scripts = ["rollback.sh", "rollback.py"]
        script_issues = []
        valid_scripts = []
        
        for script_name in required_scripts:
            script_path = scripts_dir / script_name
            
            if not script_path.exists():
                script_issues.append(f"Script not found: {script_name}")
                continue
                
            if not script_path.is_file():
                script_issues.append(f"Not a file: {script_name}")
                continue
                
            # Check executable permission
            if not script_path.stat().st_mode & 0o111:
                script_issues.append(f"Script not executable: {script_name}")
                continue
                
            # Validate script content (basic check)
            try:
                with open(script_path, 'r') as f:
                    content = f.read()
                    
                if len(content) < 100:  # Scripts should be substantial
                    script_issues.append(f"Script too short: {script_name}")
                    continue
                    
                if script_name == "rollback.sh" and "#!/bin/bash" not in content:
                    script_issues.append(f"Invalid shebang in {script_name}")
                    continue
                    
                if script_name == "rollback.py" and "#!/usr/bin/env python3" not in content:
                    script_issues.append(f"Invalid shebang in {script_name}")
                    continue
                    
                valid_scripts.append(script_name)
                
            except Exception as e:
                script_issues.append(f"Error reading {script_name}: {e}")
                
        if script_issues:
            return ValidationResult(
                validation_type="rollback_scripts",
                success=False,
                message=f"Rollback script issues: {len(script_issues)}",
                details={
                    'required_scripts': required_scripts,
                    'valid_scripts': len(valid_scripts),
                    'script_issues': len(script_issues)
                },
                errors=script_issues,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        return ValidationResult(
            validation_type="rollback_scripts",
            success=True,
            message=f"Rollback scripts validated: {len(valid_scripts)} scripts",
            details={
                'required_scripts': required_scripts,
                'valid_scripts': len(valid_scripts),
                'all_valid': True
            },
            errors=[],
            timestamp=datetime.datetime.now().isoformat()
        )
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        self.logger.info("Starting comprehensive backup validation...")
        
        validation_start_time = datetime.datetime.now()
        
        # Find latest backup session
        latest_session = self.find_latest_backup_session()
        
        if not latest_session:
            return {
                'success': False,
                'message': "No backup sessions found",
                'validations': [],
                'summary': {
                    'total_validations': 0,
                    'passed': 0,
                    'failed': 1,
                    'can_proceed': False
                }
            }
            
        # Run all validations
        validations = [
            self.validate_backup_structure(),
            self.validate_backup_manifest(latest_session),
            self.validate_file_integrity(latest_session),
            self.validate_checksums(),
            self.validate_archive_integrity(),
            self.validate_rollback_scripts()
        ]
        
        self.validation_results.extend(validations)
        
        # Calculate summary
        passed = sum(1 for v in validations if v.success)
        failed = sum(1 for v in validations if not v.success)
        
        validation_summary = {
            'total_validations': len(validations),
            'passed': passed,
            'failed': failed,
            'can_proceed': failed == 0,
            'duration_seconds': (datetime.datetime.now() - validation_start_time).total_seconds()
        }
        
        # Save validation report
        self.save_validation_report(validation_summary)
        
        return {
            'success': validation_summary['can_proceed'],
            'message': f"Validation completed: {passed}/{len(validations)} passed",
            'validations': [asdict(v) for v in validations],
            'summary': validation_summary
        }
        
    def save_validation_report(self, summary: Dict[str, Any]):
        """Save validation report to file"""
        report_data = {
            'validation_summary': summary,
            'validation_results': [asdict(result) for result in self.validation_results],
            'timestamp': self.timestamp
        }
        
        report_file = self.backup_root / f"validation_report_{self.timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        self.logger.info(f"Validation report saved to: {report_file}")

def main():
    """Main validation execution"""
    print("=" * 70)
    print("BACKUP VALIDATION SYSTEM")
    print("Digital Discourse Monitor v5.0.0")
    print("Comprehensive Backup Integrity Validation")
    print("=" * 70)
    
    # Create validator
    validator = BackupValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Print results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total validations: {results['summary']['total_validations']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Duration: {results['summary']['duration_seconds']:.2f} seconds")
    
    if results['success']:
        print("\n✅ ALL VALIDATIONS PASSED")
        print("🚀 BACKUP SYSTEM IS READY")
        print("✅ CONSOLIDATION CAN PROCEED SAFELY")
    else:
        print("\n❌ VALIDATION FAILURES DETECTED")
        print("🛑 BACKUP SYSTEM ISSUES FOUND")
        print("⚠️  RESOLVE ISSUES BEFORE CONSOLIDATION")
        
    # Print individual validation results
    print("\n" + "=" * 70)
    print("INDIVIDUAL VALIDATION RESULTS")
    print("=" * 70)
    
    for validation in results['validations']:
        status_emoji = "✅" if validation['success'] else "❌"
        print(f"{status_emoji} {validation['validation_type'].upper()}: {validation['message']}")
        
        if validation['errors']:
            print("   Errors:")
            for error in validation['errors'][:3]:  # Show first 3 errors
                print(f"     - {error}")
            if len(validation['errors']) > 3:
                print(f"     ... and {len(validation['errors']) - 3} more errors")
        print()
        
    print("=" * 70)
    
    return results['success']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)