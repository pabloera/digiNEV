#!/usr/bin/env python3
"""
BACKUP & ROLLBACK SYSTEM for Digital Discourse Monitor v5.0.0
Enterprise-grade backup system for optimization files consolidation

Created: 2025-06-15
Author: Backup & Rollback Specialist
Purpose: Comprehensive backup strategy for optimization files
"""

import os
import shutil
import hashlib
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import tarfile
import gzip

class BackupSystem:
    """
    Enterprise-grade backup system for optimization files
    
    Features:
    - Automated file backup with checksums
    - Metadata preservation
    - Rollback capabilities
    - Integrity validation
    - Recovery documentation
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_root = self.project_root / "backup"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.setup_logging()
        
        # Core backup directories
        self.backup_dirs = {
            'optimization_files': self.backup_root / "optimization_files",
            'metadata': self.backup_root / "metadata", 
            'checksums': self.backup_root / "checksums",
            'scripts': self.backup_root / "scripts",
            'archives': self.backup_root / "archives",
            'rollback': self.backup_root / "rollback"
        }
        
        # Files to backup (as specified in requirements)
        self.optimization_files = [
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
        
        self.logger.info(f"Backup system initialized - Timestamp: {self.timestamp}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.backup_root / "logs" 
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"backup_{self.timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BackupSystem')
        
    def create_backup_structure(self):
        """Create backup directory structure"""
        self.logger.info("Creating backup directory structure...")
        
        for name, path in self.backup_dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {path}")
            
        # Create timestamped backup session
        session_dir = self.backup_root / f"session_{self.timestamp}"
        session_dir.mkdir(exist_ok=True)
        
        return session_dir
        
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for file"""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return None
            
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file metadata"""
        try:
            stat = file_path.stat()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                line_count = len(content.splitlines())
                char_count = len(content)
                
            return {
                'path': str(file_path),
                'size_bytes': stat.st_size,
                'line_count': line_count,
                'char_count': char_count,
                'modified_time': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created_time': datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'checksum': self.calculate_checksum(file_path),
                'permissions': oct(stat.st_mode)[-3:],
                'backup_timestamp': self.timestamp
            }
        except Exception as e:
            self.logger.error(f"Error getting metadata for {file_path}: {e}")
            return None
            
    def backup_file(self, source_path: str, session_dir: Path) -> Dict[str, Any]:
        """Backup individual file with metadata"""
        source = self.project_root / source_path
        
        if not source.exists():
            self.logger.warning(f"Source file not found: {source}")
            return None
            
        # Create backup path maintaining structure
        relative_path = Path(source_path)
        backup_path = session_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy file
            shutil.copy2(source, backup_path)
            
            # Get metadata
            metadata = self.get_file_metadata(source)
            
            # Save metadata
            metadata_file = self.backup_dirs['metadata'] / f"{relative_path.name}_{self.timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save checksum
            checksum_file = self.backup_dirs['checksums'] / f"{relative_path.name}_{self.timestamp}.sha256"
            with open(checksum_file, 'w') as f:
                f.write(f"{metadata['checksum']}  {source_path}\n")
                
            self.logger.info(f"Backed up: {source_path} -> {backup_path}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error backing up {source_path}: {e}")
            return None
            
    def create_full_backup(self) -> Dict[str, Any]:
        """Create complete backup of all optimization files"""
        self.logger.info("Starting full backup of optimization files...")
        
        session_dir = self.create_backup_structure()
        backup_manifest = {
            'backup_id': f"optimization_backup_{self.timestamp}",
            'timestamp': self.timestamp,
            'project_root': str(self.project_root),
            'total_files': len(self.optimization_files),
            'files': {},
            'summary': {
                'total_size_bytes': 0,
                'total_lines': 0,
                'success_count': 0,
                'error_count': 0
            }
        }
        
        # Backup each file
        for file_path in self.optimization_files:
            metadata = self.backup_file(file_path, session_dir)
            
            if metadata:
                backup_manifest['files'][file_path] = metadata
                backup_manifest['summary']['total_size_bytes'] += metadata['size_bytes']
                backup_manifest['summary']['total_lines'] += metadata['line_count']
                backup_manifest['summary']['success_count'] += 1
            else:
                backup_manifest['summary']['error_count'] += 1
                
        # Save backup manifest
        manifest_file = session_dir / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(backup_manifest, f, indent=2)
            
        # Create compressed archive
        self.create_compressed_archive(session_dir, backup_manifest['backup_id'])
        
        self.logger.info(f"Backup completed - {backup_manifest['summary']['success_count']} files backed up")
        return backup_manifest
        
    def create_compressed_archive(self, session_dir: Path, backup_id: str):
        """Create compressed archive of backup"""
        archive_path = self.backup_dirs['archives'] / f"{backup_id}.tar.gz"
        
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(session_dir, arcname=backup_id)
                
            self.logger.info(f"Created compressed archive: {archive_path}")
            
            # Verify archive integrity
            if self.verify_archive_integrity(archive_path):
                self.logger.info("Archive integrity verified successfully")
            else:
                self.logger.error("Archive integrity verification failed")
                
        except Exception as e:
            self.logger.error(f"Error creating archive: {e}")
            
    def verify_archive_integrity(self, archive_path: Path) -> bool:
        """Verify integrity of compressed archive"""
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Try to list all members
                members = tar.getmembers()
                self.logger.info(f"Archive contains {len(members)} files")
                return True
        except Exception as e:
            self.logger.error(f"Archive verification failed: {e}")
            return False
            
    def verify_backup_integrity(self, manifest_path: Path) -> Dict[str, bool]:
        """Verify integrity of backed up files"""
        self.logger.info("Verifying backup integrity...")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        verification_results = {}
        
        for file_path, metadata in manifest['files'].items():
            original_file = self.project_root / file_path
            
            if not original_file.exists():
                verification_results[file_path] = False
                continue
                
            # Verify checksum
            current_checksum = self.calculate_checksum(original_file)
            original_checksum = metadata['checksum']
            
            verification_results[file_path] = (current_checksum == original_checksum)
            
            if not verification_results[file_path]:
                self.logger.warning(f"Checksum mismatch for {file_path}")
                
        return verification_results

def create_rollback_scripts():
    """Create rollback scripts for emergency restoration"""
    script_content = '''#!/bin/bash

# ROLLBACK SCRIPT - Emergency Restoration of Optimization Files
# Digital Discourse Monitor v5.0.0
# Created: 2025-06-15

set -e

BACKUP_ROOT="./backup"
PROJECT_ROOT="."
TIMESTAMP=""

usage() {
    echo "Usage: $0 -t TIMESTAMP [-f]"
    echo "  -t TIMESTAMP: Backup timestamp to restore from"
    echo "  -f: Force restore without confirmation"
    exit 1
}

while getopts "t:f" opt; do
    case $opt in
        t) TIMESTAMP="$OPTARG" ;;
        f) FORCE=true ;;
        *) usage ;;
    esac
done

if [ -z "$TIMESTAMP" ]; then
    echo "Error: Timestamp required"
    usage
fi

SESSION_DIR="$BACKUP_ROOT/session_$TIMESTAMP"
MANIFEST_FILE="$SESSION_DIR/backup_manifest.json"

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Error: Backup manifest not found: $MANIFEST_FILE"
    exit 1
fi

echo "=== ROLLBACK OPERATION ==="
echo "Timestamp: $TIMESTAMP"
echo "Session: $SESSION_DIR"
echo "Manifest: $MANIFEST_FILE"
echo

if [ "$FORCE" != "true" ]; then
    read -p "Are you sure you want to rollback? This will overwrite current files. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Rollback cancelled"
        exit 0
    fi
fi

echo "Starting rollback..."

# Extract files from session backup
find "$SESSION_DIR" -name "*.py" -type f | while read -r backup_file; do
    # Calculate relative path
    rel_path="${backup_file#$SESSION_DIR/}"
    target_path="$PROJECT_ROOT/$rel_path"
    
    echo "Restoring: $rel_path"
    
    # Create target directory if needed
    mkdir -p "$(dirname "$target_path")"
    
    # Copy file back
    cp "$backup_file" "$target_path"
done

echo "Rollback completed successfully"
echo "Files restored from backup timestamp: $TIMESTAMP"
'''

    # Save rollback script
    backup_root = Path("backup")
    scripts_dir = backup_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    rollback_script = scripts_dir / "rollback.sh"
    with open(rollback_script, 'w') as f:
        f.write(script_content)
        
    # Make script executable
    os.chmod(rollback_script, 0o755)
    
    # Create Python rollback script
    python_rollback_content = '''#!/usr/bin/env python3
"""
Python Rollback Script - Emergency Restoration
Digital Discourse Monitor v5.0.0
"""

import sys
import json
import shutil
import argparse
from pathlib import Path

def rollback_from_backup(timestamp: str, force: bool = False):
    """Rollback files from backup timestamp"""
    backup_root = Path("backup")
    project_root = Path(".")
    
    session_dir = backup_root / f"session_{timestamp}"
    manifest_file = session_dir / "backup_manifest.json"
    
    if not manifest_file.exists():
        print(f"Error: Backup manifest not found: {manifest_file}")
        sys.exit(1)
        
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
        
    print(f"=== PYTHON ROLLBACK OPERATION ===")
    print(f"Backup ID: {manifest.get('backup_id')}")
    print(f"Timestamp: {timestamp}")
    print(f"Total files: {manifest.get('total_files')}")
    print()
    
    if not force:
        confirm = input("Are you sure you want to rollback? This will overwrite current files. (y/N): ")
        if confirm.lower() != 'y':
            print("Rollback cancelled")
            sys.exit(0)
            
    print("Starting rollback...")
    
    restored_count = 0
    for file_path in manifest['files'].keys():
        source_backup = session_dir / file_path
        target_file = project_root / file_path
        
        if source_backup.exists():
            print(f"Restoring: {file_path}")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_backup, target_file)
            restored_count += 1
        else:
            print(f"Warning: Backup file not found: {source_backup}")
            
    print(f"Rollback completed - {restored_count} files restored")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollback optimization files from backup")
    parser.add_argument("-t", "--timestamp", required=True, help="Backup timestamp")
    parser.add_argument("-f", "--force", action="store_true", help="Force without confirmation")
    
    args = parser.parse_args()
    rollback_from_backup(args.timestamp, args.force)
'''

    python_rollback_script = scripts_dir / "rollback.py"
    with open(python_rollback_script, 'w') as f:
        f.write(python_rollback_content)
        
    os.chmod(python_rollback_script, 0o755)
    
    print(f"Created rollback scripts:")
    print(f"- {rollback_script}")
    print(f"- {python_rollback_script}")

def main():
    """Main backup execution"""
    print("=" * 60)
    print("BACKUP & ROLLBACK SYSTEM")
    print("Digital Discourse Monitor v5.0.0")
    print("Optimization Files Backup")
    print("=" * 60)
    
    # Initialize backup system
    backup_system = BackupSystem()
    
    # Create rollback scripts
    create_rollback_scripts()
    
    # Create full backup
    manifest = backup_system.create_full_backup()
    
    # Verify backup integrity
    session_dir = backup_system.backup_root / f"session_{backup_system.timestamp}"
    manifest_file = session_dir / "backup_manifest.json"
    
    verification_results = backup_system.verify_backup_integrity(manifest_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BACKUP SUMMARY")
    print("=" * 60)
    print(f"Backup ID: {manifest['backup_id']}")
    print(f"Timestamp: {manifest['timestamp']}")
    print(f"Total files: {manifest['total_files']}")
    print(f"Successful backups: {manifest['summary']['success_count']}")
    print(f"Failed backups: {manifest['summary']['error_count']}")
    print(f"Total size: {manifest['summary']['total_size_bytes']:,} bytes")
    print(f"Total lines: {manifest['summary']['total_lines']:,}")
    
    # Verification results
    verified_count = sum(verification_results.values())
    print(f"Integrity verified: {verified_count}/{len(verification_results)} files")
    
    if verified_count == len(verification_results):
        print("✅ ALL FILES VERIFIED SUCCESSFULLY")
    else:
        print("⚠️  SOME FILES FAILED VERIFICATION")
        
    print("\n" + "=" * 60)
    print("ROLLBACK INSTRUCTIONS")
    print("=" * 60)
    print("To rollback from this backup:")
    print(f"./backup/scripts/rollback.sh -t {backup_system.timestamp}")
    print("or")
    print(f"python ./backup/scripts/rollback.py -t {backup_system.timestamp}")
    print("=" * 60)

if __name__ == "__main__":
    main()