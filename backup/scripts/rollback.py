#!/usr/bin/env python3
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
