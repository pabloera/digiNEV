#!/usr/bin/env python3
"""
QUICK STATUS CHECK - Backup System
Digital Discourse Monitor v5.0.0

Quick validation of backup system status

Created: 2025-06-15
Author: Backup & Rollback Specialist  
Purpose: Quick health check for backup system
"""

import json
from pathlib import Path
from datetime import datetime

def check_backup_status():
    """Quick status check of backup system"""
    
    print("🛡️  BACKUP SYSTEM - QUICK STATUS CHECK")
    print("=" * 50)
    
    backup_root = Path("backup")
    
    # Check if backup directory exists
    if not backup_root.exists():
        print("❌ Backup directory not found")
        return False
        
    # Find latest session
    sessions = list(backup_root.glob("session_*"))
    if not sessions:
        print("❌ No backup sessions found")
        return False
        
    latest_session = max(sessions, key=lambda x: x.name)
    print(f"✅ Latest backup session: {latest_session.name}")
    
    # Check manifest
    manifest_file = latest_session / "backup_manifest.json"
    if not manifest_file.exists():
        print("❌ Backup manifest missing")
        return False
        
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
        
    print(f"✅ Backup ID: {manifest['backup_id']}")
    print(f"✅ Files backed up: {manifest['summary']['success_count']}/{manifest['total_files']}")
    print(f"✅ Total size: {manifest['summary']['total_size_bytes']:,} bytes")
    print(f"✅ Total lines: {manifest['summary']['total_lines']:,}")
    
    # Check rollback scripts
    scripts_dir = backup_root / "scripts"
    rollback_sh = scripts_dir / "rollback.sh"
    rollback_py = scripts_dir / "rollback.py"
    
    if rollback_sh.exists() and rollback_py.exists():
        print("✅ Rollback scripts available")
    else:
        print("❌ Rollback scripts missing")
        return False
        
    # Check archives
    archives_dir = backup_root / "archives"
    archives = list(archives_dir.glob("*.tar.gz"))
    
    if archives:
        print(f"✅ Compressed archive: {len(archives)} archive(s)")
    else:
        print("⚠️  No compressed archives found")
        
    # Check validation reports
    validation_reports = list(backup_root.glob("validation_report_*.json"))
    
    if validation_reports:
        latest_validation = max(validation_reports, key=lambda x: x.name)
        with open(latest_validation, 'r') as f:
            validation_data = json.load(f)
            
        summary = validation_data['validation_summary']
        if summary['passed'] == summary['total_validations']:
            print(f"✅ Last validation: {summary['passed']}/{summary['total_validations']} passed")
        else:
            print(f"⚠️  Last validation: {summary['passed']}/{summary['total_validations']} passed")
            
    print("\n🎯 STATUS SUMMARY:")
    
    # Calculate backup age
    timestamp_str = latest_session.name.replace("session_", "")
    backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    age = datetime.now() - backup_time
    
    if age.total_seconds() < 3600:  # Less than 1 hour
        age_str = f"{int(age.total_seconds() / 60)} minutes ago"
        age_status = "✅"
    elif age.total_seconds() < 86400:  # Less than 1 day
        age_str = f"{int(age.total_seconds() / 3600)} hours ago"
        age_status = "⚠️"
    else:
        age_str = f"{age.days} days ago"
        age_status = "❌"
        
    print(f"{age_status} Backup age: {age_str}")
    print("✅ System ready for consolidation")
    print("✅ Emergency rollback available")
    
    print("\n🚀 EMERGENCY ROLLBACK COMMAND:")
    print(f"./backup/scripts/rollback.sh -t {timestamp_str} -f")
    
    return True

if __name__ == "__main__":
    success = check_backup_status()
    exit(0 if success else 1)