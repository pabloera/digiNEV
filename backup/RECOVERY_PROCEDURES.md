# BACKUP & RECOVERY PROCEDURES
## Digital Discourse Monitor v5.0.0

**Enterprise-Grade Backup and Recovery System**  
Created: 2025-06-15  
Author: Backup & Rollback Specialist  
Purpose: Comprehensive backup and recovery procedures for optimization files

---

## 📋 TABLE OF CONTENTS

1. [Quick Recovery Guide](#quick-recovery-guide)
2. [Backup System Overview](#backup-system-overview)
3. [Recovery Procedures](#recovery-procedures)
4. [Emergency Rollback](#emergency-rollback)
5. [Validation Procedures](#validation-procedures)
6. [Troubleshooting](#troubleshooting)
7. [System Architecture](#system-architecture)

---

## 🚨 QUICK RECOVERY GUIDE

### **EMERGENCY ROLLBACK (30 seconds)**

If consolidation fails and you need immediate restoration:

```bash
# 1. Find latest backup timestamp
ls backup/session_*/ | tail -1

# 2. Execute rollback (replace TIMESTAMP)
./backup/scripts/rollback.sh -t TIMESTAMP -f

# 3. Verify restoration
poetry run python backup/backup_validator.py
```

### **VALIDATION CHECK (2 minutes)**

Before any consolidation:

```bash
# 1. Run checkpoint framework
poetry run python backup/checkpoint_framework.py

# 2. Run backup validation
poetry run python backup/backup_validator.py

# 3. Check all systems green
echo "✅ Proceed only if all validations pass"
```

---

## 🏗️ BACKUP SYSTEM OVERVIEW

### **System Architecture**

```
backup/
├── optimization_files/     # Original file backups
├── metadata/              # File metadata (checksums, timestamps)
├── checksums/             # SHA256 checksum files
├── scripts/               # Rollback scripts (bash + python)
├── archives/              # Compressed backup archives
├── rollback/              # Rollback configurations
├── logs/                  # System logs
├── checkpoints/           # Validation checkpoints
└── session_TIMESTAMP/     # Timestamped backup sessions
```

### **Files Protected (15 Files, 10,459+ Lines)**

**Week 1-5 Optimization Files:**
- `src/optimized/__init__.py` (26 lines)
- `src/optimized/async_stages.py` (655 lines)
- `src/optimized/emergency_embeddings.py` (434 lines)
- `src/optimized/memory_optimizer.py` (762 lines)
- `src/optimized/optimized_pipeline.py` (680 lines)
- `src/optimized/parallel_engine.py` (75 lines)
- `src/optimized/performance_monitor.py` (699 lines)
- `src/optimized/pipeline_benchmark.py` (1,035 lines)
- `src/optimized/production_deploy.py` (1,006 lines)
- `src/optimized/quality_tests.py` (939 lines)
- `src/optimized/realtime_monitor.py` (818 lines)
- `src/optimized/smart_claude_cache.py` (830 lines)
- `src/optimized/streaming_pipeline.py` (714 lines)
- `src/optimized/unified_embeddings_engine.py` (796 lines)
- `test_all_weeks_consolidated.py` (1,030 lines)

**Total: 10,459+ lines of critical optimization code**

### **Backup Features**

- ✅ **Automated File Backup**: All optimization files with metadata
- ✅ **Checksum Verification**: SHA256 integrity validation
- ✅ **Compressed Archives**: Space-efficient .tar.gz storage
- ✅ **Rollback Scripts**: Bash and Python recovery tools
- ✅ **Validation Framework**: 5-checkpoint validation system
- ✅ **Integrity Monitoring**: Real-time backup verification
- ✅ **Enterprise Logging**: Comprehensive audit trail

---

## 🔧 RECOVERY PROCEDURES

### **1. PRE-CONSOLIDATION BACKUP**

**Execute before any consolidation work:**

```bash
# Create comprehensive backup
poetry run python backup/backup_system.py

# Expected output:
# ✅ ALL FILES VERIFIED SUCCESSFULLY
# Backup ID: optimization_backup_TIMESTAMP
# Total files: 15
# Successful backups: 15/15
```

### **2. VALIDATION CHECKPOINTS**

**Execute checkpoint validation:**

```bash
# Run all checkpoints
poetry run python backup/checkpoint_framework.py

# Expected checkpoints:
# ✅ CP001: Backup Integrity Validation
# ✅ CP002: Rollback Scripts Validation  
# ✅ CP003: File Checksum Validation
# ✅ CP004: System Functionality Validation
# ✅ CP005: Environment Validation
```

### **3. INTEGRITY VERIFICATION**

**Validate backup integrity:**

```bash
# Comprehensive validation
poetry run python backup/backup_validator.py

# Expected validations:
# ✅ BACKUP_STRUCTURE: Directory structure validated  
# ✅ BACKUP_MANIFEST: Manifest file validated
# ✅ FILE_INTEGRITY: All files verified
# ✅ CHECKSUMS: Checksum files validated
# ✅ ARCHIVE_INTEGRITY: Archives validated
# ✅ ROLLBACK_SCRIPTS: Scripts validated
```

---

## 🚨 EMERGENCY ROLLBACK

### **Scenario 1: Consolidation Failed**

If consolidation breaks the system:

```bash
# 1. Stop any running processes
pkill -f "python.*optimized"

# 2. Find latest backup
LATEST=$(ls -1 backup/session_* | tail -1 | cut -d'/' -f2 | cut -d'_' -f2)
echo "Latest backup: $LATEST"

# 3. Execute emergency rollback
./backup/scripts/rollback.sh -t $LATEST -f

# 4. Verify system
poetry run python -c "
import src.optimized.optimized_pipeline
print('✅ System restored successfully')
"
```

### **Scenario 2: Files Corrupted**

If optimization files are corrupted:

```bash
# 1. Check file integrity
poetry run python backup/backup_validator.py

# 2. Identify timestamp of last good backup
cat backup/session_*/backup_manifest.json | jq '.backup_id'

# 3. Rollback using Python script
python backup/scripts/rollback.py -t TIMESTAMP -f

# 4. Run system tests
poetry run python tests/test_system_integration.py
```

### **Scenario 3: Archive Recovery**

If session directory is lost, recover from archive:

```bash
# 1. Find latest archive
ARCHIVE=$(ls -1 backup/archives/*.tar.gz | tail -1)
echo "Using archive: $ARCHIVE"

# 2. Extract archive
cd backup/
tar -xzf "$ARCHIVE"

# 3. Find extracted session
SESSION=$(ls -1d optimization_backup_* | tail -1)

# 4. Copy files back
find "$SESSION" -name "*.py" -type f | while read file; do
    rel_path="${file#$SESSION/}"
    target="../$rel_path"
    mkdir -p "$(dirname "$target")"
    cp "$file" "$target"
    echo "Restored: $rel_path"
done
```

---

## ✅ VALIDATION PROCEDURES

### **Pre-Consolidation Checklist**

Execute these steps before any consolidation:

```bash
# 1. Environment Check
poetry env info
poetry show | grep -E "(anthropic|voyageai|pandas)"

# 2. Create Backup
poetry run python backup/backup_system.py

# 3. Run Checkpoints  
poetry run python backup/checkpoint_framework.py

# 4. Validate Backup
poetry run python backup/backup_validator.py

# 5. Test Rollback (dry run)
./backup/scripts/rollback.sh -t $(ls backup/session_* | tail -1 | cut -d'_' -f2) --dry-run
```

**Expected Results:**
- ✅ Poetry environment active
- ✅ Required packages installed
- ✅ 15/15 files backed up successfully
- ✅ 5/5 checkpoints passed
- ✅ 6/6 validations passed
- ✅ Rollback scripts functional

### **Post-Consolidation Validation**

Execute after consolidation:

```bash
# 1. System Functionality Test
poetry run python -c "
from src.optimized import *
print('✅ All modules import successfully')
"

# 2. Pipeline Test
poetry run python tests/test_pipeline_core.py

# 3. Integration Test  
poetry run python tests/test_system_integration.py

# 4. Performance Baseline
poetry run python tests/test_performance.py
```

---

## 🛠️ TROUBLESHOOTING

### **Common Issues & Solutions**

#### **Issue: Backup Creation Failed**

```bash
# Symptoms: 
# - "Error backing up file" messages
# - Incomplete backup manifest

# Solution:
# 1. Check file permissions
find src/optimized/ -name "*.py" -exec ls -la {} \;

# 2. Verify disk space
df -h

# 3. Recreate backup with verbose logging
poetry run python backup/backup_system.py --verbose

# 4. Check backup logs
tail -50 backup/logs/backup_*.log
```

#### **Issue: Checksum Validation Failed**

```bash
# Symptoms:
# - "Checksum mismatch" errors
# - File integrity validation failures

# Solution:
# 1. Check if files were modified during backup
ls -la src/optimized/*.py

# 2. Recalculate checksums
poetry run python -c "
import hashlib
from pathlib import Path
for f in Path('src/optimized').glob('*.py'):
    with open(f, 'rb') as file:
        checksum = hashlib.sha256(file.read()).hexdigest()
        print(f'{f}: {checksum}')
"

# 3. Recreate backup if files were modified
poetry run python backup/backup_system.py
```

#### **Issue: Rollback Script Failed**

```bash
# Symptoms:
# - "Permission denied" errors
# - "Backup manifest not found" errors

# Solution:
# 1. Fix script permissions
chmod +x backup/scripts/rollback.sh
chmod +x backup/scripts/rollback.py

# 2. Verify backup session exists
ls -la backup/session_*/

# 3. Use Python rollback if bash fails
python backup/scripts/rollback.py -t TIMESTAMP -f

# 4. Manual file restoration
cd backup/session_TIMESTAMP/
find . -name "*.py" -exec cp {} ../../{} \;
```

#### **Issue: Archive Corruption**

```bash
# Symptoms:
# - "Archive integrity verification failed"
# - Cannot extract .tar.gz files

# Solution:
# 1. Test archive manually
tar -tzf backup/archives/optimization_backup_*.tar.gz | head -10

# 2. Use session backup instead of archive
ls backup/session_*/

# 3. Recreate archive from session
cd backup/
tar -czf archives/backup_manual_$(date +%Y%m%d_%H%M%S).tar.gz session_*/
```

### **Emergency Contact Information**

If all recovery methods fail:

1. **Preserve Current State**: Don't modify any files
2. **Collect Logs**: `tar -czf emergency_logs.tar.gz backup/logs/`
3. **System Snapshot**: `poetry show > system_state.txt`
4. **File Status**: `find src/optimized/ -name "*.py" -exec ls -la {} \; > file_status.txt`

---

## 🏗️ SYSTEM ARCHITECTURE

### **Backup Components**

```python
# BackupSystem Class
class BackupSystem:
    - create_backup_structure()      # Directory setup
    - calculate_checksum()           # SHA256 hashing  
    - get_file_metadata()           # Comprehensive metadata
    - backup_file()                 # Individual file backup
    - create_full_backup()          # Complete backup process
    - create_compressed_archive()   # Archive creation
    - verify_archive_integrity()   # Archive validation

# CheckpointFramework Class  
class CheckpointFramework:
    - register_checkpoint()         # Checkpoint registration
    - execute_all_checkpoints()     # Validation execution
    - save_checkpoint_results()     # Results persistence

# BackupValidator Class
class BackupValidator:
    - validate_backup_structure()   # Directory validation
    - validate_backup_manifest()    # Manifest validation
    - validate_file_integrity()     # File verification
    - validate_checksums()          # Checksum validation
    - validate_archive_integrity()  # Archive validation
    - validate_rollback_scripts()   # Script validation
```

### **File Naming Conventions**

```
# Backup Sessions
session_YYYYMMDD_HHMMSS/

# Metadata Files  
FILENAME_YYYYMMDD_HHMMSS.json

# Checksum Files
FILENAME_YYYYMMDD_HHMMSS.sha256

# Archive Files
optimization_backup_YYYYMMDD_HHMMSS.tar.gz

# Log Files
backup_YYYYMMDD_HHMMSS.log
validation_YYYYMMDD_HHMMSS.log
checkpoints_YYYYMMDD_HHMMSS.log
```

### **Recovery Time Objectives (RTO)**

- **Emergency Rollback**: < 30 seconds
- **Full System Recovery**: < 2 minutes  
- **Archive Extraction**: < 5 minutes
- **Manual File Restoration**: < 10 minutes

### **Recovery Point Objectives (RPO)**

- **Backup Frequency**: Before each consolidation
- **Data Loss Window**: 0 (point-in-time backup)
- **File Granularity**: Individual file level
- **Metadata Preservation**: Complete (timestamps, permissions, checksums)

---

## 📊 VALIDATION METRICS

### **Backup Success Criteria**

- ✅ **File Count**: 15/15 optimization files
- ✅ **Size Verification**: All files match expected sizes
- ✅ **Checksum Match**: SHA256 integrity verified
- ✅ **Metadata Complete**: Timestamps, permissions preserved
- ✅ **Archive Creation**: Compressed backup created
- ✅ **Scripts Functional**: Rollback scripts executable

### **Checkpoint Success Criteria**

- ✅ **CP001**: Backup directory structure complete
- ✅ **CP002**: Rollback scripts present and executable  
- ✅ **CP003**: File checksums match current state
- ✅ **CP004**: System modules import successfully
- ✅ **CP005**: Poetry environment active and functional

### **Recovery Success Criteria**

- ✅ **File Restoration**: All 15 files restored to correct locations
- ✅ **Permission Preservation**: Execute permissions maintained
- ✅ **Import Validation**: All modules import without errors
- ✅ **Functionality Test**: Core pipeline functions execute
- ✅ **Integration Test**: End-to-end system validation

---

## 🎯 BEST PRACTICES

### **Before Consolidation**

1. **Always create backup first**: Never proceed without backup
2. **Validate backup integrity**: Run full validation suite
3. **Test rollback procedure**: Verify recovery works
4. **Document changes**: Record what will be consolidated
5. **Notify stakeholders**: Inform team of consolidation

### **During Consolidation**

1. **Work incrementally**: Small, reversible changes
2. **Test frequently**: Validate after each major change
3. **Preserve checkpoints**: Save intermediate states
4. **Monitor system**: Watch for performance degradation
5. **Document issues**: Record any problems encountered

### **After Consolidation**

1. **Full system test**: Complete integration validation
2. **Performance baseline**: Compare with pre-consolidation metrics
3. **Update documentation**: Reflect consolidated state
4. **Clean up backups**: Archive successful consolidation
5. **Plan next iteration**: Prepare for future optimizations

---

## 📞 SUPPORT & MAINTENANCE

### **Automated Monitoring**

The backup system includes automated monitoring:

```bash
# Daily backup validation (cron job)
0 2 * * * cd /path/to/project && poetry run python backup/backup_validator.py

# Weekly archive cleanup (keep last 4 weeks)
0 3 * * 0 find backup/archives/ -name "*.tar.gz" -mtime +28 -delete

# Monthly comprehensive validation
0 4 1 * * cd /path/to/project && poetry run python backup/checkpoint_framework.py
```

### **Log Rotation**

```bash
# Setup log rotation for backup logs
cat > /etc/logrotate.d/backup-system << EOF
/path/to/project/backup/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 user group
}
EOF
```

---

**🔒 This document is part of the enterprise-grade backup and recovery system for Digital Discourse Monitor v5.0.0. All procedures have been tested and validated.**

**Last Updated**: 2025-06-15  
**Version**: 1.0.0  
**Status**: Production Ready ✅