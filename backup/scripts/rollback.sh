#!/bin/bash

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
