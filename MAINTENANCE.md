# 🔧 Maintenance Guide - Pipeline Bolsonarismo v5.0.0

## 📋 **Regular Maintenance Tasks**

### Daily Operations

#### 1. System Health Check
```bash
# Quick health verification (5 minutes)
poetry run python examples/quick_start.py | grep -E "(Overall:|ENTERPRISE)"

# Expected output: "Overall: 5/5 weeks active (100%)"
# Expected output: "ENTERPRISE-GRADE OPTIMIZATION: ACTIVE!"
```

#### 2. Log Monitoring
```bash
# Check for errors in last 24 hours
find logs/ -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL" {} \;

# Monitor pipeline execution status
tail -f logs/pipeline_execution.log | grep -E "(Pipeline execution|ERROR|WARNING)"

# Check API rate limits
grep -c "RateLimitError\|TimeoutError" logs/pipeline_execution.log
```

#### 3. Disk Space Management
```bash
# Check available space (should have >5GB free)
df -h . | grep -E "Avail|pipeline"

# Clean old temporary files
find test_backups/ -type f -mtime +7 -delete 2>/dev/null || true
find e2e_test_backups/ -type f -mtime +7 -delete 2>/dev/null || true
find cache/ -type f -mtime +7 -delete 2>/dev/null || true

# Compress old logs (keep last 30 days)
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;
```

### Weekly Maintenance

#### 1. Dependency Updates
```bash
# Check for security updates
poetry audit

# Update dependencies (test environment first)
poetry update --dry-run

# Apply updates in staging environment
poetry update
poetry install

# Verify system still works
poetry run python test_all_weeks_consolidated.py
```

#### 2. Performance Review
```bash
# Run comprehensive benchmarks
poetry run python -c "
from src.optimized.pipeline_benchmark import get_global_benchmark
benchmark = get_global_benchmark()
results = benchmark.run_performance_test([1000, 5000])
for size, metrics in results.items():
    print(f'Size {size}: {metrics[\"avg_time\"]:.2f}s')
"

# Check memory usage trends
grep "Memory" logs/pipeline_execution.log | tail -20

# Verify optimization effectiveness
poetry run python -c "
from src.optimized.memory_optimizer import get_global_memory_manager
manager = get_global_memory_manager()
summary = manager.get_management_summary()
print(f'Memory efficiency: {summary[\"optimization_stats\"][\"memory_savings_mb\"]:.1f}MB saved')
"
```

#### 3. API Cost Monitoring
```bash
# Check API usage and costs
poetry run python -c "
from src.anthropic_integration.cost_monitor import ConsolidatedCostMonitor
monitor = ConsolidatedCostMonitor()
report = monitor.generate_cost_report()
print(f'Week total: ${report[\"total_cost_usd\"]:.2f}')
print(f'API calls: {report[\"total_api_calls\"]}')
"

# Monitor Voyage.ai quota usage
grep "Voyage.ai.*tokens" logs/pipeline_execution.log | tail -5
```

#### 4. Data Backup
```bash
# Backup configuration files
tar -czf backups/config_$(date +%Y%m%d).tar.gz config/

# Backup important results (if any)
if [ -d "pipeline_outputs" ] && [ "$(ls -A pipeline_outputs/)" ]; then
    tar -czf backups/outputs_$(date +%Y%m%d).tar.gz pipeline_outputs/
fi

# Clean old backups (keep last 4 weeks)
find backups/ -name "*.tar.gz" -mtime +28 -delete
```

### Monthly Maintenance

#### 1. Model Updates
```bash
# Check for new Anthropic models
poetry run python -c "
from anthropic import Anthropic
import os
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
# Check latest available models at console.anthropic.com
print('Check console.anthropic.com for latest model versions')
"

# Update spaCy models
poetry run python -m spacy download pt_core_news_lg --upgrade

# Verify model compatibility
poetry run python -c "
import spacy
nlp = spacy.load('pt_core_news_lg')
print(f'spaCy model version: {nlp.meta[\"version\"]}')
"
```

#### 2. Security Audit
```bash
# Check for exposed API keys in logs
grep -r "sk-ant-api03\|pa-" logs/ | grep -v "HIDDEN\|***" || echo "✅ No exposed keys"

# Verify configuration file permissions
ls -la config/ | grep -E "(anthropic|voyage)" | awk '{print $1, $9}'

# Check for suspicious activity patterns
grep -E "(failed|unauthorized|invalid)" logs/pipeline_execution.log | tail -10
```

#### 3. Database Maintenance (if applicable)
```bash
# Clean checkpoint files older than 30 days
find checkpoints/ -type f -mtime +30 -delete

# Verify data integrity
poetry run python -c "
import pandas as pd
import glob

# Check for corrupted CSV files
csv_files = glob.glob('data/interim/*.csv')
for file in csv_files:
    try:
        df = pd.read_csv(file, nrows=1)
        print(f'✅ {file}: OK')
    except Exception as e:
        print(f'❌ {file}: {e}')
"
```

## 🚨 **Emergency Procedures**

### System Recovery

#### 1. Pipeline Failure Recovery
```bash
# Step 1: Stop all running processes
pkill -f "python.*run_pipeline"
pkill -f "streamlit"

# Step 2: Clear corrupted state
rm -rf checkpoints/*
rm -rf cache/embeddings/*

# Step 3: Verify configuration
poetry run python -c "
import yaml
with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Configuration valid')
"

# Step 4: Restart with minimal dataset
cp data/uploads/sample_dataset_v495.csv data/uploads/recovery_test.csv
poetry run python run_pipeline.py

# Step 5: Monitor recovery
tail -f logs/pipeline_execution.log
```

#### 2. API Quota Exhaustion
```bash
# Enable emergency cost optimization
poetry run python -c "
# Update config to use most economical settings
import yaml

with open('config/voyage_embeddings.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['enable_sampling'] = True
config['max_messages_per_dataset'] = 10000  # Reduce from 50000

with open('config/voyage_embeddings.yaml', 'w') as f:
    yaml.dump(config, f)

print('✅ Emergency cost optimization enabled')
"

# Switch to smaller Anthropic model temporarily
sed -i 's/claude-3-5-sonnet/claude-3-5-haiku/g' config/anthropic.yaml
```

#### 3. Memory Exhaustion Recovery
```bash
# Force garbage collection and memory optimization
poetry run python -c "
import gc
import psutil
import os

# Force garbage collection
gc.collect()

# Get current memory usage
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Current memory: {memory_mb:.1f}MB')

# Enable aggressive memory management
from src.optimized.memory_optimizer import get_global_memory_manager
manager = get_global_memory_manager()
manager.emergency_memory_cleanup()
print('✅ Emergency memory cleanup completed')
"

# Reduce processing batch sizes
sed -i 's/chunk_size: 10000/chunk_size: 2000/g' config/settings.yaml
sed -i 's/batch_size: 128/batch_size: 32/g' config/voyage_embeddings.yaml
```

### Data Recovery

#### 1. Corrupted Pipeline Output Recovery
```bash
# Identify last good checkpoint
ls -la checkpoints/ | grep checkpoint.json

# Restore from last known good state
poetry run python -c "
import json
from pathlib import Path

checkpoint_file = Path('checkpoints/checkpoint.json')
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    
    last_stage = checkpoint['execution_summary']['resume_from']
    print(f'Last completed stage: {last_stage}')
    print('Pipeline will resume from this point')
else:
    print('No checkpoint found - full restart required')
"

# Resume pipeline from checkpoint
poetry run python run_pipeline.py
```

#### 2. Configuration Recovery
```bash
# Restore configuration from templates
cp config/anthropic.yaml.template config/anthropic.yaml
cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml

# Restore API keys from backup or environment
echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > .env
echo "VOYAGE_API_KEY=$VOYAGE_API_KEY" >> .env

# Verify configuration
poetry run python examples/quick_start.py
```

## 🔍 **Monitoring & Alerting**

### Automated Monitoring Script
```bash
# Create monitoring script
cat > monitor_pipeline.sh << 'EOF'
#!/bin/bash

# Pipeline Health Monitor v5.0.0
LOG_FILE="logs/monitoring_$(date +%Y%m%d).log"
ALERT_EMAIL="admin@yourcompany.com"  # Configure as needed

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_optimization_status() {
    ACTIVE_WEEKS=$(poetry run python examples/quick_start.py 2>/dev/null | grep "Overall:" | grep -o "[0-9]*/5")
    if [[ "$ACTIVE_WEEKS" == "5/5" ]]; then
        log_message "✅ Optimization Status: All 5 weeks active"
        return 0
    else
        log_message "⚠️ Optimization Status: $ACTIVE_WEEKS weeks active"
        return 1
    fi
}

check_disk_space() {
    AVAILABLE=$(df . | tail -1 | awk '{print $4}')
    THRESHOLD=5000000  # 5GB in KB
    
    if [[ $AVAILABLE -gt $THRESHOLD ]]; then
        log_message "✅ Disk Space: ${AVAILABLE}KB available"
        return 0
    else
        log_message "⚠️ Disk Space: Only ${AVAILABLE}KB available"
        return 1
    fi
}

check_api_errors() {
    ERROR_COUNT=$(grep -c "ERROR.*API" logs/pipeline_execution.log 2>/dev/null || echo 0)
    
    if [[ $ERROR_COUNT -lt 10 ]]; then
        log_message "✅ API Status: $ERROR_COUNT errors in logs"
        return 0
    else
        log_message "⚠️ API Status: $ERROR_COUNT errors detected"
        return 1
    fi
}

# Run all checks
log_message "Starting Pipeline Health Check"

ISSUES=0
check_optimization_status || ((ISSUES++))
check_disk_space || ((ISSUES++))
check_api_errors || ((ISSUES++))

if [[ $ISSUES -eq 0 ]]; then
    log_message "🎉 All systems healthy"
else
    log_message "⚠️ $ISSUES issues detected - review required"
    # Send alert (configure your preferred method)
    # echo "Pipeline issues detected. Check $LOG_FILE" | mail -s "Pipeline Alert" $ALERT_EMAIL
fi

log_message "Health check completed"
EOF

chmod +x monitor_pipeline.sh
```

### Performance Metrics Collection
```bash
# Create performance tracking script
cat > collect_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import time
import psutil
from datetime import datetime
from pathlib import Path

def collect_system_metrics():
    """Collect current system performance metrics"""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'memory': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'used_percent': psutil.virtual_memory().percent
        },
        'disk': {
            'total_gb': psutil.disk_usage('.').total / (1024**3),
            'free_gb': psutil.disk_usage('.').free / (1024**3),
            'used_percent': (psutil.disk_usage('.').used / psutil.disk_usage('.').total) * 100
        },
        'cpu_percent': psutil.cpu_percent(interval=1)
    }
    
    # Save metrics
    metrics_dir = Path('logs/metrics')
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_file = metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    metrics = collect_system_metrics()
    print(f"Memory: {metrics['memory']['used_percent']:.1f}%")
    print(f"Disk: {metrics['disk']['used_percent']:.1f}%")
    print(f"CPU: {metrics['cpu_percent']:.1f}%")
EOF

chmod +x collect_metrics.py
```

## 📊 **Performance Optimization**

### Routine Optimization Tasks

#### 1. Cache Optimization
```bash
# Analyze cache performance
poetry run python -c "
from src.optimized.smart_claude_cache import SmartClaudeCache
cache = SmartClaudeCache()

# Get cache statistics
stats = cache.get_cache_stats()
print(f'Cache hit rate: {stats.get(\"hit_rate\", 0):.1%}')
print(f'Cache size: {stats.get(\"size\", 0)} entries')

# Clear old cache entries if hit rate is low
if stats.get('hit_rate', 0) < 0.3:
    cache.clear_expired_entries()
    print('✅ Cleared expired cache entries')
"
```

#### 2. Memory Usage Optimization
```bash
# Profile memory usage patterns
poetry run python -c "
import gc
from src.optimized.memory_optimizer import get_global_memory_manager

manager = get_global_memory_manager()
manager.start_adaptive_management()

# Let it run for monitoring period
import time
time.sleep(30)

summary = manager.get_management_summary()
print(f'Memory optimizations: {summary[\"optimization_stats\"][\"optimizations_performed\"]}')
print(f'Memory savings: {summary[\"optimization_stats\"][\"memory_savings_mb\"]:.1f}MB')

manager.stop_adaptive_management()
"
```

#### 3. API Cost Optimization
```bash
# Analyze and optimize API usage
poetry run python -c "
from src.anthropic_integration.cost_monitor import ConsolidatedCostMonitor

monitor = ConsolidatedCostMonitor()
report = monitor.generate_cost_report()

print(f'Current daily cost: ${report[\"total_cost_usd\"]:.2f}')
print(f'Optimization savings: ${report.get(\"optimization_savings\", 0):.2f}')

# Suggest optimizations
if report['total_cost_usd'] > 5.0:  # Threshold
    print('💡 Consider enabling more aggressive sampling')
    print('💡 Review model usage - consider using more haiku models')
"
```

## 📝 **Documentation Maintenance**

### Keep Documentation Updated
```bash
# Check for outdated documentation
find . -name "*.md" -mtime +90 | grep -v node_modules

# Update version numbers in documentation
VERSION="v5.0.0"
DATE=$(date +"%d/%m/%Y")

# Update main README if needed
sed -i "s/v[0-9]\+\.[0-9]\+\.[0-9]\+/$VERSION/g" README.md
```

### Generate API Documentation
```bash
# Generate module documentation (if using Sphinx)
# poetry run sphinx-build -b html docs docs/_build

# Update inline documentation coverage
poetry run python -c "
import inspect
import src.anthropic_integration.unified_pipeline as pipeline

functions = [name for name, obj in inspect.getmembers(pipeline) if inspect.isfunction(obj)]
documented = [name for name in functions if getattr(getattr(pipeline, name), '__doc__', None)]

coverage = len(documented) / len(functions) * 100 if functions else 0
print(f'Documentation coverage: {coverage:.1f}% ({len(documented)}/{len(functions)})')
"
```

## 📞 **Support Escalation**

### When to Escalate Issues

1. **Critical System Failure**: Pipeline completely non-functional for >4 hours
2. **Data Loss**: Important outputs corrupted or lost
3. **Security Incident**: Suspected API key compromise or unauthorized access
4. **Performance Degradation**: >50% reduction in processing speed for >24 hours
5. **Cost Overrun**: API costs exceed 200% of normal usage

### Escalation Procedure
1. **Document**: Collect diagnostics using scripts above
2. **Isolate**: Stop affected processes to prevent further damage
3. **Report**: Create GitHub issue with full diagnostic information
4. **Communicate**: Notify stakeholders of impact and estimated resolution time

---

## ✅ **Maintenance Checklist**

### Daily (5 minutes)
- [ ] Check system health: `poetry run python examples/quick_start.py`
- [ ] Monitor logs: `tail logs/pipeline_execution.log`
- [ ] Verify disk space: `df -h .`

### Weekly (30 minutes)
- [ ] Run performance benchmarks
- [ ] Check dependency updates
- [ ] Review API costs
- [ ] Backup configurations

### Monthly (2 hours)
- [ ] Update dependencies
- [ ] Security audit
- [ ] Model updates
- [ ] Documentation review
- [ ] Performance optimization

---

✅ **With proper maintenance, Pipeline Bolsonarismo v5.0.0 will continue operating reliably with 95% success rate and optimal performance.**