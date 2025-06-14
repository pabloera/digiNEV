# 🔧 Troubleshooting Guide - Pipeline Bolsonarismo v5.0.0

## 🚨 **Common Issues & Solutions**

### 📊 **Pipeline Execution Issues**

#### Problem: "Pipeline execution failed" or stages not completing
```bash
# Symptoms
❌ EXECUÇÃO FALHOU
📊 Datasets processados: 0
```

**Solutions:**
1. **Check API Keys**
   ```bash
   # Verify environment variables
   echo $ANTHROPIC_API_KEY | head -c 20
   
   # Test API connectivity
   poetry run python -c "
   from anthropic import Anthropic
   import os
   client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
   response = client.messages.create(
       model='claude-3-5-haiku-20241022',
       max_tokens=10,
       messages=[{'role': 'user', 'content': 'test'}]
   )
   print('✅ API working')
   "
   ```

2. **Check Data Format**
   ```bash
   # Verify CSV structure
   head -1 data/uploads/your_dataset.csv
   
   # Required columns:
   # message_id,datetime,body,url,hashtag,channel,is_fwrd,mentions,sender,media_type,domain,body_cleaned,source_dataset,hash_id
   ```

3. **Clear Corrupted Checkpoints**
   ```bash
   # Remove existing checkpoints
   rm -rf checkpoints/*
   
   # Restart pipeline
   poetry run python run_pipeline.py
   ```

#### Problem: "Configuration is required" error
```bash
# Symptoms
ERROR - Configuração é obrigatória para inicializar o pipeline
```

**Solutions:**
1. **Check Config Files**
   ```bash
   # Verify configuration files exist
   ls -la config/
   
   # Should show:
   # anthropic.yaml (not .template)
   # voyage_embeddings.yaml (not .template)
   # settings.yaml
   ```

2. **Recreate Config Files**
   ```bash
   # Copy templates
   cp config/anthropic.yaml.template config/anthropic.yaml
   cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml
   
   # Edit with your API keys
   nano config/anthropic.yaml
   nano config/voyage_embeddings.yaml
   ```

### 🔧 **Optimization System Issues**

#### Problem: "Some optimization systems not available"
```bash
# Symptoms
⚠️ Week X optimization not applied: ImportError
```

**Solutions:**
1. **Install Optimization Dependencies**
   ```bash
   # Install missing optimization packages
   poetry install --with optimization
   
   # Verify installation
   poetry show | grep -E "(psutil|memory-profiler|cachetools)"
   ```

2. **Check Python Path**
   ```bash
   # Verify src path is correct
   poetry run python -c "
   import sys
   from pathlib import Path
   src_path = Path.cwd() / 'src'
   print(f'Src path: {src_path}')
   print(f'Exists: {src_path.exists()}')
   if str(src_path) not in sys.path:
       sys.path.insert(0, str(src_path))
   print('✅ Path configured')
   "
   ```

### 💾 **Memory Issues**

#### Problem: "Memory error" or system freezing
```bash
# Symptoms
MemoryError: Unable to allocate array
Process killed (out of memory)
```

**Solutions:**
1. **Enable Memory Optimization**
   ```bash
   # Start adaptive memory management
   poetry run python -c "
   from src.optimized.memory_optimizer import get_global_memory_manager
   manager = get_global_memory_manager()
   manager.start_adaptive_management()
   print(f'Target memory: {manager.target_memory_gb}GB')
   "
   ```

2. **Reduce Dataset Size**
   ```bash
   # Process smaller chunks
   poetry run python -c "
   import pandas as pd
   df = pd.read_csv('data/uploads/large_dataset.csv')
   
   # Create smaller sample
   sample = df.sample(n=10000)  # Adjust size as needed
   sample.to_csv('data/uploads/sample_dataset.csv', index=False)
   print(f'Sample created: {len(sample)} records')
   "
   ```

3. **Adjust Memory Settings**
   ```bash
   # Edit config/settings.yaml
   # Reduce chunk_size: 5000 (from 10000)
   # Reduce batch_size: 64 (from 128)
   
   # Set memory limits
   export PYTHONHASHSEED=0
   export OMP_NUM_THREADS=1
   ulimit -v 8000000  # Limit to 8GB
   ```

### 🌐 **API Issues**

#### Problem: "API timeout" or "Rate limited"
```bash
# Symptoms
TimeoutError: Request timed out
RateLimitError: Rate limit exceeded
```

**Solutions:**
1. **Increase Timeout Settings**
   ```bash
   # Edit config/anthropic.yaml
   timeout_seconds: 180  # Increase from 120
   max_retries: 5        # Increase from 3
   
   # Edit config/voyage_embeddings.yaml  
   timeout_seconds: 120  # Increase from 60
   ```

2. **Reduce Request Frequency**
   ```bash
   # Enable smart caching to reduce API calls
   poetry run python -c "
   from src.optimized.smart_claude_cache import SmartClaudeCache
   cache = SmartClaudeCache()
   print(f'Cache enabled: {cache is not None}')
   "
   
   # Check cache hit rate
   grep "cache hit" logs/pipeline_execution.log
   ```

3. **Use Cost Optimization**
   ```bash
   # Enable Voyage.ai sampling (96% cost reduction)
   # In config/voyage_embeddings.yaml:
   enable_sampling: true
   max_messages_per_dataset: 50000
   ```

### 🔤 **spaCy Model Issues**

#### Problem: "pt_core_news_lg not found"
```bash
# Symptoms
OSError: [E050] Can't find model 'pt_core_news_lg'
```

**Solutions:**
1. **Reinstall spaCy Model**
   ```bash
   # Download Portuguese model
   poetry run python -m spacy download pt_core_news_lg
   
   # Verify installation
   poetry run python -c "
   import spacy
   nlp = spacy.load('pt_core_news_lg')
   print('✅ spaCy model loaded successfully')
   print(f'Model version: {nlp.meta[\"version\"]}')
   "
   ```

2. **Manual Installation**
   ```bash
   # If automatic download fails
   wget https://github.com/explosion/spacy-models/releases/download/pt_core_news_lg-3.8.0/pt_core_news_lg-3.8.0.tar.gz
   poetry run pip install pt_core_news_lg-3.8.0.tar.gz
   ```

3. **Alternative Models**
   ```bash
   # Use smaller model as fallback
   poetry run python -m spacy download pt_core_news_sm
   
   # Update config to use smaller model (edit in code)
   # Change 'pt_core_news_lg' to 'pt_core_news_sm'
   ```

### 📊 **Dashboard Issues**

#### Problem: Dashboard won't start or shows errors
```bash
# Symptoms
ModuleNotFoundError: No module named 'streamlit'
Address already in use: Port 8501
```

**Solutions:**
1. **Install Dashboard Dependencies**
   ```bash
   # Install Streamlit and Plotly
   poetry add streamlit plotly dash-bootstrap-components
   
   # Or install manually
   poetry run pip install streamlit==1.45.1 plotly==5.17.0
   ```

2. **Use Alternative Port**
   ```bash
   # Start on different port
   poetry run streamlit run src/dashboard/start_dashboard.py --server.port 8502
   
   # Or kill existing process
   lsof -ti:8501 | xargs kill -9
   ```

3. **Check Data Files**
   ```bash
   # Verify dashboard data exists
   ls -la src/dashboard/data/
   
   # Create sample data if missing
   poetry run python examples/quick_start.py
   ```

### 🗂️ **File & Permission Issues**

#### Problem: "Permission denied" or "File not found"
```bash
# Symptoms
PermissionError: [Errno 13] Permission denied
FileNotFoundError: No such file or directory
```

**Solutions:**
1. **Fix Permissions**
   ```bash
   # Make scripts executable
   chmod +x run_pipeline.py
   chmod +x src/dashboard/start_dashboard.py
   
   # Fix directory permissions
   chmod -R 755 data/
   chmod -R 755 pipeline_outputs/
   chmod -R 755 logs/
   ```

2. **Create Missing Directories**
   ```bash
   # Create all required directories
   mkdir -p data/{uploads,interim}
   mkdir -p pipeline_outputs
   mkdir -p logs/pipeline
   mkdir -p cache/embeddings
   mkdir -p checkpoints
   mkdir -p src/dashboard/data/{uploads,dashboard_results}
   ```

3. **Check Disk Space**
   ```bash
   # Check available space
   df -h .
   
   # Clean temporary files if needed
   rm -rf test_backups/
   rm -rf e2e_test_backups/
   find . -name "*.pyc" -delete
   ```

## 🔍 **Diagnostic Commands**

### System Health Check
```bash
# Comprehensive system check
poetry run python -c "
import sys, platform, os
from pathlib import Path

print('=== SYSTEM DIAGNOSTICS ===')
print(f'Python: {sys.version}')
print(f'Platform: {platform.system()} {platform.release()}')
print(f'Working directory: {Path.cwd()}')

# Check critical files
critical_files = [
    'config/settings.yaml',
    'config/anthropic.yaml', 
    'config/voyage_embeddings.yaml',
    'src/anthropic_integration/unified_pipeline.py',
    'run_pipeline.py'
]

for file in critical_files:
    path = Path(file)
    status = '✅' if path.exists() else '❌'
    print(f'{status} {file}')

# Check environment
print(f'ANTHROPIC_API_KEY: {\"Set\" if os.getenv(\"ANTHROPIC_API_KEY\") else \"Missing\"}')
print(f'VOYAGE_API_KEY: {\"Set\" if os.getenv(\"VOYAGE_API_KEY\") else \"Missing\"}')
"
```

### Optimization Status Check
```bash
# Check all optimization weeks
poetry run python examples/quick_start.py | grep -E "(Week [1-5]|Overall:|ENTERPRISE)"
```

### Log Analysis
```bash
# View recent errors
tail -50 logs/pipeline_execution.log | grep -E "(ERROR|CRITICAL)"

# Check API calls
grep -c "Cliente Anthropic inicializado" logs/pipeline_execution.log

# Monitor memory usage
grep "Memory" logs/pipeline_execution.log | tail -10
```

### Performance Monitoring
```bash
# Check processing times
grep "Pipeline execution completed" logs/pipeline_execution.log

# Monitor cache performance  
grep "cache hit" logs/pipeline_execution.log | wc -l

# Check optimization effectiveness
poetry run python test_all_weeks_consolidated.py | grep "PASSED\|FAILED"
```

## 📞 **Getting Help**

### Before Reporting Issues
1. **Run Diagnostics**: Execute all diagnostic commands above
2. **Check Logs**: Review `logs/pipeline_execution.log` for errors
3. **Verify Installation**: Re-run installation verification steps
4. **Document Symptoms**: Note exact error messages and steps to reproduce

### Log Collection Script
```bash
# Collect diagnostic information
cat > collect_diagnostics.sh << 'EOF'
#!/bin/bash
echo "=== Pipeline Bolsonarismo Diagnostics ==="
echo "Date: $(date)"
echo "System: $(uname -a)"
echo "Python: $(python3 --version)"
echo "Poetry: $(poetry --version)"
echo

echo "=== Configuration Files ==="
ls -la config/
echo

echo "=== Environment Variables ==="
env | grep -E "(ANTHROPIC|VOYAGE)" | sed 's/=.*/=***HIDDEN***/'
echo

echo "=== Recent Errors ==="
tail -20 logs/pipeline_execution.log | grep -E "(ERROR|CRITICAL)" || echo "No recent errors"
echo

echo "=== Disk Usage ==="
df -h .
echo

echo "=== Python Packages ==="
poetry show | head -20
EOF

chmod +x collect_diagnostics.sh
./collect_diagnostics.sh > diagnostics_$(date +%Y%m%d_%H%M%S).txt
```

### Contact Information
- **GitHub Issues**: https://github.com/pabloera/authorigram/issues
- **Documentation**: Check `README.md`, `CLAUDE.md`, and `pipeline_optimization.md`
- **Examples**: Run scripts in `examples/` directory for working configurations

---

⚡ **Remember**: Most issues are resolved by verifying API keys, checking file permissions, and ensuring all dependencies are properly installed via Poetry.