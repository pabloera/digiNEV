# 🚀 Installation Guide - Pipeline Bolsonarismo v5.0.0

## 📋 **Prerequisites**

### System Requirements
- **Python**: 3.12+ (tested with 3.12.5)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for data processing
- **OS**: macOS, Linux, or Windows with WSL2

### Required External Services
- **Anthropic API**: Account with API key for Claude models
- **Voyage.ai API**: Account with API key for embeddings (optional but recommended)

## 🛠️ **Step-by-Step Installation**

### 1. Clone Repository
```bash
git clone https://github.com/pabloera/authorigram.git
cd authorigram
```

### 2. Install Poetry (Dependency Manager)
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
```

### 3. Setup Python Environment
```bash
# Create and activate virtual environment
poetry install

# Install optimization dependencies
poetry install --with optimization

# Install development tools (optional)
poetry install --with dev

# Install Jupyter support (optional)  
poetry install --with jupyter
```

### 4. Install spaCy Language Model
```bash
# Download Portuguese language model (required for Stage 07)
poetry run python -m spacy download pt_core_news_lg

# Verify installation
poetry run python -c "import spacy; nlp = spacy.load('pt_core_news_lg'); print('✅ spaCy model installed')"
```

### 5. Configure API Keys
```bash
# Copy configuration templates
cp config/anthropic.yaml.template config/anthropic.yaml
cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml

# Create environment file
touch .env
echo "ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE" >> .env
echo "VOYAGE_API_KEY=pa-YOUR_KEY_HERE" >> .env  # Optional but recommended
```

### 6. Prepare Data Directories
```bash
# Create required directories
mkdir -p data/uploads
mkdir -p data/interim  
mkdir -p pipeline_outputs
mkdir -p logs/pipeline
mkdir -p cache/embeddings
mkdir -p checkpoints
mkdir -p src/dashboard/data/uploads
```

### 7. Validation Test
```bash
# Test system components
poetry run python examples/quick_start.py

# Test optimization systems
poetry run python test_all_weeks_consolidated.py

# Expected output: "5/5 weeks active (100%)"
```

## ⚙️ **Configuration Details**

### API Configuration Files

#### `config/anthropic.yaml`
```yaml
# Core Anthropic settings
api_key: "${ANTHROPIC_API_KEY}"
default_model: "claude-3-5-haiku-20241022"
timeout_seconds: 120
max_retries: 3

# Stage-specific models (optional overrides)
stages:
  stage_05_political: "claude-3-5-haiku-20241022"
  stage_08_sentiment: "claude-3-5-sonnet-20241022"
  stage_15_network: "claude-sonnet-4-20250514"
  stage_16_qualitative: "claude-3-5-sonnet-20241022"
  stage_17_review: "claude-3-5-sonnet-20241022"
  stage_18_topics: "claude-sonnet-4-20250514"
  stage_20_validation: "claude-3-5-haiku-20241022"
```

#### `config/voyage_embeddings.yaml`
```yaml
# Voyage.ai embeddings configuration
api_key: "${VOYAGE_API_KEY}"
model: "voyage-3.5-lite"  # Most cost-effective
batch_size: 128
enable_sampling: true     # 96% cost reduction
max_messages_per_dataset: 50000
timeout_seconds: 60
```

### Environment Variables
```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...    # From console.anthropic.com

# Optional but recommended  
VOYAGE_API_KEY=pa-...                 # From dash.voyageai.com

# Development (optional)
DEBUG=false
LOG_LEVEL=INFO
CACHE_TTL_HOURS=24
```

## 🗂️ **Data Setup**

### Input Data Format
Place CSV files in `data/uploads/` with the following required columns:
```csv
message_id,datetime,body,url,hashtag,channel,is_fwrd,mentions,sender,media_type,domain,body_cleaned,source_dataset,hash_id
```

### Sample Data
```bash
# Download sample dataset (if available)
wget -O data/uploads/sample_dataset.csv "YOUR_SAMPLE_DATA_URL"

# Or create minimal test data
echo "message_id,datetime,body,url,hashtag,channel,is_fwrd,mentions,sender,media_type,domain,body_cleaned,source_dataset,hash_id" > data/uploads/test_data.csv
echo "1,2023-01-01 10:00:00,Test message,,#test,test_channel,False,,test_user,text,,Test message,test_dataset,abc123" >> data/uploads/test_data.csv
```

## 🔍 **Verification Steps**

### 1. Component Health Check
```bash
poetry run python -c "
from src.optimized.optimized_pipeline import get_global_optimized_pipeline
from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
import yaml

# Test optimization components
pipeline = get_global_optimized_pipeline()
print(f'✅ Optimization system: {\"Active\" if pipeline else \"Inactive\"}')

# Test configuration loading
with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)
print(f'✅ Configuration loaded: {len(config)} sections')

print('🎉 System health check passed!')
"
```

### 2. API Connectivity Test
```bash
poetry run python -c "
import os
from anthropic import Anthropic

# Test Anthropic API
if os.getenv('ANTHROPIC_API_KEY'):
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    try:
        response = client.messages.create(
            model='claude-3-5-haiku-20241022',
            max_tokens=10,
            messages=[{'role': 'user', 'content': 'Hi'}]
        )
        print('✅ Anthropic API: Connected')
    except Exception as e:
        print(f'❌ Anthropic API: {e}')
else:
    print('⚠️ Anthropic API: Key not configured')
"
```

### 3. Pipeline Execution Test
```bash
# Run with test data
poetry run python run_pipeline.py

# Expected: Pipeline loads with all optimizations active
# Look for: "5/5 weeks active (100%)"
```

## 🚨 **Common Installation Issues**

### spaCy Model Download Fails
```bash
# Alternative download method
poetry run python -c "
import spacy.cli
spacy.cli.download('pt_core_news_lg')
"

# Manual download (if automatic fails)
# Visit: https://github.com/explosion/spacy-models/releases
# Download pt_core_news_lg-3.8.0.tar.gz
poetry run pip install pt_core_news_lg-3.8.0.tar.gz
```

### Poetry Installation Issues
```bash
# Clear Poetry cache
poetry cache clear pypi --all

# Reinstall dependencies
rm poetry.lock
poetry install

# Use pip as fallback (not recommended for production)
pip install -r requirements.txt
```

### API Key Issues
```bash
# Verify .env file
cat .env

# Test key format (Anthropic keys start with 'sk-ant-api03-')
echo $ANTHROPIC_API_KEY | grep "^sk-ant-api03-"

# Test key permissions at console.anthropic.com
```

### Memory Issues
```bash
# Enable memory optimization
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1

# Reduce chunk size in config/settings.yaml
# processing -> chunk_size: 5000 (instead of 10000)
```

## 📞 **Support**

### Log Files
- Pipeline execution: `logs/pipeline_execution.log`
- Component errors: `logs/pipeline/`
- Optimization status: Check console output

### Diagnostic Commands
```bash
# System information
poetry run python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.system()} {platform.release()}')
print(f'Architecture: {platform.machine()}')
"

# Dependency check
poetry show | grep -E "(anthropic|voyageai|spacy|pandas)"

# Memory and disk usage
df -h .
free -h  # Linux/macOS: vm_stat | head -5
```

### Getting Help
1. Check `TROUBLESHOOTING.md` for common issues
2. Review logs in `logs/` directory
3. Run diagnostic commands above
4. Check GitHub issues: https://github.com/pabloera/authorigram/issues

---

✅ **Installation Complete!** You should now have a fully functional Pipeline Bolsonarismo v5.0.0 system ready for data analysis.