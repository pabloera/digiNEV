# 📚 API Reference - Pipeline Bolsonarismo v5.0.0

## 🏗️ **Architecture Overview**

### Core Components
```python
# Main Pipeline Entry Point
run_pipeline.py               # Main execution script
└── UnifiedAnthropicPipeline  # 22-stage pipeline engine
    ├── Optimization Layers   # v5.0.0 performance enhancements
    ├── Stage Processors      # Individual stage implementations
    └── Integration Services  # API and external service integrations
```

### Pipeline Stages (22 Total)
| Stage | Function | Input | Output | Technology |
|-------|----------|-------|---------|------------|
| 01 | `chunk_processing()` | Raw CSV | Chunked data | Native |
| 02 | `encoding_validation()` | Chunked data | Validated encoding | Enhanced |
| 03 | `deduplication()` | Validated data | Deduplicated | Enhanced |
| 04 | `feature_validation()` | Deduplicated | Validated features | Native |
| 04b | `statistical_analysis_pre()` | Features | Pre-stats | Enhanced |
| 05 | `political_analysis()` | Clean data | Political labels | Anthropic |
| 06 | `text_cleaning()` | Raw text | Clean text | Enhanced |
| 06b | `statistical_analysis_post()` | Clean text | Post-stats | Enhanced |
| 07 | `linguistic_processing()` | Clean text | NLP features | spaCy |
| 08 | `sentiment_analysis()` | Text | Sentiment scores | Anthropic |
| 09 | `topic_modeling()` | Text | Topic clusters | Voyage.ai |
| 10 | `tfidf_extraction()` | Text | TF-IDF vectors | Voyage.ai |
| 11 | `clustering()` | Vectors | Clusters | Voyage.ai |
| 12 | `hashtag_normalization()` | Hashtags | Normalized | Native |
| 13 | `domain_analysis()` | URLs | Domain info | Native |
| 14 | `temporal_analysis()` | Timestamps | Time patterns | Native |
| 15 | `network_analysis()` | Relations | Network graph | Anthropic |
| 16 | `qualitative_analysis()` | Text | Qualitative codes | Anthropic |
| 17 | `smart_pipeline_review()` | Pipeline data | Quality report | Anthropic |
| 18 | `topic_interpretation()` | Topics | Interpretations | Anthropic |
| 19 | `semantic_search()` | Embeddings | Search index | Voyage.ai |
| 20 | `pipeline_validation()` | Final data | Validation report | Anthropic |

## 🚀 **Main API Classes**

### UnifiedAnthropicPipeline
**Main pipeline orchestrator for all 22 stages**

```python
from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline

# Initialize
pipeline = UnifiedAnthropicPipeline(config, base_path)

# Execute complete pipeline
results = pipeline.run_complete_pipeline(dataset_paths)

# Execute specific stages
results = pipeline.run_stage(stage_name, dataframe)
```

#### Methods
- `run_complete_pipeline(dataset_paths: List[str]) -> Dict[str, Any]`
- `run_stage(stage_name: str, df: pd.DataFrame) -> pd.DataFrame`
- `get_pipeline_health() -> Dict[str, Any]`
- `_initialize_components() -> bool`

#### Configuration
```python
config = {
    "anthropic": {"api_key": "sk-ant-...", "default_model": "claude-3-5-haiku-20241022"},
    "voyage_embeddings": {"api_key": "pa-...", "model": "voyage-3.5-lite"},
    "processing": {"chunk_size": 10000, "timeout_seconds": 120},
    "data": {"path": "data/uploads", "output_path": "pipeline_outputs"}
}
```

### Optimization System (v5.0.0)
**Performance enhancement layers applied to original pipeline**

```python
# Week 1-2: Emergency Cache + Advanced Caching
from src.optimized.optimized_pipeline import get_global_optimized_pipeline
optimized_pipeline = get_global_optimized_pipeline()

# Week 3: Parallelization + Streaming
from src.optimized.parallel_engine import get_global_parallel_engine
from src.optimized.streaming_pipeline import get_global_streaming_pipeline
parallel_engine = get_global_parallel_engine()
streaming_pipeline = get_global_streaming_pipeline()

# Week 4: Real-time Monitoring
from src.optimized.realtime_monitor import get_global_performance_monitor
monitor = get_global_performance_monitor()
monitor.start_monitoring()

# Week 5: Adaptive Memory Management
from src.optimized.memory_optimizer import get_global_memory_manager
memory_manager = get_global_memory_manager()
memory_manager.start_adaptive_management()
```

## 🎯 **Core Stage APIs**

### Stage 05: Political Analysis
```python
from src.anthropic_integration.political_analyzer import PoliticalAnalyzer

analyzer = PoliticalAnalyzer(config)
result_df = analyzer.analyze_political_content(df)

# Output columns added:
# - political_category: "neutro", "direita", "esquerda", "centro"
# - political_subcategory: Detailed classification
# - political_confidence: Confidence score (0-1)
# - political_reasoning: Analysis reasoning
```

### Stage 08: Sentiment Analysis
```python
from src.anthropic_integration.sentiment_analyzer import AnthropicSentimentAnalyzer

analyzer = AnthropicSentimentAnalyzer(config)
result_df = analyzer.analyze_sentiment(df)

# Output columns added:
# - sentiment_score: Numerical score (-1 to 1)
# - sentiment_label: "positive", "negative", "neutral"
# - sentiment_confidence: Confidence score (0-1)
# - sentiment_reasoning: Analysis reasoning
```

### Stage 07: Linguistic Processing (spaCy)
```python
from src.anthropic_integration.spacy_nlp_processor import SpacyNLPProcessor

processor = SpacyNLPProcessor(config)
result_df = processor.process_linguistic_features(df)

# Output columns added:
# - tokens: Tokenized text
# - lemmas: Lemmatized tokens
# - pos_tags: Part-of-speech tags
# - named_entities: Named entity recognition
# - linguistic_complexity: Complexity score
```

### Stages 09-11, 19: Voyage.ai Integration
```python
from src.anthropic_integration.voyage_topic_modeler import VoyageTopicModeler
from src.anthropic_integration.semantic_tfidf_analyzer import SemanticTfidfAnalyzer
from src.anthropic_integration.voyage_clustering_analyzer import VoyageClusteringAnalyzer
from src.anthropic_integration.semantic_search_engine import SemanticSearchEngine

# Topic Modeling (Stage 09)
topic_modeler = VoyageTopicModeler(config)
result_df = topic_modeler.extract_topics(df)

# TF-IDF Extraction (Stage 10)
tfidf_analyzer = SemanticTfidfAnalyzer(config)
result_df = tfidf_analyzer.extract_semantic_tfidf(df)

# Clustering (Stage 11)
clustering_analyzer = VoyageClusteringAnalyzer(config)
result_df = clustering_analyzer.perform_clustering(df)

# Semantic Search (Stage 19)
search_engine = SemanticSearchEngine(config)
search_index = search_engine.build_search_index(df)
results = search_engine.search(query, top_k=10)
```

## 🛠️ **Utility APIs**

### Configuration Management
```python
from src.anthropic_integration.base import AnthropicBase

# Load configuration
base = AnthropicBase(config)
config_loaded = base.load_configuration()

# Enhanced configuration with stage-specific models
enhanced_config = base.get_enhanced_config("stage_05_political")
```

### Cost Monitoring
```python
from src.anthropic_integration.cost_monitor import ConsolidatedCostMonitor

monitor = ConsolidatedCostMonitor()
cost_summary = monitor.get_cost_summary()
cost_report = monitor.generate_cost_report()

# Cost data structure:
{
    "total_cost_usd": 1.23,
    "api_calls": 456,
    "tokens_used": 78900,
    "cost_by_stage": {...},
    "optimization_savings": 0.45
}
```

### Cache Management
```python
from src.optimized.smart_claude_cache import SmartClaudeCache
from src.optimized.emergency_embeddings import EmergencyEmbeddingsCache

# Claude API Cache
claude_cache = SmartClaudeCache()
cached_response = claude_cache.get(prompt_hash)
claude_cache.set(prompt_hash, response, ttl=3600)

# Voyage.ai Embeddings Cache
embeddings_cache = EmergencyEmbeddingsCache()
cached_embeddings = embeddings_cache.get_embeddings(text_list)
```

### Memory Optimization
```python
from src.optimized.memory_optimizer import get_global_memory_manager

memory_manager = get_global_memory_manager()

# Start adaptive management
memory_manager.start_adaptive_management()

# Get current status
status = memory_manager.get_management_summary()
print(f"Current memory: {status['management_status']['current_memory_gb']:.2f}GB")
print(f"Within target: {status['management_status']['memory_within_target']}")

# Stop management
memory_manager.stop_adaptive_management()
```

## 📊 **Data Schemas**

### Input Data Schema
```python
Required columns:
{
    "message_id": "str",           # Unique identifier
    "datetime": "datetime",        # Message timestamp
    "body": "str",                 # Message content
    "url": "str",                  # Optional URL
    "hashtag": "str",             # Optional hashtags
    "channel": "str",             # Source channel
    "is_fwrd": "bool",            # Is forwarded message
    "mentions": "str",            # User mentions
    "sender": "str",              # Sender username
    "media_type": "str",          # Media type
    "domain": "str",              # URL domain
    "body_cleaned": "str",        # Pre-cleaned text
    "source_dataset": "str",      # Source identifier
    "hash_id": "str"              # Content hash
}
```

### Output Data Schema (After Pipeline)
```python
Additional columns added by pipeline:
{
    # Stage 05 - Political Analysis
    "political_category": "str",
    "political_subcategory": "str", 
    "political_confidence": "float",
    "political_reasoning": "str",
    
    # Stage 07 - Linguistic Processing
    "tokens": "list",
    "lemmas": "list",
    "pos_tags": "list",
    "named_entities": "list",
    "linguistic_complexity": "float",
    
    # Stage 08 - Sentiment Analysis
    "sentiment_score": "float",
    "sentiment_label": "str",
    "sentiment_confidence": "float",
    
    # Stage 09 - Topic Modeling
    "topic_id": "int",
    "topic_label": "str",
    "topic_probability": "float",
    
    # Stage 11 - Clustering
    "cluster_id": "int",
    "cluster_label": "str",
    "cluster_distance": "float",
    
    # Plus 50+ additional analysis columns...
}
```

## 🔧 **Configuration Reference**

### config/settings.yaml
```yaml
# Main configuration file
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  default_model: "claude-3-5-haiku-20241022"
  timeout_seconds: 120
  max_retries: 3

voyage_embeddings:
  api_key: "${VOYAGE_API_KEY}"
  model: "voyage-3.5-lite"
  batch_size: 128
  enable_sampling: true
  max_messages_per_dataset: 50000

processing:
  chunk_size: 10000
  concurrent_limit: 5
  memory_limit_gb: 8

data:
  path: "data/uploads"
  interim_path: "data/interim"
  output_path: "pipeline_outputs"
  dashboard_path: "src/dashboard/data"
```

### Stage-Specific Models
```yaml
# config/anthropic.yaml - Model overrides
stages:
  stage_05_political: "claude-3-5-haiku-20241022"    # Fast, cost-effective
  stage_08_sentiment: "claude-3-5-sonnet-20241022"   # More accurate
  stage_15_network: "claude-sonnet-4-20250514"       # Most capable
  stage_18_topics: "claude-sonnet-4-20250514"        # Complex reasoning
  stage_20_validation: "claude-3-5-haiku-20241022"   # Fast validation
```

## 🔄 **Error Handling**

### Exception Types
```python
# Custom exceptions
from src.anthropic_integration.exceptions import (
    PipelineExecutionError,
    ConfigurationError,
    APITimeoutError,
    DataValidationError
)

try:
    results = pipeline.run_complete_pipeline(datasets)
except PipelineExecutionError as e:
    print(f"Pipeline failed: {e}")
    print(f"Failed stage: {e.stage}")
    print(f"Error details: {e.details}")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
```

### Retry Logic
```python
# Built-in retry with exponential backoff
from src.anthropic_integration.progressive_timeout_manager import ProgressiveTimeoutManager

timeout_manager = ProgressiveTimeoutManager()
result = timeout_manager.execute_with_retry(
    func=stage_function,
    args=(dataframe,),
    max_retries=3,
    base_timeout=60
)
```

## 📈 **Performance Monitoring**

### Real-time Monitoring
```python
from src.optimized.realtime_monitor import get_global_performance_monitor

monitor = get_global_performance_monitor()
monitor.start_monitoring()

# Get current metrics
status = monitor.get_current_status()
print(f"Health score: {status['health_score']}/100")
print(f"Cache hit rate: {status['cache_hit_rate']:.2%}")
print(f"Error rate: {status['error_rate']:.2%}")

monitor.stop_monitoring()
```

### Benchmarking
```python
from src.optimized.pipeline_benchmark import get_global_benchmark

benchmark = get_global_benchmark()
results = benchmark.run_performance_test(
    test_sizes=[100, 500, 1000],
    iterations=3
)

for size, metrics in results.items():
    print(f"Size {size}: {metrics['avg_time']:.2f}s, {metrics['throughput']:.1f} records/s")
```

---

## 📞 **Support & Examples**

### Quick Examples
```bash
# Run examples
poetry run python examples/quick_start.py
poetry run python examples/optimization_usage.py

# API testing
poetry run python test_all_weeks_consolidated.py
```

### Documentation Links
- **Installation**: `INSTALLATION.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`
- **Project Overview**: `README.md`
- **AI Instructions**: `CLAUDE.md`
- **Optimization Details**: `pipeline_optimization.md`

---

✅ **This API reference covers all major components and usage patterns for Pipeline Bolsonarismo v5.0.0**