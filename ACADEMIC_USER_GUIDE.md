# Digital Discourse Monitor v5.0.0: Academic User Guide

## Quick Start Guide for Social Science Researchers

### 🎓 About This Tool

The Digital Discourse Monitor is an academic research system designed for social scientists studying Brazilian political discourse, authoritarianism, and digital violence. This tool analyzes Telegram messages (2019-2023) to identify patterns in political communication and detect signs of digital authoritarianism.

**Designed for**: Social science researchers, PhD students, research centers
**Purpose**: Academic research on Brazilian political discourse analysis
**Technical Level**: Beginner-friendly with advanced features available

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Verify Your System
Your computer needs:
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Disk Space**: 10GB free space
- **Operating System**: macOS, Linux, or Windows with WSL

### Step 2: Academic Deployment
Run the automated academic deployment:
```bash
# Quick deployment validation
poetry run python academic_deploy.py --validate

# Full academic deployment
poetry run python academic_deploy.py --environment research
```

### Step 3: Run Your First Analysis
```bash
# Complete analysis pipeline
poetry run python run_pipeline.py

# Launch research dashboard
poetry run python src/dashboard/start_dashboard.py
```

**Expected Time**: 15-30 minutes for first run (depending on dataset size)

---

## 📊 Understanding Your Results

### Political Classification System
The system categorizes messages using Brazilian political taxonomy:

| Category | Portuguese Term | Description |
|----------|----------------|-------------|
| Far Right | Extrema Direita | Authoritarian, anti-democratic discourse |
| Right | Direita | Conservative political positions |
| Center-Right | Centro-Direita | Moderate conservative views |
| Center | Centro/Neutro | Non-partisan or balanced content |
| Center-Left | Centro-Esquerda | Moderate progressive views |
| Left | Esquerda | Progressive political positions |

### Key Metrics Generated
- **Political Orientation Distribution**: Percentage breakdown by category
- **Sentiment Analysis**: Emotional tone (positive, negative, neutral)
- **Topic Modeling**: Main discussion themes (e.g., economy, security, health)
- **Temporal Evolution**: How discourse changes over time
- **Network Analysis**: Coordination patterns and influence networks

---

## 📈 Research Dashboard Features

### Main Dashboard Sections

1. **Volume Analysis**
   - Message count trends over time
   - Political category distribution
   - Regional/temporal patterns

2. **Content Analysis**
   - Top hashtags and mentions
   - Most shared domains and links
   - Semantic topic clusters

3. **Political Analysis**
   - Political orientation breakdown
   - Sentiment by political category
   - Authoritarian discourse indicators

4. **Quality Metrics**
   - Data processing statistics
   - Confidence scores
   - Validation results

### Interpreting Results

**High Confidence Indicators**:
- Sentiment scores > 0.7 or < -0.7
- Political classification with >80% confidence
- Topics with >100 representative messages

**Research Notes**:
- All Portuguese political categories are preserved for citation
- Results include confidence intervals for academic rigor
- Temporal analysis shows discourse evolution patterns

---

## 💰 Cost Management for Academic Use

### Budget Configuration
The system is optimized for academic budgets:
- **Default Budget**: $50/month
- **Auto-Protection**: Stops before exceeding budget
- **Cost Monitoring**: Real-time tracking and alerts

### Optimization Features
- **40% Cost Reduction**: Through intelligent caching
- **Sample Processing**: Analyze representative samples for large datasets
- **Efficient Models**: Uses cost-optimized AI models (claude-3-5-haiku, voyage-3.5-lite)

### Managing Costs
```bash
# Check current costs
poetry run python -c "
from src.anthropic_integration.cost_monitor import get_cost_monitor
monitor = get_cost_monitor()
print(monitor.get_daily_summary())
"

# Set custom budget
poetry run python -c "
from src.academic_config import get_academic_config
config = get_academic_config()
config.set_monthly_budget(25.0)  # $25/month
"
```

---

## 🔬 Research Best Practices

### Data Preparation
1. **File Format**: Use CSV files with UTF-8 encoding
2. **Required Columns**: Ensure 'content' column contains message text
3. **Sample Size**: Start with 1,000-10,000 messages for testing

### Analysis Workflow
1. **Exploratory Phase**: Run complete pipeline on sample
2. **Validation Phase**: Check results for research quality
3. **Full Analysis**: Process complete dataset
4. **Documentation**: Export results for citation

### Quality Assurance
- **Validation Rate**: System maintains 95% success rate
- **Reproducibility**: Same results with same input data
- **Confidence Scores**: All analyses include confidence metrics
- **Error Handling**: Automatic recovery from processing issues

---

## 📋 Common Research Scenarios

### Scenario 1: Political Discourse Evolution
**Research Question**: How did pro-government discourse change 2019-2023?

**Steps**:
1. Filter data by time periods
2. Run political classification
3. Analyze temporal trends in dashboard
4. Export time-series data for statistical analysis

### Scenario 2: Authoritarian Rhetoric Detection
**Research Question**: What patterns indicate authoritarian discourse?

**Steps**:
1. Use political classification for "Extrema Direita" category
2. Analyze sentiment patterns in authoritarian messages
3. Examine network coordination patterns
4. Extract qualitative examples for discourse analysis

### Scenario 3: Topic Modeling Research
**Research Question**: What were the main discussion themes?

**Steps**:
1. Run topic modeling pipeline
2. Review AI-generated topic interpretations
3. Validate topics with manual coding sample
4. Analyze topic evolution over time

---

## 🛠️ Technical Support for Researchers

### System Validation
```bash
# Comprehensive system check
poetry run python academic_deploy.py --validate

# Test optimization systems
poetry run python test_all_weeks_consolidated.py

# Memory usage monitoring
poetry run python -c "
from src.optimized.memory_optimizer import get_global_memory_manager
manager = get_global_memory_manager()
print(manager.get_memory_summary())
"
```

### Common Issues and Solutions

**Issue**: Out of memory errors
**Solution**: Reduce dataset size or enable streaming mode
```bash
poetry run python run_pipeline.py --streaming --chunk-size 500
```

**Issue**: API rate limits
**Solution**: Enable intelligent caching and reduce API calls
```bash
poetry run python -c "
from src.optimized.smart_claude_cache import get_smart_cache
cache = get_smart_cache()
print(f'Cache hit rate: {cache.get_hit_rate():.1%}')
"
```

**Issue**: Cost concerns
**Solution**: Use academic sampling mode
```bash
poetry run python run_pipeline.py --academic-mode --sample-rate 0.1
```

### Getting Help
1. **System Validation**: Run `academic_deploy.py --validate` first
2. **Log Files**: Check `logs/` directory for detailed error information
3. **Configuration**: Verify `config/academic_settings.yaml` settings
4. **Memory Issues**: Monitor with built-in memory optimizer

---

## 📊 Exporting Results for Publication

### Data Export Formats
- **CSV**: Structured data for statistical analysis
- **JSON**: Detailed results with metadata
- **Dashboard**: Interactive visualizations for presentations

### Citation-Ready Outputs
- All Portuguese political categories preserved
- Confidence intervals included
- Processing statistics documented
- Version information for reproducibility

### Example Export Commands
```bash
# Export political analysis results
poetry run python -c "
import pandas as pd
df = pd.read_csv('pipeline_outputs/05_political_analyzed.csv')
df.to_csv('research_export_political.csv', index=False)
"

# Export dashboard data
poetry run python src/dashboard/start_dashboard.py --export-mode
```

---

## 🔄 System Updates and Maintenance

### Regular Maintenance
```bash
# System health check
poetry run python scripts/maintenance_tools.py validate

# Clear old cache files
poetry run python scripts/maintenance_tools.py cleanup

# Update dependencies
poetry update
```

### Version Information
- **Current Version**: v5.0.0 (Final Consolidation)
- **Optimization Level**: Week 1-5 complete integration
- **Academic Optimization**: Enabled
- **Memory Target**: 4GB for academic computing environments

---

## 📚 Academic Resources

### Research Applications
- **Political Science**: Discourse analysis, polarization studies
- **Communication Studies**: Digital rhetoric, media influence
- **Sociology**: Social movement analysis, digital authoritarianism
- **Computational Social Science**: Automated content analysis

### Methodological Considerations
- **Sampling**: System uses stratified sampling for large datasets
- **Validation**: Human validation recommended for 10% of results
- **Bias Detection**: Monitor political classification balance
- **Temporal Effects**: Consider platform changes over analysis period

### Publications and Citations
When using this tool in academic work:
1. Cite the tool version (v5.0.0)
2. Document analysis parameters used
3. Include confidence intervals in results
4. Acknowledge Portuguese category preservation
5. Report any data preprocessing steps

---

## 🎯 Summary: Ready for Research

Your Digital Discourse Monitor v5.0.0 is now configured for academic research with:

✅ **95% Success Rate**: Reliable analysis for research reproducibility
✅ **$50/Month Budget**: Cost-effective for academic use
✅ **4GB Memory Optimization**: Works on standard academic computers
✅ **Portuguese Category Preservation**: Maintains research authenticity
✅ **Week 1-5 Optimizations**: Complete performance enhancement
✅ **Academic Dashboard**: Research-focused visualization tools

**Next Steps**:
1. Run your first analysis on sample data
2. Explore the research dashboard
3. Export initial results for validation
4. Scale up to your complete dataset

**Support**: Use `academic_deploy.py --validate` for any system issues.

---

*Digital Discourse Monitor v5.0.0 - Designed for Brazilian Political Discourse Research*
*© 2025 - Academic Research Tool for Social Scientists*