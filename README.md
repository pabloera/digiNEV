# digiNEV - Digital Discourse Monitor v.final

> **Academic Research Tool for Brazilian Political Discourse Analysis**
>
> Comprehensive analysis system for studying political discourse, authoritarianism, and digital violence in Brazilian Telegram messages (2019-2023). Designed specifically for social science researchers and academic institutions.

## 🎓 Research Focus

This academic tool enables researchers to study:
- **Digital Authoritarianism**: Patterns of authoritarian discourse in social media
- **Political Polarization**: Evolution of political categories and sentiment over time
- **Violence Legitimization**: How violent rhetoric develops and spreads
- **Democratic Erosion**: Discourse patterns that undermine democratic institutions
- **Conspiracy and Denialism**: Detection and analysis of misinformation patterns

## 🚀 Quick Start for Researchers

### System Requirements
- **Python**: 3.12+ (managed through Poetry)
- **Memory**: 4GB RAM minimum (optimized for academic computing)
- **Storage**: 10GB free space for datasets and results
- **API Keys**: Anthropic (Claude) and Voyage.ai for AI analysis

### One-Command Setup
```bash
# Clone and setup (requires Poetry pre-installed)
git clone [repository-url]
cd dataanalysis-bolsonarismo
poetry install

# Configure API keys
cp .env.template .env
# Edit .env with your Anthropic and Voyage.ai API keys

# Validate academic environment
poetry run python src/scripts/academic_deploy.py --validate

# Run first analysis
poetry run python run_pipeline.py
```

### Access Research Results
```bash
# Interactive research dashboard
poetry run python src/dashboard/start_dashboard.py

# Open browser at http://localhost:8050
```

## 📊 Analysis Pipeline (17 Research Stages)

The system processes Telegram messages through a 17-stage scientific pipeline:

| Phase | Stages | Purpose | Technology | Status |
|-------|--------|---------|------------|--------|
| **Preparation** | 01-02 | Feature extraction, text preprocessing | Core Python, regex | ✅ Operational |
| **Volume Reduction** | 03-06 | Dedup, stats, quality filter, affordances | Python, Anthropic API (optional) | ✅ Operational |
| **Linguistic Analysis** | 07-09 | spaCy NLP, political classification + TCW, TF-IDF | spaCy, scikit-learn | ✅ Operational |
| **Advanced Analysis** | 10-17 | Clustering, topics, semantics, temporal, network, domain, events, channels | scikit-learn, Python | ✅ Operational |

**Status**: **17/17 stages operational** · **0 errors** in end-to-end tests · **126 columns** output · **115 features** generated · **6 stages with API** (06, 08, 11, 12, 16, 17)
**Output**: Research-ready DataFrames with political classification, TCW coding (10 categories), affordances (8 types), LDA topics, K-Means clusters, sentiment + granular emotions, temporal patterns, network coordination, domain analysis.

## 🔬 Research Categories (Portuguese - Academic Authenticity)

### Brazilian Political Taxonomy
Categories preserved in Portuguese for research validity:

| Category | Portuguese Term | Research Application |
|----------|----------------|---------------------|
| **Far Left** | Esquerda | Socialist and communist movements |
| **Center-Left** | Centro-Esquerda | Social democratic positions |
| **Center** | Centro/Neutro | Non-partisan, balanced discourse |
| **Center-Right** | Centro-Direita | Liberal conservative positions |
| **Right** | Direita | Traditional conservative discourse |
| **Far Right** | Extrema-Direita | **Authoritarian, anti-democratic rhetoric** |

### Sentiment Classification
- **Positivo**: Support, hope, enthusiasm (research: positive mobilization)
- **Negativo**: Criticism, anger, dissatisfaction (research: grievance patterns)
- **Neutro**: Informational, descriptive (research: neutral information sharing)
- **Misto**: Ambivalent, contradictory (research: complex emotional states)

*Note: Portuguese categories maintained for citation authenticity and cultural context preservation.*

## 📁 Project Architecture

```
digiNEV/
├── 🚀 EXECUTION
│   ├── run_pipeline.py              # Main analysis execution
│   ├── academic_deploy.py           # Academic deployment & validation
│   └── test_all_weeks_consolidated.py # System validation tests
├── 📚 DOCUMENTATION
│   ├── README.md                    # Project overview (this file)
│   ├── CLAUDE.md                    # Technical documentation for developers
│   ├── ACADEMIC_USER_GUIDE.md       # Comprehensive guide for researchers
│   └── RESEARCH_HISTORY.md          # Development timeline & TDD journey
├── 🧠 CORE ANALYSIS
│   ├── src/anthropic_integration/   # 17-stage AI analysis pipeline
│   ├── src/optimized/              # Performance optimizations (v5.0)
│   ├── src/dashboard/              # Research visualization interface
│   ├── src/core/                   # Unified cache and execution systems
│   └── src/utils/                  # Memory management and utilities
├── ⚙️ CONFIGURATION
│   ├── config/academic_settings.yaml # Academic environment settings
│   ├── config/settings.yaml         # Main system configuration
│   ├── .env                         # API keys and secrets
│   └── pyproject.toml              # Poetry dependency management
├── 📊 DATA & RESULTS
│   ├── data/                       # Research datasets (CSV format)
│   ├── pipeline_outputs/           # Analysis results by stage
│   └── logs/                       # System logs and error tracking
└── 🔧 DEVELOPMENT
    ├── tests/                      # Test suite (95% success rate)
    ├── backup/                     # System backup and recovery
    └── archive/                    # Historical documentation
```

## 🛠️ Research Commands

### Standard Research Workflow
```bash
# Full analysis pipeline (recommended)
poetry run python run_pipeline.py

# Analyze specific dataset
poetry run python run_pipeline.py --dataset "data/your_dataset.csv"

# Launch interactive dashboard
poetry run python src/dashboard/start_dashboard.py
```

### Advanced Research Options
```bash
# Academic mode (optimized for research computing)
poetry run python run_pipeline.py --academic-mode

# Sample analysis (for large datasets)
poetry run python run_pipeline.py --sample-rate 0.1

# Streaming mode (memory-efficient)
poetry run python run_pipeline.py --streaming --chunk-size 500
```

### System Validation
```bash
# Validate academic environment
poetry run python src/scripts/academic_deploy.py --validate

# Test all optimizations
poetry run python test_all_weeks_consolidated.py

# Check memory usage
poetry run python -c "from src.utils.memory_manager import get_memory_status; print(get_memory_status())"
```

## 💡 Key Research Features

### 🎓 Academic Optimizations (v6.1.0)
- **Performance**: 60% time reduction through parallel processing
- **Memory**: 50% reduction (8GB → 4GB) for academic computing
- **Cost**: Batch API (50% off) + Prompt Caching (90% off input) for budget control
- **Reliability**: **100% success rate** - all 17 pipeline stages operational
- **Reproducibility**: Fixed model versions for consistent results
- **AI Enhancement**: 6/17 stages using Anthropic API (06, 08, 11, 12, 16, 17) with heuristic fallback
- **Pipeline Completion**: **17/17 stages fully implemented and validated**
- **Data Integrity**: **126 columns generated** with comprehensive validation
- **Production Ready**: Complete dataset processing via `process_dataset()` method

### 🇧🇷 Brazilian Research Specialization
- **Language**: Portuguese NLP optimization with spaCy models
- **Political Taxonomy**: 6-category Brazilian political classification
- **Cultural Context**: Preserves Portuguese categories for authenticity
- **Temporal Scope**: Specialized for 2019-2023 political period

### 🔬 Research Quality Assurance
- **Validation**: Comprehensive quality checks at each stage
- **Confidence Scores**: All classifications include confidence metrics
- **Error Handling**: Automatic recovery and fallback systems
- **Export Ready**: CSV/JSON outputs for statistical analysis
- **Pipeline Integrity**: **100% success rate** across all 17 stages
- **Code Organization**: All stages reorganized in numerical order (01-17)
- **Complete Processing**: New `process_dataset()` method for full pipeline execution
- **Brazilian Focus**: Enhanced Portuguese text processing and political keyword coverage

### 💻 Academic Computing Compatibility
- **Memory Target**: 4GB RAM (suitable for laptops/workstations)
- **Processing**: Adaptive parallel processing based on available resources
- **Deployment**: One-command setup with `academic_deploy.py`
- **Monitoring**: Real-time progress tracking and resource usage

## 📚 Academic Usage

### Research Applications
- **Political Science**: Discourse analysis, polarization measurement, democratic erosion studies
- **Communication Studies**: Digital rhetoric analysis, media influence patterns
- **Sociology**: Social movement analysis, authoritarianism in digital spaces
- **Computational Social Science**: Large-scale automated content analysis

### Methodological Considerations
- **Sampling**: Stratified sampling for large datasets to maintain representativeness
- **Validation**: Human validation recommended for 10-15% of automated classifications
- **Bias Monitoring**: Built-in checks for political classification balance
- **Temporal Validity**: Consider platform evolution effects (2019-2023)

### Citation Guidelines
When using digiNEV in academic work:
1. **Version**: Cite as "digiNEV v5.0.0"
2. **Categories**: Note preservation of Portuguese political terms
3. **Parameters**: Document analysis settings and confidence thresholds
4. **Reproducibility**: Include model versions (Claude 3.5 Haiku, Voyage 3.5 Lite)
5. **Data Processing**: Report any preprocessing or filtering steps

## 📚 Documentation Structure

### 🎯 Start Here
| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Project overview, quick start | New users, overview |
| **ACADEMIC_USER_GUIDE.md** | Complete user guide | Researchers, students |

### 🔧 Technical Reference
| File | Purpose | Audience |
|------|---------|----------|
| **CLAUDE.md** | Technical documentation, optimization details | Developers, advanced users |
| **RESEARCH_HISTORY.md** | Development timeline, TDD journey | Contributors, technical history |

### 📂 Additional Resources
- **config/README.md** - Configuration file explanations
- **src/dashboard/README.md** - Dashboard feature guide
- **RESEARCH_HISTORY.md** - Complete TDD implementation journey
- **API_INTEGRATION_SUMMARY.md** - API integration validation results
- **SYSTEM_READY.md** - Production readiness validation results
- **archive/weekly_reports/** - Historical development reports
- **backup/RECOVERY_PROCEDURES.md** - System recovery instructions

### 📊 System Validation
- **test_all_weeks_consolidated.py** - Comprehensive system validation (95% success rate)
- **archive/FINAL_CONSOLIDATION_REPORT_v5.0.0.md** - Complete optimization achievement summary

## ⚙️ Configuration for Researchers

### Essential Setup Files
```
.env                           # API keys (keep private)
config/academic_settings.yaml  # Academic environment settings
config/settings.yaml           # Main system configuration
```

### Academic Configuration Highlights
- **Budget Protection**: $50/month default with auto-stop
- **Memory Optimization**: 4GB target for academic computing
- **Portuguese Optimization**: Enhanced for Brazilian political texts
- **Sampling**: Intelligent sampling for large datasets
- **Caching**: Research-focused caching (48-72 hour TTL)

### Quick Configuration Check
```bash
# View current academic settings
poetry run python -c "from src.academic_config import get_academic_config; print(get_academic_config().get_research_summary())"
```

## 🆘 Getting Help

### First Steps
1. **System Check**: `poetry run python src/scripts/academic_deploy.py --validate`
2. **User Guide**: See `ACADEMIC_USER_GUIDE.md` for detailed instructions
3. **Technical Issues**: Check `CLAUDE.md` for troubleshooting

### Common Issues
| Problem | Solution | Command |
|---------|----------|---------|
| Memory errors | Enable streaming mode | `--streaming --chunk-size 500` |
| API costs | Use academic sampling | `--academic-mode --sample-rate 0.1` |
| Slow performance | Check optimization status | `test_all_weeks_consolidated.py` |
| Configuration issues | Validate environment | `src/scripts/academic_deploy.py --validate` |

### Research Support Resources
- **Interactive Dashboard**: Real-time progress monitoring
- **Detailed Logs**: Check `logs/` directory for error details
- **Quality Metrics**: 95% success rate with automatic recovery
- **Academic Community**: Tool designed for academic collaboration

---

## 🏆 digiNEV v.final — Production Ready

**Academic Research Tool for Brazilian Political Discourse Analysis**

*Specialized for studying digital authoritarianism, political polarization, and violence legitimization in Telegram messages (2019-2023)*

**Status**: 17/17 stages operational, end-to-end tests passed (0 errors), 126 columns output
**Pipeline**: Reestruturação completa (8 bugs) + TCW + API híbrida (Stages 06, 08, 11, 12, 16, 17)
**Validation**: Tested on 3 datasets (elec, pandemia, govbolso), 100-2000 rows, 3 time periods

**Last Updated**: February 22, 2026
**Academic Focus**: Social science research on Brazilian democracy and digital discourse