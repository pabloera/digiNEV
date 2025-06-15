# Digital Discourse Monitor v5.0.0

> **Brazilian Political Discourse Analysis for Social Science Research**
> 
> Automated analysis system for Telegram messages (2019-2023) focused on political discourse, denialism, and digital authoritarianism. Designed for academic research centers studying violence and authoritarianism in Brazilian society.

## 🎓 Research Context

This tool was developed for **social scientists** and **researchers** studying:
- Political communication and digital authoritarianism
- Denial discourse and conspiracy theories
- Violence legitimization in digital spaces
- Democratic erosion patterns in Brazil (2019-2023)

## 🚀 Quick Start for Researchers

### Prerequisites
- Python 3.12+
- 4GB RAM minimum
- Anthropic API key (for AI analysis)
- Voyage.ai API key (for semantic analysis)

### Installation
```bash
# Clone repository
git clone [repository-url]
cd dataanalysis-bolsonarismo

# Setup environment
poetry install
cp .env.template .env
# Edit .env with your API keys

# Run complete analysis
poetry run python run_pipeline.py
```

### View Results
```bash
# Launch interactive dashboard
poetry run python src/dashboard/start_dashboard.py
```

## 📊 Research Pipeline (22 Stages)

The system analyzes messages through comprehensive stages:

1. **Data Processing** (Stages 1-4): Chunking, encoding, deduplication, validation
2. **Political Analysis** (Stage 5): Hierarchical Brazilian political categorization
3. **Text Cleaning** (Stage 6): Intelligent content preservation
4. **Linguistic Analysis** (Stage 7): Portuguese NLP with spaCy
5. **Sentiment Analysis** (Stage 8): Political context-aware sentiment
6. **Topic Modeling** (Stage 9): Semantic topic discovery
7. **Content Analysis** (Stages 10-14): TF-IDF, clustering, hashtags, domains, temporal
8. **Advanced Analysis** (Stages 15-18): Network analysis, qualitative coding, topic interpretation
9. **Search & Validation** (Stages 19-20): Semantic search, pipeline validation

## 🔬 Analysis Categories (Portuguese - Research Validity)

### Political Categories
- **Esquerda**: Partidos e movimentos de esquerda
- **Centro-Esquerda**: Social-democracia, centro-esquerda moderada
- **Centro**: Posições centristas, moderadas
- **Centro-Direita**: Conservadorismo liberal, direita tradicional
- **Direita**: Conservadorismo social, direita tradicional
- **Extrema-Direita**: Movimentos autoritários, ultranacionalismo

### Sentiment Analysis
- **Positivo**: Apoio, esperança, entusiasmo
- **Negativo**: Crítica, revolta, descontentamento
- **Neutro**: Informativo, descritivo
- **Misto**: Ambivalência, contradições

## 📁 Project Structure

```
digiNEV/
├── run_pipeline.py              # Main execution script
├── src/
│   ├── anthropic_integration/   # AI analysis modules
│   ├── dashboard/              # Research visualization
│   └── optimized/              # Performance enhancements
├── config/                     # Configuration files
├── data/                       # Research datasets
└── docs/                       # Documentation
```

## 🛠️ For Researchers and Students

### Basic Usage
```bash
# Analyze your dataset
poetry run python run_pipeline.py --dataset "data/your_messages.csv"

# View results in browser
poetry run python src/dashboard/start_dashboard.py
```

### Research Commands
```bash
# Political analysis only
poetry run python src/main.py --stage 05_political_analysis

# Generate research report
poetry run python src/main.py --generate-report

# Export results for statistical analysis
poetry run python src/main.py --export-csv
```

## 💡 Academic Features

- **Reproducible Results**: Fixed model versions for consistent analysis
- **Cost-Efficient**: Optimized for academic budgets
- **Brazilian Context**: Specialized for Brazilian political discourse
- **Validation**: Quality checks for research reliability
- **Documentation**: Complete methodology documentation

## 📚 Citation and Research Use

This tool is designed for academic research on digital authoritarianism and political discourse in Brazil. All analysis categories maintain Portuguese terminology to preserve research validity and cultural context.

## 🔧 Configuration

Essential files for researchers:
- `.env` - API keys and basic settings
- `config/settings.yaml` - Main configuration
- `config/anthropic.yaml` - AI analysis settings

## 📞 Support

For researchers using this tool:
- Check `CLAUDE.md` for detailed technical documentation
- Use the dashboard for interactive exploration
- Review `logs/` directory for analysis details

---

**digiNEV v5.0.0** - Academic tool for analyzing Brazilian political discourse and digital authoritarianism patterns.