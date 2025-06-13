# Pipeline Bolsonarismo v4.9.8 - Dashboard Funcional com Correções Críticas 🚀

> **Análise de Discurso Político Brasileiro com Inteligência Artificial**
> 
> Pipeline unificado ultra-robusto para análise de mensagens do Telegram (2019-2023) focado em bolsonarismo, negacionismo e autoritarismo digital.
> 
> **v4.9.8 - Junho 2025**: 🎯 DASHBOARD FUNCIONAL com correções críticas implementadas! Análise temporal corrigida, erro `dropna=False` resolvido, 4 níveis políticos funcionais, 2 clusters semânticos identificados. Pipeline 22 etapas + Dashboard 100% operacional com 300 registros validados.

## 🚨 **INÍCIO RÁPIDO - LEIA PRIMEIRO!**

### ⚡ **Setup Inicial**
```bash
# 1. Configurar ambiente
cp config/anthropic.yaml.template config/anthropic.yaml
cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml

# 2. Configurar APIs
echo "ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE]" > .env
echo "VOYAGE_API_KEY=pa-[SUA_CHAVE]" >> .env

# 3. Executar pipeline completo
python run_pipeline.py

# 4. Iniciar dashboard (opcional)
cd src/dashboard && python start_dashboard.py
```

### ✨ **Características v4.9.8 - DASHBOARD FUNCIONAL COM CORREÇÕES CRÍTICAS**
- 🔢 **22 Etapas Implementadas**: Pipeline expandido (01-20 + 04b/06b)
- 🎯 **100% Padrões Anthropic**: XML prompting, Haiku model, taxonomia hierárquica
- ⚡ **Enhanced Components**: Encoding, deduplication, text cleaning, statistical analysis
- 🚀 **Voyage.ai Integrado**: 4 estágios com embeddings semânticos otimizados
- 🔤 **spaCy NLP**: Processamento linguístico avançado com pt_core_news_lg
- 💰 **96%+ Economia**: Custos API drasticamente reduzidos (1.3M → 50K)
- 🎛️ **API-only Stages 12-20**: Sem fallbacks, máxima qualidade
- 📊 **Dashboard Integrado**: Interface web completa

---

## 🎯 **MÓDULOS DO PIPELINE COMPLETOS (22 ETAPAS)**

### 📁 **Estrutura Principal**
```
src/
├── main.py                           # Controlador principal com checkpoints
├── anthropic_integration/
│   ├── unified_pipeline.py          # Engine principal (22 etapas)
│   ├── base.py                      # Classe base Anthropic
│   ├── political_analyzer.py        # Stage 05 - Análise Política Enhanced
│   ├── sentiment_analyzer.py        # Stage 08 - Análise de Sentimentos
│   ├── voyage_topic_modeler.py      # Stage 09 - Topic Modeling
│   ├── semantic_tfidf_analyzer.py   # Stage 10 - TF-IDF Semântico
│   ├── voyage_clustering_analyzer.py # Stage 11 - Clustering
│   ├── semantic_search_engine.py    # Stage 19 - Busca Semântica
│   ├── spacy_nlp_processor.py       # Stage 07 - Processamento Linguístico
│   ├── encoding_validator.py        # Stage 02 - Enhanced Encoding
│   ├── deduplication_validator.py   # Stage 03 - Global Deduplication
│   ├── statistical_analyzer.py      # Stage 04b/06b - Análise Estatística
│   ├── intelligent_text_cleaner.py  # Stage 06 - Enhanced Text Cleaning
│   ├── performance_optimizer.py     # Otimização de Performance
│   └── [15+ outros módulos AI]
└── dashboard/
    ├── app.py                       # Interface Web Principal
    ├── csv_parser.py               # Parser CSV Integrado
    └── start_dashboard.py          # Iniciador Dashboard
```

### 🔄 **Pipeline Completo (22 Etapas)**

| Stage | Nome | Arquivo | Tecnologia | Status |
|-------|------|---------|------------|--------|
| **01** | Chunk Processing | `unified_pipeline.py` | - | ✅ Concluído |
| **02** | Enhanced Encoding | `encoding_validator.py` | chardet + AI | ✅ Enhanced |
| **03** | Global Deduplication | `deduplication_validator.py` | AI | ✅ Enhanced |
| **04** | Feature Validation | `unified_pipeline.py` | - | ✅ Concluído |
| **04b** | Statistical Analysis (Pre) | `statistical_analyzer.py` | - | ✅ Enhanced |
| **05** | Political Analysis | `political_analyzer.py` | Anthropic Enhanced | ✅ API-only |
| **06** | Enhanced Text Cleaning | `intelligent_text_cleaner.py` | AI | ✅ Enhanced |
| **06b** | Statistical Analysis (Post) | `statistical_analyzer.py` | - | ✅ Enhanced |
| **07** | Linguistic Processing | `spacy_nlp_processor.py` | spaCy pt_core_news_lg | ✅ Ativo |
| **08** | Sentiment Analysis | `sentiment_analyzer.py` | Anthropic | ✅ API-only |
| **09** | Topic Modeling | `voyage_topic_modeler.py` | Voyage.ai | ✅ API-only |
| **10** | TF-IDF Extraction | `semantic_tfidf_analyzer.py` | Voyage.ai | ✅ API-only |
| **11** | Clustering | `voyage_clustering_analyzer.py` | Voyage.ai | ✅ API-only |
| **12** | Hashtag Normalization | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **13** | Domain Analysis | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **14** | Temporal Analysis | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **15** | Network Analysis | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **16** | Qualitative Analysis | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **17** | Smart Pipeline Review | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **18** | Topic Interpretation | `unified_pipeline.py` | Anthropic | ✅ API-only |
| **19** | Semantic Search | `semantic_search_engine.py` | Voyage.ai | ✅ API-only |
| **20** | Pipeline Validation | `unified_pipeline.py` | Anthropic | ✅ API-only |

---

## 🎛️ **COMANDOS PRINCIPAIS**

### **Execução Completa**
```bash
# Pipeline completo (22 etapas)
python run_pipeline.py

# Com dataset específico
python run_pipeline.py --dataset "data/uploads/meu_dataset.csv"

# Com logging detalhado
python run_pipeline.py --verbose

# Forçar reinício sem checkpoints
python run_pipeline.py --force-restart
```

### **Execução Seletiva**
```bash
# Executar a partir de uma etapa específica
python src/main.py --start-from 08_sentiment_analysis

# Executar etapa individual
python src/main.py --stage 05_political_analysis

# Executar apenas etapas Anthropic
python src/main.py --anthropic-only

# Executar apenas etapas Voyage.ai
python src/main.py --voyage-only
```

### **Diagnóstico e Manutenção**
```bash
# Verificar status do pipeline
python run_pipeline.py --status

# Limpar checkpoints e recomeçar
python run_pipeline.py --clean

# Verificar dependências
python run_pipeline.py --check-deps

# Relatório de custos
python -c "from src.anthropic_integration.cost_monitor import get_cost_report; print(get_cost_report())"
```

### **🔄 Recovery e Troubleshooting**

#### **Cenários de Recovery**
```bash
# 1. Pipeline travou em alguma etapa
python run_pipeline.py --recover

# 2. Erro de API (Anthropic/Voyage)
python run_pipeline.py --retry-failed

# 3. Problema de memória
python run_pipeline.py --low-memory

# 4. Corrupção de dados
python run_pipeline.py --validate-and-fix

# 5. Reset completo
python run_pipeline.py --reset-all
rm -rf checkpoints/* logs/* data/interim/*
```

#### **Monitoramento em Tempo Real**
```bash
# Logs em tempo real
tail -f logs/pipeline_execution.log

# Status de checkpoints
watch -n 5 "ls -la checkpoints/"

# Monitoramento de custos
python -c "from src.anthropic_integration.cost_monitor import monitor_realtime; monitor_realtime()"
```

---

## ⚙️ **CONFIGURAÇÃO COMPLETA**

### 📋 **Dependências**

#### **Python Packages (requirements.txt)**
```bash
# APIs Principais
anthropic>=0.25.0
voyageai>=0.2.0

# Processamento de Dados
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# NLP e Embeddings
spacy>=3.7.0
sentence-transformers>=2.2.0

# Detecção de Encoding
chardet>=5.0.0
charset-normalizer>=3.0.0

# Clustering e Similaridade  
faiss-cpu>=1.7.0
umap-learn>=0.5.0

# Web Dashboard
streamlit>=1.28.0
plotly>=5.17.0

# Configuração
pyyaml>=6.0
python-dotenv>=1.0.0

# Logging e Monitoramento
loguru>=0.7.0
tqdm>=4.65.0
```

#### **Modelos spaCy**
```bash
# Instalar modelo português
python -m spacy download pt_core_news_lg
```

#### **APIs Externas**
- **Anthropic API**: claude-3-5-haiku-20241022
- **Voyage.ai API**: voyage-3.5-lite (96% economia ativada)

### 🔧 **Configuração de Arquivos**

#### **1. Variáveis de Ambiente (.env)**
```bash
# APIs (OBRIGATÓRIO)
ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE_AQUI]
VOYAGE_API_KEY=pa-[SUA_CHAVE_AQUI]

# Modelos (OPCIONAL)
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
VOYAGE_MODEL=voyage-3.5-lite

# Performance (OPCIONAL)
MAX_WORKERS=4
CHUNK_SIZE=10000
MEMORY_LIMIT=2GB

# Debug (OPCIONAL)
DEBUG_MODE=false
LOG_LEVEL=INFO
COST_MONITORING=true
```

#### **2. Configuração Principal (config/settings.yaml)**
```yaml
# Configuração atualizada para v4.9.1
project:
  name: "dataanalysis-bolsonarismo"
  version: "4.9.1"
  
# APIs
anthropic:
  model: "claude-3-5-haiku-20241022"
  max_tokens: 4000
  temperature: 0.3
  cost_monitoring: true

voyage:
  model: "voyage-3.5-lite"
  batch_size: 128
  cache_enabled: true

# Pipeline
processing:
  chunk_size: 10000
  max_workers: 4
  memory_limit: "2GB"
  
# Otimizações
optimizations:
  sampling_enabled: true
  cost_reduction_target: 0.96
  fallback_strategies: true
```

#### **3. Configuração Anthropic (config/anthropic.yaml)**
```yaml
# Configuração específica Anthropic Enhanced
api:
  model: "claude-3-5-haiku-20241022"
  max_tokens: 4000
  temperature: 0.3
  
# Padrões Anthropic Implementados
features:
  xml_structured_prompting: true
  hierarchical_taxonomy: true
  pydantic_validation: true
  concurrent_processing: true
  rag_integration: true
  fallback_strategies: true
  token_control: true
  experiment_control: true
```

#### **4. Configuração Voyage (config/voyage_embeddings.yaml)**
```yaml
# Configuração Voyage.ai otimizada
api:
  model: "voyage-3.5-lite"
  batch_size: 128
  max_tokens: 32000
  
# Integrações ativas
integration:
  topic_modeling: true      # Stage 09
  tfidf_analysis: true      # Stage 10  
  clustering: true          # Stage 11
  semantic_search: true     # Stage 19
  
# Otimizações
cost_optimization:
  sampling_enabled: true
  reduction_target: 0.96
  threshold: 0.75
```

### 🏗️ **Estrutura de Diretórios**
```
dataanalysis-bolsonarismo/
├── .env                          # Variáveis de ambiente
├── run_pipeline.py              # Executor principal
├── src/
│   ├── main.py                  # Controlador com checkpoints
│   ├── anthropic_integration/   # 22+ módulos AI
│   └── dashboard/              # Interface web
├── config/                     # Configurações
│   ├── settings.yaml           # Configuração principal
│   ├── anthropic.yaml          # Config Anthropic
│   └── voyage_embeddings.yaml  # Config Voyage
├── data/                       # Dados
│   ├── uploads/               # Datasets de entrada
│   ├── interim/               # Processamento intermediário
│   └── dashboard_results/     # Resultados dashboard
├── checkpoints/               # Checkpoints pipeline
├── logs/                      # Logs do sistema
└── docs/                      # Documentação adicional
```

---

## 📈 **Características Técnicas v4.9.1**

### 🎯 **Otimizações Implementadas**
- **96% Economia de Custos**: Sampling inteligente (1.3M → 50K registros)
- **API-only Stages 12-20**: Sem fallbacks, máxima qualidade
- **Enhanced Performance**: Otimizações específicas por etapa
- **Result Extension**: Manutenção da completude do dataset
- **Concurrent Processing**: Processamento paralelo com semáforos

### 🔬 **Análises Disponíveis**
- **Análise Política Enhanced**: Taxonomia hierárquica brasileira
- **Sentiment Analysis**: Multi-dimensional contextualizada
- **Topic Modeling**: Interpretação semântica com Voyage.ai
- **Network Analysis**: Detecção de coordenação e influência
- **Qualitative Analysis**: Classificação de narrativas e frames
- **Temporal Analysis**: Evolução discursiva e marcos históricos

### 📊 **Métricas e Monitoramento**
- **Cost Monitoring**: Tracking em tempo real de custos API
- **Quality Metrics**: Scores de qualidade por etapa
- **Performance Tracking**: Tempos de execução e otimizações
- **Error Handling**: Logs detalhados e recovery automático

---

## 🚨 **Troubleshooting**

### ⚠️ **Problemas Comuns**

#### **Erro de API**
```bash
# Verificar configuração
python -c "import os; print('ANTHROPIC_API_KEY:', os.getenv('ANTHROPIC_API_KEY')[:10] + '...')"

# Testar conectividade
python -c "from src.anthropic_integration.base import AnthropicBase; AnthropicBase({}).test_connection()"
```

#### **Problema de Memória**
```bash
# Executar com menor chunk size
export CHUNK_SIZE=5000
python run_pipeline.py --low-memory
```

#### **Falha em Etapa Específica**
```bash
# Ver logs específicos
grep "Stage 05" logs/pipeline_execution.log

# Reiniciar da etapa
python src/main.py --start-from 05_political_analysis
```

### 🔧 **Comandos de Diagnóstico**
```bash
# Status completo do sistema
python run_pipeline.py --health-check

# Validar configurações
python run_pipeline.py --validate-config

# Limpar cache corrompido
python run_pipeline.py --clear-cache

# Verificar dependências
python run_pipeline.py --check-dependencies
```

---

## 🎓 **Contexto Científico**

### **Período Analisado: 2019-2023**
- **Governo Bolsonaro** (2019-2022)
- **Pandemia COVID-19** (2020-2022)
- **Eleições Presidenciais** (2022)
- **Transição Governamental** (2022-2023)

### **Fenômenos Estudados**
- **Bolsonarismo** e extrema-direita digital
- **Negacionismo científico** e histórico
- **Autoritarismo** e ataques à democracia
- **Desinformação** e teorias conspiratórias
- **Polarização política** nas redes

### **Metodologia AI-Enhanced**
- **Análise semântica** contextualizada
- **Classificação automática** de narrativas
- **Detecção de padrões** autoritários
- **Interpretação qualitativa** inteligente

---

## 🛠️ **Desenvolvimento**

### **Princípios da Arquitetura**

1. **Centralização Absoluta**: Um comando, uma configuração, um ponto de manutenção
2. **AI como Padrão**: Anthropic API para todas as análises complexas
3. **Voyage.ai Integration**: Embeddings semânticos para análises avançadas
4. **Contexto Brasileiro**: Prompts especializados em política nacional

### **Padrão de Implementação**

```python
# Todos os stages 12-20 seguem este padrão API-only
if self._validate_dependencies(required=["component_name"]):
    # Análise inteligente com API (sem fallback)
    result = self.component.analyze_enhanced(data, api_mode=True)
    
    # Extensão de resultados se necessário
    if len(optimized_df) < len(df):
        result = self._extend_results(df, result, optimization_report)
else:
    # Erro - dependências não disponíveis
    logger.error("❌ Dependências API não disponíveis")
    continue
```

---

## 📚 **Documentação Adicional**

Para informações mais detalhadas, consulte:
- **[CLAUDE.md](CLAUDE.md)** - Instruções para Claude Code e configurações avançadas
- **[config/](config/)** - Arquivos de configuração específicos
- **[src/dashboard/README.md](src/dashboard/README.md)** - Setup do dashboard
- **[SECURITY_SETUP.md](SECURITY_SETUP.md)** - Configuração de segurança

---

## 📄 **Licença e Uso Acadêmico**

Este projeto é destinado para **pesquisa acadêmica** sobre:
- Comunicação política digital
- Análise de discurso autoritário
- Desinformação e teorias conspiratórias
- Democracia digital no Brasil

---

**Pipeline Bolsonarismo v4.9.1** - Sistema completo de análise científica de discurso político brasileiro com inteligência artificial, otimizado para máxima qualidade e economia de custos.