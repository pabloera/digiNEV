# 🌍 INTERNATIONALIZATION RULES - DIGITAL DISCOURSE MONITOR v5.0.0

## 📋 CONVERSION RULES AND TERMINOLOGY MAPPING

### 🎯 **PROJECT TRANSFORMATION STRATEGY**

**Goal**: Convert the entire project from Portuguese to English for international collaboration, following software engineering best practices.

**Scope**: Complete codebase internationalization including:
- Code structure (functions, classes, variables)
- Documentation (README, technical docs)
- Configuration files
- Comments and docstrings
- Log messages and outputs
- Directory structure (where applicable)

---

## 🔄 **TERMINOLOGY MAPPING TABLE**

### **🏷️ Core Project Terms:**

| Portuguese | English | Context |
|------------|---------|---------|
| Monitor do Discurso Digital | Digital Discourse Monitor | Project name |
| análise de discurso | discourse analysis | Core functionality |
| discurso político | political discourse | Domain focus |
| polarização | polarization | Analysis concept |
| negacionismo | denialism | Research topic |
| autoritarismo | authoritarianism | Analysis focus |

### **🔧 Technical Infrastructure:**

| Portuguese | English | Usage |
|------------|---------|--------|
| pipeline | pipeline | Keep (international term) |
| estágio/etapa | stage | Pipeline steps |
| processamento | processing | Data operations |
| validação | validation | Quality control |
| otimização | optimization | Performance |
| monitoramento | monitoring | System oversight |
| configuração | configuration | Settings |
| relatório | report | Output documents |

### **📊 Data Analysis Terms:**

| Portuguese | English | Category |
|------------|---------|----------|
| análise estatística | statistical_analysis | Function naming |
| limpeza de texto | text_cleaning | Processing step |
| extração de características | feature_extraction | ML operation |
| classificação política | political_classification | Analysis type |
| análise de sentimento | sentiment_analysis | NLP task |
| modelagem de tópicos | topic_modeling | ML technique |
| agrupamento | clustering | Algorithm |
| busca semântica | semantic_search | Search functionality |

### **🏗️ System Components:**

| Portuguese | English | Module Type |
|------------|---------|-------------|
| carregador de configuração | config_loader | System module |
| processador de dados | data_processor | Processing module |
| analisador | analyzer | Analysis component |
| validador | validator | Validation component |
| otimizador | optimizer | Performance module |
| gerenciador de memória | memory_manager | System utility |
| cache unificado | unified_cache | Storage system |

---

## 📝 **NAMING CONVENTIONS**

### **🐍 Python Code Standards:**

1. **Functions and Variables**: `snake_case`
   ```python
   # BEFORE (Portuguese)
   def analisar_sentimento(texto):
       resultado_analise = processar_texto(texto)
   
   # AFTER (English)
   def analyze_sentiment(text):
       analysis_result = process_text(text)
   ```

2. **Classes**: `PascalCase`
   ```python
   # BEFORE
   class AnalisadorSentimento:
   
   # AFTER  
   class SentimentAnalyzer:
   ```

3. **Constants**: `UPPER_SNAKE_CASE`
   ```python
   # BEFORE
   LIMITE_MAXIMO = 1000
   
   # AFTER
   MAX_LIMIT = 1000
   ```

4. **Module Names**: `snake_case`
   ```python
   # BEFORE
   analise_estatistica.py
   
   # AFTER
   statistical_analysis.py
   ```

### **📁 File and Directory Naming:**

| Portuguese Path | English Path | Rationale |
|----------------|--------------|-----------|
| `src/anthropic_integration/` | `src/anthropic_integration/` | Keep (proper nouns) |
| `src/utils/` | `src/utils/` | Keep (standard) |
| `configuracao/` | `config/` | Standard abbreviation |
| `documentacao/` | `docs/` | Standard abbreviation |
| `dados/` | `data/` | Standard term |

---

## 🎯 **FUNCTION MAPPING STRATEGY**

### **Priority 1 - Core Pipeline Functions:**

| Original Function | New Function | Module |
|------------------|--------------|---------|
| `processamento_chunk()` | `process_chunks()` | data_processor |
| `validacao_encoding()` | `validate_encoding()` | encoding_validator |
| `deduplicacao()` | `deduplicate_data()` | deduplication_processor |
| `analise_politica()` | `analyze_political_content()` | political_analyzer |
| `limpeza_texto()` | `clean_text()` | text_cleaner |
| `processamento_linguistico()` | `process_linguistics()` | linguistic_processor |
| `analise_sentimento()` | `analyze_sentiment()` | sentiment_analyzer |
| `modelagem_topicos()` | `model_topics()` | topic_modeler |
| `extracao_tfidf()` | `extract_tfidf()` | tfidf_extractor |
| `agrupamento()` | `perform_clustering()` | clustering_analyzer |

### **Priority 2 - Configuration and Utils:**

| Original Function | New Function | Module |
|------------------|--------------|---------|
| `carregar_configuracao()` | `load_configuration()` | config_loader |
| `validar_qualidade()` | `validate_quality()` | quality_validator |
| `otimizar_memoria()` | `optimize_memory()` | memory_optimizer |
| `gerenciar_cache()` | `manage_cache()` | cache_manager |
| `monitorar_performance()` | `monitor_performance()` | performance_monitor |

---

## 📋 **VARIABLE NAMING STANDARDS**

### **Common Patterns:**

| Portuguese Pattern | English Pattern | Example |
|-------------------|-----------------|---------|
| `dados_*` | `data_*` | `dados_processados` → `processed_data` |
| `resultado_*` | `result_*` | `resultado_analise` → `analysis_result` |
| `configuracao_*` | `config_*` | `configuracao_api` → `api_config` |
| `relatorio_*` | `report_*` | `relatorio_final` → `final_report` |
| `estatisticas_*` | `stats_*` | `estatisticas_texto` → `text_stats` |

### **Domain-Specific Terms:**

| Portuguese Variable | English Variable | Context |
|-------------------|------------------|---------|
| `mensagens_telegram` | `telegram_messages` | Data source |
| `conteudo_politico` | `political_content` | Classified data |
| `sentimento_analise` | `sentiment_analysis` | Analysis result |
| `topicos_identificados` | `identified_topics` | ML output |
| `grupos_semanticos` | `semantic_clusters` | Clustering result |

---

## 🔧 **CONFIGURATION FILES STRATEGY**

### **YAML Structure Conversion:**

```yaml
# BEFORE (Portuguese)
projeto:
  nome: monitor-discurso-digital
  descricao: "Monitor do Discurso Digital"
  
processamento:
  tamanho_chunk: 10000
  limite_memoria: "4GB"
  
# AFTER (English)
project:
  name: digital-discourse-monitor
  description: "Digital Discourse Monitor"
  
processing:
  chunk_size: 10000
  memory_limit: "4GB"
```

---

## 📚 **DOCUMENTATION TRANSLATION GUIDELINES**

### **1. README.md Structure:**
- Project title in English
- Description emphasizing international scope
- Installation instructions in English
- Usage examples with English variable names
- Contribution guidelines for international collaborators

### **2. Technical Documentation:**
- All docstrings in English
- Comments explaining complex Brazilian political context
- API documentation in English
- Architecture diagrams with English labels

### **3. Code Comments:**
- Brief comments in English
- Detailed context about Brazilian politics where necessary
- Links to Portuguese research papers with English summaries

---

## 🚀 **IMPLEMENTATION PHASES**

### **Phase 1: Core Infrastructure (High Priority)**
- [ ] Rename core functions and classes
- [ ] Update main pipeline file
- [ ] Convert configuration files
- [ ] Update import statements

### **Phase 2: Analysis Modules (High Priority)**
- [ ] Convert analysis functions
- [ ] Update class hierarchies
- [ ] Translate module docstrings
- [ ] Update variable names

### **Phase 3: Documentation (Medium Priority)**
- [ ] Translate README.md
- [ ] Convert technical documentation
- [ ] Update code comments
- [ ] Create English API documentation

### **Phase 4: Validation (Low Priority)**
- [ ] Test all functionality
- [ ] Validate import chains
- [ ] Check configuration loading
- [ ] Generate final report

---

## ✅ **QUALITY STANDARDS**

### **Code Quality Checks:**
1. All function names follow `snake_case`
2. All class names follow `PascalCase`
3. All docstrings are in English
4. No mixed language in single files
5. Configuration keys are in English
6. Log messages are in English (with Portuguese context where needed)

### **Documentation Quality:**
1. Clear English technical writing
2. Proper explanation of Brazilian political context
3. International collaboration guidelines
4. Comprehensive API documentation
5. Installation instructions for global audience

---

## 📊 **SUCCESS METRICS**

- [ ] 100% of function names converted to English
- [ ] 100% of class names converted to English  
- [ ] 100% of configuration keys converted to English
- [ ] 100% of main documentation in English
- [ ] 0 mixed-language files
- [ ] All imports working correctly
- [ ] All tests passing
- [ ] Pipeline executing successfully

---

**Author:** Pablo Emanuel Romero Almada, Ph.D.  
**Date:** 2025-06-14  
**Version:** 5.0.0  
**Purpose:** International collaboration and open-source readiness