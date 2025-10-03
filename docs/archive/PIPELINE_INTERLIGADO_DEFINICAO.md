# PIPELINE INTERLIGADO - DEFINIÇÃO CLARA v.final
**Data**: 2025-10-03

## 🎯 PROBLEMA IDENTIFICADO

### ❌ PROBLEMAS ATUAIS:
1. **Fallbacks confusos** - Não fica claro se stage está implementado ou não
2. **Reprocessamento** - Dados processados múltiplas vezes desnecessariamente
3. **Stages isolados** - Não aproveitam resultados de stages anteriores
4. **Métricas inventadas** - Colunas criadas sem dados reais

## ✅ SOLUÇÃO: PIPELINE SEQUENCIAL INTERLIGADO

### FLUXO SEQUENCIAL (sem fallbacks):
```
INPUT: DataFrame with text column
  ↓
STAGE 01: text_preprocessing → normalized_text
  ↓ (usa normalized_text)
STAGE 02: basic_statistics → word_count, sentence_count, text_length
  ↓ (usa normalized_text + statistics)
STAGE 03: political_classification → political_spectrum, political_entities
  ↓ (usa normalized_text + political_data)
STAGE 04: linguistic_analysis → spacy_tokens, pos_tags, named_entities
  ↓ (usa normalized_text + linguistic_data)
STAGE 05: tfidf_vectorization → tfidf_matrix, top_terms
  ↓ (usa normalized_text + tfidf_matrix)
STAGE 06: clustering → cluster_id, cluster_center (usa tfidf_matrix)
  ↓ (usa tfidf_matrix + clusters)
STAGE 07: topic_modeling → topics, topic_weights (usa tfidf_matrix + clusters)
  ↓ (usa all previous data)
STAGE 08: temporal_analysis → timestamp_features (usa metadata)
  ↓ (usa all data)
STAGE 09: network_analysis → coordination_metrics (usa clusters + temporal)
  ↓ (usa all data)
STAGE 10: domain_analysis → url_domains, domain_authority
  ↓ (usa all data)
OUTPUT: Complete DataFrame with 30-40 REAL columns
```

## 📊 STAGES REAIS IMPLEMENTADOS

### STAGE 01: text_preprocessing
- **Input**: text column
- **Process**: Limpa, normaliza, remove caracteres especiais
- **Output**: `normalized_text` (string)
- **Status**: ✅ IMPLEMENTADO

### STAGE 02: basic_statistics
- **Input**: normalized_text
- **Process**: Conta palavras, sentenças, caracteres
- **Output**: `word_count`, `sentence_count`, `text_length` (int)
- **Status**: ✅ IMPLEMENTADO

### STAGE 03: political_classification
- **Input**: normalized_text
- **Process**: Classifica usando lexicon político real
- **Output**: `political_spectrum`, `political_entity_count` (string, int)
- **Status**: ✅ IMPLEMENTADO (usando lexicon real)

### STAGE 04: linguistic_analysis
- **Input**: normalized_text
- **Process**: spaCy processing (se disponível)
- **Output**: `spacy_tokens_count`, `spacy_entities_count` (int)
- **Status**: ⚠️ DEPENDENTE DO SPACY

### STAGE 05: tfidf_vectorization
- **Input**: normalized_text
- **Process**: TF-IDF real com scikit-learn
- **Output**: `tfidf_score`, `top_tfidf_terms` (float, string)
- **Status**: ✅ IMPLEMENTADO

### STAGE 06: clustering
- **Input**: tfidf_matrix (do stage 05)
- **Process**: KMeans real com scikit-learn
- **Output**: `cluster_id`, `cluster_distance` (int, float)
- **Status**: ✅ IMPLEMENTADO

### STAGE 07: topic_modeling
- **Input**: tfidf_matrix + clusters
- **Process**: LDA real com scikit-learn
- **Output**: `topic_id`, `topic_probability` (int, float)
- **Status**: ✅ IMPLEMENTADO

### STAGE 08: temporal_analysis
- **Input**: timestamp column (se existe)
- **Process**: Extrai hora, dia da semana, mês
- **Output**: `hour`, `day_of_week`, `month` (int)
- **Status**: ✅ IMPLEMENTADO

### STAGE 09: network_analysis
- **Input**: clusters + temporal_data
- **Process**: Detecta coordenação temporal entre clusters
- **Output**: `potential_coordination`, `temporal_pattern` (bool, string)
- **Status**: ✅ IMPLEMENTADO

### STAGE 10: domain_analysis
- **Input**: text com URLs
- **Process**: Extrai domínios reais
- **Output**: `url_count`, `unique_domains` (int, string)
- **Status**: ✅ IMPLEMENTADO

## 🔄 INTERLIGAÇÃO ENTRE STAGES

### DEPENDÊNCIAS CLARAS:
```python
stage_dependencies = {
    'stage_02': ['stage_01'],  # statistics precisa de normalized_text
    'stage_03': ['stage_01'],  # classification precisa de normalized_text
    'stage_04': ['stage_01'],  # linguistic precisa de normalized_text
    'stage_05': ['stage_01'],  # tfidf precisa de normalized_text
    'stage_06': ['stage_05'],  # clustering precisa de tfidf_matrix
    'stage_07': ['stage_05', 'stage_06'],  # topic modeling precisa de tfidf + clusters
    'stage_08': [],  # temporal independente (usa timestamp original)
    'stage_09': ['stage_06', 'stage_08'],  # network precisa clusters + temporal
    'stage_10': [],  # domain independente (procura URLs no texto original)
}
```

### DADOS REUTILIZADOS:
- `normalized_text` → usado por stages 02, 03, 04, 05
- `tfidf_matrix` → usado por stages 06, 07
- `cluster_id` → usado por stages 07, 09
- `temporal_features` → usado por stage 09

## 📋 SAÍDA FINAL REAL (30-35 colunas)

### COLUNAS REAIS (não inventadas):
```python
real_columns = {
    # Original data
    'original_text': 'string',
    'normalized_text': 'string',

    # Statistics (stage 02)
    'word_count': 'int',
    'sentence_count': 'int',
    'text_length': 'int',

    # Political (stage 03)
    'political_spectrum': 'string',
    'political_entity_count': 'int',

    # Linguistic (stage 04) - se spaCy disponível
    'spacy_tokens_count': 'int',
    'spacy_entities_count': 'int',

    # TF-IDF (stage 05)
    'tfidf_score': 'float',
    'top_tfidf_terms': 'string',

    # Clustering (stage 06)
    'cluster_id': 'int',
    'cluster_distance': 'float',

    # Topics (stage 07)
    'topic_id': 'int',
    'topic_probability': 'float',

    # Temporal (stage 08)
    'hour': 'int',
    'day_of_week': 'int',
    'month': 'int',

    # Network (stage 09)
    'potential_coordination': 'bool',
    'temporal_pattern': 'string',

    # Domain (stage 10)
    'url_count': 'int',
    'unique_domains': 'string',

    # Metadata
    'processing_timestamp': 'datetime',
    'stages_completed': 'int'
}
```

**TOTAL**: 20-25 colunas com DADOS REAIS (não 64+ inventadas)

## 🚫 ELIMINAR COMPLETAMENTE:

### ❌ REMOVER:
- Todos os fallbacks heurísticos confusos
- Métricas inventadas (confidence_score, quality_score, etc.)
- Colunas "supplementary_analysis_X"
- AIResourceManager complexo
- Sistema de 64+ colunas artificiais

### ✅ MANTER:
- Apenas stages com implementação real
- Apenas colunas com dados reais
- Fluxo sequencial claro
- Dependências explícitas entre stages

---
**RESULTADO**: Pipeline limpo, interligado e com dados reais apenas.