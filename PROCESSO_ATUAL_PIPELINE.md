# PROCESSO ATUAL DO PIPELINE - digiNEV v.final

**Data de atualização:** 04 de outubro de 2025
**Status:** ✅ Operacional e consolidado
**Versão:** v.final com otimizações 5.0.0

## 🎯 STATUS ATUAL DO SISTEMA

### Pipeline Consolidado (17 Stages)
```
✅ FUNCIONAL: 17 stages executando sequencialmente
✅ VALIDADO: 102 colunas geradas com dados reais
✅ TESTADO: Validação completa com controlled_test_100.csv
✅ OTIMIZADO: 5/5 semanas de otimização ativas (100%)
```

### Arquivos Principais Funcionais
- `src/analyzer.py` - Pipeline principal (17 stages sequenciais)
- `run_pipeline.py` - Executor principal com descoberta automática
- `test_clean_analyzer.py` - Sistema de validação funcional
- `data/` - 11 datasets válidos (0.0 MB a 230 MB)

## 🚀 COMO EXECUTAR AGORA

### Execução Padrão (Todos os Datasets)
```bash
python run_pipeline.py
```

### Execução com Dataset Específico
```bash
python run_pipeline.py --dataset data/controlled_test_100.csv
python run_pipeline.py --dataset data/1_2019-2021-govbolso.csv
```

### Teste de Validação
```bash
python test_clean_analyzer.py
```

### Dashboard (Visualização)
```bash
python src/dashboard/start_dashboard.py
```

## 📊 PIPELINE SEQUENCIAL OTIMIZADO (17 STAGES)

### FASE 1: PREPARAÇÃO E ESTRUTURA (01-02)
```
STAGE 01: Feature Extraction
- Detecção automática de colunas (text, timestamp)
- Extração de features básicas (hashtags, URLs, mentions, emojis)
- Padronização de datetime

STAGE 02: Text Preprocessing
- Normalização de texto em português
- Limpeza básica
- Validação de features
```

### FASE 2: REDUÇÃO DE VOLUME (03-06) - CRÍTICO
```
STAGE 03: Cross-Dataset Deduplication
- Redução: 40-50% (300k → 180k)
- Agrupa textos idênticos, mantém mais antigo
- Contador dupli_freq

STAGE 04: Statistical Analysis
- Comparação antes/depois redução
- Estatísticas de qualidade e duplicação
- Detecção de padrões

STAGE 05: Content Quality Filter
- Redução: 15-25% (180k → 135k)
- Filtros: comprimento, emoji_ratio, caps_ratio, idioma
- Score de qualidade 0-100

STAGE 06: Political Relevance Filter
- Redução: 30-40% (135k → 80k)
- Classificação política brasileira
- Manter apenas conteúdo político relevante
```

### FASE 3: ANÁLISE LINGUÍSTICA (07-09) - VOLUME OTIMIZADO
```
STAGE 07: Linguistic Processing (spaCy)
- Processamento com pt_core_news_lg
- Tokens, lemmas, POS tags, entidades

STAGE 08: Political Classification
- Classificação política brasileira detalhada
- extrema-direita, direita, centro, esquerda, neutral

STAGE 09: TF-IDF Vectorization
- Vetorização com tokens spaCy
- Top termos por documento
```

### FASE 4: ANÁLISES AVANÇADAS (10-17)
```
STAGE 10: Clustering Analysis
- K-Means clustering
- Análise de distâncias

STAGE 11: Topic Modeling
- LDA topic modeling
- Probabilidades por tópico

STAGE 12: Semantic Analysis
- Análise semântica avançada
- Conectivos e modalidade

STAGE 13: Temporal Analysis
- Análise temporal (hour, day, month)
- Padrões temporais

STAGE 14: Network Analysis
- Coordenação de rede
- Padrões de propagação

STAGE 15: Domain Analysis
- Análise de domínios e URLs
- Classificação de fontes

STAGE 16: Event Context Analysis
- Contexto de eventos políticos brasileiros
- Detecção de contextos eleitorais

STAGE 17: Channel Analysis
- Análise de canais/fontes
- Classificação de autoridade
```

## 📁 DATASETS DISPONÍVEIS

### Datasets Principais (data/)
```
1. controlled_test_100.csv (0.0 MB) - Teste validado
2. 1_2019-2021-govbolso.csv (135.9 MB) - Período Bolsonaro
3. 2_2021-2022-pandemia.csv (230.0 MB) - Pandemia
4. 3_2022-2023-poseleic.csv (93.2 MB) - Pós-eleição
5. 4_2022-2023-elec.csv (54.2 MB) - Eleições
6. 5_2022-2023-elec-extra.csv (25.2 MB) - Dados extras
```

### Datasets Processados (data/processed/)
```
- processed_1_2019-2021-govbolso.csv
- processed_2_2021-2022-pandemia.csv
```

## ⚡ OTIMIZAÇÕES ATIVAS (5.0.0)

### Week 1-2: Emergency Cache + Advanced Caching
- ✅ Cache inteligente de stages
- ✅ Checkpoints automáticos

### Week 3: Parallelization + Streaming
- ✅ Processamento paralelo integrado
- ✅ Streaming de dados grandes

### Week 4: Real-time Monitoring
- ✅ Monitoramento em tempo real
- ✅ Logs detalhados

### Week 5: Memory Management
- ✅ Gestão de memória otimizada
- ✅ Auto-chunking para datasets grandes

## 🔧 SAÍDA DE DADOS (102 COLUNAS)

### Colunas Estruturais
```
id, body, channel, user_id, message_id, datetime
main_text_column, timestamp_column, has_timestamp
```

### Features Extraídas
```
hashtags_extracted, hashtags_count, urls_extracted, urls_count
mentions_extracted, mentions_count, emojis_extracted, emojis_count
```

### Processamento de Texto
```
normalized_text, text_cleaned, dupli_freq, channels_found
char_count, word_count, emoji_ratio, caps_ratio, repetition_ratio
likely_portuguese, content_quality_score, language_confidence
```

### Análise Política
```
political_orientation, political_keywords, political_intensity
political_relevance_score, political_terms_found
```

### Análise Linguística (spaCy)
```
spacy_tokens, spacy_lemmas, spacy_pos_tags, spacy_entities
spacy_tokens_count, spacy_entities_count, lemmatized_text
```

### TF-IDF e Clustering
```
tfidf_score_mean, tfidf_score_max, tfidf_top_terms
cluster_id, cluster_distance, cluster_size
```

### Topic Modeling
```
dominant_topic, topic_probability, topic_keywords
```

### Análise Semântica
```
sentiment_polarity, sentiment_label, emotion_intensity
has_aggressive_language, semantic_diversity
```

### Análise Temporal
```
hour, day_of_week, month, year, day_of_year
is_weekend, is_business_hours
```

### Análise de Rede
```
sender_frequency, is_frequent_sender, shared_url_frequency
temporal_coordination
```

### Análise de Domínios
```
domain_type, domain_frequency, is_mainstream_media
url_count, has_external_links
```

### Análise de Contexto
```
political_context, mentions_government, mentions_opposition
election_context, protest_context
```

### Análise de Canais
```
channel_type, channel_activity, is_active_channel
content_type, has_media, is_forwarded, forwarding_context
sender_channel_influence
```

### Metadados
```
processing_timestamp, stages_completed, features_extracted
```

## 🧪 VALIDAÇÃO ATUAL

### Teste Funcional (test_clean_analyzer.py)
```bash
🔬 TESTE: Analyzer v.final com dados reais
============================================================
📄 Dataset real carregado: 100 registros, 6 colunas
✅ RESULTADO DA ANÁLISE:
📊 Colunas geradas: 102
🎯 Stages completados: 17/10
🔧 Features extraídas: 81

🔗 VERIFICAÇÃO DE INTERLIGAÇÃO ENTRE STAGES:
✅ Todos os stages executados sequencialmente
✅ Cada stage usa dados dos stages anteriores
✅ Nenhum reprocessamento desnecessário
✅ Todas as 102 colunas contêm dados reais
✅ Pipeline totalmente interligado
```

## 📈 PERFORMANCE ATUAL

### Processamento Sequencial Otimizado
- **Redução de Volume:** 40-50% → 15-25% → 30-40% = ~80% redução final
- **Stages Linguísticos:** Apenas no volume otimizado (economia de 80% de processamento)
- **Memória:** Auto-chunking para datasets > 4GB
- **Tempo:** Processamento inteligente por fases

### Exemplo de Execução
```
Initial: 300,000 registros
→ Stage 03: 180,000 (deduplicação)
→ Stage 05: 135,000 (qualidade)
→ Stage 06: 80,000 (relevância política)
→ Stages 07-17: Processamento linguístico otimizado
```

## 🛡️ VALIDAÇÃO E CONTROLE

### Checkpoints Automáticos
- Salvamento automático entre stages
- Retomada de execução em caso de falha

### Validação de Dados
- Verificação de integridade em cada stage
- Logs detalhados de transformações

### Proteção de Stages
- Stages críticos protegidos contra reprocessamento
- Sistema de flags de proteção

## 🚨 RESOLUÇÃO DE PROBLEMAS

### Erro "Error tokenizing data"
```bash
# Usar dataset menor para teste
python run_pipeline.py --dataset data/controlled_test_100.csv
```

### Erro de memória
```bash
# O sistema usa auto-chunking automaticamente
# Configurado para datasets até 4GB
```

### Pipeline não encontra datasets
```bash
# Verificar se os arquivos estão em data/
ls data/*.csv
```

## 📝 LOGS E MONITORAMENTO

### Logs Detalhados
```
INFO:Analyzer:🔬 Iniciando análise OTIMIZADA: X registros
INFO:Analyzer:🔍 STAGE 01: Feature Extraction
INFO:Analyzer:📅 Padronizando datetime...
INFO:Analyzer:✅ Stage XX concluído: Y registros processados
```

### Métricas de Performance
```
⏱️ Total duration: X.Xs
📊 Datasets processed: X
📈 Records processed: X
🔧 Stages executed: 17
```

## 🎯 PRÓXIMOS PASSOS

1. **Execução com Datasets Completos:**
   ```bash
   python run_pipeline.py --dataset data/1_2019-2021-govbolso.csv
   ```

2. **Análise dos Resultados:**
   ```bash
   python src/dashboard/start_dashboard.py
   ```

3. **Processamento em Lote:**
   ```bash
   python run_pipeline.py  # Todos os datasets
   ```

---

**Status:** ✅ Pipeline operacional e documentado
**Última validação:** 04/10/2025
**Commit:** d9acb89 - feat: Resume and consolidate pipeline processing