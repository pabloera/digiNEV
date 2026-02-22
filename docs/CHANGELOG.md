# CHANGELOG

## [6.2.0] - 2026-02-22 — Full API Integration (6 Stages) + Topic Naming + Channel Classification

### New Features
- **Stage 11 API Integration**: Topic modeling híbrido (LDA + API)
  - LDA descobre clusters de tópicos heuristicamente (100% msgs)
  - API nomeia tópicos com rótulos descritivos em português (1 chamada)
  - API reclassifica mensagens com topic_confidence < 0.4
  - 2 novas colunas: `topic_label`, `topic_confidence`
  - Resultado: tópicos nomeados como "Notícias Políticas Lula" vs "topic_0_..."
- **Stage 16 API Integration**: Contexto de eventos híbrido
  - Heurística detecta political_context, frames Entman, menções (100%)
  - API detecta referências indiretas a eventos específicos brasileiros
  - 2 novas colunas: `event_confidence`, `specific_event`
  - Eventos: 8_de_janeiro, cpi_covid, eleicao_2022, stf_inquerito, impeachment
  - Resultado: 22 eventos específicos detectados em 500 msgs
- **Stage 17 API Integration**: Análise de canais híbrida
  - Classificação por nome do canal (keyword matching, 100%)
  - API classifica canais 'general' por amostra de conteúdo (5 msgs)
  - Categorias API: conspiracy, military, activism (além das existentes)
  - 2 novas colunas: `channel_confidence`, `channel_theme`
  - Resultado: 100% dos "general" reclassificados com tipo e tema

### Bug Fixes
- **`_parse_api_json_response`**: Agora aceita dict/list como input (não só str)
- **`import os`**: Adicionado nos Stages 11 e 17 para acesso ao `os.environ`

### Validation (6 Stages com API)
- **200 rows**: 17/17 stages, 0 erros, 126 colunas, 187s
- **500 rows**: 17/17 stages, 0 erros, 126 colunas, 384s
- Stage 08: neutral 9.7%, confidence mean=0.842
- Stage 11: 5 tópicos, "Notícias Políticas Lula" nomeado via API
- Stage 12: sarcasmo=17, anger=0.265
- Stage 16: 22 eventos (8_janeiro=18, stf=2, eleição=1, impeachment=1)
- Stage 17: 0 canais "general" restantes (100% reclassificados)

---

## [6.1.0] - 2026-02-22 — API Expansion (Stages 08, 12) + Generic Batch API

### New Features
- **Stage 08 API Integration**: Classificação política híbrida (heurística + API)
  - Heurística classifica 100% das mensagens via léxico unificado
  - API reclassifica mensagens "neutral" com political_confidence < 0.4
  - Nova coluna `political_confidence` (0.0-1.0)
  - Resultado: ~30-40% dos "neutral" reclassificados → orientação política real
- **Stage 12 API Integration**: Análise semântica/sentimento híbrida
  - Sentimento LIWC-PT + API para ambiguidade (sentiment_confidence < 0.5)
  - 5 novas colunas: `sentiment_confidence`, `emotion_anger`, `emotion_fear`, `emotion_hope`, `emotion_disgust`, `emotion_sarcasm`
  - Detecção de sarcasmo/ironia contextual via API
  - Resultado: ~20-30% dos "neutral" reclassificados + emoções granulares
- **Generic API Methods**: Métodos reutilizáveis para qualquer stage
  - `_api_classify_sync()`: Chamada síncrona com prompt caching
  - `_api_submit_batch()`: Submit para Batch API (50% desconto)
  - `_api_poll_batch()`: Polling com status logging
  - `_api_process_low_confidence()`: Orquestrador genérico (heurística → API)
  - `_api_batch_process()`: Processamento batch genérico
  - `_parse_api_json_response()`: Parser JSON multi-estratégia

### Bug Fixes
- **Batch API header**: Corrigido `prompt-caching-2024-07-31` → `message-batches-2024-09-24` no Stage 06

### Validation
- **200 rows test**: 17/17 stages, 0 erros, 120 colunas, 144s
- **500 rows test**: 17/17 stages, 0 erros, 120 colunas, 388s
- Stage 08: neutral 40% → 9.4% (API reclassificou ~75% dos neutral)
- Stage 12: sarcasmo detectado em 17 msgs, emoções granulares funcionais
- Fallback testado: sem API key → 100% heurística, pipeline não falha

### Performance (500 rows, 3 stages com API)
- Stage 06 (affordances): ~220 msgs API (73.8% low confidence)
- Stage 08 (político): ~120 msgs API (40% neutral → 9.4%)
- Stage 12 (sentimento): ~150 msgs API (50% low confidence)
- Tempo total: 388s (vs 3.4s sem API)
- Colunas: 113 → 120 (+7 novas)

## [6.0.0] - 2026-02-22 — Reestruturação Pipeline + Modularização

### Bug Fixes (TARETAs 1-5)
- **Stage 04**: `caps_ratio` e `emoji_ratio` agora calculados sobre `body` (texto cru), não `normalized_text` (que é lowercase sem emojis)
- **Stage 04**: Detecção de hashtags usa `hashtags_extracted` (Stage 01) em vez de regex sobre texto sem `#`
- **Stage 06**: URL detection via `urls_extracted` (Stage 01) em vez de regex sobre texto sem `://`
- **Stage 07**: spaCy recebe `body` (texto cru) em vez de `normalized_text` — NER, POS e sentence splitting restaurados
- **Stage 07**: Fallback linguístico gera `spacy_lemmas` para consistência downstream
- **Stages 09, 11, 12**: Corrigido `'tokens'` → `'lemmatized_text'`/`'spacy_tokens'` (nomes corretos)
- **Stage 10**: Corrigido `'text_length'` → `'char_count'` (feature existente)

### New Features (TARETAs 6-9)
- **TCW Integration** (Stage 08): 217 códigos TCW (3-dígito) integrados ao Stage 08 via token matching
  - 181 termos únicos, 10 categorias temáticas
  - Colunas: `tcw_codes`, `tcw_categories`, `tcw_agreement`
  - Matching via set() intersection sobre spacy_lemmas (O(1)/token)
- **Léxico expandido**: +2 macrotemas (`corrupcao_transparencia`, `politica_externa`) no `lexico_unified_system.json`
- **Keywords expandido**: +2 categorias (`cat11_corrupcao`, `cat12_politica_externa`) no `political_keywords_dict.py`
- **Token matching reformulado** (Stage 08): `_classify_political_orientation`, `_extract_political_keywords`, `_calculate_political_intensity` usam set() lookup quando spaCy lemmas disponíveis

### Modularização (TAREFA 11)
- **19 arquivos** criados em `src/stages/`: 17 stage modules + `helpers.py` + `__init__.py`
- **STAGE_REGISTRY**: lista ordenada de (número, nome, função) para orquestração
- **21 helper functions** extraídas para `stages/helpers.py`
- **3327 linhas** de código modularizado (1:1 com métodos inline em analyzer.py)
- `analyzer.py` permanece como **source of truth** (versão autoritativa)

### Validation
- **4 testes ponta-a-ponta**: 100, 500, 1000, 2000 rows
- **3 datasets**: 4_elec, 2_pandemia, 1_govbolso (períodos 2019, 2021, 2022-23)
- **Resultados**: 17/17 stages, 0 erros, 113 colunas, 102 features em todos os testes
- **TCW coverage**: 22-46% dos registros classificados (varia por dataset)
- **Categorias políticas ativas**: 9-12/12 categorias (varia por período)

### Performance
- 100 rows: 0.7s | 500 rows: 3.4s | 1000 rows: 7.6s | 2000 rows: 6.1s
- Fallbacks funcionais: sem ANTHROPIC_API_KEY → heurística; sem hdbscan → KMeans

## [5.0.1] - 2025-09-30

### Fixed
- Created missing `api_error_handler.py` module that was breaking entire pipeline
- Fixed logger initialization in `unified_pipeline.py` (undefined logger in exception handlers)
- Fixed CSV separator issue in data loading (confirmed comma separator is correct)
- Fixed import issues in `stage_validator.py` (missing Path import)
- Fixed checkpoint recovery with gzip compression support

### Added
- Implemented `stage_validator.py` for inter-stage validation with memory management
- Implemented `fallback_config.py` for robust multi-source configuration
- Created `test_integration_complete.py` for comprehensive integration testing
- Generated analysis report with political classification and sentiment analysis
- Added memory optimization with 95.5% reduction capability

### Improved
- Pipeline now 100% functional (was 0% before fixes)
- All 23 stages executing successfully
- Dashboard integration working on port 8501
- Memory management integrated throughout pipeline
- Error recovery system with checkpoints

### Performance
- Pipeline execution: 20.4s for 100 records
- Processing rate: 16.6 texts/second for sentiment analysis
- Memory usage: ~668 MB (optimized from initial load)
- Cache hit rate: 50% for sentiment analysis
- API cost: $0.0004 (within academic budget)

## [5.0.0] - 2025-09-28

### Initial Release
- 22-stage pipeline for Brazilian political discourse analysis
- Integration with Claude 3.5 Haiku and Voyage.ai
- Portuguese-optimized NLP with spaCy
- Academic research focus with budget constraints
- Dashboard visualization with Streamlit