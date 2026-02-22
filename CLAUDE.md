# digiNEV v.final - Brazilian Political Discourse Analysis

## 🎯 CONTEXTO
**Tipo**: Pesquisa Acadêmica em Ciências Sociais
**Foco**: Análise sociológica de discurso político brasileiro
**Dataset**: Mensagens Telegram (2019-2023)
**Specs**: 4GB RAM | Portuguese-optimized | 17 stages científicos otimizados | Consolidado

## 🏗️ Sistema Científico Consolidado v.final

### Pipeline Científico Otimizado (17 estágios) - IMPLEMENTADO E VALIDADO
**FASE 1: PREPARAÇÃO (01-02)**
1. **Feature Extraction (01)**: Detecção automática de colunas e features
2. **Text Preprocessing (02)**: Limpeza básica de texto em português

**FASE 2: REDUÇÃO DE VOLUME (03-06) - CRÍTICO PARA PERFORMANCE**
3. **Cross-Dataset Deduplication (03)**: Eliminação de duplicatas (redução 40-50%)
4. **Statistical Analysis (04)**: Análise estatística comparativa
5. **Content Quality Filter (05)**: Filtro de qualidade (redução 15-25%)
6. **Political Relevance Filter (06)**: Filtro de relevância política (redução 30-40%)

**FASE 3: ANÁLISE LINGUÍSTICA (07-09) - VOLUME OTIMIZADO**
7. **Linguistic Processing (07)**: Processamento linguístico avançado com spaCy
8. **Political Classification (08)**: Classificação política brasileira
9. **TF-IDF Vectorization (09)**: TF-IDF com tokens spaCy

**FASE 4: ANÁLISES AVANÇADAS (10-17)**
10. **Clustering Analysis (10)**: Clustering baseado em features linguísticas
11. **Topic Modeling (11)**: Topic modeling com embeddings
12. **Semantic Analysis (12)**: Análise semântica avançada
13. **Temporal Analysis (13)**: Análise temporal
14. **Network Analysis (14)**: Coordenação e padrões de rede
15. **Domain Analysis (15)**: Análise de domínios e URLs
16. **Event Context (16)**: Detecção de contextos políticos
17. **Channel Analysis (17)**: Classificação de canais/fontes

**Stack**: Python | scikit-learn | spaCy pt_core_news_sm | pandas | numpy | Anthropic Claude API

### API Integration (v6.2) — Fev 2026
- **6 stages com API híbrida**: Stage 06, 08, 11, 12, 16, 17
- **Padrão**: Heurística 100% → confidence score → API para baixa confiança → merge
- **Modelo**: `claude-sonnet-4-20250514` (Sonnet 4)
- **Batch API**: Suportada (50% desconto), ativável via `USE_BATCH_API=true` no `.env`
- **Prompt Caching**: Ativo (90% desconto no input repetido)
- **Fallback**: Sem API key → 100% heurística (pipeline NUNCA falha)

| Stage | Função | Threshold | Colunas novas |
|-------|--------|-----------|---------------|
| S06 | Affordances | confidence < 0.6 | (reclassifica categorias existentes) |
| S08 | Classificação política | confidence < 0.4 | `political_confidence` |
| S11 | Topic modeling (LDA + API) | confidence < 0.4 | `topic_label`, `topic_confidence` |
| S12 | Sentimento/emoções | confidence < 0.5 | `sentiment_confidence`, `emotion_anger/fear/hope/disgust`, `emotion_sarcasm` |
| S16 | Contexto de eventos | confidence < 0.5 | `event_confidence`, `specific_event` |
| S17 | Classificação de canais | tipo = 'general' | `channel_confidence`, `channel_theme` |

- **Resultados reais (500 rows)**: S08 neutral 40%→9.4% | S11 tópicos nomeados ("Notícias Políticas Lula") | S12 +sarcasmo/emoções | S16 22 eventos detectados (8_janeiro=18) | S17 100% "general" reclassificados
- **Métodos genéricos**: `_api_classify_sync()`, `_api_submit_batch()`, `_api_poll_batch()`, `_api_process_low_confidence()`, `_parse_api_json_response()`

### Modularização (TAREFA 11) — Fev 2026
- Cada stage extraído como módulo independente em `src/stages/stage_XX.py`
- Registry de stages: `from stages import STAGE_REGISTRY`
- Helpers compartilhados: `from stages.helpers import _calculate_emoji_ratio, ...`
- `src/analyzer.py` = **source of truth** (versão autoritativa inline)
- `src/stages/` = versão modular de referência, 1:1 com os métodos inline
- 19 arquivos: 17 stages + helpers.py + __init__.py (3327 linhas total)

### Reestruturação do Pipeline (TARETAs 1-10) — Fev 2026
- **8 bugs corrigidos**: spaCy input, caps/emoji/hashtag sobre body, token names, URL detection
- **TCW integrado** no Stage 08 (217 códigos, 10 categorias, 181 termos)
- **Léxico expandido**: +2 macrotemas (corrupção, política externa) no lexico_unified_system.json
- **Keywords expandido**: +2 categorias (cat11_corrupcao, cat12_politica_externa)
- **Token matching** via set() lookup com spaCy lemmas (O(1) por token)

## 🚀 Execução

### Uso Programático
```python
from src.analyzer import Analyzer

analyzer = Analyzer()
output = analyzer.analyze(df)  # Retorna dict
result_df = output['data']     # DataFrame com 126 colunas (113 base + 13 API)
print(f"Stages: {output['stages_completed']}/17")
print(f"Colunas: {output['columns_generated']}")
```

### Teste Rápido com Dados Reais
```python
import pandas as pd
from src.analyzer import Analyzer

df = pd.read_csv('path/to/dataset.csv', nrows=500, sep=',',
                  quotechar='"', quoting=1, on_bad_lines='skip')
analyzer = Analyzer()
output = analyzer.analyze(df)
print(f"Rows: {len(df)} → {output['total_records']} (pós-filtro)")
```

## 🔧 Características Principais

### Classificação Política (Stage 05)
- **Categorias**: extrema-direita, direita, centro, esquerda, neutral, unknown
- **Léxico político brasileiro** integrado
- **Classificação baseada** em análise de conteúdo real

### Recursos Implementados
- **spaCy**: Processamento linguístico em português (pt_core_news_lg)
- **scikit-learn**: TF-IDF, K-Means clustering, LDA topic modeling
- **Python puro**: Análise estatística, temporal e de redes
- **Regex otimizado**: Extração de features em português brasileiro

## 📁 Estrutura

```
├── src/
│   ├── analyzer.py              # Pipeline principal (17 stages inline) — SOURCE OF TRUTH
│   ├── lexicon_loader.py        # Carregador de léxico político
│   ├── core/                    # Recursos de classificação
│   │   ├── lexico_unified_system.json  # Léxico unificado (12 macrotemas)
│   │   ├── political_keywords_dict.py  # Keywords políticas (12 categorias)
│   │   ├── tcw_codes.json              # TCW: 217 códigos, 181 termos
│   │   └── tcw_categories.json         # TCW: 10 categorias temáticas
│   ├── stages/                  # Módulos extraídos (TAREFA 11)
│   │   ├── __init__.py          # STAGE_REGISTRY + imports
│   │   ├── helpers.py           # 21 funções utilitárias compartilhadas
│   │   ├── stage_01.py          # Feature Extraction
│   │   ├── stage_02.py          # Text Preprocessing
│   │   ├── ...                  # Stages 03-17
│   │   └── stage_17.py          # Channel Analysis
│   └── dashboard/               # Dashboard acadêmico
├── config/                      # Configuração unificada
│   └── settings.yaml            # Configurações principais
├── data/                        # Datasets de pesquisa
└── run_pipeline.py              # Script principal de execução
```

### Regras Estruturais
- TODO código científico em `/src`
- Configuração distribuída em `/config`
- NUNCA criar `.fixed`, `.new`, `.updated`
- SEMPRE editar arquivos in-place

## 🔬 Aplicações de Pesquisa
- Evolução da polarização política (2019-2023)
- Padrões de legitimação da violência
- Marcadores do discurso autoritário
- Análise de coordenação em rede
- Indicadores de erosão democrática

## 📊 Saída de Dados
- **126 colunas** geradas pelo pipeline sequencial de 17 stages (113 base + 13 API)
- Classificação política (extrema-direita, direita, centro-direita, neutral) + `political_confidence`
- Análise estatística (word_count, char_count, sentence_count, caps_ratio, emoji_ratio)
- Features extraídas (hashtags, URLs, mentions, emojis — sobre body cru)
- Deduplicação cross-dataset com contador de frequência
- Filtros de qualidade com scores 0-100
- Affordances (8 categorias: ataque, interação, mídia_social, mobilização, etc.)
- spaCy: tokens, lemmas, entities, lemmatized_text (sobre body cru)
- Classificação política com token matching via set() sobre spacy_lemmas
- TCW: tcw_codes (3-digit), tcw_categories (10 cat.), tcw_agreement (1-3)
- TF-IDF com scores e top termos (sobre lemmatized_text)
- Clustering K-Means com distâncias calculadas
- Topic modeling LDA com `topic_label` (nomeado via API) + `topic_confidence`
- Sentimento: `sentiment_label` + emoções granulares (anger, fear, hope, disgust, sarcasm)
- Eventos: `specific_event` (8_janeiro, stf_inquerito, eleicao_2022, etc.) + `event_confidence`
- Canais: `channel_type` (conspiracy, military, activism, etc.) + `channel_confidence`, `channel_theme`
- Análise temporal, network, domínios

### Resultados de Validação — v6.0 (heurística pura, Fev 2026)

| Teste | Dataset | Rows in→out | Stages | Errors | Colunas | Tempo |
|-------|---------|-------------|--------|--------|---------|-------|
| 1 | 4_elec (100) | 100→67 | 17/17 | 0 | 113 | 0.7s |
| 2 | 4_elec (500) | 500→298 | 17/17 | 0 | 113 | 3.4s |
| 3 | 2_pandemia (1000) | 1000→705 | 17/17 | 0 | 113 | 7.6s |
| 4 | 1_govbolso (2000) | 2000→717 | 17/17 | 0 | 113 | 6.1s |

### Resultados de Validação — v6.2 (6 API stages, Fev 2026)

| Teste | Dataset | Rows in→out | Stages | Errors | Colunas | Tempo |
|-------|---------|-------------|--------|--------|---------|-------|
| E2E quick | 4_elec (100) | 100→67 | 17/17 | 0 | 126 | ~90s |
| E2E standard | 4_elec (200) | 200→120 | 17/17 | 0 | 126 | ~187s |
| E2E full | 4_elec (500) | 500→298 | 17/17 | 0 | 126 | ~384s |

## 🧪 Testes
```bash
# Teste ponta-a-ponta v6.2 (com API)
python test_e2e_pipeline.py --quick      # 100 rows, ~90s
python test_e2e_pipeline.py              # 200 rows, ~3min (default)
python test_e2e_pipeline.py --full       # 500 rows, ~6min
python test_e2e_pipeline.py --no-api     # 100% heurístico

# Execução com dados reais
python run_pipeline.py
```

## 💡 Diretrizes de Desenvolvimento

### Princípios Fundamentais
1. **TESTAR SEMPRE** - Cada mudança testada imediatamente
2. **DADOS REAIS** - Usar datasets reais, não sintéticos
3. **REFATORAR INCREMENTALMENTE** - Pequenas mudanças validadas
4. **FALHAR RÁPIDO** - Detectar problemas cedo
5. **MEDIR IMPACTO** - Comparar performance antes/depois

### Workflow Obrigatório
```python
# Baseline → Mudança → Teste → Validação → Commit
df_original = pd.read_csv('data/controlled_test_100.csv', sep=';')
baseline_results = pipeline.process_dataset(df_original.copy())
# ... código modificado ...
new_results = pipeline.process_dataset(df_test.copy())
assert len(new_results) == len(baseline_results)
```

## 🔧 Políticas de Implementação

### Refatoração de Módulos
```python
# 1. Branch → 2. Módulo isolado → 3. Teste → 4. Integração → 5. Consolidação
refactoring_checklist = {
    'political_analyzer.py': {'tested': False, 'integrated': False},
    'sentiment_analyzer.py': {'tested': False, 'integrated': False}
}
# Consolidar APENAS se todos passaram
```

### Debugging e Validação
```python
# Logging detalhado
logging.debug(f"Input: {df.shape}, Columns: {df.columns.tolist()}")

# Tratamento de erros com contexto
try:
    result = complex_operation(data)
except Exception as e:
    logging.error(f"Erro: {e}, Context: {data.shape}")
    raise

# Validação pragmática
def validate_dataframe(df, stage_name):
    validations = {'not_empty': len(df) > 0, 'has_text': 'text' in df.columns}
    failed = [k for k, v in validations.items() if not v]
    if failed: logging.warning(f"Stage {stage_name}: {failed}")
    return df
```

### Otimização e Cache
```python
# Medir antes de otimizar
@measure_performance
def expensive_function(): pass

# Cache inteligente
@lru_cache(maxsize=10)
def cached_operation(cache_key): pass

# Monitoramento de memória
def check_memory(expected_gb=2.0):
    mem_gb = psutil.Process().memory_info().rss / 1024**3
    if mem_gb > expected_gb: gc.collect()
```

## 📝 Controle de Mudanças

### Estrutura do Changelog
```markdown
## [2025-09-30] - Sprint Atual
### ✅ Adicionado: Pipeline 22 estágios, validação dados reais
### 🔄 Modificado: political_analyzer.py otimização (linha 45-67)
### 🐛 Corrigido: Bug memória stage_15, encoding UTF-8
### 📊 Métricas: Tempo 45s→31s, Memória 2.1GB→1.4GB
```

### Automação
```python
class ChangelogManager:
    def add_change(self, type, description, details=None):
        # Buffer automático com timestamp
    def commit_to_changelog(self, version=None):
        # Consolidação por tipo: added/changed/fixed/removed
```

## 🎭 Orquestração de Tarefas

### Padrão de Orquestração
```python
@dataclass
class Task:
    name: str
    function: Callable
    dependencies: List[str] = None
    retry_count: int = 3
    timeout: float = 300
    critical: bool = True

class PragmaticOrchestrator:
    def add_task(self, task: Task): # Registrar tarefa
    async def run_task(self, task_name: str): # Executar com retry/timeout
    async def orchestrate(self): # Executar respeitando dependências
```

### Exemplo de Uso
```python
orchestrator = PragmaticOrchestrator()
orchestrator.add_task(Task("load_data", lambda: pd.read_csv(...), critical=True))
orchestrator.add_task(Task("validate", validate_func, dependencies=["load_data"]))
results = await orchestrator.orchestrate()
```

### Monitoramento
```python
class OrchestratorMonitor:
    def print_status(self): # Status visual das tarefas
    def get_metrics(self): # Métricas de sucesso/falha
```

## 🔄 Regras de Desenvolvimento

### Política de Atualizações
**ANTES** de modificar: LISTAR → PRESERVAR → COMENTAR → TESTAR

### Edição de Código
```python
# ❌ NUNCA: Deletar arquivo inteiro, reescrever do zero
# ✅ SEMPRE: Identificar trecho exato, mostrar "linhas X-Y", verificar impactos
```

### Verificação de Integração
- [ ] Função alterada: onde é chamada?
- [ ] Import modificado: quais arquivos importam?
- [ ] Output alterado: verificar pipelines dependentes

### Implementação
```python
# Dados reais obrigatórios
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dados reais necessários: {data_path}")

# Guardrails sempre
assert data is not None and len(data) > 0
assert required_columns.issubset(data.columns)
```

### Continuidade de Pipeline
```python
# Pipeline atual:
# [✓] Etapa 1: Coleta → [✓] Etapa 2: Limpeza → [►] Etapa 3: ALTERANDO
```

## ⚠️ Checklist Crítico
- [ ] Arquivo em `/src`?
- [ ] Nome preservado?
- [ ] Código comentado?
- [ ] CHANGELOG atualizado?
- [ ] Linguagem acadêmica?

## 🚫 Proibições
- ❌ Inventar funções sem verificar
- ❌ Criar fora de `/src`
- ❌ Usar linguagem comercial
- ❌ Criar `.fixed`/`.new`
- ❌ Deletar sem preservar

## 📊 Dados e Arquivos

### Datasets de Pesquisa
- `data/1_2019-2021-govbolso.csv` (135.9 MB) - Período Bolsonaro
- `data/2_2021-2022-pandemia.csv` (230.0 MB) - Pandemia
- `data/3_2022-2023-poseleic.csv` (93.2 MB) - Pós-eleição
- `data/4_2022-2023-elec.csv` (54.2 MB) - Eleições
- `data/5_2022-2023-elec-extra.csv` (25.2 MB) - Dados extras
- `data/controlled_test_100.csv` (0.0 MB) - Teste validado

### Arquivos Críticos
**Sistema Principal:**
- `/src/analyzer.py` - Pipeline consolidado 17 estágios otimizados (SOURCE OF TRUTH)
- `/run_pipeline.py` - Script de execução principal
- `/test_e2e_pipeline.py` - Teste ponta-a-ponta v6.2 (6 API stages, 4 modos)

**Dashboard:**
- `/src/dashboard/data_analysis_dashboard.py` - Dashboard principal
- `/src/dashboard/start_dashboard.py` - Iniciador do dashboard

## 📝 Atualizações Recentes

### Fev 2026 — API Integration v6.2 (6 Stages)
- ✅ **6 stages com API híbrida**: S06 (affordances), S08 (político), S11 (topic naming), S12 (sentimento), S16 (eventos), S17 (canais)
- ✅ **126 colunas** output (113 base + 13 API)
- ✅ **Teste E2E** completo: `test_e2e_pipeline.py` (quick/standard/full/stress)
- ✅ **Validação**: 17/17 stages, 0 erros em 100/200/500 rows
- ✅ **Resultados API**: neutral 9.4%, tópicos nomeados, 22 eventos, canais reclassificados

### Fev 2026 — Reestruturação + Modularização (v6.0)
- ✅ **8 bugs corrigidos** no pipeline (spaCy input, caps/emoji/hashtag, token names, URL detection)
- ✅ **TCW integrado** no Stage 08 (217 códigos, 10 categorias, 181 termos únicos)
- ✅ **Léxico expandido** com macrotemas corrupção e política externa
- ✅ **Token matching** reformulado: set() lookup com spaCy lemmas → O(1)/token
- ✅ **Modularização completa** (TAREFA 11): 19 arquivos em src/stages/
- ✅ **4 testes ponta-a-ponta** em 3 datasets diferentes, 0 erros

### Out 2025 — Pipeline Consolidado
- ✅ Pipeline otimizado em 17 stages sequenciais
- ✅ Sistema de deduplicação cross-dataset (redução 40-50%)
- ✅ Filtros de qualidade e relevância política
- ✅ Classificação política brasileira integrada
- ✅ Dashboard unificado disponível

---
**Version**: v6.2 (API Integration: 6 stages hybrid) | **RAM**: 4GB | **Colunas**: 126 | **Focus**: Análise discurso político brasileiro