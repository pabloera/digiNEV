# digiNEV v.final - Brazilian Political Discourse Analysis

## 🎯 CONTEXTO
**Tipo**: Pesquisa Acadêmica em Ciências Sociais
**Foco**: Análise sociológica de discurso político brasileiro
**Dataset**: Mensagens Telegram (2019-2023)
**Specs**: 4GB RAM | Portuguese-optimized | 14 stages científicos | Consolidado

## 🏗️ Sistema Científico Consolidado v.final

### Pipeline Científico (14 estágios) - IMPLEMENTADO
1. **Feature Extraction (01)**: Detecção automática de colunas e features
2. **Text Preprocessing (02)**: Limpeza básica de texto em português
3. **Linguistic Processing (03)**: Processamento linguístico avançado com spaCy
4. **Statistical Analysis (04)**: Análise estatística com dados spaCy
5. **Political Classification (05)**: Classificação política brasileira
6. **TF-IDF Vectorization (06)**: TF-IDF com tokens spaCy
7. **Clustering Analysis (07)**: Clustering baseado em features linguísticas
8. **Topic Modeling (08)**: Topic modeling com embeddings
9. **Temporal Analysis (09)**: Análise temporal
10. **Network Analysis (10)**: Coordenação e padrões de rede
11. **Domain Analysis (11)**: Análise de domínios e URLs
12. **Semantic Analysis (12)**: Análise semântica avançada
13. **Event Context (13)**: Detecção de contextos políticos
14. **Channel Analysis (14)**: Classificação de canais/fontes

**Stack**: Python | scikit-learn | spaCy pt_core_news_lg | Streamlit

## 🚀 Execução

### Analyzer v.final
```bash
# Execução direta
python run_pipeline.py

# Teste com dados controlados
python test_clean_analyzer.py

# Dashboard acadêmico
python -m src.dashboard.start_dashboard
```

### Uso Programático
```python
from src.analyzer import Analyzer

analyzer = Analyzer()
results = analyzer.analyze_dataset(df)
print(f"Colunas geradas: {results['columns_generated']}")
print(f"Stages completados: {results['stats']['stages_completed']}/14")
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
├── src/                         # Sistema científico consolidado
│   ├── analyzer.py              # Analyzer v.final (núcleo principal) - 14 stages
│   ├── lexicon_loader.py        # Carregador de léxico político
│   └── dashboard/               # Dashboard acadêmico
│       ├── start_dashboard.py   # Iniciador do dashboard
│       ├── data_analysis_dashboard.py  # Dashboard principal
│       └── [outros dashboards]  # Dashboards especializados
├── config/                      # Configuração unificada
│   ├── settings.yaml            # Configurações principais
│   ├── processing.yaml          # Configurações de processamento
│   └── [outras configs]         # Configurações específicas
├── data/                        # Datasets de pesquisa
├── run_pipeline.py              # Script principal de execução
└── test_clean_analyzer.py       # Teste do sistema
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
- **30+ colunas reais** geradas pelo pipeline sequencial de 14 stages
- Classificação política (extrema-direita, direita, centro, esquerda, neutral)
- Análise estatística descritiva (word_count, char_count, sentence_count)
- Features extraídas automaticamente (hashtags, URLs, mentions, emojis)
- TF-IDF com scores reais e top termos por documento
- Clustering K-Means com distâncias calculadas
- Topic modeling LDA com probabilidades reais
- Análise temporal (hour, day_of_week, month) quando disponível
- Coordenação de rede detectada por cluster e tempo
- Análise de domínios e URLs com classificação
- Análise semântica avançada com conectivos e modalidade
- Contexto de eventos políticos brasileiros
- Análise de canais/fontes com autoridade e padrões

## 🧪 Testes
```bash
# Teste do pipeline consolidado
python test_clean_analyzer.py

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
- `batch_analyzer/data/1_2019-2021-govbolso.csv` (135.88 MB)
- `batch_analyzer/data/2_2021-2022-pandemia.csv` (229.96 MB)
- `data/controlled_test_100.csv` (teste local)

### Arquivos Críticos
**Sistema Principal:**
- `/src/analyzer.py` - Pipeline consolidado 14 estágios
- `/run_pipeline.py` - Script de execução principal
- `/test_clean_analyzer.py` - Teste do sistema

**Dashboard:**
- `/src/dashboard/data_analysis_dashboard.py` - Dashboard principal
- `/src/dashboard/start_dashboard.py` - Iniciador do dashboard

## 📝 Atualizações Recentes (Out 2025)
- ✅ Pipeline consolidado em 14 stages sequenciais
- ✅ Analyzer.py implementado com todos os estágios funcionais
- ✅ Classificação política brasileira integrada
- ✅ Dashboard unificado disponível
- ✅ Sistema testado e validado
- ✅ Documentação atualizada para refletir realidade

---
**Version**: v.final | **RAM**: 4GB | **Focus**: Análise discurso político brasileiro consolidado