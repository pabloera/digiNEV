# PROJECT RULES - Projeto Bolsonarismo

## 🔥 REGRAS FIXAS E IMUTÁVEIS DO PROJETO

Estas são as **REGRAS ABSOLUTAS** que NUNCA podem ser violadas ao trabalhar com o projeto Bolsonarismo.

---

## ⚡ REGRA #1: PROCESSAMENTO EM CHUNKS OBRIGATÓRIO

### 🚨 REGRA CRÍTICA DE MEMÓRIA

**NUNCA, JAMAIS, EM HIPÓTESE ALGUMA carregue arquivos completos dos DATASETS_FULL**

```python
# ❌ PROIBIDO ABSOLUTO - CAUSA CRASH DO SISTEMA
df = pd.read_csv('data/DATASETS_FULL/qualquer_arquivo.csv', sep=';')
df = pd.read_csv('data/DATASETS_FULL/1_2019-2021-govbolso.csv')
data = open('data/DATASETS_FULL/arquivo.csv').read()

# ✅ ÚNICA FORMA PERMITIDA - SEMPRE CHUNKS
from src.data.processors.chunk_processor import ChunkProcessor
processor = ChunkProcessor(chunk_size=10000)
for chunk in processor.process_file('data/DATASETS_FULL/arquivo.csv'):
    # Processar apenas este chunk
    pass
```

### 📊 Tamanhos de Chunk OBRIGATÓRIOS

| Uso | Chunk Size | Quando Usar |
|-----|------------|-------------|
| **API Anthropic** | 5,000 linhas | Processamento com IA |
| **Análises Complexas** | 10,000 linhas | Operações pesadas |
| **Operações Simples** | 20,000 linhas | Máquinas potentes |
| **Pouca Memória** | 1,000 linhas | Sistemas limitados |

### 🔒 CONSEQUÊNCIAS DE VIOLAÇÃO

- **Sistema trava/crash**
- **Out of Memory errors**
- **Performance degradada**
- **Pipeline corrompido**

---

## ⚡ REGRA #2: ESTRUTURA DE DADOS FIXA

### 📁 ÚNICA FONTE DE DADOS PERMITIDA

```
data/
└── DATASETS_FULL/                 # ÚNICA localização válida
    ├── 1_2019-2021-govbolso.csv   # ✅ Permitido
    ├── 2_2021-2022-pandemia.csv   # ✅ Permitido
    ├── 3_2022-2023-poseleic.csv   # ✅ Permitido
    ├── 4_2022-2023-elec.csv       # ✅ Permitido
    ├── 5_2022-2023-elec-extra.csv # ✅ Permitido
    └── channels_name.csv           # ✅ Permitido
```

### ❌ ESTRUTURAS PROIBIDAS

```
❌ data/raw/
❌ data/processed/
❌ data/interim/
❌ data/external/
❌ data/temp/
❌ data/backup/
❌ Qualquer outro diretório em data/
```

---

## ⚡ REGRA #3: EXECUÇÃO CENTRALIZADA

### 🎯 ÚNICO PONTO DE ENTRADA

```bash
# ✅ ÚNICA FORMA PERMITIDA DE EXECUTAR O PROJETO
python run_pipeline.py

# ✅ Comandos permitidos
python run_pipeline.py --stages 01_validate_data 03_clean_text
python run_pipeline.py --single 04_sentiment_analysis
python run_pipeline.py --list
```

### ❌ EXECUÇÕES PROIBIDAS

```bash
❌ python src/pipeline/stages/stage_01_validate_data.py
❌ python scripts/qualquer_script.py
❌ python src/anthropic_integration/qualquer_modulo.py
❌ Execução direta de qualquer script individual
```

---

## ⚡ REGRA #4: INTEGRAÇÃO ANTHROPIC CENTRALIZADA

### 🤖 PADRÃO OBRIGATÓRIO

```python
# ✅ SEMPRE usar a integração centralizada
from src.pipeline.runner import PipelineRunner

runner = PipelineRunner()
if runner.anthropic_integration:
    # API disponível - usar métodos Anthropic
    runner.run_pipeline()
else:
    # Fallback - usar métodos tradicionais
    runner.run_pipeline()
```

### ❌ INTEGRAÇÕES PROIBIDAS

```python
❌ from anthropic import Anthropic  # Integração direta
❌ import openai                    # Outras APIs
❌ Qualquer integração fora de src/anthropic_integration/
```

---

## ⚡ REGRA #5: ESTRUTURA DE CÓDIGO FIXA

### 🏗️ ARQUITETURA IMUTÁVEL (ATUALIZADA JANEIRO 2025)

```
src/
├── anthropic_integration/   # ✅ CENTRO: 28 componentes API (pipeline_validator integrado)
├── data/
│   ├── processors/         # ✅ chunk_processor.py (essencial)
│   ├── transformers/       # ✅ Apenas módulos consolidados
│   └── utils/              # ✅ encoding_fixer.py (crítico)
└── preprocessing/          # ✅ stopwords_pt.txt (dados essenciais)
```

### 📁 ARQUIVOS ÓRFÃOS ARQUIVADOS (Janeiro 2025)

```
archive/scripts_non_pipeline/  # 🗂️ 15 scripts movidos
├── src/preprocessing/         # 2 scripts arquivados
├── src/data/processors/       # 2 scripts arquivados  
└── src/data/transformers/     # 11 scripts arquivados
```

### ❌ MODIFICAÇÕES PROIBIDAS

- **❌ Não criar novos diretórios em `src/`**
- **❌ Não mover módulos entre diretórios**
- **❌ Não criar scripts fora da estrutura**
- **❌ Não duplicar funcionalidades**

---

## ⚡ REGRA #6: PIPELINE VALIDATOR INTEGRADO (JANEIRO 2025)

### 🔍 VALIDAÇÃO HOLÍSTICA OBRIGATÓRIA

```python
# ✅ PIPELINE_VALIDATOR AGORA É AUTOMÁTICO
# Executado automaticamente no final de cada pipeline completo

pipeline = UnifiedAnthropicPipeline(config, project_root)
results = pipeline.run_complete_pipeline(datasets)

# Validação final automática inclui:
# 1. CompletePipelineValidator.validate_complete_pipeline() (70% peso)
# 2. api_integration.execute_comprehensive_pipeline_validation() (30% peso)
# 3. Score final combinado
# 4. Critérios de sucesso ≥ 0.7
```

### 📊 COMPONENTES ANTHROPIC INTEGRATION

| Componente | Status | Função |
|------------|--------|---------|
| **pipeline_validator** | ✅ INTEGRADO | Validação holística final |
| **28 componentes API** | ✅ ATIVOS | Pipeline completo |
| **Sistema semântico** | ✅ ATIVO | Busca e análise avançada |

---

## ⚡ REGRA #7: CONFIGURAÇÃO API

### 🔑 CONFIGURAÇÃO OBRIGATÓRIA

```bash
# ✅ Configuração correta da API
echo "ANTHROPIC_API_KEY=sk-ant-api03-xxxxx" > .env
```

```yaml
# config/settings.yaml - Configuração padrão
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  model: "claude-3-haiku-20240307"
  max_tokens_per_request: 2000
  temperature: 0.3
```

---

## ⚡ REGRA #7: GESTÃO DE LOGS

### 📝 SISTEMA DE LOGS PADRONIZADO

```
logs/
├── pipeline/           # ✅ Logs do pipeline apenas
└── anthropic/         # ✅ Custos e tracking API
```

### ❌ LOGS PROIBIDOS

- **❌ Não criar logs personalizados fora de `logs/`**
- **❌ Não manter logs antigos (limpeza automática)**
- **❌ Não logar informações sensíveis**

---

## ⚡ REGRA #8: DESENVOLVIMENTO

### 🔧 PADRÕES DE DESENVOLVIMENTO

```python
# ✅ Estrutura obrigatória para novos stages
def run_stage(config, stage_config, base_dir, logger, **params):
    """
    Args:
        config: Configuração global
        stage_config: Configuração da etapa  
        base_dir: Diretório base do projeto
        logger: Logger configurado
        **params: Parâmetros específicos
        
    Returns:
        Dict com resultados da etapa
    """
    logger.info(f"Iniciando stage")
    
    # SEMPRE usar chunks se processar dados
    if 'data_file' in params:
        from src.data.processors.chunk_processor import ChunkProcessor
        processor = ChunkProcessor(chunk_size=10000)
        # Processar em chunks...
    
    return {
        'status': 'completed',
        'metrics': {},
        'output_path': None
    }
```

### ❌ PADRÕES PROIBIDOS

- **❌ Não criar funções que carregam arquivos completos**
- **❌ Não criar classes sem herdar de AnthropicBase quando usar API**
- **❌ Não ignorar tratamento de erros**

---

## 🚨 VIOLAÇÕES ABSOLUTAMENTE PROIBIDAS

### ❌ NUNCA FAZER

1. **Carregar arquivos completos**: `pd.read_csv()` direto em DATASETS_FULL
2. **Criar estruturas de dados alternativas**: Novos diretórios em `data/`
3. **Executar scripts individuais**: Fora do `run_pipeline.py`
4. **Modificar arquitetura**: Mover ou criar diretórios em `src/`
5. **Integração API externa**: Fora de `anthropic_integration/`
6. **Ignorar chunks**: Qualquer processamento sem ChunkProcessor
7. **Logs personalizados**: Fora da estrutura padrão
8. **Backups manuais**: Sistema automatizado já existe

---

## ✅ SEMPRE FAZER

### 🎯 PRÁTICAS OBRIGATÓRIAS

1. **Usar ChunkProcessor** para TODOS os arquivos de dados
2. **Executar via `run_pipeline.py`** apenas
3. **Configurar API Anthropic** antes de usar
4. **Seguir estrutura de stages** para novas funcionalidades
5. **Usar logging padrão** do projeto
6. **Respeitar checkpoints** do pipeline
7. **Documentar mudanças** em GUIDELINES.md
8. **Testar com dados pequenos** antes de processar datasets completos

---

## 🔒 ENFORCEMENT

### 🛡️ Como Garantir Cumprimento

1. **Code Review Obrigatório**: Toda mudança deve seguir estas regras
2. **Validação Automática**: Pipeline falha se regras forem violadas
3. **Documentação Atualizada**: Sempre manter PROJECT_RULES.md atualizado
4. **Treinamento**: Todos devem conhecer estas regras antes de contribuir

### ⚠️ CONSEQUÊNCIAS DE VIOLAÇÃO

- **Sistema instável**
- **Perda de dados**
- **Performance degradada**
- **Falhas de pipeline**
- **Necessidade de rollback**

---

## 📚 REFERÊNCIAS OBRIGATÓRIAS

Antes de QUALQUER trabalho no projeto, LEIA:

1. **PROJECT_RULES.md** (este arquivo) - **OBRIGATÓRIO**
2. **CLAUDE.md** - Instruções para Claude Code
3. **GUIDELINES.md** - Diretrizes detalhadas
4. **README.md** - Visão geral do projeto

---

## 📝 CONTROLE DE VERSÃO

- **Versão**: 4.1 (Estrutura Limpa + Pipeline Validator Integrado)
- **Última Atualização**: 06 Janeiro 2025
- **Status**: **REGRAS FIXAS - NÃO MODIFICAR SEM APROVAÇÃO**

### 🔄 CHANGELOG v4.1 (06/01/2025)

- ✅ **Arquivamento de scripts órfãos**: 15 scripts movidos para `archive/scripts_non_pipeline/`
- ✅ **Integração pipeline_validator**: CompletePipelineValidator agora é parte do fluxo principal
- ✅ **Estrutura limpa**: Mantidos apenas scripts essenciais (4 scripts + dados)
- ✅ **Validação robusta**: Score combinado com critérios de qualidade ≥ 0.7
- ✅ **28 componentes ativos**: Todos os scripts em anthropic_integration funcionais

---

**⚠️ ATENÇÃO: Estas regras são IMUTÁVEIS e devem ser seguidas por TODOS que trabalham no projeto. Violações podem causar instabilidade do sistema e perda de dados.**