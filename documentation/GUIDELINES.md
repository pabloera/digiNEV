# GUIDELINES - Projeto Bolsonarismo

## Diretrizes de Desenvolvimento e Uso

Este documento estabelece as diretrizes para trabalhar com o projeto Bolsonarismo, que está **completamente centralizado na integração Anthropic**.

## 🚨 ATENÇÃO: LEIA PROJECT_RULES.md PRIMEIRO

**OBRIGATÓRIO**: Antes de prosseguir, leia o arquivo `PROJECT_RULES.md` que contém as **REGRAS FIXAS E IMUTÁVEIS** do projeto. Este documento (GUIDELINES.md) complementa as regras, mas PROJECT_RULES.md tem precedência absoluta.

---

## 📋 Visão Geral

O projeto Bolsonarismo é uma análise abrangente do discurso político brasileiro em canais do Telegram (2019-2023), com **arquitetura v4.0 centralizada na API Anthropic**.

### Princípios Fundamentais

1. **🔥 PROCESSAMENTO EM CHUNKS OBRIGATÓRIO**: NUNCA carregue arquivos completos - sempre use `ChunkProcessor`
2. **Centralização Anthropic**: Todas as 13 etapas do pipeline utilizam integração Anthropic como método principal  
3. **Simplicidade**: Estrutura linear e focada, zero redundância
4. **Fonte Única de Dados**: Apenas `data/DATASETS_FULL/` como origem de dados
5. **Execução Unificada**: Um único entry point (`run_pipeline.py`)

---

## 🗂️ Estrutura do Projeto

```
📁 dataanalysis-bolsonarismo/
├── 🎯 run_pipeline.py              # ÚNICO entry point
├── 📋 CLAUDE.md                    # Instruções para Claude Code
├── 📋 GUIDELINES.md                # Este arquivo
├── 📋 README.md                    # Documentação principal
├── 📊 data/DATASETS_FULL/          # ÚNICA fonte de dados
├── ⚙️ config/                      # Configurações do projeto
├── 📝 logs/                        # Logs do pipeline
├── 🗂️ archive/scripts_non_pipeline/ # Scripts arquivados
└── 🧠 src/                         # Código-fonte
    ├── 🤖 anthropic_integration/   # CENTRO: Integração API
    ├── ⚡ pipeline/                # Pipeline runner + stages
    ├── 🔧 data/                    # Processamento de dados
    └── 📚 preprocessing/           # Pré-processamento
```

---

## 🚀 Como Usar o Projeto

### Setup Inicial

```bash
# 1. Configurar API Anthropic
echo "ANTHROPIC_API_KEY=sk-ant-api03-xxxxx" > .env

# 2. Instalar dependências
pip install -r pyproject.toml  # ou poetry install

# 3. Executar pipeline completo
python run_pipeline.py
```

### Comandos Principais

```bash
# Executar pipeline completo (13 etapas)
python run_pipeline.py

# Executar etapas específicas
python run_pipeline.py --stages 01_validate_data 03_clean_text

# Executar uma única etapa
python run_pipeline.py --single 04_sentiment_analysis

# Listar todas as etapas
python run_pipeline.py --list

# Resumir execução anterior
python run_pipeline.py  # Detecta checkpoint automaticamente
```

---

## 📊 Trabalhando com Dados

### Fonte Única de Dados

**REGRA OBRIGATÓRIA**: Use APENAS chunks para processar dados de `data/DATASETS_FULL/`:

```python
# ✅ MÉTODO OBRIGATÓRIO: SEMPRE usar ChunkProcessor
from src.data.processors.chunk_processor import ChunkProcessor

# Configurar processamento em chunks
processor = ChunkProcessor(chunk_size=10000)  # Ajustar conforme memória disponível

# Processar arquivo em chunks
results = []
for chunk in processor.process_file('data/DATASETS_FULL/1_2019-2021-govbolso.csv'):
    # Processar cada chunk individualmente
    processed_chunk = process_chunk(chunk)
    results.append(processed_chunk)

# ❌ NUNCA FAZER: Carregar arquivo completo
# df = pd.read_csv('data/DATASETS_FULL/arquivo.csv', sep=';')  # PROIBIDO!

# ✅ CHUNK SIZES RECOMENDADOS:
# - 10,000 linhas: Para análises complexas
# - 5,000 linhas: Para processamento com API Anthropic
# - 20,000 linhas: Para operações simples
```

### Datasets Disponíveis

1. **1_2019-2021-govbolso.csv** - Período do Governo Bolsonaro
2. **2_2021-2022-pandemia.csv** - Período da Pandemia
3. **3_2022-2023-poseleic.csv** - Período Pós-Eleições
4. **4_2022-2023-elec.csv** - Período Eleitoral
5. **5_2022-2023-elec-extra.csv** - Período Eleitoral Estendido
6. **channels_name.csv** - Lista de canais

---

## 🤖 Integração Anthropic

### Arquitetura Centralizada

O projeto utiliza **integração Anthropic centralizada** através de:

- **`src/anthropic_integration/pipeline_integration.py`** - Orquestrador principal
- **`src/anthropic_integration/base.py`** - Cliente API base
- **17 módulos especializados** - Funcionalidades específicas

### Padrão de Uso

```python
# O pipeline detecta automaticamente a integração Anthropic
from src.pipeline.runner import PipelineRunner

runner = PipelineRunner()
if runner.anthropic_integration:
    # Usa métodos API Anthropic
    runner.run_pipeline()
else:
    # Usa métodos tradicionais como fallback
    runner.run_pipeline()
```

### Configuração API

```yaml
# config/settings.yaml
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  model: "claude-3-haiku-20240307"
  max_tokens_per_request: 2000
  temperature: 0.3
```

---

## ⚡ Pipeline de 13 Etapas

### Core Processing Stages

1. **01_validate_data** - Validação estrutural + detecção de encoding
2. **02_fix_encoding** - Correção otimizada de encoding
3. **02b_deduplication** - Deduplicação inteligente com contagem de frequência
4. **01b_feature_extraction** - Extração de features via API
5. **03_clean_text** - Limpeza textual contextualizada

### Analysis Stages

6. **04_sentiment_analysis** - Análise de sentimentos multicamadas
7. **05_topic_modeling** - Modelagem de tópicos LDA com interpretação
8. **06_tfidf_extraction** - Extração TF-IDF ponderada por frequência
9. **07_clustering** - Clustering com validação API
10. **08_hashtag_normalization** - Normalização de hashtags
11. **09_domain_extraction** - Extração de domínios ponderada
12. **10_temporal_analysis** - Análise temporal
13. **11_network_structure** - Estrutura de redes
14. **12_qualitative_analysis** - Análise qualitativa via API
15. **13_review_reproducibility** - Revisão de reprodutibilidade

---

## 🔧 Desenvolvimento

### Adicionando Nova Funcionalidade

1. **Para funcionalidades API**: Adicionar em `src/anthropic_integration/`
2. **Para etapas do pipeline**: Modificar em `src/pipeline/stages/`
3. **Para processamento de dados**: Adicionar em `src/data/`

### Estrutura de um Stage

```python
# src/pipeline/stages/stage_XX_nome.py
def run_stage(config, stage_config, base_dir, logger, **params):
    """
    Executa a etapa XX do pipeline
    
    Args:
        config: Configuração global
        stage_config: Configuração da etapa
        base_dir: Diretório base do projeto
        logger: Logger configurado
        **params: Parâmetros específicos
        
    Returns:
        Dict com resultados da etapa
    """
    logger.info(f"Iniciando stage XX")
    
    # Implementação da etapa
    result = {
        'status': 'completed',
        'metrics': {},
        'output_path': None
    }
    
    return result
```

### Integração com Anthropic

```python
# Exemplo de uso da integração centralizada
from src.anthropic_integration.base import AnthropicBase

class MeuModulo(AnthropicBase):
    def __init__(self, config):
        super().__init__(config)
    
    def processar(self, data):
        prompt = f"Analise este texto: {data}"
        resposta = self.create_message(
            prompt=prompt,
            stage='meu_stage',
            operation='analise_texto'
        )
        return resposta
```

---

## 📝 Logs e Monitoramento

### Sistema de Logs

- **Pipeline logs**: `logs/pipeline/`
- **Anthropic costs**: Monitoramento automático de custos
- **Checkpoints**: Salvamento automático de progresso

### Estrutura de Logs

```
logs/
├── pipeline/
│   ├── pipeline_YYYYMMDD_HHMMSS.log
│   └── api_checkpoints/
└── anthropic/
    └── cost_tracking.json
```

---

## 🚨 Regras Importantes

### ✅ O Que FAZER

1. **SEMPRE use ChunkProcessor para todos os datasets em `data/DATASETS_FULL/`**
2. **Execute via `python run_pipeline.py`**
3. **Configure API Anthropic no arquivo `.env`**
4. **Use chunk_size apropriado (5K-20K linhas)**
5. **Mantenha logs para debugging**
6. **Processe dados em batches, nunca arquivo completo**

### ❌ O Que NÃO FAZER

1. **❌ NUNCA carregue arquivo completo com `pd.read_csv()` - SEMPRE usar chunks**
2. **❌ Não criar diretórios em `data/` além de `DATASETS_FULL/`**
3. **❌ Não executar scripts individuais fora do pipeline**
4. **❌ Não ignorar limitações de memória - usar chunks menores se necessário**
5. **❌ Não processar múltiplos datasets simultaneamente sem chunks**
6. **❌ Não criar scripts fora da estrutura centralizada**

---

## 🛠️ Troubleshooting

### Problemas Comuns

**Erro: "API Anthropic não configurada"**
```bash
# Solução
echo "ANTHROPIC_API_KEY=sua_chave_aqui" > .env
```

**Erro: "Dataset não encontrado"**
```bash
# Verificar se arquivo existe
ls data/DATASETS_FULL/
# Usar path correto: data/DATASETS_FULL/nome_arquivo.csv
```

**Pipeline lento ou com problemas de memória**
```python
# SEMPRE ajustar chunk_size conforme disponibilidade de memória
from src.data.processors.chunk_processor import ChunkProcessor

# Para máquinas com pouca memória
processor = ChunkProcessor(chunk_size=1000)

# Para processamento com API Anthropic (evitar timeout)
processor = ChunkProcessor(chunk_size=5000)

# Para operações simples em máquinas potentes
processor = ChunkProcessor(chunk_size=20000)

# NUNCA processar arquivo completo - mesmo pequenos
```

**Erro: "Memory Error" ou "Out of Memory"**
```python
# Solução: Reduzir drasticamente o chunk_size
processor = ChunkProcessor(chunk_size=500)  # Chunks muito pequenos
```

---

## 📚 Referências

- **CLAUDE.md** - Instruções específicas para Claude Code
- **README.md** - Documentação geral do projeto
- **config/settings.yaml** - Configurações detalhadas
- **src/anthropic_integration/README.md** - Documentação da integração API

---

## 🔄 Versionamento

- **v4.0** - Arquitetura centralizada Anthropic (atual)
- **v3.0** - Pipeline otimizado híbrido
- **v2.0** - Pipeline tradicional
- **v1.0** - Versão inicial

---

*Última atualização: Janeiro 2025*
*Autor: Projeto Bolsonarismo Team*