# Arquitetura Centralizada do Pipeline Bolsonarismo 2025

## Visão Geral

Este documento descreve a arquitetura completamente centralizada e integrada com Anthropic API implementada em 2025 para o projeto de análise de discurso político brasileiro. A nova arquitetura elimina scripts separados e centraliza toda a execução através de um sistema unificado de 13 stages.

## Princípios Fundamentais

### 1. **Centralização Absoluta**
- **Uma única execução**: `python run_centralized_pipeline.py`
- **Configuração única**: `config/settings.yaml`
- **Factory centralizada**: `src/pipeline/stage_factory.py`
- **Executor unificado**: `src/pipeline/pipeline_executor.py`

### 2. **API Anthropic como Padrão**
- **Todos os 13 stages** têm integração Anthropic
- **Fallback robusto** para métodos tradicionais apenas quando necessário
- **Inteligência artificial** para análise semântica e contextual
- **Processamento de chunks** otimizado para grandes datasets

### 3. **Eliminação de Scripts Separados**
- **Não há mais** scripts individuais por stage
- **Funções centralizadas** nos módulos principais
- **Atualizações** apenas nos arquivos principais
- **Manutenção simplificada** e consistente

## Arquitetura do Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PONTO DE ENTRADA ÚNICO                          │
│                 run_centralized_pipeline.py                        │
│                                                                     │
│  • Argumentos CLI                                                  │
│  • Configuração global                                             │
│  • Controle de execução                                            │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                 ORQUESTRADOR PRINCIPAL                             │
│                 src/pipeline/runner.py                             │
│                                                                     │
│  • Gerenciamento de stages                                         │
│  • Checkpoint e recuperação                                        │
│  • Logging e relatórios                                            │
│  • Integração Anthropic central                                    │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
    ┌─────────────────▼─────────────────┐    ┌─────────────────────────┐
    │      FACTORY DE STAGES            │    │   EXECUTOR CENTRALIZADO │
    │  src/pipeline/stage_factory.py    │◄──►│src/pipeline/pipeline_   │
    │                                   │    │       executor.py       │
    │  • Instanciação de todos stages   │    │                         │
    │  • Detecção Anthropic             │    │  • Métodos únicos       │
    │  • Configuração dinâmica          │    │  • Processamento dados  │
    │  • Fallback inteligente           │    │  • Integração AI        │
    └─────────────────┬─────────────────┘    └─────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                 MÓDULOS ANTHROPIC (13 STAGES)                      │
│              src/anthropic_integration/                            │
│                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐      │
│  │ smart_encoding_ │ │ intelligent_    │ │ semantic_tfidf_ │      │
│  │    fixer.py     │ │ deduplicator.py │ │   analyzer.py   │ ...  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘      │
│                                                                     │
│  • Herança de AnthropicBase                                        │
│  • Processamento semântico                                         │
│  • Análise contextual brasileira                                   │
│  • Fallback para métodos tradicionais                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Fluxo de Dados

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Raw    │───►│Interim  │───►│Processed│───►│Results  │───►│ Final   │
│ Data    │    │ Data    │    │  Data   │    │ Analysis│    │ Report  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
Stages 1-3     Stages 4-8     Stages 9-11   Stages 12-13   Dashboard
(Prep + Clean) (Analysis)     (Network +    (Qualitative  (Visualization)
               (AI Enhanced)   Temporal)     + Review)
```

## Detalhamento dos Componentes

### 1. **Stage Factory (`src/pipeline/stage_factory.py`)**

#### Responsabilidades:
- **Instanciação dinâmica** de todos os 13 stages
- **Detecção automática** da disponibilidade Anthropic
- **Configuração inteligente** baseada em `settings.yaml`
- **Fallback robusto** quando API não disponível

#### Padrão de Implementação:
```python
def _create_stage_XX(self, **kwargs) -> Any:
    """Stage XX: Descrição"""
    use_anthropic = self.config.get('config_section', {}).get('use_anthropic', False)
    
    if use_anthropic and self.anthropic_available:
        try:
            from src.anthropic_integration.module_name import ModuleClass
            self.logger.info("🤖 Stage XX: Usando análise inteligente")
            return ModuleClass(self.config)
        except Exception as e:
            self.logger.warning(f"Falha na AI Stage XX: {e}. Usando método tradicional.")
    
    # Fallback tradicional APENAS para funções simples
    self.logger.info("🔧 Stage XX: Usando método básico")
    return None  # Delega para implementação básica
```

#### Stages Mapeados:
- **01_validate_data**: Validação estrutural (sem AI para performance)
- **02_fix_encoding**: Correção inteligente de encoding
- **02b_deduplication**: Deduplicação semântica
- **01b_feature_extraction**: Extração inteligente de características
- **03_clean_text**: Limpeza contextual preservando significado
- **04_sentiment_analysis**: Análise multi-dimensional de sentimentos
- **05_topic_modeling**: Interpretação semântica de tópicos
- **06_tfidf_extraction**: TF-IDF com agrupamento temático
- **07_clustering**: Validação e interpretação de clusters
- **08_hashtag_normalization**: Normalização semântica de hashtags
- **09_domain_extraction**: Classificação e análise de credibilidade
- **10_temporal_analysis**: Detecção e interpretação de eventos
- **11_network_structure**: Análise de comunidades e influência
- **12_qualitative_analysis**: Classificação de conspiração e negacionismo
- **13_review_reproducibility**: Revisão inteligente de qualidade

### 2. **Pipeline Executor (`src/pipeline/pipeline_executor.py`)**

#### Responsabilidades:
- **Execução centralizada** de todos os stages
- **Processamento sequencial** com checkpoints
- **Integração direta** com módulos Anthropic
- **Gerenciamento de dados** entre stages

#### Métodos Centralizados:
```python
def execute_stage_XX_description(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Stage XX: Descrição - Funcionalidade específica"""
    stage_instance = self.stage_factory.create_stage('XX_stage_name')
    
    if hasattr(stage_instance, 'method_intelligent'):  # Usando AI
        return stage_instance.method_intelligent(df, **params)
    else:  # Método tradicional APENAS para tarefas simples
        return self._execute_traditional_method(df)
```

### 3. **Módulos Anthropic (`src/anthropic_integration/`)**

#### Arquitetura dos Módulos:

```python
class IntelligentModule(AnthropicBase):
    """
    Módulo inteligente para análise específica
    Herda funcionalidades comuns de AnthropicBase
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Configurações específicas do módulo
    
    def analyze_intelligent(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Método principal de análise inteligente"""
        # Processamento de chunks
        # Chamadas à API Anthropic
        # Análise contextual brasileira
        # Fallback em caso de erro
```

#### Módulos Implementados:

1. **`smart_encoding_fixer.py`**
   - Correção contextual de encoding
   - Detecção inteligente de problemas
   - Preservação de conteúdo político

2. **`intelligent_deduplicator.py`**
   - Deduplicação semântica
   - Análise de similaridade contextual
   - Preservação de variações importantes

3. **`semantic_tfidf_analyzer.py`**
   - TF-IDF com interpretação semântica
   - Agrupamento temático inteligente
   - Extração de termos politicamente relevantes

4. **`intelligent_domain_analyzer.py`**
   - Classificação automática de fontes
   - Análise de credibilidade
   - Detecção de padrões de desinformação

5. **`smart_temporal_analyzer.py`**
   - Detecção automática de eventos
   - Correlação com contexto histórico brasileiro
   - Análise de campanhas coordenadas

6. **`intelligent_network_analyzer.py`**
   - Detecção de comunidades com interpretação
   - Análise de influência e propagação
   - Identificação de comportamento coordenado

7. **`smart_pipeline_reviewer.py`**
   - Revisão inteligente de qualidade
   - Análise de vieses metodológicos
   - Recomendações de melhorias

### 4. **Configuração Centralizada (`config/settings.yaml`)**

#### Estrutura de Configuração:
```yaml
# Configuração global Anthropic
anthropic:
  model: "claude-3-haiku-20240307"
  max_tokens: 4000
  temperature: 0.3
  cost_monitoring: true
  fallback_enabled: true

# Configuração por stage
stage_name:
  use_anthropic: true/false
  param1: value1
  param2: value2
```

#### Stages com Anthropic Habilitado:
- **02_fix_encoding**: `use_anthropic: true`
- **02b_deduplication**: `use_anthropic: true`
- **01b_feature_extraction**: `use_anthropic: true`
- **03_clean_text**: `use_anthropic: true`
- **04_sentiment_analysis**: `use_anthropic: true`
- **05_topic_modeling**: `use_anthropic_interpretation: true`
- **06_tfidf_extraction**: `use_anthropic: true`
- **07_clustering**: `use_anthropic_validation: true`
- **08_hashtag_normalization**: `use_anthropic: true`
- **09_domain_extraction**: `use_anthropic: true`
- **10_temporal_analysis**: `use_anthropic: true`
- **11_network_structure**: `use_anthropic: true`
- **12_qualitative_analysis**: `use_anthropic_classification: true`
- **13_pipeline_review**: `use_anthropic: true`

## Padrões de Implementação

### 1. **Padrão de Fallback Inteligente**

Todos os stages seguem o padrão:
```python
if use_anthropic and ANTHROPIC_AVAILABLE:
    try:
        # Implementação com Anthropic API
        result = anthropic_module.intelligent_analysis(data)
    except Exception as e:
        logger.warning(f"API falhou: {e}. Usando método tradicional.")
        result = traditional_method(data)
else:
    result = traditional_method(data)
```

**Importante**: Métodos tradicionais são implementados **APENAS** para:
- **Carregamento de arquivos** (leitura CSV básica)
- **Funções muito simples** (contagem, validação estrutural básica)
- **Operações de I/O** (salvar checkpoints)

### 2. **Padrão de Processamento de Chunks**

Para datasets grandes:
```python
from src.data.processors.chunk_processor import ChunkProcessor

processor = ChunkProcessor(chunk_size=10000)
for chunk in processor.process_file('large_file.csv'):
    chunk_result = anthropic_module.process_chunk(chunk)
    results.append(chunk_result)
```

### 3. **Padrão de Análise Contextual Brasileira**

Todos os prompts incluem contexto específico:
```python
prompt = f"""
Analise os dados do Telegram brasileiro (2019-2023):

CONTEXTO: 
- Governo Bolsonaro
- Pandemia COVID-19
- Eleições 2022
- Movimento bolsonarista
- Negacionismo e autoritarismo

DADOS: {data_sample}

Responda em JSON com análise específica...
"""
```

## Benefícios da Arquitetura

### 1. **Centralização Total**
- ✅ **Um comando**: `python run_centralized_pipeline.py`
- ✅ **Uma configuração**: `config/settings.yaml`
- ✅ **Um ponto de manutenção**: Arquivos principais
- ✅ **Sem scripts separados**: Eliminação de fragmentação

### 2. **Inteligência Artificial Integrada**
- ✅ **Análise semântica**: Interpretação contextual do conteúdo
- ✅ **Compreensão política**: Contexto brasileiro específico
- ✅ **Qualidade superior**: Resultados mais precisos e relevantes
- ✅ **Automação inteligente**: Redução de intervenção manual

### 3. **Robustez e Flexibilidade**
- ✅ **Fallback automático**: Continua funcionando sem API
- ✅ **Configuração dinâmica**: Habilitar/desabilitar AI por stage
- ✅ **Processamento otimizado**: Chunks para grandes datasets
- ✅ **Monitoramento de custos**: Controle de uso da API

### 4. **Manutenibilidade**
- ✅ **Código centralizado**: Mudanças em poucos arquivos
- ✅ **Padrões consistentes**: Mesma estrutura em todos stages
- ✅ **Logging unificado**: Rastreamento completo da execução
- ✅ **Testes integrados**: Validação do pipeline completo

## Comandos de Uso

### Execução Completa
```bash
# Pipeline completo com AI
python run_centralized_pipeline.py

# Pipeline sem AI (apenas métodos simples)
python run_centralized_pipeline.py --no-anthropic
```

### Execução Seletiva
```bash
# Stages específicos
python run_centralized_pipeline.py --stages 02_fix_encoding 06_tfidf_extraction

# Stage único
python run_centralized_pipeline.py --single 04_sentiment_analysis
```

### Informações e Debug
```bash
# Listar stages com status AI
python run_centralized_pipeline.py --list

# Debug detalhado
python run_centralized_pipeline.py --log-level DEBUG

# Simulação (dry run)
python run_centralized_pipeline.py --dry-run
```

## Diretrizes de Desenvolvimento

### 1. **Regra da Centralização**
- **NUNCA** criar scripts separados para stages
- **SEMPRE** implementar funcionalidades nos módulos principais
- **OBRIGATÓRIO** usar o stage factory para instanciação

### 2. **Regra da Inteligência Artificial**
- **PADRÃO**: Implementar com Anthropic API
- **EXCEÇÃO**: Métodos tradicionais apenas para tarefas triviais
- **PROIBIDO**: Análise complexa sem AI

### 3. **Regra da Configuração**
- **ÚNICA FONTE**: `config/settings.yaml`
- **PARÂMETRO OBRIGATÓRIO**: `use_anthropic: true/false`
- **ATUALIZAÇÃO**: Apenas nos arquivos principais

### 4. **Regra do Fallback**
- **SEMPRE** implementar fallback robusto
- **APENAS** para funções muito simples
- **LOGGING** obrigatório quando fallback é usado

## Conclusão

A arquitetura centralizada de 2025 representa uma evolução completa do pipeline Bolsonarismo, eliminando a fragmentação anterior e estabelecendo um sistema unificado, inteligente e maintível. A integração profunda com Anthropic API garante análises de alta qualidade específicas para o contexto político brasileiro, mantendo robustez através de fallbacks inteligentes para operações simples.

Esta implementação segue rigorosamente os princípios de:
- **Centralização absoluta**
- **Inteligência artificial como padrão**
- **Simplicidade operacional**
- **Manutenibilidade sustentável**

O resultado é um pipeline que **não requer scripts separados**, **atualiza-se nos arquivos principais** e **utiliza AI para análise complexa**, conforme especificado nos requisitos do projeto.