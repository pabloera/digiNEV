# PIPELINE ANTHROPIC INTEGRATION
## Sistema Unificado de Análise com IA (v4.1 - Janeiro 2025)

### 🎯 EXECUÇÃO CENTRALIZADA

**PONTO DE ENTRADA ÚNICO:**
- `run_pipeline.py` - **Único script permitido** (raiz do projeto)

**ORQUESTRADOR PRINCIPAL:**
- `unified_pipeline.py` - **28 componentes integrados** (pipeline_validator incluído)

### 📊 STATUS DOS COMPONENTES (28/28 ATIVOS)

**ETAPAS DO PIPELINE:**

00. `system_validator.py` - Validação do sistema
01. `encoding_validator.py` - Validação de encoding (correção integrada)
02b. `deduplication_validator.py` - Deduplicação inteligente
01b. `feature_extractor.py` - Extração de features abrangentes (otimizado para datasets pré-processados)
03. `intelligent_text_cleaner.py` - Limpeza contextual de texto
04. `sentiment_analyzer.py` - Análise de sentimento político
05. `topic_interpreter.py` - Modelagem e interpretação de tópicos
06. `semantic_tfidf_analyzer.py` - TF-IDF com análise semântica
07. `cluster_validator.py` - Clustering validado
08. `semantic_hashtag_analyzer.py` - Normalização de hashtags
09. `intelligent_domain_analyzer.py` - Análise de domínios e credibilidade
10. `smart_temporal_analyzer.py` - Análise temporal inteligente
11. `intelligent_network_analyzer.py` - Análise de redes e influência
12. `qualitative_classifier.py` - Classificação acadêmica qualitativa
13. `smart_pipeline_reviewer.py` - Revisão e reprodutibilidade
14. **`pipeline_validator.py` - 🆕 VALIDAÇÃO HOLÍSTICA INTEGRADA**

**SISTEMA SEMÂNTICO AVANÇADO:**
- `semantic_search_engine.py` - Motor de busca semântica
- `intelligent_query_system.py` - Sistema de queries inteligentes
- `content_discovery_engine.py` - Descoberta de padrões
- `analytics_dashboard.py` - Dashboard analítico
- `temporal_evolution_tracker.py` - Rastreamento temporal

**COMPONENTES DE APOIO:**
- `base.py` - Classe base para todos os módulos
- `pipeline_integration.py` - Integração coordenada
- `voyage_embeddings.py` - Integração com Voyage.ai
- `api_error_handler.py` - Tratamento de erros da API
- `cost_monitor.py` - Monitoramento de custos

### 🚀 EXECUÇÃO (v4.1 - JANEIRO 2025):

```bash
# ✅ ÚNICA FORMA PERMITIDA (respeitando PROJECT_RULES.md):
python run_pipeline.py

# ❌ MÉTODOS ANTIGOS REMOVIDOS:
# python src/run_centralized_pipeline.py  # <- Arquivo removido
# python src/*.py                         # <- Viola regras
```

### 🔍 NOVA VALIDAÇÃO AUTOMÁTICA:

O **pipeline_validator** agora é **executado automaticamente** no final:

```python
# Validação final automática inclui:
# 1. CompletePipelineValidator.validate_complete_pipeline() (70% peso)
# 2. api_integration.execute_comprehensive_pipeline_validation() (30% peso)
# 3. Score final combinado
# 4. Critérios de sucesso ≥ 0.7
```

### 📁 ESTRUTURA LIMPA (Janeiro 2025):

- ✅ **28 componentes ativos** neste diretório
- ✅ **100% funcionalidade** integrada
- ✅ **Pipeline_validator** agora parte do fluxo principal
- ✅ **15 scripts órfãos** arquivados em `archive/scripts_non_pipeline/`

---

**📋 Atualizado em:** 06 Janeiro 2025  
**Versão:** v4.1 (Estrutura Limpa + Pipeline Validator Integrado)  
**Status:** Todos os 28 componentes funcionais e integrados

### ✅ STATUS: 100% INTEGRADO COM ANTHROPIC API

Todos os módulos utilizam a API Anthropic como método principal, com fallbacks tradicionais para robustez.

## 💰 **OTIMIZAÇÃO DE CUSTOS VOYAGE.AI - CONSOLIDADA (Janeiro 2025)**

### 🎯 **CONFIGURAÇÃO ATIVA:**
- **Modelo:** `voyage-3.5-lite` (mais econômico)
- **Amostragem:** Inteligente ativa (96% redução)
- **Limite:** 50.000 mensagens por dataset
- **Custo:** $0.00 (gratuito até 200M tokens)

### 📊 **ECONOMIA IMPLEMENTADA:**
```yaml
Cenário Anterior: 100M tokens = GRATUITO
Cenário Atual:    3M tokens = GRATUITO (97% economia)
Escalabilidade:   66x mais execuções possíveis
```

### 🔧 **CONFIGURAÇÃO TÉCNICA:**
```yaml
embeddings:
  model: "voyage-3.5-lite"
  cost_optimization:
    enable_sampling: true
    max_messages_per_dataset: 50000
    sampling_strategy: "strategic"
```

### 📋 **RELATÓRIOS AUTOMATIZADOS:**
- **Custo estimado por dataset**
- **Uso da cota gratuita**
- **Recomendações de otimização**
- **Métricas de economia em tempo real**

### 📊 ADAPTAÇÃO PARA DATASETS PRÉ-PROCESSADOS

**Feature Extractor Otimizado:**

**Colunas Existentes Esperadas:**
`datetime, body, url, hashtag, channel, is_fwrd, mentions, sender, media_type, domain, body_cleaned`

**Features NÃO re-extraídas (já existem):**
- ❌ URLs, hashtags, domínios - usa colunas existentes
- ❌ Emojis - já foram removidos dos datasets
- ❌ Detecção básica de mídia - usa `media_type`

**Features NOVAS extraídas:**
- ✅ **Análise semântica**: political_alignment, sentiment_category, discourse_type
- ✅ **Métricas textuais**: text_length, word_count, sentence_count
- ✅ **Análise de menções**: mention_count, mention_purpose, echo_chamber_score
- ✅ **Detecção de coordenação**: coordination_probability, bot_indicators
- ✅ **Contexto temporal**: hour_of_day, is_election_period, days_to_election
- ✅ **Risco e credibilidade**: misinformation_risk, violence_indicators

### 🔧 CONFIGURAÇÃO

Configure a chave da API no arquivo `config/settings.yaml`:

```yaml
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  model: "claude-3-5-haiku-20241022"
```