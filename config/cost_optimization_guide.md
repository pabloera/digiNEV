# Guia de Otimização de Custos Voyage.ai
## Implementação Ativa - Janeiro 2025

### 🎯 **STATUS: IMPLEMENTADO E ATIVADO**

A otimização de custos foi completamente implementada e está **ATIVA** no sistema.

## 📊 **Redução de Custos Implementada**

### Cenário ANTES:
- **Mensagens processadas:** 1.3M (100%)
- **Custo estimado:** $36-60 USD
- **Processamento:** Completo sem filtros

### Cenário DEPOIS (Implementado):
- **Mensagens processadas:** ~50K por dataset (3.8%)
- **Custo estimado:** $1.5-3 USD
- **Redução de custos:** **90-95%**

## ⚙️ **Configurações Ativas**

### 1. Amostragem Inteligente
```yaml
cost_optimization:
  enable_sampling: true              # ✅ ATIVADO
  max_messages_per_dataset: 50000    # Limite por dataset
  sampling_strategy: "strategic"     # Amostragem estratégica
  min_text_length: 50               # Mínimo 50 caracteres
  require_political_keywords: true   # ✅ Apenas conteúdo político
```

### 2. Estratégia de Amostragem Estratégica
- **70%** mensagens de alta importância (hashtags, menções, palavras-chave)
- **30%** amostra aleatória para diversidade
- **Score composto** baseado em:
  - Comprimento do texto (30%)
  - Número de hashtags (20%)
  - Número de menções (20%)
  - Palavras-chave políticas (30%)

### 3. Períodos Temporais Otimizados
```yaml
key_periods:
  - 2019 Q1: 10% sample rate (Início governo Bolsonaro)
  - 2020 Mar-Jun: 30% sample rate (COVID-19)
  - 2022 Out-Dez: 50% sample rate (Eleições)
  - 2023 Jan: 40% sample rate (8 de Janeiro)
```

### 4. Pipeline Otimizado
```yaml
integration:
  deduplication: false      # ✅ DESABILITADO (economia)
  topic_modeling: true      # ✅ MANTIDO (qualidade)
  clustering: true          # ✅ MANTIDO (descoberta)
  tfidf_analysis: false     # ✅ DESABILITADO (economia)
```

## 🔧 **Implementações Técnicas**

### 1. Amostragem Automática
- `apply_cost_optimized_sampling()` implementado
- Filtros automáticos por qualidade e relevância
- Extensão inteligente para dataset completo

### 2. Configuração Otimizada
- `batch_size`: 8 → 128 (melhor throughput)
- `similarity_threshold`: 0.8 → 0.75 (performance)
- Cache ativado para reutilização

### 3. Relatórios de Economia
- Métricas de custo em todos os relatórios
- Sample ratio tracking
- Estimativas de economia em tempo real

## 📈 **Qualidade Mantida**

### Análises Preservadas:
- ✅ **Topic modeling semântico** (alta qualidade)
- ✅ **Clustering de padrões** (descoberta)
- ✅ **Extensão para dataset completo** (inferência)

### Análises Otimizadas:
- ❌ **Deduplicação semântica** → Usa métodos tradicionais
- ❌ **TF-IDF semântico** → Usa TF-IDF tradicional
- ✅ **Análise semântica principal** → Amostra + extensão

## 🚀 **Como Funciona na Prática**

1. **Carrega dataset completo** (1.3M mensagens)
2. **Aplica filtros de qualidade** (remove spam, textos curtos)
3. **Filtra por relevância política** (apenas conteúdo político)
4. **Amostragem estratégica** (50K mensagens mais importantes)
5. **Análise Voyage.ai** (apenas na amostra)
6. **Extensão inteligente** (inferência para dataset completo)
7. **Resultado final** (insights para todas as mensagens)

## 💰 **Monitoramento de Custos**

### Relatórios Incluem:
- `sampling_enabled`: true/false
- `sample_ratio`: 0.038 (3.8%)
- `original_messages`: 1300000
- `processed_messages`: 50000
- `cost_reduction_estimate`: "96.2%"

## 📝 **Próximos Passos**

1. ✅ **Configuração ativada** - `voyage_embeddings.yaml`
2. ✅ **Pipeline implementado** - Otimizações no `unified_pipeline.py`
3. ✅ **Métodos de amostragem** - `voyage_embeddings.py`
4. 🔄 **Pronto para execução** - Execute `python run_pipeline.py`

---

**Resultado:** Sistema otimizado que reduz custos em 90-95% mantendo qualidade analítica alta através de amostragem inteligente e extensão por inferência.