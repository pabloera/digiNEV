# 🚀 Novo Fluxo de Feature Extraction - Pipeline v4.6

## 📋 **Visão Geral**

O sistema de extração de features foi completamente reestruturado em **duas etapas distintas** para maior eficiência e precisão:

- **01b: Feature Validation** - Validação e enriquecimento básico (local)
- **01c: Political Analysis** - Análise política profunda (via API Anthropic)

---

## 🔄 **Novo Fluxo do Pipeline**

```
Dados → 01_validate_data → 02b_deduplication → 
01b_feature_validation → 01c_political_analysis → 03_clean_text → ...
```

### **Etapas Detalhadas:**

1. **01_validate_data**: Validação estrutural e encoding
2. **02b_deduplication**: Deduplicação inteligente
3. **🆕 01b_feature_validation**: Validação de features existentes + enriquecimento básico
4. **🆕 01c_political_analysis**: Análise política profunda via API
5. **03_clean_text**: Limpeza de texto (usa dados de 01c)

---

## 🔧 **Etapa 01b: Feature Validation**

### **Objetivo**
Validar features já existentes e adicionar enriquecimentos básicos **sem duplicação**.

### **Módulo**: `feature_validator.py`

### **Funcionalidades**:

#### **✅ Validação Inteligente**
- **hashtags**: Valida se existem, corrige formato (adiciona # se necessário)
- **urls**: Valida existência, extrai domínios se não existirem
- **media_type**: Revisa baseado em conteúdo real de `body` e `body_cleaned`

#### **📊 Enriquecimento Básico**
- **Métricas de texto**: comprimento, palavras, emojis, caps ratio
- **Padrões estruturais**: mensagens encaminhadas, menções (@usuario)
- **Qualidade**: flags de spam, mensagens muito curtas/longas
- **Links Telegram**: detecção de links t.me

#### **🚫 Não Duplica**
- Se `hashtag` ou `hashtags` existem → **não extrai novamente**
- Se `url` ou `urls` existem → **não extrai novamente** 
- Se `media_type` existe → **não cria** `has_photo`, `has_video`, `has_audio`

### **Saída**: `data/interim/01b_features_validated/`

---

## 🏛️ **Etapa 01c: Political Analysis**

### **Objetivo**
Análise política profunda e contextualizada do discurso brasileiro.

### **Módulo**: `political_analyzer.py`

### **Funcionalidades**:

#### **🤖 Via API Anthropic**
- **Alinhamento político**: bolsonarista/antibolsonarista/neutro
- **Teorias conspiratórias**: Detecção baseada em contexto 2019-2023
- **Negacionismo**: COVID, vacinas, urnas eletrônicas
- **Tom emocional**: raiva/medo/esperança/tristeza/alegria
- **Sinais de coordenação**: linguagem padronizada, hashtags coordenadas
- **Risco de desinformação**: baixo/médio/alto

#### **📚 Fallback Tradicional**
- **Análise léxica**: Baseada em `brazilian_political_lexicon.yaml`
- **Padrões conhecidos**: Governo Bolsonaro, oposição, militarismo
- **Detecção rápida**: Para casos sem API disponível

#### **💾 Cache Inteligente**
- **Hash MD5**: Evita reprocessar textos idênticos
- **Performance**: Acelera análises de mensagens repetidas

### **Saída**: `data/interim/01c_politically_analyzed/`

---

## 🗂️ **Léxico Político Brasileiro**

### **Arquivo**: `config/brazilian_political_lexicon.yaml`

### **Categorias Incluídas**:
- **governo_bolsonaro**: presidente, capitão, mito, patriota
- **oposição**: lula, pt, esquerda, comunista
- **militarismo**: forças armadas, militares, intervenção militar, quartel ⭐
- **teorias_conspiração**: urna fraudada, globalismo, deep state
- **saúde_negacionismo**: tratamento precoce, ivermectina, cloroquina
- **mobilização**: acordem, despertem, manifestação
- **indicadores_emocionais**: raiva, medo, esperança, urgência

---

## 📈 **Vantagens do Novo Sistema**

### **1. Eficiência**
- ✅ Não duplica extrações já existentes
- ✅ Processamento local para validações básicas
- ✅ API apenas para análise que agrega valor real

### **2. Modularidade**
- ✅ Etapas independentes e especializadas
- ✅ Fallbacks robustos quando API indisponível
- ✅ Fácil manutenção e evolução

### **3. Precisão**
- ✅ Prompts contextualizados para realidade brasileira
- ✅ Léxico especializado em discurso político 2019-2023
- ✅ Validação de qualidade das respostas da API

### **4. Performance**
- ✅ Cache inteligente evita reprocessamento
- ✅ Lotes otimizados (10 textos para análise política)
- ✅ Processamento paralelo quando possível

---

## 🔧 **Configuração**

### **settings.yaml**
```yaml
# Etapa 01b: Feature Validation
feature_validation:
  use_anthropic: false  # Local processing
  validate_existing: true
  enrich_basic: true

# Etapa 01c: Political Analysis
political_analysis:
  use_anthropic: true  # API required for best results
  batch_size: 10
  confidence_threshold: 0.7
  use_cache: true
```

---

## 🧪 **Como Testar**

### **1. Executar Pipeline Completo**
```bash
python run_pipeline.py
```

### **2. Verificar Etapas Específicas**
```python
# Teste de validação de features
from src.anthropic_integration.feature_validator import FeatureValidator
validator = FeatureValidator()
enriched_df, report = validator.validate_and_enrich_features(df)

# Teste de análise política
from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
analyzer = PoliticalAnalyzer()
analyzed_df, report = analyzer.analyze_political_discourse(df)
```

### **3. Verificar Arquivos de Saída**
- `data/interim/01b_features_validated/` - Features validadas
- `data/interim/01c_politically_analyzed/` - Análise política

---

## ⚠️ **Compatibilidade**

### **Retrocompatibilidade**
- ✅ Pipeline antigo (`01b_feature_extraction`) mantido como fallback
- ✅ Configurações existentes continuam funcionando
- ✅ Estrutura de dados preservada

### **Migração Gradual**
- ✅ Novo sistema ativo por padrão
- ✅ Rollback possível via configuração
- ✅ Logs claros indicam qual sistema está sendo usado

---

## 🎯 **Próximos Passos**

1. **Monitorar performance** das novas etapas
2. **Coletar feedback** sobre qualidade da análise política
3. **Refinar léxico** baseado em resultados reais
4. **Otimizar cache** para datasets grandes
5. **Adicionar métricas** de eficiência

---

## 📝 **Changelog**

### **v4.6 - Janeiro 2025**
- ✅ Divisão em duas etapas especializadas
- ✅ Feature validation sem duplicação
- ✅ Análise política contextualizada
- ✅ Léxico político brasileiro completo
- ✅ Cache inteligente implementado
- ✅ Fallbacks robustos para estabilidade

---

*Este documento reflete as melhorias implementadas baseadas na análise profunda dos requisitos e melhores práticas da API Anthropic.*