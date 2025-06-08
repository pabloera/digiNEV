# 📁 Configurações Centralizadas

Este diretório contém **todas as configurações** do projeto Bolsonarismo de forma centralizada.

## 🎯 **Estrutura Consolidada (Janeiro 2025)**

A partir de **06/01/2025**, todas as configurações foram consolidadas neste diretório único:

```
config/
├── README.md                        # Este arquivo
├── anthropic.yaml.template          # Template para configuração Anthropic
├── logging.yaml                     # Configuração de logs
├── processing.yaml                  # Configuração do pipeline
├── settings.yaml                    # Configurações gerais
├── voyage_embeddings.yaml           # Configuração ativa Voyage.ai
├── voyage_embeddings.yaml.template  # Template Voyage.ai
├── cost_optimization_guide.md       # Guia de otimização de custos
├── voyage_pricing_analysis.md       # Análise de preços Voyage.ai
└── timeline_bolsonaro.md           # Timeline política (contexto)
```

## ✅ **Mudanças Realizadas**

- ❌ **Removido:** `src/config/` (pasta duplicada desatualizada)
- ✅ **Mantido:** `config/` (pasta raiz atualizada e completa)
- 🔧 **Atualizado:** `system_validator.py` para usar apenas `config/`

## 📋 **Arquivos Principais**

### **settings.yaml**
Configurações gerais do projeto incluindo:
- Configurações de API (Anthropic/Voyage)
- Parâmetros de processamento
- Estrutura de diretórios

### **processing.yaml**
Configurações específicas do pipeline:
- Parâmetros de cada etapa
- Configurações de deduplicação
- Estatísticas de dataset

### **logging.yaml**
Configuração de logs:
- Níveis de log por módulo
- Formatação de saída
- Arquivos de log

### **voyage_embeddings.yaml**
Configuração **ativa** do Voyage.ai:
- Otimização de custos habilitada
- Amostragem inteligente (50K mensagens max)
- Modelo voyage-3.5-lite (econômico)

## 🔧 **Como Usar**

### **1. Configuração Básica**
```bash
# Copiar templates
cp config/anthropic.yaml.template config/anthropic.yaml
cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml.custom

# Editar com suas credenciais
nano config/anthropic.yaml
```

### **2. Referências no Código**
O código sempre busca configurações em:
```python
config_path = "config/settings.yaml"  # ✅ Correto
```

### **3. Validação**
O sistema valida automaticamente se os arquivos existem:
```python
from anthropic_integration.system_validator import SystemValidator
validator = SystemValidator()
validator.validate_config_files()
```

## 💰 **Otimização de Custos**

O arquivo `voyage_embeddings.yaml` já está configurado para **máxima economia**:

- ✅ **Amostragem ativada:** máximo 50K mensagens por dataset
- ✅ **Modelo econômico:** voyage-3.5-lite (200M tokens grátis)
- ✅ **Batch otimizado:** 128 mensagens por requisição
- ✅ **Filtros ativos:** apenas conteúdo político relevante

**Economia estimada:** 90-95% dos custos originais

## 📚 **Documentação Adicional**

- `cost_optimization_guide.md` - Guia completo de otimização
- `voyage_pricing_analysis.md` - Análise detalhada de custos
- `timeline_bolsonaro.md` - Contexto político para análise

## 🔄 **Histórico de Consolidação**

- **Antes:** Duas pastas (`config/` e `src/config/`) com duplicação
- **Depois:** Uma pasta centralizada (`config/`) com tudo atualizado
- **Benefícios:** Menor confusão, melhor manutenção, estrutura mais limpa

---
**✅ Estrutura consolidada e testada - Janeiro 2025**