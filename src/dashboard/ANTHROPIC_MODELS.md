# 🤖 Modelos Anthropic Disponíveis no Dashboard

## 📋 **Lista Completa de Modelos (Atualizada 2025)**

O dashboard agora suporta **todos os modelos Claude** mais recentes da Anthropic:

### 🏆 **Claude 4 (Série Mais Recente)**

| Modelo | ID da API | Descrição | Uso Recomendado |
|--------|-----------|-----------|-----------------|
| **Claude Opus 4** | `claude-opus-4-20250514` | Modelo mais capaz | Análise complexa, tarefas críticas |
| **Claude Sonnet 4** | `claude-sonnet-4-20250514` | Alto desempenho balanceado | **Recomendado para o projeto** |

### 🧠 **Claude 3.7**

| Modelo | ID da API | Descrição | Característica Especial |
|--------|-----------|-----------|------------------------|
| **Claude 3.7 Sonnet** | `claude-3-7-sonnet-20250219` | Pensamento estendido | Extended thinking toggleável |

### ⚡ **Claude 3.5**

| Modelo | ID da API | Descrição | Performance |
|--------|-----------|-----------|-------------|
| **Claude 3.5 Sonnet v2** | `claude-3-5-sonnet-20241022` | Versão mais recente | Alta inteligência |
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20240620` | Versão anterior | Estável e confiável |
| **Claude 3.5 Haiku** | `claude-3-5-haiku-20241022` | Modelo mais rápido | Velocidade máxima |

### 💎 **Claude 3 (Legacy)**

| Modelo | ID da API | Descrição | Status |
|--------|-----------|-----------|--------|
| **Claude 3 Opus** | `claude-3-opus-20240229` | Modelo poderoso | Legacy, ainda funcional |
| **Claude 3 Sonnet** | `claude-3-sonnet-20240229` | Modelo balanceado | Legacy |
| **Claude 3 Haiku** | `claude-3-haiku-20240307` | Modelo rápido | Legacy |

### 🏷️ **Aliases (Atalhos)**

| Alias | Modelo Atual | Vantagem |
|-------|--------------|----------|
| `claude-opus-4-0` | `claude-opus-4-20250514` | Sempre aponta para a versão mais recente |
| `claude-sonnet-4-0` | `claude-sonnet-4-20250514` | Auto-atualização |
| `claude-3-7-sonnet-latest` | `claude-3-7-sonnet-20250219` | Conveniência |
| `claude-3-5-sonnet-latest` | `claude-3-5-sonnet-20241022` | Sempre atual |
| `claude-3-5-haiku-latest` | `claude-3-5-haiku-20241022` | Auto-update |
| `claude-3-opus-latest` | `claude-3-opus-20240229` | Legacy latest |

---

## 💰 **Preços por Modelo (USD/Million Tokens)**

### **Claude 4**
| Modelo | Input | Output | Melhor Para |
|--------|-------|--------|-------------|
| **Opus 4** | $15/MTok | $75/MTok | Análise crítica, máxima qualidade |
| **Sonnet 4** | $3/MTok | $15/MTok | **Custo-benefício ideal** |

### **Claude 3.7 & 3.5** 
| Modelo | Input | Output | Observação |
|--------|-------|--------|------------|
| **3.7 Sonnet** | $3/MTok | $15/MTok | Pensamento estendido |
| **3.5 Sonnet** | $3/MTok | $15/MTok | Versão estável |
| **3.5 Haiku** | $0.80/MTok | $4/MTok | **Mais econômico** |

### **Claude 3 (Legacy)**
| Modelo | Input | Output | Status |
|--------|-------|--------|--------|
| **3 Opus** | $15/MTok | $75/MTok | Legacy, caro |
| **3 Sonnet** | $3/MTok | $15/MTok | Legacy |
| **3 Haiku** | $0.25/MTok | $1.25/MTok | Legacy econômico |

---

## 🎯 **Recomendações por Caso de Uso**

### **📊 Para Análise de Dados Bolsonarismo**

#### **🥇 Opção Recomendada: Claude Sonnet 4**
- **ID**: `claude-sonnet-4-20250514`
- **Por quê**: Melhor custo-benefício para análise de dados
- **Custo**: $3/MTok input, $15/MTok output
- **Vantagens**: Alta qualidade, velocidade boa, preço justo

#### **🥈 Alternativa Econômica: Claude 3.5 Haiku**
- **ID**: `claude-3-5-haiku-20241022`
- **Por quê**: Mais barato para datasets grandes
- **Custo**: $0.80/MTok input, $4/MTok output
- **Vantagens**: Muito rápido, econômico

#### **🥉 Máxima Qualidade: Claude Opus 4**
- **ID**: `claude-opus-4-20250514`
- **Por quê**: Melhor qualidade absoluta
- **Custo**: $15/MTok input, $75/MTok output
- **Quando usar**: Análises críticas, pesquisa acadêmica

---

## 📈 **Estimativas de Custo para o Projeto**

### **Dataset Típico: 100K mensagens**

| Modelo | Tokens Estimados | Custo Input | Custo Total Est. |
|--------|------------------|-------------|------------------|
| **Sonnet 4** | ~50M tokens | $150 | **$300-400** |
| **Haiku 3.5** | ~50M tokens | $40 | **$80-120** |
| **Opus 4** | ~50M tokens | $750 | **$1,500-2,000** |

### **Dataset Grande: 1M mensagens**

| Modelo | Tokens Estimados | Custo Input | Custo Total Est. |
|--------|------------------|-------------|------------------|
| **Sonnet 4** | ~500M tokens | $1,500 | **$3,000-4,000** |
| **Haiku 3.5** | ~500M tokens | $400 | **$800-1,200** |
| **Opus 4** | ~500M tokens | $7,500 | **$15,000-20,000** |

---

## 🔧 **Como Configurar no Dashboard**

### **1. Acessar Configurações**
```
Dashboard → ⚙️ Configurações → Aba "API"
```

### **2. Selecionar Modelo**
- **Dropdown**: Lista todos os 16 modelos disponíveis
- **Default**: Claude Sonnet 4 (recomendado)
- **Informações**: Descrição automática do modelo selecionado

### **3. Configurar Tokens**
- **Max Tokens**: 1000-8192 (Claude 4 suporta mais)
- **Recomendado**: 4000 tokens para análises completas

### **4. Estimativa de Custo**
- **Slider**: Tamanho do dataset (1K → 1M mensagens)
- **Cálculo Automático**: Baseado no modelo selecionado
- **Preview**: Custo estimado em tempo real

---

## 💡 **Dicas de Uso**

### **🎯 Para Desenvolvimento/Teste**
- Use **Claude 3.5 Haiku** para testes rápidos
- Económico e suficientemente inteligente

### **📊 Para Produção**
- Use **Claude Sonnet 4** para equilíbrio ideal
- Qualidade profissional com custo controlado

### **🏆 Para Pesquisa Crítica**
- Use **Claude Opus 4** quando a qualidade é fundamental
- Máxima inteligência disponível

### **⚡ Para Datasets Enormes**
- **Otimize com sampling** (já implementado no pipeline)
- **Considere Haiku 3.5** para economia
- **Use aliases** para auto-atualização

---

## 🔄 **Atualização Automática**

O dashboard é atualizado automaticamente com:
- ✅ **Novos modelos** conforme lançados pela Anthropic
- ✅ **Preços atualizados** conforme mudanças de pricing
- ✅ **Funcionalidades novas** (como extended thinking)
- ✅ **Aliases dinâmicos** que apontam para versões mais recentes

---

## 🚀 **Status Atual**

**✅ 16 modelos Claude disponíveis**  
**✅ Informações completas de preço e características**  
**✅ Estimativas de custo em tempo real**  
**✅ Recomendações inteligentes por caso de uso**  
**✅ Interface intuitive para seleção**  

**🎉 Dashboard pronto com todos os modelos Claude 2024-2025!**