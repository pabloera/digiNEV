# 📁 Configurações do Pipeline v5.0.0

## 🏗️ **ESTRUTURA HIERÁRQUICA v5.0.0**

### **🎯 Configuração Principal**
- `settings.yaml` - **Configuração mestre** com referências hierárquicas
- `core.yaml` - **Configurações essenciais** consolidadas (modelos, processamento, qualidade)
- `master.yaml` - **Configuração legada** (mantida para compatibilidade)

### **🌍 Ambientes Específicos** 
- `environments/development.yaml` - **Desenvolvimento** (dados reduzidos, logs verbose)
- `environments/production.yaml` - **Produção** (qualidade máxima, processamento completo)  
- `environments/testing.yaml` - **Testes** (dados controlados, execução determinística)

### **⚙️ Componentes Específicos**
- `api_limits.yaml` - Limites e configurações de APIs
- `logging.yaml` - Configuração de logs estruturada
- `processing.yaml` - Parâmetros de processamento (v5.0.0)
- `timeout_management.yaml` - Sistema de timeout inteligente
- `paths.yaml` - Estrutura de diretórios e caminhos
- `network.yaml` - Configurações de rede e dashboard

### **🔑 Templates de APIs**
- `anthropic.yaml.template` - Template configuração Anthropic
- `voyage_embeddings.yaml.template` - Template configuração Voyage.ai (otimizada)

### **📚 Dados Específicos**
- `brazilian_political_lexicon.yaml` - Léxico político brasileiro

## 🚀 **Setup por Ambiente**

### **🔧 Configuração Básica**
```bash
# 1. Definir ambiente (development | production | testing)
export BOLSONARISMO_ENV=development

# 2. Configurar APIs no .env (raiz do projeto)
echo "ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE]" > ../.env
echo "VOYAGE_API_KEY=pa-[SUA_CHAVE]" >> ../.env

# 3. Copiar templates necessários
cp anthropic.yaml.template anthropic.yaml
cp voyage_embeddings.yaml.template voyage_embeddings.yaml
```

### **🌍 Ambientes Disponíveis**
```bash
# DESENVOLVIMENTO - Dados reduzidos, execução rápida
export BOLSONARISMO_ENV=development

# PRODUÇÃO - Qualidade máxima, processamento completo  
export BOLSONARISMO_ENV=production

# TESTES - Dados controlados, execução determinística
export BOLSONARISMO_ENV=testing
```

### **📋 Hierarquia de Configuração**
```
1. Environment Variables (ANTHROPIC_API_KEY, etc.) [HIGHEST]
2. environments/{environment}.yaml (overrides específicos)
3. settings.yaml (configuração mestre)
4. core.yaml + component files (api_limits.yaml, etc.)  
5. master.yaml (configuração base) [LOWEST]
```

---
**Referência**: Configuração completa no [README.md](../README.md#configuração-completa)