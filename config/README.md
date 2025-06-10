# 📁 Configurações do Pipeline v4.9.1

## 🔧 Arquivos Principais

### **Configuração Base**
- `settings.yaml` - Configuração principal do pipeline
- `logging.yaml` - Configuração de logs estruturada
- `processing.yaml` - Parâmetros de processamento (v4.9.1)
- `timeout_management.yaml` - Sistema de timeout inteligente

### **APIs e Integração**
- `anthropic.yaml.template` - Template configuração Anthropic
- `voyage_embeddings.yaml` - Configuração Voyage.ai (otimizada)

### **Dados Específicos**
- `brazilian_political_lexicon.yaml` - Léxico político brasileiro

## 🚀 Setup Rápido

```bash
# 1. Copiar template Anthropic
cp anthropic.yaml.template anthropic.yaml

# 2. Configurar APIs no .env (raiz do projeto)
echo "ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE]" > ../.env
echo "VOYAGE_API_KEY=pa-[SUA_CHAVE]" >> ../.env

# 3. voyage_embeddings.yaml já está configurado e otimizado
# Verifique se as configurações atendem suas necessidades
```

---
**Referência**: Configuração completa no [README.md](../README.md#configuração-completa)