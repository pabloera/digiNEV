# 🔒 Configuração de Segurança - APIs

## ⚠️ IMPORTANTE: NUNCA COMMITE CHAVES DE API!

## 🔧 Setup Rápido

### 1. Configurar Variáveis de Ambiente
```bash
# Criar arquivo .env (OBRIGATÓRIO)
echo "ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE_AQUI]" > .env
echo "VOYAGE_API_KEY=pa-[SUA_CHAVE_AQUI]" >> .env

# Verificar configuração
python -c "import os; print('✅ APIs configuradas') if os.getenv('ANTHROPIC_API_KEY') and os.getenv('VOYAGE_API_KEY') else print('❌ APIs não configuradas')"
```

### 2. Copiar Templates de Configuração
```bash
# Templates necessários
cp config/anthropic.yaml.template config/anthropic.yaml
cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml
```

## 🔑 APIs Necessárias

### Anthropic API
- **Modelo**: claude-3-5-haiku-20241022
- **Uso**: Stages 05, 08, 12-18, 20 (API-only)
- **Custo**: Otimizado com 96% economia

### Voyage.ai API  
- **Modelo**: voyage-3.5-lite
- **Uso**: Stages 09-11, 19 (embeddings semânticos)
- **Custo**: Batch otimizado 128 vs 8

## 🛡️ Segurança

### Arquivos que NUNCA devem ser commitados:
- `.env` (já no .gitignore)
- `config/anthropic.yaml` (se contém chaves)
- `config/voyage_embeddings.yaml` (se contém chaves)

### Verificação de Segurança:
```bash
# Verificar se .env está no .gitignore
grep -q "\.env" .gitignore && echo "✅ .env protegido" || echo "❌ .env não protegido"
```

---
**Referência**: Configuração completa documentada no [README.md](README.md#configuração-completa)