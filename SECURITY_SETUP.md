# 🔒 Configuração de Segurança - Chaves de API

## ⚠️ IMPORTANTE: NUNCA COMMITE CHAVES DE API!

Este projeto utiliza APIs externas que requerem autenticação. Siga estas instruções para configurar com segurança:

## 🔧 Configuração Inicial

### 1. Copie o template de configuração:
```bash
cp .env.template .env
```

### 2. Configure suas chaves no arquivo `.env`:

#### Anthropic API (obrigatória)
1. Acesse: https://console.anthropic.com/
2. Crie uma conta ou faça login
3. Gere uma API key
4. Substitua `[SUA_CHAVE_ANTHROPIC_AQUI]` no arquivo `.env`

#### Voyage.ai API (opcional - para análise semântica)
1. Acesse: https://www.voyageai.com/
2. Crie uma conta
3. Gere uma API key
4. Substitua `[SUA_CHAVE_VOYAGE_AQUI]` no arquivo `.env`

#### Pinecone API (opcional - para armazenamento vetorial)
1. Acesse: https://www.pinecone.io/
2. Crie uma conta
3. Gere uma API key
4. Descomente e substitua `[SUA_CHAVE_PINECONE_AQUI]` no arquivo `.env`

## 🛡️ Medidas de Segurança Implementadas

- ✅ Arquivo `.env` incluído no `.gitignore`
- ✅ Template `.env.template` sem chaves reais
- ✅ Placeholders seguros em todos os arquivos
- ✅ Avisos de segurança na documentação

## ❌ O QUE NUNCA FAZER

- ❌ Commitar o arquivo `.env` com chaves reais
- ❌ Incluir chaves de API em código fonte
- ❌ Compartilhar chaves em issues ou PRs
- ❌ Usar chaves em arquivos de configuração commitados

## ✅ BOAS PRÁTICAS

- ✅ Use variáveis de ambiente
- ✅ Mantenha chaves em arquivo `.env` local
- ✅ Use diferentes chaves para dev/prod
- ✅ Revogue chaves expostas imediatamente
- ✅ Monitore uso das APIs regularmente

## 🚨 SE CHAVES FORAM EXPOSTAS

1. **Revogue imediatamente** nas respectivas plataformas
2. **Gere novas chaves**
3. **Atualize seu arquivo `.env` local**
4. **Verifique logs** para uso indevido

## 📞 Suporte

Se tiver dúvidas sobre configuração de segurança, consulte:
- Documentação oficial das APIs
- Canal de suporte do projeto
- Issues do GitHub (SEM incluir chaves!)

---

**🔒 Segurança em primeiro lugar! Proteja suas chaves de API como senhas.**