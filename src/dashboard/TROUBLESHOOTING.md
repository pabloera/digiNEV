# 🔧 Troubleshooting do Dashboard

## ❌ Problema: "Connection error - Is Streamlit still running?"

### Causa
Este erro geralmente ocorre quando:
1. O processo Streamlit foi interrompido
2. Há problemas de importação de módulos
3. A porta está sendo usada por outro processo

### ✅ Soluções

#### Solução 1: Usar a versão corrigida
```bash
streamlit run src/dashboard/app_fixed.py --server.port 8503
```

#### Solução 2: Matar processos existentes
```bash
# Encontrar processos streamlit
ps aux | grep streamlit

# Matar processos (substitua PID pelo número do processo)
kill -9 PID

# Ou matar todos os processos streamlit
pkill -f streamlit
```

#### Solução 3: Usar porta diferente
```bash
streamlit run src/dashboard/app_fixed.py --server.port 8504
```

#### Solução 4: Limpar cache do Streamlit
```bash
# Limpar cache
streamlit cache clear

# Remover diretório de cache
rm -rf ~/.streamlit/
```

### 🎯 Dashboard Recomendado

Use sempre a versão **`app_fixed.py`** que tem:
- ✅ Tratamento de erros de importação
- ✅ Modo demo funcional
- ✅ Verificação de disponibilidade do pipeline
- ✅ Interface simplificada mas completa

### 🚀 Como Executar

```bash
# Do diretório do projeto
cd /Users/pabloalmada/development/project/dataanalysis-bolsonarismo

# Executar dashboard
streamlit run src/dashboard/app_fixed.py --server.port 8503

# Acessar no navegador
# http://localhost:8503
```

### 📊 Funcionalidades Disponíveis

1. **Upload & Processamento**: Upload múltiplo de CSVs
2. **Visão Geral**: Métricas e gráficos consolidados  
3. **Análise por Etapa**: Visualizações específicas
4. **Comparação**: Compare diferentes datasets
5. **Sobre**: Informações do projeto

### 🎭 Modo Demo

Se o pipeline não estiver disponível, o dashboard automaticamente:
- Detecta a indisponibilidade
- Oferece modo demo com dados simulados
- Mantém todas as visualizações funcionais

### 🔍 Debug

Para verificar problemas:

```bash
# Verificar porta em uso
lsof -i :8503

# Verificar processos Python
ps aux | grep python

# Logs do Streamlit
streamlit run app_fixed.py --logger.level debug
```

### 📝 URLs de Acesso

Quando o dashboard estiver rodando, estará disponível em:
- **Local**: http://localhost:8503
- **Rede**: http://IP_DA_REDE:8503

Pressione **Ctrl+C** no terminal para parar o servidor.