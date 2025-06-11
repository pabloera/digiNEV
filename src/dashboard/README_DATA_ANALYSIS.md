# Dashboard de Análise de Dados - Pipeline Bolsonarismo v4.9.7

## 🎯 Visão Geral

Dashboard **completamente redesenhado** para focar exclusivamente na apresentação dos **RESULTADOS das análises de dados** geradas pelos stages do pipeline. Não é mais um dashboard de monitoramento técnico, mas sim uma ferramenta de **análise e insights** sobre o discurso político brasileiro.

## 🔄 Mudança de Paradigma

### ❌ **ANTES (Dashboard de Monitoramento):**
- Foco no pipeline e sua execução
- Métricas de performance técnica
- Monitoramento de custos de API
- Status de stages e processos
- Alertas técnicos e logs

### ✅ **AGORA (Dashboard de Análise de Dados):**
- Foco nos **resultados das análises**
- Insights sobre **discurso político brasileiro**
- Visualizações de **conteúdo e padrões**
- Descobertas sobre **comportamento político**
- Análises **comparativas e temporais**

## 📊 Páginas de Análise Disponíveis

### 1. 📋 **Visão Geral - Análise Comparativa**
- **Volume de mensagens**: Original vs Deduplicated
- **Top 10 hashtags**: Antes vs Depois da limpeza
- **Top 10 menções**: Antes vs Depois da limpeza
- **Top 10 domínios**: Antes vs Depois da limpeza
- **Resumo das transformações** aplicadas pelo pipeline

### 2. 🏛️ **Análise Política Hierárquica (4 Níveis)**
- **Nível 1**: Classificação política básica (político/não-político)
- **Nível 2**: Alinhamento político (bolsonarista/antibolsonarista/neutro/indefinido)
- **Nível 3**: Categorias temáticas (negacionismo/autoritarismo/deslegitimação/mobilização/conspiração/informativo)
- **Nível 4**: Subcategorias detalhadas e agrupamentos semânticos
- **Correlações hierárquicas** entre os 4 níveis
- **Evolução temporal** das categorias políticas
- **Análise multidimensional** com visualizações interativas (sunburst, heatmaps)
- **Densidade de entidades políticas** por categoria
- **Características textuais** por alinhamento político

### 3. 😊 **Análise de Sentimento**
- Distribuição geral de **sentimentos**
- Análise de **scores de sentimento**
- Sentimento por **categoria política**
- Evolução temporal dos **sentimentos**

### 4. 💬 **Análise do Discurso**
- Tipos de **discurso identificados**
- Comprimento médio por **tipo de discurso**
- Padrões de **comunicação política**

### 5. 📅 **Análise Temporal**
- Atividade por **hora do dia**
- Padrões por **dia da semana**
- **Sazonalidade** do discurso político
- Evolução histórica

### 6. 🔤 **Análise Linguística**
- Distribuição de **comprimento das mensagens**
- Análise de **número de palavras**
- **Complexidade linguística** (via spaCy)
- Métricas de **diversidade lexical**

### 7. 🔍 **Análise de Agrupamentos**
- **Clusters semânticos** identificados
- Distribuição por **grupos temáticos**
- **Qualidade semântica** dos agrupamentos
- Padrões de **similaridade**

### 8. 🌐 **Análise de Redes**
- Análise de **menções** entre usuários
- Padrões de **hashtags**
- Compartilhamento de **URLs**
- **Interações** e conectividade

### 9. ⚖️ **Análise Comparativa**
- **Heatmaps** de correlações
- Comparações entre **dimensões** (política × sentimento)
- **Estatísticas comparativas**
- **Cross-analysis** multidimensional

## 🚀 Como Executar

### **Método 1: Script Automático**
```bash
# Executar o dashboard de análise
poetry run python src/dashboard/start_data_analysis.py
```

### **Método 2: Streamlit Direto**
```bash
# Executar diretamente via streamlit
poetry run streamlit run src/dashboard/data_analysis_dashboard.py --server.port 8503
```

### **Acesso:**
- **URL:** http://localhost:8503
- **Navegação:** Menu lateral com 9 seções de análise
- **Dados:** Carregamento automático do dataset processado

## 📁 Estrutura do Novo Dashboard

```
src/dashboard/
├── data_analysis_dashboard.py     # Dashboard principal de análise
├── start_data_analysis.py         # Script de inicialização
├── README_DATA_ANALYSIS.md        # Esta documentação
├── app.py                         # Dashboard antigo (monitoramento)
└── start_dashboard.py            # Script antigo
```

## 📊 Fonte de Dados

### **Dataset Principal:**
- **Arquivo:** `data/interim/sample_dataset_v495_19_pipeline_validated.csv`
- **Colunas:** 64 colunas processadas
- **Registros:** 300 mensagens analisadas
- **Período:** 2019-2021 (Governo Bolsonaro)

### **Colunas de Análise Utilizadas:**
- **Política:** `political_category`, `political_alignment`, `radicalization_level`
- **Sentimento:** `sentiment`, `sentiment_score`, `confidence`
- **Discurso:** `discourse_type`, `text_length`, `word_count`
- **Linguística:** `spacy_*` (13 features do spaCy)
- **Clustering:** `cluster_name`, `semantic_quality`
- **Temporal:** `datetime`, `timestamp`
- **Redes:** `mentions`, `hashtag`, `url`

## 🎨 Visualizações Implementadas

### **Tipos de Gráficos:**
- **📊 Barras:** Distribuições categóricas
- **🥧 Pizza:** Proporções e percentuais
- **📈 Linhas:** Evolução temporal
- **📋 Heatmaps:** Correlações e comparações
- **📊 Histogramas:** Distribuições numéricas
- **📊 Áreas:** Evolução temporal empilhada

### **Paletas de Cores:**
- **Política:** Azuis (`Blues`, `Set3`)
- **Sentimento:** Verde/Vermelho (`RdYlBu_r`)
- **Temporal:** Viridis, Plasma
- **Comparativo:** Divergentes (`RdYlBu_r`)

## 💡 Insights Automáticos

O dashboard gera automaticamente insights como:

- **Categoria política dominante** e percentual
- **Sentimento predominante** na comunicação
- **Tipo de discurso principal** identificado
- **Comprimento médio** das mensagens
- **Número de grupos temáticos** descobertos

## 🔧 Personalização e Extensão

### **Adicionar Nova Análise:**
1. Criar método `_render_nova_analise_page()`
2. Adicionar à navegação em `_render_navigation()`
3. Incluir no switch em `run()`

### **Adicionar Novo Gráfico:**
1. Usar Plotly Express (`px`) ou Graph Objects (`go`)
2. Seguir padrões de cores estabelecidos
3. Incluir títulos e labels descritivos

### **Adicionar Nova Métrica:**
1. Calcular a métrica nos dados (`self.df`)
2. Usar `st.metric()` para exibição
3. Adicionar contexto e interpretação

## ⚠️ Limitações e Considerações

### **Dados Requeridos:**
- O dashboard requer que o **pipeline tenha sido executado**
- Sem dados, mostra página de instruções
- Algumas análises dependem de colunas específicas

### **Performance:**
- Otimizado para datasets de **até 10K registros**
- Para datasets maiores, considerar **sampling**
- Gráficos complexos podem ser lentos

### **Compatibilidade:**
- Requer **Python 3.12+**
- Dependências: **streamlit**, **plotly**, **pandas**
- Testado no **macOS** e **Linux**

## 🔮 Próximas Funcionalidades

### **v5.0 (Planejado):**
- 📱 **Dashboard Mobile:** Interface responsiva
- 🔍 **Filtros Avançados:** Filtros por período, categoria, etc.
- 📊 **Exportação:** PDF, PNG, CSV dos gráficos
- 🤖 **IA Insights:** Insights automáticos com LLM
- 📈 **Benchmarking:** Comparação com outros períodos

### **v5.1 (Futuro):**
- 🌎 **Dados Geográficos:** Análise por região
- 📺 **Análise de Canais:** Insights por canal do Telegram
- 🎯 **Segmentação Avançada:** Clustering de usuários
- 📊 **Métricas Avançadas:** Influência, viralidade, etc.

## 📞 Suporte e Contribuições

### **Desenvolvido por:**
- **Pablo Emanuel Romero Almada, Ph.D.**
- **Projeto:** Análise do Discurso Político Brasileiro
- **Período:** 2019-2021 (Governo Bolsonaro)

### **Contribuições:**
1. **Issues:** Reporte bugs ou sugira análises
2. **Pull Requests:** Implemente novas visualizações
3. **Documentação:** Melhore esta documentação
4. **Dados:** Sugira novas fontes ou períodos

---

## 🎯 Filosofia do Dashboard

> **"Transformar dados em insights, insights em conhecimento, conhecimento em compreensão do comportamento político brasileiro."**

Este dashboard foi redesenhado para ser uma ferramenta de **descoberta e análise**, não de monitoramento técnico. O foco está em **responder perguntas** sobre o discurso político brasileiro, **identificar padrões** de comportamento e **gerar insights** acionáveis para pesquisadores, jornalistas e analistas políticos.

**Status:** ✅ **Produção - Totalmente Funcional** (v4.9.7)