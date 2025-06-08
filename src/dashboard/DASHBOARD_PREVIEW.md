# 📊 Dashboard Bolsonarismo - Preview da Interface

## 🏠 **Página Principal (Visão Atual)**

### **Header Principal**
```
🔬 Dashboard de Análise - Projeto Bolsonarismo
    Pipeline de Processamento com Integração Anthropic
```

### **Status de Sistema** 
```
✅ Todas as visualizações avançadas disponíveis
🔑 Chave da API Anthropic não configurada  [se não configurada]
🔑 Chave da API Voyage.ai não configurada   [se não configurada]
```

---

## 📋 **Sidebar - Menu Principal**

```
📋 Menu Principal

📤 Upload & Processamento       ← [SELECIONADO]
📊 Visão Geral  
🔍 Análise por Etapa
📈 Comparação de Datasets
🔎 Busca Semântica
⚙️ Configurações

───────────────────────────────

📊 Status do Processamento
[Vazio até arquivos serem processados]
```

---

## 📤 **Página Atual: Upload & Processamento**

### **Seção Principal (Coluna Esquerda)**

```
📤 Upload de Datasets

┌─────────────────────────────────────────────────────┐
│  📁 Arraste seus arquivos CSV aqui ou clique       │
│     para selecionar                                 │
│                                                     │
│  📋 Selecione um ou mais arquivos CSV para         │
│     análise em massa                                │
│                                                     │
│  [Clique aqui ou arraste arquivos]                 │
└─────────────────────────────────────────────────────┘

[Área aparecerá quando arquivos forem selecionados:]
✅ 3 arquivo(s) válido(s) pronto(s) para processamento
```

### **Seção Configurações (Coluna Direita)**

```
Total de Arquivos: 0

⚙️ Configurações

☑️ Usar Integração Anthropic

Tamanho do Chunk: [10000]

📋 Etapas do Pipeline
☑️ Executar todas as etapas

[Se desmarcar "todas", aparece lista:]
☐ 01_validate_data
☐ 02b_deduplication  
☐ 01b_feature_extraction
☐ 03_clean_text
☐ 04_sentiment_analysis
☐ 05_topic_modeling
☐ 06_tfidf_extraction
☐ 07_clustering
☐ 08_hashtag_normalization
☐ 09_domain_extraction
☐ 10_temporal_analysis
☐ 11_network_structure
☐ 12_qualitative_analysis
☐ 13_review_reproducibility
☐ 14_semantic_search_intelligence

🚀 [Iniciar Processamento] [DESABILITADO]
🎭 [Executar Modo Demo]    [DESABILITADO]
```

---

## 📊 **Preview: Página Visão Geral** (Após Processamento)

```
📊 Visão Geral dos Resultados

┌─────────────┬─────────────┬─────────────┬─────────────┐
│📁 Arquivos  │📝 Total de  │✅ Taxa de   │⏱️ Tempo    │
│Processados  │Registros    │Sucesso      │Total        │
│     3       │   45,234    │    95%      │  2h 35min   │
└─────────────┴─────────────┴─────────────┴─────────────┘

📈 Progresso por Etapa
[Gráfico de barras colorido mostrando status de cada etapa]

───────────────────────────────────────────────────────────

📈 Estatísticas Gerais do Dataset

┌─────────────┬─────────────┬─────────────┬─────────────┐
│📝 Total de  │📺 Canais    │📅 Período   │🔄 Taxa      │
│Mensagens    │Únicos       │Coberto      │Encaminhamento│
│   42,156    │    127      │  1,247 dias │    23.4%    │
│-15.2% (dedup)│           │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

🏷️ Top Hashtags              📺 Top Canais
[Gráfico barras horizontal]   [Gráfico pizza colorido]

⏰ Atividade Temporal
[Gráfico linha mostrando atividade por hora 0-23]

💡 Insights Principais
🔍 Alta taxa de duplicação detectada (15.2%) - considere investigar fontes de spam
🔍 Alto índice de mensagens encaminhadas (23.4%) - conteúdo viral predominante

📊 Qualidade dos Dados
[Gráficos de completude e flags de qualidade]

📊 Resumo por Dataset
[Tabela com status de cada arquivo processado]
```

---

## 🔍 **Preview: Análise por Etapa**

```
🔍 Análise Detalhada por Etapa

Selecione o dataset: [dropdown com arquivos]

┌─────────────┬─────────────┬─────────────┬─────────────┐
│01. Validação│02b. Deduplic│01b. Features│03. Limpeza  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│04. Sentimen │05. Tópicos  │06. TF-IDF   │07. Cluster  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│08. Hashtags │09. Domínios │10. Temporal │11. Rede     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│12. Qualitat │14. Busca    │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

[Conteúdo muda baseado na aba selecionada]

Exemplo - Aba "11. Rede":
🕸️ Análise de Estrutura de Rede

┌─────────────┬─────────────┬─────────────┐
│Nós na Rede  │Densidade    │Coeficiente  │
│   1,234     │   0.023     │Clustering   │
│Arestas      │Diâmetro     │   0.412     │
│   5,678     │     6       │Componentes  │
│             │             │Conectados: 3│
└─────────────┴─────────────┴─────────────┘

🕸️ Visualização da Rede de Canais
[Rede interativa com nós coloridos por centralidade]

👥 Top Influenciadores (por Centralidade)
[Scatter plot interativo]
```

---

## 🔎 **Preview: Busca Semântica**

```
🔎 Busca Semântica Inteligente

Digite sua consulta: [___________________________]
Ex: mensagens sobre vacinas com sentimento negativo

Dataset: [Todos ▼] Limiar: [0.7] Max Results: [50]

🔍 [Buscar]

[Após busca:]
✅ Encontrados 127 resultados para 'vacinas negativo'

📋 Resultados da Busca
▼ Resultado 1 - Similaridade: 0.95
  Canal: Canal_Exemplo_1
  Data: 2022-01-15  
  Sentimento: Negativo
  Texto: Lorem ipsum dolor sit amet...
  Tópicos: Vacinas, Saúde, Governo

🧠 Mapa de Conceitos Relacionados
[Rede conceitual interativa centrada na busca]
```

---

## ⚙️ **Preview: Configurações**

```
⚙️ Configurações

┌─────────┬─────────┬─────────────┬─────────────┐
│   API   │Pipeline │Visualização │ Exportação  │
└─────────┴─────────┴─────────────┴─────────────┘

🔑 Configurações da API

Chave API Anthropic: [••••••••••••••••••••]
Chave API Voyage.ai: [••••••••••••••••••••]

Modelo Anthropic: [claude-3-sonnet-20240229 ▼]
Max Tokens: [2000]

💾 [Salvar Configurações API]
```

---

## 🎨 **Características Visuais**

### **Cores e Tema**
- **Background**: Branco/Cinza claro
- **Sidebar**: Azul suave
- **Sucesso**: Verde (`#28a745`)
- **Warning**: Amarelo (`#ffc107`) 
- **Erro**: Vermelho (`#dc3545`)
- **Primário**: Azul (`#1f77b4`)

### **Componentes Interativos**
- **Gráficos**: Todos com hover, zoom, pan
- **Tabelas**: Ordenáveis e filtráveis
- **Métricas**: Cards com deltas coloridos
- **Upload**: Drag-and-drop visual

### **Responsividade**
- **Colunas**: Adaptam automaticamente
- **Gráficos**: `use_container_width=True`
- **Sidebar**: Colapsável em telas menores
- **Texto**: Tamanhos adaptativos

---

## 🚀 **Como Acessar**

```bash
cd src/dashboard
python start_dashboard.py
```

**URL**: http://localhost:8501

**🎉 Dashboard pronto com todas as funcionalidades avançadas habilitadas!**