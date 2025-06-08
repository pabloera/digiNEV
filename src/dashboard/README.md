# Dashboard do Pipeline Bolsonarismo 📊

Interface web completa para análise em massa de datasets do projeto Bolsonarismo, com visualizações interativas para cada uma das 14 etapas do pipeline.

## 🚀 Características Principais

### 📤 Upload e Processamento em Massa
- **Upload múltiplo** de arquivos CSV via drag-and-drop
- **Validação automática** da estrutura dos arquivos
- **Processamento paralelo** de múltiplos datasets
- **Monitoramento em tempo real** do progresso

### 📊 Visualizações por Etapa

O dashboard oferece visualizações específicas para cada uma das 14 etapas:

1. **Validação de Dados**: Problemas de encoding, métricas de qualidade
2. **Correção de Encoding**: Comparação antes/depois, caracteres problemáticos
3. **Deduplicação**: Clusters de duplicatas, heatmap de similaridade
4. **Feature Extraction**: Distribuição de features, matriz de correlação
5. **Limpeza de Texto**: Frequência de palavras, padrões removidos
6. **Análise de Sentimento**: Distribuição, evolução temporal, comparação por canal
7. **Modelagem de Tópicos**: Distribuição de tópicos, evolução, visualização LDA
8. **Análise TF-IDF**: Termos importantes, word clouds, t-SNE de embeddings
9. **Clustering**: Visualização de clusters, dendrogramas, análise de silhueta
10. **Análise de Hashtags**: Rede de hashtags, tendências, co-ocorrência
11. **Análise de Domínios**: Distribuição, scores de credibilidade, rede de links
12. **Análise Temporal**: Séries temporais, detecção de picos, correlação com eventos
13. **Estrutura de Rede**: Grafo de rede, medidas de centralidade, detecção de comunidades
14. **Análise Qualitativa**: Categorias de conteúdo, detecção de desinformação, alinhamento político
15. **Busca Semântica**: Mapas conceituais, redes de similaridade, insights automáticos

### 📈 Recursos Avançados

- **Comparação entre Datasets**: Análise lado a lado de múltiplos arquivos
- **Busca Semântica Inteligente**: Busca por conceitos com IA
- **Exportação de Resultados**: CSV, Excel, JSON, PDF, HTML
- **Dashboard de Visão Geral**: Métricas consolidadas de todos os datasets
- **Filtros e Agregações**: Análise seletiva de subconjuntos

### 🎨 **Visualizações Avançadas Habilitadas**

#### **🕸️ Análise de Redes**
- **Redes de Canais**: Visualização interativa usando NetworkX + Plotly
- **Co-ocorrência de Hashtags**: Rede de hashtags que aparecem juntas
- **Mapas Conceituais**: Relacionamentos semânticos entre conceitos
- **Métricas de Centralidade**: PageRank, betweenness, degree centrality

#### **🌳 Clustering Hierárquico**
- **Dendrogramas Interativos**: Usando Scipy + Plotly
- **Análise de Agrupamento**: Visualização de clusters de documentos
- **Métricas de Qualidade**: Silhouette score, inércia

#### **☁️ Análise de Texto**
- **Nuvens de Palavras**: WordCloud para termos TF-IDF
- **Análise de Frequência**: Distribuições estatísticas avançadas
- **Embeddings t-SNE**: Projeção de alta dimensionalidade

#### **📊 Estatísticas Completas**
- **Análise de Qualidade**: Métricas de completude e consistência
- **Distribuições Temporais**: Padrões por hora, dia, mês
- **Insights Automáticos**: Recomendações baseadas em IA

## 🛠️ Instalação

1. **Clone o repositório** (se ainda não tiver):
```bash
git clone https://github.com/seu-usuario/dataanalysis-bolsonarismo.git
cd dataanalysis-bolsonarismo
```

2. **Instale as dependências básicas**:
```bash
cd src/dashboard
pip install -r requirements.txt
```

3. **Instale visualizações avançadas** (opcional, mas recomendado):
```bash
python install_advanced_viz.py
```

Ou manualmente:
```bash
pip install networkx scipy wordcloud matplotlib seaborn scikit-learn
```

4. **Execute o dashboard**:
```bash
python start_dashboard.py
# ou
streamlit run app.py
```

## 🖥️ Uso

### Iniciando o Dashboard

```bash
# Do diretório do projeto
cd src/dashboard
./run_dashboard.sh

# Ou diretamente com streamlit
streamlit run app.py
```

O dashboard estará disponível em: http://localhost:8501

### Processando Arquivos

1. **Upload**: Arraste múltiplos arquivos CSV para a área de upload
2. **Configuração**: Ajuste as configurações do pipeline (usar Anthropic, tamanho de chunks, etc.)
3. **Processamento**: Clique em "Iniciar Processamento"
4. **Monitoramento**: Acompanhe o progresso na barra lateral
5. **Análise**: Explore os resultados nas diferentes páginas

### Navegação

- **📤 Upload & Processamento**: Upload e configuração inicial
- **📊 Visão Geral**: Métricas consolidadas e resumo
- **🔍 Análise por Etapa**: Visualizações detalhadas de cada etapa
- **📈 Comparação de Datasets**: Compare resultados entre arquivos
- **🔎 Busca Semântica**: Busca inteligente nos dados processados
- **⚙️ Configurações**: Configurações de API e visualização

## 🔧 Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` no diretório do dashboard:

```env
# API Keys
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here

# Dashboard
DASHBOARD_PORT=8501
```

### Personalização

O dashboard pode ser personalizado editando:
- `app.py`: Lógica principal e fluxo
- Temas do Streamlit em `.streamlit/config.toml`
- Estilos CSS inline no código

## 📊 Estrutura de Dados

O dashboard espera arquivos CSV com as seguintes colunas mínimas:
- `texto`: Conteúdo da mensagem
- `data_hora`: Timestamp da mensagem

Colunas opcionais que enriquecem a análise:
- `canal`: Nome do canal
- `url`: URLs compartilhadas
- `hashtags`: Hashtags usadas
- Etc.

## 🐛 Troubleshooting

### Problema: "Module not found"
```bash
# Reinstale as dependências
pip install -r requirements.txt
```

### Problema: "Pipeline not found"
```bash
# Certifique-se de estar no diretório correto
export PYTHONPATH=$PYTHONPATH:../../src
```

### Problema: "Out of memory"
- Reduza o tamanho dos chunks nas configurações
- Processe menos arquivos por vez
- Aumente a memória disponível para o Python

### Problema: "Visualizações avançadas não disponíveis"
```bash
# Instale as bibliotecas necessárias
python install_advanced_viz.py
# ou manualmente
pip install networkx scipy wordcloud matplotlib seaborn
```

### Problema: "Error creating network visualization"
- Certifique-se de que NetworkX está instalado
- Verifique se os dados estão no formato correto
- Reduza o tamanho da rede se muito grande

## 🤝 Contribuindo

Para adicionar novas visualizações:

1. Crie um novo método `render_[nome]_analysis()` 
2. Adicione a chamada na página apropriada
3. Use Plotly para gráficos interativos
4. Mantenha consistência visual

## 📝 Licença

Este projeto está sob a licença do projeto principal Bolsonarismo.