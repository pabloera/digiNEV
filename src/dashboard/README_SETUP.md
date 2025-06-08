# Dashboard Bolsonarismo - Configuração e Uso

## 🚀 Configuração Completa

O dashboard está totalmente integrado com o pipeline e pronto para uso.

### ✅ Status da Configuração
- **Pipeline**: ✅ Totalmente funcional
- **APIs**: ✅ Anthropic e Voyage.ai configuradas
- **Dependencies**: ✅ Todas instaladas
- **Integração**: ✅ Dashboard conectado ao pipeline

## 📋 Como Usar

### 1. Iniciar o Dashboard

```bash
# Opção 1: Script de inicialização (recomendado)
python src/dashboard/start_dashboard.py

# Opção 2: Diretamente via streamlit
streamlit run src/dashboard/app.py --server.port 8501
```

### 2. Acessar a Interface

Abra seu navegador em: **http://localhost:8501**

### 3. Funcionalidades Disponíveis

#### 📤 Upload & Processamento
- Upload múltiplo de arquivos CSV
- Validação automática dos dados
- Configuração de parâmetros do pipeline
- Execução das 14 etapas de processamento
- Modo demo para testes (quando pipeline não disponível)

#### 📊 Visão Geral
- Métricas gerais dos datasets processados
- Progresso por etapa
- Timeline de atividade
- Resumo por dataset

#### 🔍 Análise por Etapa
Visualizações detalhadas para cada etapa:
- **01. Validação**: Problemas de encoding e qualidade dos dados
- **02. Encoding**: Correções aplicadas
- **02b. Deduplicação**: Estatísticas de duplicatas
- **01b. Features**: Extração de características
- **03. Limpeza**: Padrões removidos
- **04. Sentimento**: Distribuição e evolução temporal
- **05. Tópicos**: Modelagem e palavras-chave
- **06. TF-IDF**: Termos importantes e embeddings
- **07. Clustering**: Agrupamentos e silhueta
- **08. Hashtags**: Frequência e tendências
- **09. Domínios**: Distribuição e credibilidade
- **10. Temporal**: Padrões por hora/dia
- **11. Rede**: Estrutura e influenciadores
- **12. Qualitativa**: Categorias e alinhamento político
- **14. Busca Semântica**: Conceitos e insights automáticos

#### 📈 Comparação de Datasets
- Métricas comparativas entre diferentes datasets
- Evolução temporal comparativa
- Distribuições de sentimento lado a lado

#### 🔎 Busca Semântica
- Interface de busca inteligente
- Filtros por dataset e similaridade
- Mapa de conceitos relacionados

#### ⚙️ Configurações
- Configuração de APIs (Anthropic, Voyage.ai)
- Parâmetros do pipeline
- Configurações de visualização
- Opções de exportação

## 🔧 Configuração das APIs

As APIs já estão configuradas no arquivo `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE_ANTHROPIC_AQUI]
VOYAGE_API_KEY=pa-[SUA_CHAVE_VOYAGE_AQUI]
```

## 📁 Estrutura de Dados

### Formatos Suportados
- **Arquivos**: CSV com separador `;` ou `,`
- **Encoding**: UTF-8 (correção automática aplicada)

### Colunas Esperadas
- `texto`: Conteúdo da mensagem (obrigatório)
- `data_hora`: Timestamp (obrigatório)
- `canal`: Nome do canal (opcional)
- `url`: URLs compartilhadas (opcional)
- `hashtags`: Hashtags usadas (opcional)

### Estrutura de Diretórios
```
data/
├── uploads/           # Arquivos enviados pelo dashboard
├── dashboard_results/ # Resultados processados
└── interim/          # Dados intermediários do pipeline
```

## 🚨 Modo Demo

Quando o pipeline não está disponível, o dashboard oferece um **modo demo** que:
- Simula o processamento dos dados
- Gera visualizações de exemplo
- Permite testar a interface
- Demonstra todas as funcionalidades visuais

## 📊 Visualizações

O dashboard inclui visualizações interativas usando Plotly:
- Gráficos de pizza para distribuições
- Gráficos de linha para séries temporais
- Mapas de calor para correlações
- Redes para estruturas complexas
- Treemaps para hierarquias
- Gráficos de radar para métricas multidimensionais

## 🔍 Solução de Problemas

### Pipeline não disponível
- Verifique se as dependências estão instaladas
- Confirme se as APIs estão configuradas
- Use o modo demo para testar a interface

### Erro de importação
- Execute: `pip install -r src/dashboard/requirements.txt`
- Verifique se o PYTHONPATH inclui o diretório `src`

### Erro de encoding
- O dashboard detecta e corrige automaticamente
- Suporta múltiplos formatos de CSV

## 📈 Performance

- **Chunk Processing**: Processa arquivos grandes em blocos
- **Cache**: Resultados são armazenados para consulta rápida
- **Visualizações**: Otimizadas para grandes volumes de dados
- **Estado**: Interface mantém estado entre sessões

## 🎯 Próximos Passos

1. **Carregue seus dados**: Use a aba "Upload & Processamento"
2. **Execute o pipeline**: Configure e inicie o processamento
3. **Explore os resultados**: Use as abas de análise
4. **Compare datasets**: Analise diferentes períodos
5. **Faça buscas**: Use a busca semântica para insights

---

**Dashboard totalmente configurado e pronto para análise completa do movimento bolsonarista! 🚀📊**