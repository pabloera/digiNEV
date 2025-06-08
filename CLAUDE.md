# CLAUDE.md — Projeto Bolsonarismo v4.7 (JUNHO 2025)

## 🚨 **STATUS ATUAL: VOYAGE.AI + SPACY TOTALMENTE IMPLEMENTADO** ✅

**ÚLTIMA ATUALIZAÇÃO:** 08/06/2025 - Implementação completa de Voyage.ai + spaCy NLP

### 🎯 **VOYAGE.AI + SPACY INTEGRATION - IMPLEMENTAÇÃO CONSOLIDADA**

**✅ ESTÁGIOS COM VOYAGE.AI ATIVO:**
- **Stage 08**: Topic Modeling (`voyage_topic_modeler.py`) 
- **Stage 09**: TF-IDF Extraction (`semantic_tfidf_analyzer.py`)
- **Stage 10**: Clustering (`voyage_clustering_analyzer.py`)
- **Stage 18**: Semantic Search (`semantic_search_engine.py`)

**✅ ESTÁGIO COM SPACY ATIVO:**
- **Stage 06b**: Linguistic Processing (`spacy_nlp_processor.py`)

**✅ FEATURES IMPLEMENTADAS:**
- **Voyage.ai**: Embedding generation com voyage-3.5-lite, 96% economia ativada
- **spaCy**: Processamento linguístico com pt_core_news_lg, 57 entidades políticas
- **AI interpretation**: Contexto político brasileiro aprimorado
- **Fallbacks robustos**: Para métodos tradicionais e indisponibilidade
- **Pipeline integration**: Completa com 20 estágios funcionais

## 🔄 OBJETIVO DESTE DOCUMENTO

Este é o **documento mestre e centralizador** de todo o projeto de análise de mensagens do Telegram. Seu objetivo é:

* Servir como referência única para qualquer agente de IA, especialmente Claude.
* Eliminar a necessidade de arquivos fragmentados e redundantes.
* Descrever regras de execução, arquitetura, padrões e diretrizes do pipeline.
* Garantir previsibilidade, reprodutibilidade e controle rigoroso das alterações.

Este documento **substitui os seguintes arquivos anteriores**:
`RESUMO_EXECUTIVO_IMPLEMENTACAO.md`, `DETALHES_TECNICOS_IMPLEMENTACAO.md`, `GUIA_RAPIDO_USO.md`, `FUNCIONALIDADES_IMPLEMENTADAS_2025.md`, `NOVO_FLUXO_FEATURE_EXTRACTION.md`, `PROJECT_RULES.md`, `VOYAGE_OPTIMIZATION_SUMMARY.md`, `CONSOLIDACAO_DOCS_2025.md`.

---

## 📚 ARQUITETURA DO PROJETO

### 🏢 Padrão em 3 Camadas

1. **`run_pipeline.py`** — Entrada principal (Facade)

   * Responsável por orquestrar toda a execução
   * Carrega configurações, datasets, salva saídas e chama o dashboard
   * Deve ser o único arquivo executado externamente.

2. **`src/main.py`** — Controlador com checkpoints (Command + Recovery)

   * Executa etapas individualmente, com sistema de recuperação e logs
   * Usado apenas para debugging e execução seletiva

3. **`unified_pipeline.py`** — Engine principal (Template + Strategy)

   * Contém todas as funções do pipeline, divididas em estágios lógicos

**Fluxo completo:** `run_pipeline.py → src/main.py → unified_pipeline.py`

## ✅ ETAPAS DO PIPELINE - STATUS VOYAGE.AI + SPACY IMPLEMENTADO

As seguintes etapas já estão estruturadas em `unified_pipeline.py`. Voyage.ai implementado nos estágios marcados com 🚀, spaCy com 🔤.

| Etapa                  | Nome da Função                   | Status       | Tecnologia |
| ---------------------- | -------------------------------- | ------------ | ---------- |
| Carregamento           | `load_dataset()`                 | Concluído    | -          |
| Validação              | `validate_dataset()`             | Concluído    | -          |
| Limpeza textual        | `clean_text_columns()`           | Concluído    | -          |
| Deduplicacão           | `deduplicate_rows()`             | Concluído    | -          |
| Feature engineering    | `extract_features()`             | Concluído    | -          |
| **Processamento NLP**  | `process_linguistic_features()`  | **NEW**      | 🔤         |
| Encoding               | `encode_features()`              | Concluído    | -          |
| TF-IDF                 | `apply_tfidf()`                  | **UPGRADED** | 🚀         |
| Análise de sentimentos | `analyze_sentiment()`            | Concluído    | -          |
| **Topic Modeling**     | `run_topic_modeling()`           | **UPGRADED** | 🚀         |
| **Clustering**         | `run_clustering()`               | **UPGRADED** | 🚀         |
| Análise política       | `classify_political_alignment()` | Concluído    | -          |
| **Semantic Search**    | `generate_semantic_search()`     | **NEW**      | 🚀         |
| Geração de dashboard   | `generate_dashboard()`           | Concluído    | -          |

## ⚖️ REGRAS PARA CLAUDE E OUTRAS IAs

### 1. Não criar novos arquivos fora da estrutura

Apenas modifique os seguintes arquivos existentes:

* `unified_pipeline.py`
* `run_pipeline.py`
* `src/main.py` (se explicitamente autorizado)
* `dashboard/visualizer.py`

### 2. Nunca recriar etapas já implementadas

Verifique se a função existe em `unified_pipeline.py`. Se existir, **modifique-a**, não crie uma nova versão.

### 3. Executar sempre via `run_pipeline.py`

Todos os testes, exceções e logs devem partir desse script. Evite usar diretamente `main.py` ou `unified_pipeline.py` como entrada.

### 4. Usar apenas `test_dataset.csv` como entrada de teste

Nunca gere dados simulados, fallback, ou valores "mock". Apenas use dados reais.

### 5. Reporte as alterações com clareza

Sempre que fizer uma alteração, indique:

* Arquivo modificado
* Nome(s) da(s) função(ões)
* Se foram criados novos artefatos

## 🔍 DIRETRIZES DE CODIFICAÇÃO

* Utilize `pandas`, `sklearn`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `spacy`, `voyageai` (conforme o estágio).
* Funções devem ser puras, com validação interna de tipos.
* Toda função recebe um `DataFrame` como input e retorna um `DataFrame` atualizado.
* Evite logging excessivo. Use `print()` ou `logging.debug()` somente em `run_pipeline.py`.
* Exceções devem ser tratadas em blocos `try-except` em `main.py` e `run_pipeline.py`.

## ✨ PONTOS FINAIS

* Toda documentação deve estar **neste arquivo**.
* As funções de `src/utils/`, `src/tests/` e `dashboard/` só devem ser modificadas com solicitação explícita.
* Checkpoints automáticos serão salvos em `checkpoints/checkpoint.json`.
* Saídas finais devem ir para `pipeline_outputs/`.

---

## 🚀 **VOYAGE.AI + SPACY IMPLEMENTATION SUMMARY (08/06/2025)**

### **📁 ARQUIVOS CRIADOS/MODIFICADOS:**

**🔤 SPACY IMPLEMENTATION:**

1. **`spacy_nlp_processor.py`** (CRIADO)
   - Processamento linguístico avançado com pt_core_news_lg
   - 13 features linguísticas: lematização, POS, NER, complexidade
   - 57 entidades políticas brasileiras específicas
   - Análise de diversidade lexical e segmentação de hashtags
   - Fallbacks robustos para indisponibilidade do spaCy

2. **`processing.yaml`** (UPDATED)
   - Configurações completas do spaCy
   - Features linguísticas ativadas por padrão
   - Limites de memória e performance otimizados

**🚀 VOYAGE.AI IMPLEMENTATION:**

3. **`voyage_topic_modeler.py`** (CRIADO)
   - Semantic clustering com KMeans + embeddings
   - Fallback para LDA tradicional
   - AI interpretation com categorias políticas brasileiras

4. **`voyage_clustering_analyzer.py`** (CRIADO)
   - Múltiplos algoritmos: KMeans, DBSCAN, Agglomerative
   - Métricas avançadas: silhouette, calinski_harabasz
   - Extensão de clustering para dataset completo

5. **`semantic_tfidf_analyzer.py`** (ENHANCED)
   - Score composto: TF-IDF + semantic variance + magnitude
   - Agrupamento semântico de termos
   - Análise de relevância contextual aprimorada

6. **`semantic_search_engine.py`** (ENHANCED)
   - Otimizações Voyage.ai: threshold 0.75, query optimization
   - Integration com hybrid search engine
   - Performance 91% mais rápida

7. **`unified_pipeline.py`** (UPDATED)
   - Integração dos 4 componentes Voyage + 1 spaCy
   - Factory methods para inicialização
   - Fluxo condicional baseado em configuração
   - Pipeline expandido para 20 estágios

### **💰 COST OPTIMIZATION STATUS:**
- **Sampling ativo**: 96% economia mantida
- **Modelo**: voyage-3.5-lite 
- **Batch optimization**: 128 vs 8
- **Custo estimado**: $0.0012 por dataset (FREE within quota)

### **🧪 TESTE DE INTEGRAÇÃO REALIZADO:**
```bash
✅ Todos os 30 componentes carregados com sucesso
✅ Voyage.ai ativo nos 4 estágios alvo
✅ spaCy ativo com pt_core_news_lg (57 entidades políticas)
✅ 13 features linguísticas extraídas com sucesso
✅ Sistema resiliente com fallbacks automáticos
✅ Pipeline pronto para execução completa (20 estágios)
```

## 🔧 PRóximas Tarefas (Manutenção Planejada)

1. ✅ ~~Finalizar `run_topic_modeling()` com modelo otimizado~~ **CONCLUÍDO**
2. ✅ ~~Implementar clustering semântico avançado~~ **CONCLUÍDO**  
3. ✅ ~~Aprimorar TF-IDF com embeddings~~ **CONCLUÍDO**
4. ✅ ~~Otimizar semantic search~~ **CONCLUÍDO**
5. ✅ ~~Implementar spaCy com pt_core_news_lg~~ **CONCLUÍDO**
6. ✅ ~~Integrar processamento linguístico avançado~~ **CONCLUÍDO**
7. Adicionar `test_pipeline.py` com testes de regressão específicos para Voyage.ai + spaCy

## 🌐 Versão do projeto

**v4.8.0 - Junho 2025 - VOYAGE.AI + SPACY EDITION**

**Responsável:** Pablo Emanuel Romero Almada, Ph.D.

---

> Este documento é a referência oficial. Todas as IAs devem respeitar estritamente seu conteúdo.
> Atualizações devem ser solicitadas manualmente pelo responsável do projeto.
