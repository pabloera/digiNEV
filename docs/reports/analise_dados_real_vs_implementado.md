# Inventario Real: Analises de Dados do Pipeline digiNEV

**Data:** 2026-02-06
**Dados verificados:** `data/processed/processed_1_2019-2021-govbolso.csv` (786 registros, 96 colunas)
**Metodo:** Inspecao direta dos dados processados + codigo fonte `src/analyzer.py`

---

## RESUMO EXECUTIVO

| Categoria | Quantidade |
|-----------|-----------|
| Colunas totais no DataFrame | 96 |
| Colunas com dados reais variados | 66 |
| Colunas com valores constantes/defaults | 16 |
| Colunas de metadados do pipeline | 14 |
| Colunas de affordances (Stage 06) | 0 (NAO GERADAS nos processed) |
| Dashboards implementados | 14 dashboards + 13 paginas Streamlit |
| Analises realmente funcionais | 8 de 14 |
| Analises possiveis SEM novas colunas | 12 tipos adicionais |

---

## 1. COLUNAS COM DADOS REAIS (66 colunas)

Estas colunas contem dados variados e uteis para analise.

### 1.1 Dados Originais do Dataset (10 colunas)

| Coluna | Non-null | Exemplo |
|--------|----------|---------|
| `body` | 100% | Texto original da mensagem |
| `url` | 27.7% | URLs compartilhadas |
| `hashtag` | 2.7% | Hashtags usadas |
| `channel` | 5.7% | Canal Telegram |
| `mentions` | 7.1% | Mencoes a outros usuarios |
| `sender` | 7.1% | Remetente da mensagem |
| `media_type` | 100% | text/photo |
| `domain` | 27.7% | Dominio de URLs |
| `datetime` | 100% | DD/MM/AAAA HH:MM:SS |
| `datetime_parsed` | 100% | 2019-07-06 08:24:23 |

### 1.2 Stage 01-02: Preparacao (4 colunas uteis)

| Coluna | Dados reais? | Descricao |
|--------|-------------|-----------|
| `normalized_text` | SIM (100% unico) | Texto normalizado - COLUNA HUB do pipeline |
| `emojis_extracted` | SIM (mas 100% vazio neste chunk) | Lista de emojis |
| `emojis_count` | SIM (0 neste chunk) | Contagem de emojis |

### 1.3 Stage 03: Deduplicacao (3 colunas)

| Coluna | Dados reais? | Valores |
|--------|-------------|---------|
| `dupli_freq` | SIM | 1-3 (maioria=1, unico) |
| `channels_found` | PARCIAL | 0-1 (94% = 0) |
| `date_span_days` | CONSTANTE | 100% = 0 |

### 1.4 Stage 04: Estatistica (4 colunas)

| Coluna | Dados reais? | Distribuicao |
|--------|-------------|-------------|
| `char_count` | SIM | mean=250, std=311, range=[11, 1958] |
| `word_count` | SIM | mean=46, std=57, range=[2, 401] |
| `repetition_ratio` | FRACO | 99% = 0.0 (so 6 valores unicos) |
| `semantic_diversity` | SIM | mean=0.85, std=0.12, range=[0.38, 1.0] |

### 1.5 Stage 05: Filtro (1 coluna - DEFEITUOSA)

| Coluna | Dados reais? | Problema |
|--------|-------------|---------|
| `content_quality_score` | NAO | 100% = 100 (valor constante, nao filtra nada) |

### 1.6 Stage 06: Affordances (0 colunas)

**AUSENTE nos arquivos processados.** As 10 colunas `aff_*` e `affordance_*` NAO existem nos CSVs gerados. Possivel causa: Stage 06 foi adicionado ao codigo DEPOIS do processamento de outubro/2025.

### 1.7 Stage 07: Linguistica/spaCy (7 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `spacy_tokens` | SIM | Listas reais de tokens (spaCy rodou) |
| `spacy_lemmas` | SIM | Lemmas reais (ex: "achar", "justo") |
| `spacy_pos_tags` | SIM | POS tags reais (NUM, PROPN, ADJ, VERB...) |
| `spacy_entities` | SIM | Entidades nomeadas reais (PER, LOC, ORG) |
| `spacy_tokens_count` | SIM | mean=45, correlaciona com word_count |
| `spacy_entities_count` | SIM | mean=2.4, range=[0, 22], >0: 558/786 |
| `lemmatized_text` | SIM | Texto lematizado real |

**NOTA:** spaCy RODOU com sucesso neste processamento. Entidades nomeadas sao reais.

### 1.8 Stage 08: Classificacao Politica (3 colunas)

| Coluna | Dados reais? | Distribuicao |
|--------|-------------|-------------|
| `political_orientation` | SIM (keyword-based) | neutral=396, extrema-direita=200, esquerda=162, direita=18, centro-direita=9, centro-esquerda=1 |
| `political_keywords` | SIM | Listas de keywords encontradas |
| `political_intensity` | FRACO | 3 valores unicos, 91.5% = 0.0 |

**METODO REAL:** Contagem de keywords em listas hardcoded (6-14 palavras por categoria). NAO e ML, NAO e NLP avancado. E `word in text_lower` puro.

### 1.9 Stage 09: TF-IDF (3 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `tfidf_score_mean` | SIM | mean=0.041, std=0.015, 90 valores unicos |
| `tfidf_score_max` | SIM | valores variados |
| `tfidf_top_terms` | SIM | Top 5 termos por mensagem |

**METODO REAL:** sklearn TfidfVectorizer com max_features adaptativo. Dados sao reais.

### 1.10 Stage 10: Clustering (3 colunas)

| Coluna | Dados reais? | Distribuicao |
|--------|-------------|-------------|
| `cluster_id` | SIM | 5 clusters (0-4), distribuicao: {1:397, 2:273, 3:49, 4:40, 0:27} |
| `cluster_distance` | SIM | mean=0.56, std=1.24 |
| `cluster_size` | SIM | 5 tamanhos unicos |

**METODO REAL:** sklearn KMeans. Dados sao reais mas a coluna `text_length` fantasma pode ter degradado qualidade.

### 1.11 Stage 11: Topic Modeling (3 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `dominant_topic` | SIM | 5 topicos (0-4), distribuicao razoavel |
| `topic_probability` | SIM | mean=0.67, valores variados |
| `topic_keywords` | FRACO | Keywords muito genericas: ['da', 'de', 'do'], ['que', 'bolsonaro'] |

**METODO REAL:** sklearn LatentDirichletAllocation. Rodou mas keywords sao stopwords (falta preprocessamento para LDA).

### 1.12 Stage 12: Semantica (5 colunas)

| Coluna | Dados reais? | Problema |
|--------|-------------|---------|
| `sentiment_polarity` | MUITO FRACO | 92.5% = 0.0, apenas 7 palavras positivas + 7 negativas |
| `sentiment_label` | MUITO FRACO | 99.6% = "neutral" (so 3 "positive", 0 "negative") |
| `emotion_intensity` | CONSTANTE | 100% = 0.0 (usa texto raw com pontuacao, mas normalized_text remove `!?`) |
| `has_aggressive_language` | SIM | 4.5% = True (9 palavras na lista) |
| `semantic_diversity` | SIM | Ratio de palavras unicas/total |

**PROBLEMA CRITICO:** O sentimento e quase inutil. 7 palavras positivas e 7 negativas para classificar portugues politico e totalmente insuficiente.

### 1.13 Stage 13: Temporal (5 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `hour` | SIM | 0-23 (24 valores) |
| `day_of_week` | SIM | 0-6 (7 valores) |
| `month` | CONSTANTE | 100% = 7 (so julho neste chunk) |
| `year` | CONSTANTE | 100% = 2019 (so 2019 neste chunk) |
| `day_of_year` | SIM | 14 valores unicos |

**NOTA:** Constancia de month/year e esperada para chunk pequeno de um periodo.

### 1.14 Stage 14: Network (4 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `sender_frequency` | FRACO | 92.9% null (depende de `sender` que tem 92.9% null) |
| `is_frequent_sender` | SIM | 3.2% = True |
| `temporal_coordination` | SIM | 19 valores unicos, baseado na distribuicao de hora |
| `shared_url_frequency` | CONSTANTE | 100% = 0 |

### 1.15 Stage 15: Domain (5 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `domain_type` | SIM | unknown=568, other=154, video=33, social=19, mainstream_news=11, blog=1 |
| `domain_frequency` | PARCIAL | 72.3% null (segue `domain` original) |
| `is_mainstream_media` | SIM | 6.7% = True |
| `url_count` | CONSTANTE | 100% = 0 (bug: nao conta URLs de `url` original) |
| `has_external_links` | CONSTANTE | 100% = False |

### 1.16 Stage 16: Event Context (7 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `political_context` | SIM | general=616, government=134, electoral=28, economic=6, protest=2 |
| `mentions_government` | SIM | 20.1% = True |
| `mentions_opposition` | SIM | 10.1% = True |
| `election_context` | SIM | 5.1% = True |
| `protest_context` | SIM | 12.6% = True |
| `is_weekend` | SIM | Baseado em day_of_week |
| `is_business_hours` | SIM | Baseado em hour |

**METODO REAL:** Keyword matching com ~4-6 termos por contexto. Simples mas funcional.

### 1.17 Stage 17: Channel (8 colunas)

| Coluna | Dados reais? | Qualidade |
|--------|-------------|-----------|
| `channel_type` | FRACO | 94.3% = "unknown" (pouca variacao) |
| `channel_activity` | FRACO | 94.3% null |
| `is_active_channel` | FRACO | 96.8% = False |
| `content_type` | SIM | text/photo |
| `has_media` | CONSTANTE | 100% = True (bug) |
| `is_forwarded` | CONSTANTE | 100% = False |
| `forwarding_context` | CONSTANTE | 100% = 0.0 |
| `sender_channel_influence` | SIM | 3 valores |

---

## 2. METODOS DE ANALISE REAIS NO PIPELINE

### 2.1 Metodos que PRODUZEM dados uteis

| # | Metodo | Tecnica Real | Qualidade dos Dados |
|---|--------|-------------|---------------------|
| 1 | **Normalizacao de texto** (Stage 02) | Lowercase + remoção de especiais | ALTA - base para todo o pipeline |
| 2 | **Deduplicacao** (Stage 03) | Exact text matching + contagem | ALTA - funcional |
| 3 | **Metricas textuais** (Stage 04) | len(), split(), contagem | ALTA - char_count, word_count sao confiaveis |
| 4 | **NER com spaCy** (Stage 07) | `pt_core_news_lg` NER pipeline | ALTA - entidades PER/LOC/ORG reais |
| 5 | **Lematizacao spaCy** (Stage 07) | spaCy lemmatizer PT | ALTA - lemmas reais |
| 6 | **POS Tagging** (Stage 07) | spaCy POS tagger PT | ALTA - tags reais |
| 7 | **TF-IDF** (Stage 09) | sklearn TfidfVectorizer | MEDIA - scores reais, top_terms uteis |
| 8 | **K-Means Clustering** (Stage 10) | sklearn KMeans (k=5) | MEDIA - clusters reais mas features limitadas |
| 9 | **LDA Topic Modeling** (Stage 11) | sklearn LatentDirichletAllocation | BAIXA - keywords sao stopwords (falta remocao) |
| 10 | **Analise temporal** (Stage 13) | datetime parsing + componentes | ALTA - hour, day_of_week confiaveis |
| 11 | **Classificacao de dominio** (Stage 15) | Keyword matching em URLs | MEDIA - categorias basicas mas funcionais |
| 12 | **Contexto politico** (Stage 16) | Keyword matching em texto | MEDIA - categorias basicas mas funcionais |

### 2.2 Metodos que FALHAM ou PRODUZEM LIXO

| # | Metodo | Tecnica | Problema |
|---|--------|---------|---------|
| 1 | **Sentimento** (Stage 12) | 7 palavras positivas + 7 negativas | 99.6% classificado como "neutral". INUTIL. |
| 2 | **Emocao** (Stage 12) | Contagem de `!` e `?` | 100% = 0.0 porque `normalized_text` remove pontuacao |
| 3 | **Filtro qualidade** (Stage 05) | Score composto | 100% = 100 (nao filtra nada) |
| 4 | **Classificacao politica** (Stage 08) | 6 listas de 4-6 keywords | Funciona mas com vocabulario minimo |
| 5 | **Intensidade politica** (Stage 08) | 10 palavras de intensidade | 91.5% = 0.0 (lista curta demais) |
| 6 | **Affordances** (Stage 06) | Heuristica + API | AUSENTE nos dados processados |
| 7 | **Network frequency** (Stage 14) | Value counts de sender | 92.9% null (sender quase vazio) |
| 8 | **Channel analysis** (Stage 17) | Keyword matching no nome | 94.3% = "unknown" |

---

## 3. DASHBOARDS: O QUE FUNCIONA vs O QUE NAO FUNCIONA

### 3.1 Dashboards que FUNCIONAM com dados reais

| Dashboard | Colunas que usa | Status |
|-----------|----------------|--------|
| **stage03_deduplication** | `dupli_freq`, `normalized_text`, `channels_found` | FUNCIONA - dados existem |
| **stage04_duplication_stats** | `dupli_freq`, `dataset_source` | FUNCIONA (mas cria dados sinteticos se faltam) |
| **stage07_linguistic** | `spacy_entities`, `body` | FUNCIONA - spaCy rodou |
| **stage09_tfidf** | `tfidf_top_terms`, `tfidf_score_mean`, `datetime` | FUNCIONA - dados TF-IDF existem |
| **stage10_clustering** | `cluster_id`, `spacy_tokens_count` | FUNCIONA PARCIAL - clusters existem |
| **stage13_temporal** | `datetime`, `hour`, `day_of_week` | FUNCIONA - dados temporais existem |
| **stage14_network** | `sender_frequency`, `temporal_coordination` | FUNCIONA PARCIAL - muitos nulls |

### 3.2 Dashboards que NAO funcionam

| Dashboard | Colunas que precisa | Problema |
|-----------|-------------------|---------|
| **stage06_affordances** | `aff_*`, `affordance_*` | COLUNAS NAO EXISTEM nos dados processados |
| **stage11_topic_modeling** | `dominant_topic`, `topic_keywords` | FUNCIONA mas keywords sao stopwords |
| **stage12_semantic** | `sentiment_polarity`, `emotion_intensity` | DADOS SAO LIXO (99.6% neutral, 100% zero) |
| **stage13_temporal** (parcial) | `affordances_score` | COLUNA FANTASMA - nao existe |
| **data_analysis_dashboard** | `radicalization_level`, `discourse_type` | COLUNAS NAO EXISTEM |

---

## 4. ANALISES POSSIVEIS COM COLUNAS EXISTENTES (SEM CRIAR NOVAS)

### 4.1 Analises ja implementadas nos dashboards

| # | Analise | Colunas usadas | Implementada em |
|---|---------|---------------|----------------|
| 1 | Distribuicao de duplicatas | `dupli_freq` | stage03_dashboard |
| 2 | Frequencia de entidades nomeadas | `spacy_entities` | stage07_dashboard |
| 3 | TF-IDF termos relevantes | `tfidf_top_terms`, `tfidf_score_mean` | stage09_dashboard |
| 4 | Visualizacao de clusters | `cluster_id`, `cluster_distance` | stage10_dashboard |
| 5 | Volume temporal | `hour`, `day_of_week`, `datetime` | stage13_dashboard |
| 6 | Rede de coordenacao | `sender_frequency`, `temporal_coordination` | stage14_dashboard |

### 4.2 Analises POSSIVEIS mas NAO implementadas

| # | Analise | Colunas necessarias (JA EXISTEM) | Complexidade |
|---|---------|----------------------------------|-------------|
| 1 | **Distribuicao de orientacao politica por canal** | `political_orientation` x `channel` | SIMPLES: crosstab + heatmap |
| 2 | **Evolucao temporal da orientacao politica** | `political_orientation` x `datetime_parsed` | SIMPLES: groupby month + stacked area |
| 3 | **Comprimento de texto por orientacao politica** | `word_count` x `political_orientation` | SIMPLES: boxplot |
| 4 | **Co-ocorrencia de entidades nomeadas** | `spacy_entities` (parse listas) | MEDIA: network graph de co-ocorrencia |
| 5 | **Topicos por orientacao politica** | `dominant_topic` x `political_orientation` | SIMPLES: crosstab + heatmap |
| 6 | **Diversidade lexical por cluster** | `semantic_diversity` x `cluster_id` | SIMPLES: boxplot |
| 7 | **Linguagem agressiva por contexto politico** | `has_aggressive_language` x `political_context` | SIMPLES: crosstab |
| 8 | **URLs mais compartilhadas** | `url` (value_counts direto) | SIMPLES: bar chart |
| 9 | **Dominios de midia por orientacao politica** | `domain_type` x `political_orientation` | SIMPLES: crosstab |
| 10 | **Padroes temporais de conteudo agressivo** | `has_aggressive_language` x `hour` x `day_of_week` | MEDIA: heatmap |
| 11 | **Cluster profiling textual** | `cluster_id` x `tfidf_top_terms` (agregar por cluster) | MEDIA: word cloud por cluster |
| 12 | **Mencoes governo vs oposicao temporal** | `mentions_government` x `mentions_opposition` x `datetime_parsed` | SIMPLES: line chart |

---

## 5. METODOS DE ANALISE: CLASSIFICACAO HONESTA

### 5.1 METODOS REAIS DE NLP/ML (produzem resultados confiaveis)

| Metodo | Biblioteca | Resultado |
|--------|-----------|-----------|
| spaCy NER | `pt_core_news_lg` | Entidades PER/LOC/ORG reais |
| spaCy Lemmatization | `pt_core_news_lg` | Lemmas reais |
| spaCy POS Tagging | `pt_core_news_lg` | Tags gramaticais reais |
| TF-IDF Vectorization | `sklearn` | Scores de relevancia reais |
| K-Means Clustering | `sklearn` | Agrupamentos reais (qualidade depende de features) |
| LDA Topic Modeling | `sklearn` | Topicos reais (mas precisam de limpeza de stopwords) |

### 5.2 METODOS HEURISTICOS (funcionam mas sao limitados)

| Metodo | Tecnica | Limitacao |
|--------|---------|----------|
| Classificacao politica | Keyword matching (6 listas, 4-6 palavras cada) | Vocabulario minimo, sem contexto |
| Contexto politico | Keyword matching (4-6 termos por contexto) | Categorias muito amplas |
| Tipo de dominio | String matching em URL | 5 categorias basicas |
| Linguagem agressiva | 9 palavras hardcoded | Lista muito curta |
| Deduplicacao | Exact text matching | Nao detecta parafrase |

### 5.3 METODOS QUEBRADOS (NAO produzem dados uteis)

| Metodo | Tecnica | Por que falha |
|--------|---------|--------------|
| Analise de sentimento | 7 palavras positivas + 7 negativas | Vocabulario absurdamente pequeno para PT politico. 99.6% = neutral |
| Intensidade emocional | Contagem de `!` e `?` | `normalized_text` remove toda pontuacao. 100% = 0.0 |
| Intensidade politica | 10 palavras genericas (sempre, nunca, urgente...) | Nao sao palavras politicas. 91.5% = 0.0 |
| Filtro de qualidade | Score composto de 5 metricas | Score constante 100 para todos os registros |
| Affordances (Stage 06) | Heuristica + API Anthropic | NAO FOI EXECUTADO nos dados existentes |

---

## 6. DADOS PROCESSADOS: ESTADO ATUAL

### 6.1 Arquivos existentes

| Arquivo | Registros | Periodo | Status |
|---------|-----------|---------|--------|
| `processed_1_2019-2021-govbolso.csv` | 786 | Jul/2019 | Processado (96 cols, 17 stages) |
| `processed_2_2021-2022-pandemia.csv` | 1.900 | Pandemia | Processado (96 cols, 17 stages) |
| `3_2022-2023-poseleic.csv` | ~93MB raw | Pos-eleicao | NAO PROCESSADO |
| `4_2022-2023-elec.csv` | ~55MB raw | Eleicao | NAO PROCESSADO |
| `5_2022-2023-elec-extra.csv` | ~26MB raw | Eleicao extra | NAO PROCESSADO |

**PROBLEMA:** Apenas 2 de 5 datasets foram processados. Os 3 maiores (pos-eleicao, eleicao) NAO foram processados.

### 6.2 Volume real processado

| Metrica | Valor |
|---------|-------|
| Total processado | ~2.686 registros (786 + 1.900) |
| Total raw disponivel | ~500MB (~300k mensagens estimadas) |
| Percentual processado | <1% do total |

---

## 7. CONCLUSOES

### O que FUNCIONA hoje:
1. **spaCy NLP** (tokens, lemmas, POS, NER) - dados confiaveis
2. **TF-IDF** - termos relevantes identificados
3. **Clustering** - agrupamentos criados (k=5)
4. **Temporal** - padroes de hora/dia identificados
5. **Deduplicacao** - contagens de duplicatas
6. **Keyword-based classification** - orientacao politica basica

### O que NAO funciona:
1. **Sentimento** - completamente inutil com 14 palavras
2. **Emocao** - 100% zero por bug de design (pontuacao removida)
3. **Affordances** - ausente dos dados processados
4. **Topic keywords** - LDA retorna stopwords
5. **Network** - 92.9% null pela falta de sender
6. **Channel** - 94.3% unknown pela falta de channel

### O que pode ser feito SEM novas colunas:
12 tipos de analise cruzada usando colunas existentes (Secao 4.2), incluindo:
- Cruzamento orientacao politica x temporal
- Co-ocorrencia de entidades nomeadas
- Linguagem agressiva por contexto
- Cluster profiling com TF-IDF
- Dominios de midia por orientacao politica

### O que PRECISA de correcao para funcionar:
1. **Sentimento:** Substituir as 14 palavras por um modelo real (ex: `pysentimiento` para PT)
2. **Emocao:** Usar `body` original em vez de `normalized_text` para detectar `!` e `?`
3. **Topic modeling:** Remover stopwords em PT antes do LDA
4. **Affordances:** Re-processar os dados com o Stage 06 ativo
5. **Processar 3 datasets faltantes** (~98% dos dados nao processados)

---

*Relatorio gerado por inspecao direta dos dados processados e codigo-fonte. Nenhuma metrica foi inventada.*
