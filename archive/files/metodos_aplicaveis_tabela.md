# Tabela de Métodos Científicos Validados Aplicáveis ao Dataset

## 📊 Dataset: Discurso Político Brasileiro (Telegram/WhatsApp, 2019-2023)
- **1000 mensagens** cobrindo governo Bolsonaro, pandemia e eleições 2022
- **846 textos válidos** em português
- **162 canais únicos**
- **1306 dias** de cobertura temporal

## ✅ MÉTODOS APLICÁVEIS POR STAGE

| Stage | Método Validado | Aplicabilidade | Bibliografia | Área | Justificativa com Base nos Dados |
|-------|----------------|----------------|--------------|------|-----------------------------------|
| **01 - Preprocessing** | | | | | |
| | spaCy pt_core_news_lg | ✅ ALTA | Honnibal et al. (2020) | NLP | 846 textos em português brasileiro |
| | Emoji sentiment preservation | ✅ ALTA | Kralj Novak et al. (2015) *PLoS ONE* | Comunicação Digital | Emojis detectados em >30% das mensagens |
| | | | | | |
| **02 - Text Mining** | | | | | |
| | Named Entity Recognition (PT) | ✅ ALTA | Souza et al. (2020) - BERTimbau | NLP/Política | Nomes políticos frequentes (Bolsonaro, STF, etc.) |
| | Frame Analysis | ✅ ALTA | Entman (1993) *J Communication* | Comunicação | Frames eleitorais e pandêmicos identificados |
| | Political Event Extraction | ✅ MÉDIA | Leetaru & Schrodt (2013) GDELT | Ciência Política | Períodos eleitorais claros nos dados |
| | | | | | |
| **03 - Statistical Analysis** | | | | | |
| | STL Decomposition | ✅ ALTA | Cleveland et al. (1990) *JOS* | Estatística | 407 dias únicos, padrões sazonais possíveis |
| | Changepoint Detection | ✅ ALTA | Killick et al. (2012) *JASA* | Estatística | Transições governo-pandemia-eleição visíveis |
| | Mann-Kendall Trend Test | ✅ ALTA | Mann (1945); Kendall (1975) | Estatística | Série temporal longa (3.5 anos) |
| | | | | | |
| **04 - Semantic Analysis** | | | | | |
| | BERTimbau Embeddings | ✅ ALTA | Souza et al. (2020) *STIL* | NLP | Modelo BERT treinado em português |
| | Word2Vec Político | ✅ MÉDIA | Rheault & Cochrane (2020) *AJPS* | Ciência Política | Vocabulário político consistente |
| | Moral Foundations (PT) | ✅ ALTA | Graham et al. (2009); Silveira (2018) | Psicologia Política | Discurso moral-político evidente |
| | | | | | |
| **05 - TF-IDF Analysis** | | | | | |
| | BM25 Ranking | ✅ ALTA | Robertson et al. (1995) *TREC* | IR | 768 documentos únicos suficientes |
| | PMI Collocations | ✅ ALTA | Church & Hanks (1990) *CL* | Linguística Computacional | Colocações políticas identificáveis |
| | Chi-square Feature Selection | ✅ ALTA | Manning & Schütze (1999) | NLP | Features distintivas por período |
| | | | | | |
| **06 - Clustering** | | | | | |
| | HDBSCAN | ✅ ALTA | Campello et al. (2013) *TKDD* | Data Mining | 162 canais, densidades variáveis |
| | Louvain Communities | ✅ ALTA | Blondel et al. (2008) *JSM* | Física Social | Rede de canais identificável |
| | K-means + Silhouette | ✅ MÉDIA | Rousseeuw (1987) *JCA* | Estatística | Grupos temáticos possíveis |
| | | | | | |
| **07 - Topic Modeling** | | | | | |
| | STM (Structural Topic Model) | ✅ ALTA | Roberts et al. (2014) *AJPS* | Ciência Política | Covariáveis temporais disponíveis |
| | BERTopic | ✅ ALTA | Grootendorst (2022) | NLP | Tópicos dinâmicos por período |
| | Guided LDA | ✅ MÉDIA | Jagarlamudi et al. (2012) *ECML* | ML | Seed words políticas aplicáveis |
| | | | | | |
| **08 - Evolution Analysis** | | | | | |
| | Dynamic Topic Models | ✅ ALTA | Blei & Lafferty (2006) *ICML* | ML | 407 dias únicos, evolução clara |
| | Kleinberg Burst Detection | ✅ ALTA | Kleinberg (2003) *KDD* | Data Mining | Eventos burst (eleições) detectáveis |
| | Wavelet Analysis | ✅ MÉDIA | Torrence & Compo (1998) *BAMS* | Física | Ciclos discursivos possíveis |
| | | | | | |
| **09 - Network Coordination** | | | | | |
| | Information Cascades | ✅ MÉDIA | Leskovec et al. (2007) *KDD* | Redes Sociais | 128 forwards, cascatas limitadas |
| | Cross-correlation Analysis | ✅ ALTA | Box & Jenkins (1976) | Séries Temporais | Múltiplos canais síncronos |
| | Granger Causality | ✅ MÉDIA | Granger (1969) *Econometrica* | Econometria | Causalidade entre canais testável |
| | | | | | |
| **10 - Domain/URL Analysis** | | | | | |
| | URL Categorization | ✅ ALTA | Castillo et al. (2011) *WWW* | Web Science | 348 URLs para classificar |
| | Domain Authority | ✅ MÉDIA | Page et al. (1999) PageRank | CS | 84 domínios únicos |
| | Link Co-occurrence | ✅ MÉDIA | Adamic & Glance (2005) *LinkKDD* | Redes | Padrões de compartilhamento |
| | | | | | |
| **11 - Event Context** | | | | | |
| | Critical Discourse Analysis | ✅ ALTA | Wodak (2001) *Discourse & Society* | Linguística Crítica | Contextos claros (eleição/pandemia) |
| | Event Detection (TDT) | ✅ ALTA | Allan et al. (1998) *DARPA* | IR | 316 msgs eleitorais detectáveis |
| | Narrative Analysis | ✅ ALTA | Franzosi (2010) | Sociologia | Narrativas políticas presentes |
| | | | | | |
| **12 - Channel Analysis** | | | | | |
| | Channel Influence (PageRank) | ✅ ALTA | Page et al. (1999) | CS | 162 canais para ranking |
| | KL Divergence | ✅ ALTA | Kullback & Leibler (1951) | Teoria da Informação | Comparação entre canais viável |
| | Cross-platform Analysis | ✅ ALTA | Stier et al. (2018) *SMR* | Comunicação | Multi-canal Telegram |
| | | | | | |
| **13 - Linguistic Analysis** | | | | | |
| | LIWC Portuguese | ✅ ALTA | Balage Filho et al. (2013) *PROPOR* | Psicolinguística | Adaptação PT-BR disponível |
| | Readability (Flesch PT) | ✅ ALTA | Martins et al. (1996) | Linguística Aplicada | Fórmula adaptada português |
| | Stylometry | ✅ BAIXA | Burrows (2002) *LLC* | Humanidades Digitais | 29% msgs sem autor identificado |

## ❌ MÉTODOS NÃO APLICÁVEIS

| Método | Razão | Dados Necessários |
|--------|-------|-------------------|
| **Virality Prediction** | SEM métricas de engajamento | Likes, shares, retweets |
| **Influence Metrics (engagement-based)** | SEM dados de interação | Views, reactions |
| **Conversation Analysis** | SEM estrutura de threads | Reply chains, thread IDs |
| **Full Authorship Attribution** | 29% dados faltantes em 'sender' | Autoria completa |
| **A/B Testing Analysis** | Dataset observacional apenas | Dados experimentais |

## 🎯 RECOMENDAÇÕES PRIORITÁRIAS

### TOP 5 Métodos Mais Adequados para seu Dataset:

1. **STM (Structural Topic Model)** - Roberts et al. (2014)
   - Perfeito para análise política com covariáveis temporais
   - Implementação: `stm` (R) ou `stmpy` (Python)

2. **BERTimbau** - Souza et al. (2020)
   - BERT treinado especificamente em português brasileiro
   - Implementação: `transformers` com 'neuralmind/bert-base-portuguese-cased'

3. **Critical Discourse Analysis** - Wodak (2001)
   - Ideal para períodos políticos contrastantes
   - Framework qualitativo + quantitativo

4. **HDBSCAN** - Campello et al. (2013)
   - Detecta comunidades discursivas sem K predefinido
   - Implementação: `hdbscan` Python

5. **Kleinberg Burst Detection** - Kleinberg (2003)
   - Identifica momentos críticos (eleições, crises)
   - Implementação: `pybursts`

## 📝 OBSERVAÇÕES FINAIS

- **Dataset rico**: Todos os 13 stages têm métodos aplicáveis
- **Foco recomendado**: Análise temporal-discursiva dado o período crítico (2019-2023)
- **Limitação principal**: Falta de métricas de engajamento limita análise de influência
- **Força principal**: Cobertura temporal de eventos políticos cruciais no Brasil
