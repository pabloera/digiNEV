# Consolidacao: Integracao de 3 Fases - Pipeline digiNEV

**Data:** 2026-02-06
**Arquivos modificados:** `src/analyzer.py` (1,099+, 288-), `src/lexicon_loader.py` (33+, 2-)
**Arquivos criados:** `src/core/__init__.py`, `src/core/lexico_unified_system.json`, `src/core/lexico_politico_hierarquizado.json`, `src/core/political_keywords_dict.py`

---

## Evidencia Concreta: ANTES vs DEPOIS (200 registros reais, dataset 1)

### political_orientation
| Valor | ANTES | DEPOIS |
|-------|-------|--------|
| neutral | 91 (45.5%) | 20 (10.0%) |
| direita | 5 (2.5%) | 122 (61.0%) |
| extrema-direita | 58 (29.0%) | 42 (21.0%) |
| centro-direita | 2 (1.0%) | 16 (8.0%) |
| esquerda | 43 (21.5%) | 0 |
| centro-esquerda | 1 (0.5%) | 0 |

**Mudanca:** De 45.5% "neutral" para 10.0%. Classificacao agora usa 777 termos do lexico unificado.

### political_intensity
| Metrica | ANTES | DEPOIS |
|---------|-------|--------|
| mean | 0.0100 | 0.2027 |
| registros > 0 | 18/200 (9%) | 112/200 (56%) |

**Mudanca:** De 10 palavras genericas para termos de mobilizacao_acao + autoritarismo_violencia + desinformacao_verdade do lexico.

### sentiment_label
| Valor | ANTES | DEPOIS |
|-------|-------|--------|
| neutral | 200 (100%) | 196 (98%) |
| positive | 0 | 2 (1%) |
| negative | 0 | 2 (1%) |

**Mudanca:** De 14 palavras (7+7) para ~90 LIWC-PT expandido (Balage Filho et al. 2013).

### emotion_intensity
| Metrica | ANTES | DEPOIS |
|---------|-------|--------|
| mean | 0.0000 | 0.2530 |
| registros > 0 | 0/200 (0%) | 119/200 (59.5%) |

**Mudanca:** Corrigido bug - agora usa `body` (texto original com !, ?, CAPS) em vez de `normalized_text` (que remove pontuacao).

### topic_keywords (LDA)
| Posicao | ANTES | DEPOIS |
|---------|-------|--------|
| topic_keywords[0] | ['da', 'de', 'do'] | ['brasil', 'sa', 'poli'] |
| Stopwords presentes | SIM | NENHUMA |

**Mudanca:** Adicionadas 120+ stopwords PT ao CountVectorizer do LDA.

### has_aggressive_language
| Metrica | ANTES | DEPOIS |
|---------|-------|--------|
| registros positivos | 7/200 (3.5%) | 45/200 (22.5%) |

**Mudanca:** De 9 palavras fixas para termos de autoritarismo_violencia do lexico + 20 termos de agressao.

---

## Novas Colunas Adicionadas (26 colunas novas)

### Frame Analysis - Entman (1993)
| Coluna | mean | registros > 0 |
|--------|------|---------------|
| frame_conflito | 0.0141 | 26/200 (13%) |
| frame_responsabilizacao | 0.0072 | 12/200 (6%) |
| frame_moralista | 0.0054 | 11/200 (5.5%) |
| frame_economico | 0.0042 | 10/200 (5%) |

### Domain Analysis - Page et al. (1999)
| Coluna | Valor |
|--------|-------|
| domain_trust_score mean | 0.113 |
| domain_type distribuicao | unknown=139, alternative=33, video=13, social=7, news=5, gov=3 |

### Categorias Tematicas (political_keywords_dict.py, 98 termos)
| Categoria | total_hits | registros |
|-----------|-----------|-----------|
| cat_inimigos_ideologicos | 131 | 104/200 (52%) |
| cat_identidade_politica | 85 | 74/200 (37%) |
| cat_meio_ambiente_amazonia | 18 | 18/200 (9%) |
| cat_antissistema | 11 | 10/200 (5%) |
| cat_moralidade | 11 | 11/200 (5.5%) |
| cat_violencia_seguranca | 5 | 5/200 (2.5%) |
| cat_polarizacao | 3 | 3/200 (1.5%) |
| cat_autoritarismo_regime | 2 | 2/200 (1%) |
| cat_religiao_moral | 2 | 2/200 (1%) |
| cat_pandemia_covid | 0 | 0/200 (0%) |

### Outras Novas
| Coluna | Descricao |
|--------|-----------|
| is_burst_day | Kleinberg burst detection (dias com volume > 2 desvios padrao) |
| domain_trust_score | Score de confianca 0-0.9 por tipo de dominio |

---

## Metodos Integrados e Referencias

| Metodo | Referencia | Stage | Status |
|--------|-----------|-------|--------|
| Lexico Unificado (777 termos) | Projeto digiNEV/FAPESP | Stage 08 | Integrado |
| LIWC-PT Expandido | Balage Filho et al. (2013) | Stage 12 | Integrado |
| Frame Analysis | Entman (1993), J Communication 43(4): 51-58 | Stage 16 | Integrado |
| Domain Authority | Page et al. (1999), adaptado | Stage 15 | Integrado |
| Burst Detection | Kleinberg (2003), KDD | Stage 13 | Integrado |
| HDBSCAN | McInnes et al. (2017) | Stage 10 | Integrado (fallback KMeans) |
| Keywords Tematicas | political_keywords_dict.py (10 cats) | Stage 08 | Integrado |
| Mann-Kendall Trend | Mann (1945); Kendall (1975) | Helper | Pronto |
| Information Cascades | Leskovec et al. (2007) | Helper | Pronto |
| Stopwords PT | Lista 120+ termos | Stage 11 | Integrado |

---

## Resumo de Mudancas por Arquivo

### src/analyzer.py (1,099 insercoes, 288 remocoes)
- Import LexiconLoader
- `__init__`: Carrega lexicon unificado via LexiconLoader, inicializa `_political_terms_map`
- `_load_political_lexicon()`: Usa LexiconLoader em vez de fallback hardcoded
- Stage 08: Adiciona 10 categorias tematicas (cat_*)
- Stage 10: HDBSCAN com fallback KMeans
- Stage 11: CountVectorizer com 120+ stopwords PT
- Stage 12: `emotion_intensity` usa body (texto original)
- Stage 13: Burst detection por dia
- Stage 15: Domain trust score + domain_type expandido
- Stage 16: Frame Analysis (4 frames Entman)
- Todos os helper methods expandidos com lexico unificado (777 termos)
- Novos helpers: `_analyze_political_frames()`, `_mann_kendall_trend_test()`, `_detect_information_cascades()`

### src/lexicon_loader.py (33 insercoes, 2 remocoes)
- Auto-deteccao de path (`src/core/lexico_unified_system.json`)
- Cache de termos por categoria (`_terms_by_category_cache`)
- Novo metodo `get_terms_by_category_map()`

### src/core/ (novos arquivos)
- `__init__.py` (criado)
- `lexico_unified_system.json` (777 termos, 9 macrotemas, copiado de archive)
- `lexico_politico_hierarquizado.json` (copiado de archive)
- `political_keywords_dict.py` (98 keywords, 10 categorias, copiado de archive)
