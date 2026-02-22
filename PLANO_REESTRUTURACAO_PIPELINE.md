# PLANO DE REESTRUTURAÇÃO DO PIPELINE digiNEV

**Data**: 2026-02-22
**Status**: ✅ EXECUTADO E VALIDADO (TARETAs 1-11 completas, 4 testes ponta-a-ponta)
**Arquivo principal**: `src/analyzer.py` (~3,930 linhas)

---

## DIAGNÓSTICO: 8 Bugs + spaCy Desperdiçado

O pipeline roda spaCy (Stage 07) e produz `lemmatized_text`, `spacy_tokens`, `spacy_lemmas` — mas **nenhum stage downstream os consome**. Além disso, 8 bugs concretos comprometem filtros e classificações.

### Bugs Identificados

| # | Bug | Linhas | Impacto |
|---|---|---|---|
| 1 | `'tokens'` vs `'spacy_tokens'` — nome errado | 2732, 2930, 3188 | TF-IDF, Topics, Semântica nunca usam spaCy |
| 2 | `caps_ratio` sempre 0.0 (texto já lowercase) | ~2597 via Stage 04 L.1706 | Filtro CAPS não funciona |
| 3 | `emoji_ratio` sempre 0.0 (emojis já removidos) | ~2573 via Stage 04 L.1705 | Filtro emoji não funciona |
| 4 | Detecção hashtag falha (`#` removido) | Stage 04 L.1678 | Estatística sempre zero |
| 5 | `'text_length'` não existe (é `'char_count'`) | Stage 10 L.2860 | Clustering com 3 features em vez de 4 |
| 6 | Regex URL nunca match (`://` removido) | Stage 06 L.2508 | Detecção notícia comprometida |
| 7 | spaCy recebe `normalized_text` (degradado) | Stage 07 L.1506 | NER e POS degradados |
| 8 | `lemmatized_text` nunca consumido | Pipeline inteiro | Lematização desperdiçada |

---

## DECISÃO: TCW INTEGRADO NO STAGE 08 (NÃO COMO STAGE SEPARADO)

### Evidência da Investigação de Duplicação

Comparação termo-a-termo realizada:

| Métrica | Valor |
|---|---|
| TCW termos únicos (lowercase) | 181 |
| digiNEV termos totais (léxico + keywords_dict) | 966 |
| **Termos em AMBOS** | **77 (42.5% do TCW)** |
| Termos SÓ no TCW | 104 (57.5% do TCW) |
| Termos SÓ no digiNEV | 889 |

### Por que NÃO criar stage separado:

1. **77 termos duplicados** — Stage TCW separado detectaria os mesmos termos que Stage 08 já detecta, gerando colunas redundantes
2. **Duas passadas sobre o mesmo texto** — desperdício computacional para classificar o mesmo texto duas vezes
3. **Conflito de outputs** — Stage 08 diria `stf → inimigos_ideologicos`, Stage TCW diria `stf → Instituições` — qual prevalece para downstream?
4. **Proliferação de colunas** — 13 colunas digiNEV + 4 colunas TCW = 17 colunas de classificação

### Por que INTEGRAR no Stage 08:

1. **Uma passada, múltiplos outputs** — o matching por token/lemma encontra o termo UMA vez e atribui ambos os rótulos (digiNEV discursivo + TCW temático)
2. **Zero redundância** — cada termo é processado uma única vez
3. **Léxico expandido** — os 104 termos TCW-only são ADICIONADOS ao léxico unificado
4. **Conflito preservado como dado** — ambos os rótulos coexistem na mesma estrutura (ex: `stf → {diginev: "inimigos_ideologicos", tcw_code: 140, tcw_cat: "Instituições"}`)

---

## PLANO DE EXECUÇÃO: 10 TAREFAS

### TAREFA 1: Corrigir Stage 04 — Métricas sobre `body`

**Arquivo**: `src/analyzer.py`
**O que muda**: `caps_ratio`, `emoji_ratio` e hashtag detection passam a usar `body` em vez de `normalized_text`

**Alterações**:
- L.1705-1710: Mudar input de `df[text_column]` para `df['body']` (ou coluna raw detectada) para `emoji_ratio`, `caps_ratio`
- L.1678: Substituir `df[text_column].str.contains('#')` por `df['hashtags_extracted'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)` (usar output do Stage 01)

**Resultado**: Filtros de qualidade (Stage 05) passam a funcionar — CAPS-spam e emoji-spam serão filtrados.

---

### TAREFA 2: Corrigir Stage 07 — spaCy recebe `body` (texto cru)

**Arquivo**: `src/analyzer.py`
**O que muda**: L.1506 muda de `df['normalized_text']` para `df['body']` (ou coluna raw)

**Alterações**:
- L.1506: `spacy_results = df['body'].apply(process_text_with_spacy)` (com fallback para `normalized_text`)
- Dentro de `process_text_with_spacy` (L.1470-1503): manter truncamento a 1000 chars, manter try/except

**Resultado**: NER reconhece "Bolsonaro" como PER (pessoa), POS tagging funciona com pontuação, lematização mais precisa.

---

### TAREFA 3: Corrigir Stages 09, 11, 12 — `tokens` → `spacy_tokens`

**Arquivo**: `src/analyzer.py`
**O que muda**: 3 linhas com nome de coluna errado

**Alterações**:
- L.2732: `if 'tokens' not in df.columns` → `if 'spacy_tokens' not in df.columns`
- L.2736-2738: `df['tokens']` → `df['spacy_tokens']`
- L.2930: `if 'tokens' not in df.columns` → `if 'spacy_tokens' not in df.columns`
- L.2934-2936: `df['tokens']` → `df['spacy_tokens']`
- L.3188: `if 'tokens' in df.columns` → `if 'spacy_tokens' in df.columns`
- L.3189: `df['tokens']` → `df['spacy_tokens']`

**Melhor ainda**: Todas essas linhas deveriam usar `lemmatized_text` em vez de tokens raw:
- TF-IDF: `text_data = df['lemmatized_text'].fillna('').tolist()`
- Topic Modeling: `text_data = df['lemmatized_text'].fillna('').tolist()`
- Semântica: `df['semantic_diversity'] = df['lemmatized_text'].apply(...)`

**Resultado**: TF-IDF, Topic Modeling e Semântica passam a usar lemmas do spaCy.

---

### TAREFA 4: Corrigir Stage 10 — `text_length` → `char_count`

**Arquivo**: `src/analyzer.py`
**O que muda**: L.2860

**Alteração**:
- L.2860: `'text_length'` → `'char_count'`

**Resultado**: Clustering usa 4 features em vez de 3.

---

### TAREFA 5: Corrigir Stage 06 — URL detection via `urls_extracted`

**Arquivo**: `src/analyzer.py`
**O que muda**: L.2508 (no heuristic fallback)

**Alteração**:
- Substituir `re.search(r'https?://', text_lower)` por verificação de `urls_extracted` do Stage 01
- Ex: `has_url = isinstance(row.get('urls_extracted'), list) and len(row['urls_extracted']) > 0`

**Resultado**: Detecção de notícia recupera funcionalidade de URL.

---

### TAREFA 6: Expandir `lexico_unified_system.json` com termos TCW únicos

**Arquivo**: `src/core/lexico_unified_system.json`
**O que muda**: Adicionar 2 novos macrotemas + termos em macrotemas existentes

**Novos macrotemas**:

```json
"corrupcao_transparencia": {
  "nome": "Corrupção e Transparência",
  "descricao": "Discurso anticorrupção e accountability",
  "subtemas": {
    "corrupcao_geral": {
      "palavras": ["corrupto", "corrupção", "propina", "suborno", "desvio"],
      "expressoes": ["lava jato", "delação premiada", "orçamento secreto"]
    },
    "escândalos_historicos": {
      "palavras": ["mensalão", "petrolão", "rachadinha"],
      "expressoes": ["operação lava jato", "ministério público", "polícia federal"]
    }
  }
},
"politica_externa": {
  "nome": "Política Externa e Relações Internacionais",
  "descricao": "Alinhamentos internacionais e geopolítica",
  "subtemas": {
    "relacoes_internacionais": {
      "palavras": ["diplomacia", "sanções", "isolacionismo", "alinhamento", "mercosul", "embaixada"],
      "expressoes": ["política externa", "comunidades internacionais"]
    },
    "figuras_internacionais": {
      "palavras": ["trump", "eua", "israel", "greta"],
      "expressoes": []
    }
  }
}
```

**Termos adicionais em macrotemas existentes**:
- `negacionismo.negacionismo_historico` + `ustra`, `nostalgismo autoritário`
- `autoritarismo_violencia.autoritarismo_institucional` + `inquérito das fake news`, `ativismo judicial`
- `identidade_patriotica.militarismo_ordem` + `cidadão de bem`, `excludente de ilicitude`, `armamentismo`
- `identidade_patriotica.valores_conservadores` + `aborto`, `homossexualidade`, `bancada evangélica`, `escola sem partido`, `mimimi`, `liberdade de expressão`

**NÃO incluir** termos raciais polissêmicos (preto, negro, minoria, cota) — risco alto de falsos positivos sem NER.

---

### TAREFA 7: Expandir `political_keywords_dict.py` com 2 novas categorias

**Arquivo**: `src/core/political_keywords_dict.py`
**O que muda**: Adicionar `cat11_corrupcao` e `cat12_politica_externa`

```python
'cat11_corrupcao': [
    'lava jato', 'propina', 'petrolão', 'mensalão', 'rachadinha',
    'orçamento secreto', 'delação premiada', 'ministério público',
    'polícia federal', 'corrupto', 'corrupção'
],
'cat12_politica_externa': [
    'trump', 'eua', 'israel', 'mercosul', 'diplomacia', 'sanções',
    'isolacionismo', 'alinhamento', 'política externa', 'greta'
]
```

**Resultado**: Stage 08 gera `cat_corrupcao` e `cat_politica_externa` automaticamente.

---

### TAREFA 8: Reformular Stage 08 — Token-based matching com spaCy lemmas

**Arquivo**: `src/analyzer.py`
**O que muda**: Stage 08 e seus helpers passam a usar `lemmatized_text` com token matching em `set()`

**Mudanças nos helpers**:

`_classify_political_orientation(self, text, lemmas=None)`:
- Se `lemmas` fornecido (lista de strings): match por lookup em set
- Se não: fallback para substring (compatibilidade)
- Lógica de classificação (extrema-direita/direita/etc.) mantida idêntica

`_extract_political_keywords(self, text, lemmas=None)`:
- Se `lemmas` fornecido: intersecção entre set de lemmas e set de termos do léxico
- Retorna lista de matches

`_calculate_political_intensity(self, text, lemmas=None)`:
- Mesmo padrão: set intersection quando lemmas disponível

**Mudança no Stage 08**:
```python
# Antes (L.2684):
df['political_orientation'] = df[text_column].apply(self._classify_political_orientation)

# Depois:
df['political_orientation'] = df.apply(
    lambda row: self._classify_political_orientation(
        str(row.get('normalized_text', '')),
        lemmas=row.get('spacy_lemmas', None)
    ), axis=1
)
```

**Mudança nas categorias temáticas (L.2695-2697)**:
```python
# Antes:
lambda text, terms=cat_terms: sum(1 for t in terms if t in str(text).lower())

# Depois:
lambda row, terms_set=set(cat_terms): (
    len(terms_set & set(row.get('spacy_lemmas', []))) if row.get('spacy_lemmas')
    else sum(1 for t in terms_set if t in str(row.get('normalized_text', '')).lower())
)
```

**Resultado**: Eliminação de falsos positivos (word-boundary via tokenização), cobertura morfológica automática (flexões verbais).

---

### TAREFA 9: Adicionar codificação TCW no Stage 08

**Arquivo**: `src/analyzer.py`
**O que muda**: Após a classificação política, adicionar mapeamento TCW

**Novo código no Stage 08** (após L.2698):

```python
# Codificação TCW (Tabela-Categoria-Palavra)
try:
    tcw_lookup = self._load_tcw_codes()  # Carrega tcw_codes.json

    def assign_tcw(lemmas, text):
        if not lemmas and not text:
            return [], [], 0
        # Match por lemma (prioridade) ou por texto normalizado (fallback)
        search_tokens = set(lemmas) if lemmas else set(str(text).lower().split())
        codes = []
        for token in search_tokens:
            if token in tcw_lookup:
                codes.extend(tcw_lookup[token])
        codes = list(set(codes))[:20]  # Max 20 códigos
        categories = list(set(str(c)[1] for c in codes if len(str(c)) == 3))
        agreement = _calc_agreement(codes)  # Quantas tabelas concordam
        return codes, categories, agreement

    tcw_results = df.apply(
        lambda row: assign_tcw(
            row.get('spacy_lemmas', None),
            row.get('normalized_text', '')
        ), axis=1
    )
    df['tcw_codes'] = tcw_results.apply(lambda x: x[0])
    df['tcw_categories'] = tcw_results.apply(lambda x: x[1])
    df['tcw_agreement'] = tcw_results.apply(lambda x: x[2])
except Exception as e:
    self.logger.warning(f"TCW encoding skipped: {e}")
```

**Arquivo auxiliar necessário**: Copiar `tcw_codes.json` (lookup table) para `src/core/tcw_codes.json`

**Resultado**: Stage 08 produz `tcw_codes`, `tcw_categories`, `tcw_agreement` como colunas adicionais. Sem stage separado, sem duplicação.

---

### TAREFA 10: Atualizar Stage 16 — Event Context usa `lemmatized_text`

**Arquivo**: `src/analyzer.py`
**O que muda**: L.3217 e L.3220-3233

**Alterações**:
- L.3217: Preferir `lemmatized_text` quando disponível
- L.3220-3233: Helpers de detecção de contexto recebem lemmas

**Resultado**: Detecção de contexto político e frames captura flexões verbais.

---

## ORDEM DE EXECUÇÃO

```
TAREFA 1: Corrigir Stage 04 (caps/emoji/hashtag sobre body)
TAREFA 2: Corrigir Stage 07 (spaCy recebe body)
TAREFA 3: Corrigir Stages 09/11/12 (tokens → spacy_tokens/lemmatized_text)
TAREFA 4: Corrigir Stage 10 (text_length → char_count)
TAREFA 5: Corrigir Stage 06 (URL detection via urls_extracted)
TAREFA 6: Expandir lexico_unified_system.json (+2 macrotemas, +termos)
TAREFA 7: Expandir political_keywords_dict.py (+2 categorias)
TAREFA 8: Reformular Stage 08 (token-based matching com lemmas)
TAREFA 9: Adicionar codificação TCW no Stage 08
TAREFA 10: Atualizar Stage 16 (lemmatized_text)
```

**Tarefas 1-5**: Correções de bugs (baixo risco, alto impacto)
**Tarefas 6-7**: Expansão de léxicos (médio risco, dados novos)
**Tarefas 8-9**: Reformulação do Stage 08 (alto impacto, requer testes)
**Tarefa 10**: Melhoria incremental

---

## ARQUIVOS MODIFICADOS

| Arquivo | Tarefas |
|---|---|
| `src/analyzer.py` | 1, 2, 3, 4, 5, 8, 9, 10 |
| `src/core/lexico_unified_system.json` | 6 |
| `src/core/political_keywords_dict.py` | 7 |
| `src/core/tcw_codes.json` (NOVO — copiar) | 9 |

---

## FLUXO RESULTANTE

```
FASE 1: PREPARAÇÃO
  01 → Feature Extraction [body]
  02 → Text Preprocessing [body → normalized_text]

FASE 2: REDUÇÃO DE VOLUME
  03 → Deduplicação [normalized_text]
  04 → Estatística [body para caps/emoji, normalized_text para contagens] ← CORRIGIDO
  05 → Filtro de Qualidade [métricas funcionais] ← AGORA FUNCIONA
  06 → Affordances [normalized_text, urls_extracted] ← CORRIGIDO

  ═══ Volume: ~300k → ~80k ═══

FASE 3: PROCESSAMENTO LINGUÍSTICO
  07 → spaCy [body → spacy_tokens, spacy_lemmas, lemmatized_text] ← CORRIGIDO

FASE 4: CLASSIFICAÇÃO E ANÁLISE
  08 → Classificação Unificada (digiNEV + TCW) [spacy_lemmas] ← REFORMULADO
       → political_orientation, keywords, intensity
       → cat_* (12 categorias: 10 originais + corrupcao + politica_externa)
       → tcw_codes, tcw_categories, tcw_agreement
  09 → TF-IDF [lemmatized_text] ← CORRIGIDO
  11 → Topic Modeling [lemmatized_text] ← CORRIGIDO
  12 → Semântica [lemmatized_text + body] ← CORRIGIDO
  16 → Event Context [lemmatized_text] ← MELHORADO

FASE 5: ANÁLISES ESTRUTURAIS (sem mudança)
  10 → Clustering [features numéricas, char_count] ← CORRIGIDO
  13 → Temporal [datetime]
  14 → Network [metadados]
  15 → Domain [domain/URLs]
  17 → Channel [metadados]
```

---

## VALIDAÇÃO PÓS-IMPLEMENTAÇÃO

Para cada tarefa concluída, rodar:
```bash
python test_clean_analyzer.py
python run_pipeline.py --dataset data/controlled_test_100.csv
```

Verificar:
- [ ] `caps_ratio` > 0 para textos com CAPS
- [ ] `emoji_ratio` > 0 para textos com emojis
- [ ] `spacy_entities_count` > 0 (NER funcionando)
- [ ] `lemmatized_text` ≠ `normalized_text`
- [ ] `tfidf_top_terms` contém lemmas (não formas flexionadas)
- [ ] `dominant_topic` mais coerente
- [ ] `political_keywords` sem falsos positivos ("pt" não aparece em "apto")
- [ ] `cat_corrupcao` e `cat_politica_externa` existem e > 0
- [ ] `tcw_codes` preenchido para textos com palavras-chave
- [ ] `tcw_agreement` varia entre 1 e 3
