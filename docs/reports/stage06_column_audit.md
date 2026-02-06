# Auditoria de Colunas - Stage 06: Affordances Classification

**Data da auditoria:** 2026-02-06
**Arquivo analisado:** `src/analyzer.py` (linhas 1844-2569)
**Metodo:** `_stage_06_affordances_classification` + `_stage_06_affordances_heuristic_fallback`

---

## 1. Colunas existentes no DataFrame ANTES do Stage 06

Colunas acumuladas dos stages 01-05 que entram no Stage 06:

### Do dataset original (pre-pipeline)
| Coluna | Origem |
|--------|--------|
| `body` | Dataset original |
| `url` | Dataset original |
| `hashtag` | Dataset original |
| `channel` | Dataset original |
| `is_fwrd` | Dataset original |
| `mentions` | Dataset original |
| `sender` | Dataset original |
| `media_type` | Dataset original |
| `domain` | Dataset original |

### Stage 01 - Feature Extraction
| Coluna | Descricao |
|--------|-----------|
| `datetime` | Timestamp padronizado DD/MM/AAAA HH:MM:SS |
| `main_text_column` | Nome da coluna de texto principal |
| `timestamp_column` | Nome da coluna de timestamp |
| `metadata_columns_count` | Contagem de colunas de metadados |
| `has_timestamp` | Flag booleana |
| `hashtags_extracted` | Hashtags extraidas do texto |
| `urls_extracted` | URLs extraidas do texto |
| `mentions_extracted` | Mencoes extraidas do texto |
| `emojis_extracted` | Emojis extraidos do texto |

### Stage 02 - Text Preprocessing
| Coluna | Descricao |
|--------|-----------|
| `normalized_text` | Texto limpo e normalizado (lowercase, sem especiais) |

### Stage 03 - Cross-Dataset Deduplication
| Coluna | Descricao |
|--------|-----------|
| `dupli_freq` | Frequencia de duplicacao do texto |
| `channels_found` | Numero de canais onde o texto apareceu |
| `date_span_days` | Periodo em dias entre primeira e ultima ocorrencia |

### Stage 04 - Statistical Analysis
| Coluna | Descricao |
|--------|-----------|
| `char_count` | Contagem de caracteres |
| `word_count` | Contagem de palavras |
| `emoji_ratio` | Proporcao de emojis no texto |
| `caps_ratio` | Proporcao de letras maiusculas |
| `repetition_ratio` | Proporcao de repeticoes |
| `likely_portuguese` | Deteccao de idioma portugues |

### Stage 05 - Content Quality Filter
| Coluna | Descricao |
|--------|-----------|
| `content_quality_score` | Score de qualidade (0-100) |

---

## 2. Colunas ADICIONADAS pelo Stage 06

O Stage 06 adiciona **10 colunas** ao DataFrame:

| # | Coluna | Tipo | Descricao | Usa API Anthropic? |
|---|--------|------|-----------|-------------------|
| 1 | `affordance_categories` | list[str] | Lista de categorias atribuidas (ex: `['noticia', 'opiniao']`) | Parcial* |
| 2 | `affordance_confidence` | float | Score de confianca (0.0 a 1.0) | Parcial* |
| 3 | `aff_noticia` | int (0/1) | Binario: conteudo informativo/reportagem | Parcial* |
| 4 | `aff_midia_social` | int (0/1) | Binario: posts de redes sociais | Parcial* |
| 5 | `aff_video_audio_gif` | int (0/1) | Binario: referencias a multimidia | Parcial* |
| 6 | `aff_opiniao` | int (0/1) | Binario: opinioes pessoais | Parcial* |
| 7 | `aff_mobilizacao` | int (0/1) | Binario: chamadas para acao politica | Parcial* |
| 8 | `aff_ataque` | int (0/1) | Binario: ataques pessoais/agressao verbal | Parcial* |
| 9 | `aff_interacao` | int (0/1) | Binario: respostas/conversacoes | Parcial* |
| 10 | `aff_is_forwarded` | int (0/1) | Binario: conteudo encaminhado | Parcial* |

**Coluna temporaria removida:** `_heuristic_scores` (criada e deletada dentro do stage)

\* **Parcial** = Todas as mensagens sao primeiro classificadas por heuristica (Python puro). Apenas mensagens com `affordance_confidence < 0.6` sao enviadas a API Anthropic para reclassificacao.

---

## 3. Colunas MODIFICADAS pelo Stage 06

Nenhuma coluna pre-existente e modificada. O Stage 06 apenas **adiciona** colunas novas.

---

## 4. Rastreamento de Uso: Colunas Stage 06 nos Stages Posteriores

### 4.1 Referencia nos Pipeline Stages (analyzer.py)

| Coluna | Stage 07 | Stage 08 | Stage 09 | Stage 10 | Stage 11 | Stage 12 | Stage 13 | Stage 14 | Stage 15 | Stage 16 | Stage 17 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| `affordance_categories` | - | - | - | - | - | - | - | - | - | - | - |
| `affordance_confidence` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_noticia` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_midia_social` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_video_audio_gif` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_opiniao` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_mobilizacao` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_ataque` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_interacao` | - | - | - | - | - | - | - | - | - | - | - |
| `aff_is_forwarded` | - | - | - | - | - | - | - | - | - | - | - |

**RESULTADO: NENHUMA coluna do Stage 06 e referenciada como input em qualquer stage posterior do pipeline (07-17).**

### 4.2 Referencia nos Dashboards

| Coluna | Dashboard Stage 06 | Dashboard Stage 11 | Dashboard Stage 13 | Page Affordances |
|--------|-------------------|--------------------|--------------------|-----------------|
| `affordance_categories` | INPUT (distribuicao, co-ocorrencia, Sankey) | INPUT (cruzamento topico-affordance) | - | Documentacao |
| `affordance_confidence` | INPUT (histograma, filtro, media) | - | - | Documentacao |
| `aff_noticia` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_midia_social` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_video_audio_gif` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_opiniao` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_mobilizacao` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_ataque` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_interacao` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |
| `aff_is_forwarded` | INPUT (contagem, exclusividade, temporal) | - | - | Documentacao |

**NOTA:** O `stage13_temporal_dashboard.py` referencia `affordances_score` (NAO e coluna do Stage 06 - e uma coluna inexistente/diferente).

---

## 5. Tabela Consolidada: Classificacao de Colunas

| Coluna | Consumida por Stage Pipeline? | Consumida por Dashboard? | Tipo de Uso | Classificacao |
|--------|-------------------------------|--------------------------|-------------|---------------|
| `affordance_categories` | NENHUM | Stage 06 dashboard, Stage 11 dashboard | Visualizacao, cruzamento | UTIL (dashboard only) |
| `affordance_confidence` | NENHUM | Stage 06 dashboard | Visualizacao, filtro | UTIL (dashboard only) |
| `aff_noticia` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_midia_social` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_video_audio_gif` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_opiniao` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_mobilizacao` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_ataque` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_interacao` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |
| `aff_is_forwarded` | NENHUM | Stage 06 dashboard | Contagem, temporal | UTIL (dashboard only) |

### Legenda de Classificacao

| Classificacao | Criterio | Colunas |
|---------------|----------|---------|
| **ESSENCIAL** | Usada em 2+ stages posteriores do pipeline | *Nenhuma coluna do Stage 06* |
| **UTIL** | Usada em 1 stage posterior OU em dashboards | Todas as 10 colunas (apenas dashboards) |
| **CANDIDATA A REMOCAO** | Nao usada em nenhum stage posterior | *Nenhuma, pois todas aparecem em dashboards* |

---

## 6. Estimativa de Custo API Anthropic

### Mecanismo do Stage 06

```
Fase 1: Heuristica em 100% das mensagens (custo = $0)
Fase 2: API apenas para mensagens com affordance_confidence < 0.6
Fase 3: Batches de 10 mensagens por chamada API
```

### Modelo utilizado
- **Default:** `claude-3-5-haiku-20241022` (configuravel via `ANTHROPIC_MODEL` no .env)
- **Batch API:** Disponivel com 50% desconto para >100 mensagens (`USE_BATCH_API=true`)

### Estimativa de custo por coluna

Se o Stage 06 fosse removido, TODAS as 10 colunas seriam eliminadas simultaneamente (sao geradas pelo mesmo processo). Nao e possivel remover colunas individualmente.

| Cenario | Mensagens totais | Mensagens p/ API (est. 30-40%) | Chamadas API (batches de 10) | Custo estimado |
|---------|------------------|-------------------------------|------------------------------|----------------|
| Dataset pequeno | 10.000 | ~3.500 | ~350 | ~$0.35-0.70 |
| Dataset medio | 100.000 | ~35.000 | ~3.500 | ~$3.50-7.00 |
| Dataset grande | 300.000 | ~105.000 | ~10.500 | ~$10.50-21.00 |

**Calculo base (Haiku 3.5):**
- Input: ~400 tokens/mensagem + ~200 tokens system prompt (cacheado)
- Output: ~100 tokens/resposta (JSON)
- Custo por batch de 10: ~$0.001-0.002
- Com Batch API (50% desconto): custo reduzido pela metade

### Economia ao remover Stage 06 completo

| Cenario | Economia API | Economia tempo |
|---------|-------------|----------------|
| Sem API key (heuristica only) | $0.00 | Minutos (classificacao rapida) |
| API sincrona | $0.35-21.00 (depende do dataset) | 5-60 minutos |
| Batch API | $0.18-10.50 (50% desconto) | 1-24 horas (async) |

---

## 7. Achados Criticos

### 7.1 Desconexao Pipeline-Dashboard

**TODAS as 10 colunas do Stage 06 sao "ilhas" no pipeline.** Nenhum stage posterior (07-17) as utiliza como input. As colunas so sao consumidas pelos dashboards de visualizacao.

Isso significa que:
- O pipeline pode rodar stages 07-17 **sem executar o Stage 06** sem quebrar nada
- As colunas sao uteis apenas para visualizacao/exploracao via Streamlit dashboards
- Nao ha dependencia funcional entre Stage 06 e o restante do pipeline

### 7.2 Custo vs. Valor

O Stage 06 e o **unico stage que faz chamadas API Anthropic** no pipeline principal. Se executado com API:
- Adiciona custo financeiro variavel ($0.35-$21.00)
- Adiciona latencia significativa (minutos a horas)
- O resultado nao alimenta nenhuma analise posterior automatizada

### 7.3 Redundancia com Stage 17

O Stage 17 (`_stage_17_channel_analysis`) gera a coluna `is_forwarded` a partir de `is_fwrd`, enquanto Stage 06 gera `aff_is_forwarded` por heuristica de texto. Potencial sobreposicao.

### 7.4 Coluna fantasma no Stage 13 Dashboard

O `stage13_temporal_dashboard.py` referencia `affordances_score` (linha 75, 733, 770) que **NAO e uma coluna gerada pelo Stage 06** nem por nenhum outro stage. Isso sugere um bug ou coluna removida sem atualizar o dashboard.

---

## 8. Recomendacoes

1. **Considerar tornar Stage 06 opcional** - executar apenas quando dashboards de affordances forem necessarios
2. **Priorizar modo heuristica** - O fallback heurístico (custo $0) cobre 100% das mensagens. A API melhora precisao mas com custo
3. **Investigar `affordances_score`** no stage13_temporal_dashboard - coluna inexistente referenciada
4. **Avaliar integracao** - se as colunas `aff_*` fossem inputs para stages posteriores (ex: clustering por affordance, analise temporal por tipo), o Stage 06 teria valor analitico muito maior
5. **Remover `aff_is_forwarded`** se `is_fwrd` (dataset original) ja cobre o mesmo dado de forma mais confiavel

---

*Auditoria gerada por analise estatica do codigo em `src/analyzer.py` e busca grep em todo o repositorio.*
