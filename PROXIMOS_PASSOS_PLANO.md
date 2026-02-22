# PLANO DETALHADO — Próximos Passos digiNEV

**Data**: 2026-02-22
**Status**: SALVO PARA EXECUÇÃO FUTURA
**Pré-requisito**: Pipeline 17/17 stages, 0 erros, API Anthropic funcional

---

## PARTE 1: STATUS ATUAL DA API (v6.2)

### 6 Stages com API Híbrida (heurística + Anthropic Claude)
- **API**: Anthropic Claude (claude-sonnet-4-20250514)
- **Modelo**: claude-sonnet-4-20250514
- **Padrão**: Heurística (100%) → confidence score → API para baixa confiança → merge
- **Batch API**: Disponível para datasets >100 msgs (50% desconto, USE_BATCH_API=true)
- **Prompt Caching**: Ativo em chamadas síncronas (90% desconto em input cacheado)

| Stage | Threshold | Resultado (500 rows) |
|-------|-----------|---------------------|
| S06 Affordances | confidence < 0.6 | opinião 6.2%→26.5%, ataque 8.9%→17.1% |
| S08 Político | confidence < 0.4 | neutral 40%→9.7% |
| S11 Topic Modeling | confidence < 0.4 | tópicos nomeados via API |
| S12 Sentimento | confidence < 0.5 | +sarcasmo, +emoções granulares |
| S16 Eventos | confidence < 0.5 | 22 eventos específicos detectados |
| S17 Canais | tipo = 'general' | 100% reclassificados |

### Demais 11 Stages — Heurísticos puros
- scikit-learn (TF-IDF, LDA, K-Means)
- spaCy (NER, POS, lematização)
- Python puro (regex, contagens, estatística)

---

## PARTE 2: STAGES QUE PODEM MELHORAR COM API

### 🔴 ALTA PRIORIDADE (impacto direto na qualidade da análise)

#### A. Stage 08 — Classificação Política (MAIOR IMPACTO POTENCIAL)
**Situação atual**: Classificação por keyword matching (set intersection com léxico)
**Limitações**:
- Mensagens ambíguas ou com linguagem indireta → classificadas como "neutral"
- Ironia e sarcasmo não detectados
- Léxico fixo — não adapta a neologismos ou gírias

**Melhoria com API**:
- Classificar mensagens de baixa intensidade política via API
- Mesmo padrão do Stage 06: heurística primeiro, API para baixa confiança
- Prompt: classificar orientação política + intensidade + contexto
- **Estimativa**: 30-50% das mensagens "neutral" podem ser reclassificadas

**Implementação**:
```python
# Padrão: heurística + API para baixa confiança
political_intensity = df['political_intensity']  # já calculado
low_intensity = political_intensity < 0.3  # candidatas à API
# Enviar low_intensity ao Claude para reclassificação
```
**Esforço**: ~4-6h | **Custo API**: ~$0.50-2.00 por 10k mensagens (Sonnet)

#### B. Stage 12 — Análise Semântica (SENTIMENTO/EMOÇÃO)
**Situação atual**: Sentimento calculado por contagem de palavras positivas/negativas
**Limitações**:
- Léxico de sentimento limitado (português)
- Não capta contexto ou negação ("não é bom" → detecta "bom" como positivo)
- Sem detecção de emoções específicas (raiva, medo, esperança)

**Melhoria com API**:
- Análise de sentimento contextual via Claude
- Detecção de emoções granulares (anger, fear, hope, disgust)
- Detecção de sarcasmo e ironia
- **Estimativa**: Precisão de sentimento pode subir de ~60% para ~85%

**Esforço**: ~3-4h | **Custo API**: ~$0.30-1.50 por 10k mensagens

#### C. Stage 16 — Detecção de Contexto de Evento
**Situação atual**: Detecção por keywords fixas (eleição, protesto, pandemia)
**Limitações**:
- Keywords não captam referências indiretas ("aquilo lá em Brasília")
- Sem identificação de eventos específicos (ex: "8 de janeiro")

**Melhoria com API**:
- Classificar contexto político com compreensão semântica
- Detectar referências a eventos históricos específicos
- **Estimativa**: 20-40% mais eventos detectados

**Esforço**: ~2-3h | **Custo API**: ~$0.20-0.80 por 10k mensagens

---

### 🟡 MÉDIA PRIORIDADE

#### D. Stage 11 — Topic Modeling
**Situação atual**: LDA com CountVectorizer (bag of words)
**Limitações**: LDA é probabilístico e tópicos podem ser incoerentes
**Melhoria com API**: Usar Claude para rotular/nomear tópicos após LDA
- LDA gera clusters → Claude nomeia cada tópico com base nas keywords
- Não substitui LDA, apenas melhora a interpretabilidade
**Esforço**: ~1-2h | **Custo**: Mínimo (1 chamada API por tópico)

#### E. Stage 17 — Análise de Canal
**Situação atual**: Classificação de canais por keywords no nome
**Melhoria com API**: Classificar canais pela amostra de conteúdo
- Enviar 5-10 mensagens representativas de cada canal ao Claude
- Classificar como: propagandístico, informativo, conspiratório, religioso, etc.
**Esforço**: ~2-3h | **Custo**: Mínimo (1 chamada por canal)

---

### 🟢 BAIXA PRIORIDADE (nice-to-have)

#### F. Stage 14 — Network Analysis
- API poderia analisar padrões de coordenação entre mensagens similares
- Baixo impacto vs custo

#### G. Stage 15 — Domain Analysis
- API poderia classificar credibilidade de domínios desconhecidos
- Já tem heurística funcional

---

## PARTE 3: DASHBOARD (PRIORIDADE ALTA)

### 3.1 Corrigir Referências "22 stages" → "17 stages"
**Arquivos afetados**: `src/dashboard/pages/2_🔄_Pipeline.py` e outros
**Esforço**: 30 min

### 3.2 Criar Páginas Faltantes (Stages 15-17)
**Faltam**:
- `15_🌐_Domínios.py` — Visualização de análise de domínios
- `16_📰_Eventos.py` — Contextos de eventos detectados
- `17_📡_Canais.py` — Análise de canais/fontes
**Esforço**: ~5h total (use templates existentes como referência)

### 3.3 Atualizar Pipeline Monitor
**Arquivo**: `2_🔄_Pipeline.py` (timestamp Sep 20, pré-reestruturação)
**Esforço**: ~1-2h

---

## PARTE 4: INFRAESTRUTURA

### 4.1 Dependabot (GitHub Security)
- 8 vulnerabilidades detectadas (1 critical, 6 high, 1 moderate)
- Ativar Dependabot em GitHub Settings → Security
- Criar `.github/dependabot.yml`
**Esforço**: 1h

### 4.2 Modelo API Atualizado ✅
- ~~`claude-3-5-haiku-20241022`~~ → `claude-sonnet-4-20250514` (atualizado hoje)
- `.env` atualizado
- `analyzer.py` default atualizado

### 4.3 CI-CD Security Check
- `safety check` está `continue-on-error: true` → tornar bloqueante
**Esforço**: 30 min

---

## PARTE 5: ANÁLISE DE DADOS (OBJETIVO PRINCIPAL)

### 5.1 Processamento Completo dos Datasets
Com o pipeline validado e API funcional, processar datasets inteiros:

| Dataset | Tamanho | Rows est. | Tempo est. (sem API) | Tempo est. (com API) |
|---------|---------|-----------|---------------------|---------------------|
| 4_elec | 54MB | ~200k | ~10 min | ~2-4h |
| 2_pandemia | 230MB | ~800k | ~40 min | ~8-16h |
| 1_govbolso | 136MB | ~500k | ~25 min | ~5-10h |
| 3_poseleic | 93MB | ~350k | ~17 min | ~3-7h |

**Recomendação**: Processar sem API primeiro (rápido), depois usar API apenas para análise refinada em subsets

### 5.2 Análise Comparativa Cross-Dataset
Com os 4 datasets processados, comparar:
- Evolução da polarização política 2019→2023
- Variação de categorias TCW por período
- Padrões de affordances (ataque, mobilização) por contexto político
- Distribuição de tópicos LDA por período

### 5.3 Exportação para Análise Estatística
- Gerar CSVs consolidados para R/SPSS
- Formato: 1 row por mensagem, todas as 113 colunas
- Filtros por período, canal, orientação política

---

## ORDEM DE EXECUÇÃO RECOMENDADA

### Sprint 1 (imediato — ~8h)
1. ✅ ~~Modelo API atualizado~~ (feito)
2. Dashboard: corrigir "22 stages" (30 min)
3. Dashboard: páginas 15-17 (5h)
4. Dependabot (1h)
5. CI-CD security (30 min)

### Sprint 2 (curto prazo — ~12h) ✅ CONCLUÍDO
6. ✅ ~~Stage 08: API para classificação política~~ (neutral 40%→9.4%)
7. ✅ ~~Stage 12: API para sentimento contextual~~ (+5 emoções granulares + sarcasmo)
8. ✅ ~~Testes de validação com API expandida~~ (200 + 500 rows, 0 erros, 120 colunas)

### Sprint 3 (médio prazo — ~8h) ✅ CONCLUÍDO
9. ✅ ~~Stage 16: API para detecção de contexto~~ (22 eventos detectados em 500 msgs)
10. ✅ ~~Stage 11: API para rotulação de tópicos~~ ("Notícias Políticas Lula" etc.)
11. ✅ ~~Stage 17: API para classificação de canais~~ (100% "general" → classificados)

### Sprint 4 (análise de dados — tempo variável)
12. Processar datasets completos (sem API primeiro)
13. Análise comparativa cross-dataset
14. Exportação para análise estatística em R

---

## CUSTOS ESTIMADOS (API Anthropic)

| Operação | Msgs | Custo est. |
|----------|------|------------|
| Stage 06 (affordances, 10k msgs) | ~7k (70% low conf) | $0.50-2.00 |
| Stage 08 (político, 10k msgs) | ~3k (30% neutral) | $0.30-1.00 |
| Stage 12 (sentimento, 10k msgs) | ~10k (todos) | $0.80-3.00 |
| **Total por 10k mensagens** | | **$1.60-6.00** |
| **Total para 1M mensagens** | | **$160-600** |

**Recomendação de budget**: $50-100 para análise completa com API seletiva (apenas baixa confiança)
