# GUIA DE USO RÁPIDO - digiNEV v.final

**Atualizado:** 04 de outubro de 2025
**Status:** ✅ Sistema operacional e validado

## 🚀 EXECUÇÃO IMEDIATA

### Comando Principal
```bash
python run_pipeline.py
```
> Processa automaticamente todos os 11 datasets encontrados em `data/`

### Teste Rápido (2 minutos)
```bash
python test_clean_analyzer.py
```
> Valida o sistema completo com dados controlados

### Dataset Específico
```bash
python run_pipeline.py --dataset data/controlled_test_100.csv
python run_pipeline.py --dataset data/1_2019-2021-govbolso.csv
```

### Dashboard
```bash
python src/dashboard/start_dashboard.py
```
> Visualização dos resultados processados

## 📊 O QUE ESPERAR

### Resultado do Processamento
```
✅ EXECUTION COMPLETED
⏱️  Total duration: X.Xs
📊 Datasets processed: X
📈 Records processed: X
🔧 Stages executed: 17
📊 Final progress: 100.0%
```

### Colunas Geradas: **102 colunas**
- **Estruturais:** id, body, channel, datetime, etc.
- **Features:** hashtags, URLs, mentions, emojis
- **Qualidade:** dupli_freq, content_quality_score
- **Política:** political_orientation, political_keywords
- **Linguística:** spacy_tokens, lemmatized_text
- **TF-IDF:** tfidf_top_terms, tfidf_score_max
- **Clustering:** cluster_id, cluster_distance
- **Temporal:** hour, day_of_week, month
- **Rede:** coordination_score, temporal_pattern
- **Semântica:** sentiment_polarity, emotion_intensity

## 🔄 PIPELINE EM 4 FASES

### Fase 1: Preparação (Stages 01-02)
```
01. Feature Extraction → Detecta colunas automaticamente
02. Text Preprocessing → Limpa e normaliza texto
```

### Fase 2: Redução Inteligente (Stages 03-06)
```
03. Deduplication → Remove 40-50% duplicatas
04. Statistical Analysis → Análise comparativa
05. Quality Filter → Remove 15-25% baixa qualidade
06. Political Filter → Mantém apenas conteúdo político (30-40% redução)
```

### Fase 3: Linguística Otimizada (Stages 07-09)
```
07. spaCy Processing → Tokens, lemmas, POS, entidades
08. Political Classification → Categorias políticas brasileiras
09. TF-IDF → Vetorização e top termos
```

### Fase 4: Análises Avançadas (Stages 10-17)
```
10. Clustering → K-Means clustering
11. Topic Modeling → LDA topic modeling
12. Semantic Analysis → Análise semântica
13. Temporal Analysis → Padrões temporais
14. Network Analysis → Coordenação de rede
15. Domain Analysis → Análise de domínios
16. Event Context → Contextos políticos
17. Channel Analysis → Análise de canais
```

## 📁 DATASETS DISPONÍVEIS

```
data/
├── controlled_test_100.csv (0.0 MB) ← TESTE VALIDADO
├── 1_2019-2021-govbolso.csv (135.9 MB)
├── 2_2021-2022-pandemia.csv (230.0 MB)
├── 3_2022-2023-poseleic.csv (93.2 MB)
├── 4_2022-2023-elec.csv (54.2 MB)
└── 5_2022-2023-elec-extra.csv (25.2 MB)
```

## ⚡ OTIMIZAÇÕES ATIVAS (100%)

- ✅ **Week 1-2:** Cache inteligente + checkpoints
- ✅ **Week 3:** Processamento paralelo + streaming
- ✅ **Week 4:** Monitoramento em tempo real
- ✅ **Week 5:** Gestão de memória + auto-chunking

## 🔧 LOGS E MONITORAMENTO

### Logs Típicos de Sucesso
```
INFO:Analyzer:🔬 Iniciando análise OTIMIZADA: X registros
INFO:Analyzer:🔍 STAGE 01: Feature Extraction
INFO:Analyzer:✅ Features detectadas: ['hashtags', 'urls', 'mentions']
INFO:Analyzer:🔄 STAGE 03: Cross-Dataset Deduplication
INFO:Analyzer:✅ Deduplicação concluída: X → Y registros (Z% redução)
INFO:Analyzer:🎯 STAGE 05: Content Quality Filter
INFO:Analyzer:✅ Filtro aplicado: X → Y registros (Z% redução)
INFO:Analyzer:✅ Análise OTIMIZADA concluída: 102 colunas, 17 stages
```

### Verificação de Sucesso
```bash
python test_clean_analyzer.py
```
Deve mostrar:
```
✅ TESTE CONCLUÍDO COM SUCESSO!
✅ Analyzer v.final está funcionalmente correto
✅ Pipeline interligado e sem reprocessamento
✅ Apenas dados reais nas colunas geradas
```

## 🚨 RESOLUÇÃO RÁPIDA DE PROBLEMAS

### Erro "No datasets found"
```bash
ls data/*.csv  # Verificar se existem arquivos
```

### Erro "Error tokenizing data"
```bash
# Testar com dataset menor primeiro
python run_pipeline.py --dataset data/controlled_test_100.csv
```

### Erro de memória
- Sistema usa auto-chunking automaticamente
- Configurado para até 4GB RAM

### Pipeline interrompido
- Sistema retoma automaticamente do último checkpoint
- Use `python run_pipeline.py` para continuar

## 📈 ANÁLISE DOS RESULTADOS

### Arquivo de Saída
```
src/dashboard/data/dashboard_results/pipeline_results_YYYYMMDD_HHMMSS.json
```

### Principais Métricas
- **Redução de volume:** ~80% total (300k → 60k registros típico)
- **Colunas geradas:** 102 colunas com dados reais
- **Classificação política:** extrema-direita, direita, centro, esquerda, neutral
- **Quality score:** 0-100 (média ~85 para dados filtrados)
- **Duplicação:** Detectada e quantificada com dupli_freq

### Dashboard
```bash
python src/dashboard/start_dashboard.py
```
- Acesse via navegador (URL mostrada no terminal)
- Visualizações automáticas dos dados processados
- Gráficos de distribuição política, temporal, qualidade

## 🎯 COMANDOS ESSENCIAIS

```bash
# Execução completa
python run_pipeline.py

# Teste rápido
python test_clean_analyzer.py

# Dataset específico
python run_pipeline.py --dataset data/controlled_test_100.csv

# Dashboard
python src/dashboard/start_dashboard.py

# Verificar dados
ls data/*.csv

# Verificar resultados
ls src/dashboard/data/dashboard_results/
```

## 💡 DICAS DE USO

1. **Primeira execução:** Sempre usar `controlled_test_100.csv` para validar
2. **Datasets grandes:** Sistema processa automaticamente em chunks
3. **Interrupção:** Pipeline retoma do último checkpoint
4. **Logs detalhados:** Acompanhar progresso em tempo real
5. **Resultados:** Verificar dashboard para visualizações

---

**Sistema validado e operacional** ✅
**Última validação:** 04/10/2025
**Commit:** d9acb89